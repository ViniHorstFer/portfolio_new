"""
sources.py — Raw external data fetchers (BCB CDI + CVM daily fund NAVs).

baixar_cdi() is copied VERBATIM from download_benchmarks.py. The CVM monthly
fetcher (fetch_cvm_months) is a focused rewrite of baixar_dados_cvm() that pulls
a SPECIFIC set of YYYYMM months instead of a rolling period — this is what makes
incremental updates possible.
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
from dateutil.relativedelta import relativedelta

def baixar_cdi(period):

    serie_codigo = 12
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie_codigo}/dados"

    start_date = datetime.today() - relativedelta(months=period)
    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = datetime.today().strftime('%d/%m/%Y')
    
    params = {
        'formato': 'csv',
        'dataInicial': start_date_str,
        'dataFinal': end_date_str
    }
    headers = {'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    
    if response.status_code == 200:
        data = StringIO(response.text)
        cdi_df = pd.read_csv(data, sep=";", decimal=",", encoding="latin1")
        
        # Garantindo que a coluna 'valor' seja tratada como string antes de substituir ',' por '.'
        cdi_df['valor'] = pd.to_numeric(cdi_df['valor'].astype(str).str.replace(',', '.', regex=False), errors='coerce') / 100
        
        # Convertendo para datetime
        cdi_df['data'] = pd.to_datetime(cdi_df['data'], format='%d/%m/%Y')
        
        # Definindo 'data' como índice
        cdi_df.set_index('data', inplace=True)

        cdi_df['CDI_ACUM'] = 0
        cdi_df['CDI_ACUM'].iloc[1:] = ((1 + cdi_df['valor'].iloc[1:]).cumprod() - 1) * 100

        cdi_df.rename(columns={'valor': 'CDI'}, inplace=True)

        cdi_diario = cdi_df['CDI']

        cdi_mensal = cdi_df['CDI'].resample('ME').apply(lambda x: (1 + x).prod() - 1)

        return cdi_diario, cdi_mensal
    else:
        print(f"Erro ao acessar a API: {response.status_code}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CVM daily NAV fetcher (incremental: fetch a specific set of months)
# ---------------------------------------------------------------------------

CVM_BASE_CURRENT = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
CVM_BASE_HIST = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/HIST/"

NAV_COLUMNS = ['CNPJ_FUNDO', 'VL_PATRIM_LIQ', 'NR_COTST', 'VL_QUOTA',
               'ALAVANCAGEM', 'MOVIMENTACAO']

_CVM_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
}


def _parse_cvm_csv(raw_bytes, yyyymm, cnpjs):
    """Parse one monthly CVM CSV (bytes) -> filtered DataFrame or None."""
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=";", decimal=",",
                     encoding="latin1", low_memory=False)
    # CVM renamed the CNPJ column from 202312 onward (class structure reform).
    cnpj_col = 'CNPJ_FUNDO' if int(yyyymm) <= 202311 else 'CNPJ_FUNDO_CLASSE'
    if cnpj_col not in df.columns:
        # fall back to whichever exists
        cnpj_col = next((c for c in ('CNPJ_FUNDO', 'CNPJ_FUNDO_CLASSE')
                         if c in df.columns), None)
        if cnpj_col is None:
            return None
    df = df[df[cnpj_col].isin(cnpjs)]
    if df.empty:
        return None
    if cnpj_col != 'CNPJ_FUNDO':
        df = df.rename(columns={cnpj_col: 'CNPJ_FUNDO'})
    return df


def fetch_cvm_months(months, cnpjs, log=print):
    """
    Download CVM daily-NAV data for a SPECIFIC list of months.

    Parameters
    ----------
    months : iterable[int]
        Months as YYYYMM integers (e.g. [202505, 202506]).
    cnpjs : iterable[str]
        Fund CNPJs to keep.
    log : callable
        Logging function.

    Returns
    -------
    pd.DataFrame indexed by DT_COMPTC with columns NAV_COLUMNS (may be empty).
    """
    cnpjs = list(cnpjs)
    months = sorted({int(m) for m in months})
    frames = []
    hist_year_cache = {}  # year -> {yyyymm: csv_bytes} for annual HIST zips

    for yyyymm in months:
        year = yyyymm // 100
        is_hist = year <= 2020
        try:
            if is_hist:
                if year not in hist_year_cache:
                    url = f"{CVM_BASE_HIST}inf_diario_fi_{year}.zip"
                    log(f"  CVM HIST {year}: {url}")
                    r = requests.get(url, headers=_CVM_HEADERS, timeout=120)
                    if r.status_code != 200 or len(r.content) < 1000:
                        log(f"  ! HIST {year} unavailable (status {r.status_code})")
                        hist_year_cache[year] = {}
                    else:
                        bucket = {}
                        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                            for nm in zf.namelist():
                                if nm.endswith('.csv') and 'inf_diario_fi_' in nm:
                                    key = nm.split('inf_diario_fi_')[-1].replace('.csv', '')
                                    bucket[key] = zf.read(nm)
                        hist_year_cache[year] = bucket
                raw = hist_year_cache[year].get(str(yyyymm))
                if raw is None:
                    log(f"  ! {yyyymm} not in HIST {year} archive")
                    continue
                parsed = _parse_cvm_csv(raw, yyyymm, cnpjs)
                if parsed is not None and not parsed.empty:
                    frames.append(parsed)
            else:
                url = f"{CVM_BASE_CURRENT}inf_diario_fi_{yyyymm}.zip"
                log(f"  CVM {yyyymm}: {url}")
                r = requests.get(url, headers=_CVM_HEADERS, timeout=90)
                if r.status_code != 200 or len(r.content) < 1000:
                    log(f"  ! {yyyymm} unavailable (status {r.status_code})")
                    continue
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    for nm in zf.namelist():
                        if nm.endswith('.csv'):
                            parsed = _parse_cvm_csv(zf.read(nm), yyyymm, cnpjs)
                            if parsed is not None and not parsed.empty:
                                frames.append(parsed)
        except requests.exceptions.Timeout:
            log(f"  ! timeout on {yyyymm}")
        except zipfile.BadZipFile:
            log(f"  ! corrupt zip on {yyyymm}")
        except Exception as e:  # noqa
            log(f"  ! error on {yyyymm}: {e}")

    if not frames:
        return pd.DataFrame(columns=NAV_COLUMNS,
                            index=pd.DatetimeIndex([], name='DT_COMPTC'))

    combined = pd.concat(frames, ignore_index=True)
    combined.drop(columns=['ID_SUBCLASSE'], errors='ignore', inplace=True)

    for col in ['VL_PATRIM_LIQ', 'NR_COTST', 'VL_TOTAL', 'VL_QUOTA',
                'CAPTC_DIA', 'RESG_DIA']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    if {'VL_TOTAL', 'VL_PATRIM_LIQ'}.issubset(combined.columns):
        combined['ALAVANCAGEM'] = combined['VL_TOTAL'] / combined['VL_PATRIM_LIQ']
    else:
        combined['ALAVANCAGEM'] = np.nan

    if {'CAPTC_DIA', 'RESG_DIA'}.issubset(combined.columns):
        combined['MOVIMENTACAO'] = combined['CAPTC_DIA'] - combined['RESG_DIA']
    else:
        combined['MOVIMENTACAO'] = np.nan

    combined['DT_COMPTC'] = pd.to_datetime(combined['DT_COMPTC'], errors='coerce')
    combined = combined.dropna(subset=['DT_COMPTC'])
    combined = combined.set_index('DT_COMPTC')

    for col in NAV_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined[NAV_COLUMNS]
