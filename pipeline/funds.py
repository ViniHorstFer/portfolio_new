"""
funds.py — Incremental investment-fund NAVs and metrics.

Strategy:
  * Cache the RAW concatenated NAV long-frame (pre-gross-up, real funds only).
  * Every run, re-pull ONLY the current month + N prior months for all tracked
    funds (current month grows daily; prior months can be revised by CVM).
  * If a NEW CNPJ appears in Fund_Guide, backfill that fund's full history over
    the older months only — the rest stays incremental.
  * Trim the cache to the rolling lookback window so it stays bounded.
Gross-up of tax-exempt funds, the synthetic CDI "fund", and the metric math are
preserved exactly from the original standalone script.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from . import config
from . import release_io
from .sources import fetch_cvm_months, NAV_COLUMNS
from .metrics import gross_up_tax_exempt_navs, calculate_fund_metrics

logger = logging.getLogger("pipeline.funds")


def _month_list(start_dt, end_dt) -> list:
    """Inclusive list of YYYYMM ints between two dates."""
    months = []
    cur = pd.Timestamp(start_dt.year, start_dt.month, 1)
    end = pd.Timestamp(end_dt.year, end_dt.month, 1)
    while cur <= end:
        months.append(cur.year * 100 + cur.month)
        cur += pd.DateOffset(months=1)
    return months


def _refetch_months(today) -> list:
    """Current month + N prior months."""
    months = []
    for k in range(config.CVM_REFETCH_PRIOR_MONTHS + 1):
        m = pd.Timestamp(today.year, today.month, 1) - pd.DateOffset(months=k)
        months.append(m.year * 100 + m.month)
    return sorted(months)


def _index_yyyymm(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year * 100 + idx.month


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (CNPJ_FUNDO, date); keep the obs with the most cotistas."""
    if df.empty:
        return df
    out = df.reset_index()
    date_col = out.columns[0]
    out = out.sort_values("NR_COTST", ascending=False)
    out = out.drop_duplicates(subset=["CNPJ_FUNDO", date_col], keep="first")
    out = out.set_index(date_col)
    out.index.name = "DT_COMPTC"
    return out.sort_index()


def _trim(df: pd.DataFrame, today) -> pd.DataFrame:
    cutoff = pd.Timestamp(today) - pd.DateOffset(months=config.FUND_LOOKBACK_MONTHS)
    return df[df.index >= cutoff]


def update_fund_navs(cfg, release, tracked_cnpjs, full_rebuild: bool) -> pd.DataFrame:
    """Return up-to-date RAW NAVs (pre-gross-up, real funds only)."""
    today = datetime.today()
    all_months = _month_list(today - pd.DateOffset(months=config.FUND_LOOKBACK_MONTHS), today)
    refetch = _refetch_months(today)

    cache = None if full_rebuild else release_io.load_pickle(
        cfg, release, config.CACHE_ASSETS["funds_nav_raw"]
    )

    if cache is None or cache.empty:
        logger.info("Funds: FULL fetch over %d months", len(all_months))
        navs = fetch_cvm_months(all_months, tracked_cnpjs, log=logger.info)
        return _trim(_dedup(navs), today)

    # Ensure cache has a proper datetime index
    if not isinstance(cache.index, pd.DatetimeIndex):
        cache.index = pd.to_datetime(cache.index, errors="coerce")
        cache = cache[cache.index.notna()]
    cache.index.name = "DT_COMPTC"

    cached_cnpjs = set(cache["CNPJ_FUNDO"].unique())
    new_cnpjs = [c for c in tracked_cnpjs if c not in cached_cnpjs]
    logger.info("Funds: incremental. refetch months=%s, new funds=%d",
                refetch, len(new_cnpjs))

    # 1) Re-pull the refetch window for ALL tracked funds.
    fresh_recent = fetch_cvm_months(refetch, tracked_cnpjs, log=logger.info)

    # 2) Backfill new funds over the older months only.
    older = [m for m in all_months if m not in refetch]
    if new_cnpjs and older:
        logger.info("Funds: backfilling %d new funds over %d older months",
                    len(new_cnpjs), len(older))
        fresh_new = fetch_cvm_months(older, new_cnpjs, log=logger.info)
    else:
        fresh_new = pd.DataFrame(columns=NAV_COLUMNS,
                                 index=pd.DatetimeIndex([], name="DT_COMPTC"))

    # 3) Drop the refetch-window rows from the cache (replaced wholesale).
    cache_yyyymm = _index_yyyymm(cache.index)
    cache_keep = cache[~np.isin(cache_yyyymm, refetch)]

    combined = pd.concat([cache_keep, fresh_recent, fresh_new])
    combined = _dedup(combined)
    combined = _trim(combined, today)
    return combined


def run(cfg, release, fund_guide: pd.DataFrame, cdi_series: pd.Series,
        full_rebuild: bool) -> list:
    """Execute the funds stage. Returns the list of output file paths."""
    logger.info("=" * 70)
    logger.info("FUNDS  (full_rebuild=%s)", full_rebuild)
    logger.info("=" * 70)

    tracked = fund_guide["CNPJ"].dropna().astype(str).unique().tolist()
    logger.info("Tracked funds: %d", len(tracked))

    navs_raw = update_fund_navs(cfg, release, tracked, full_rebuild)
    logger.info("Raw NAVs: %s funds=%d span=%s→%s",
                navs_raw.shape, navs_raw["CNPJ_FUNDO"].nunique(),
                navs_raw.index.min(), navs_raw.index.max())

    # Persist the raw cache BEFORE downstream transforms.
    cache_path = config.SHEETS_DIR / config.CACHE_ASSETS["funds_nav_raw"]
    joblib.dump(navs_raw, cache_path)

    # --- Gross-up tax-exempt funds (verbatim logic) ---
    navs_adjusted = gross_up_tax_exempt_navs(navs_raw, fund_guide)

    # --- Synthetic CDI "fund" (verbatim construction) ---
    cdi_quota_df = pd.DataFrame(columns=NAV_COLUMNS)
    cdi_quota_df["VL_QUOTA"] = cdi_series.add(1).cumprod()
    cdi_quota_df["CNPJ_FUNDO"] = config.CDI_CNPJ
    cdi_quota_df["VL_PATRIM_LIQ"] = 0
    cdi_quota_df["NR_COTST"] = 0
    cdi_quota_df["ALAVANCAGEM"] = 0
    cdi_quota_df["MOVIMENTACAO"] = 0

    navs_adjusted = pd.concat([navs_adjusted, cdi_quota_df], ignore_index=False)

    # App file: adjusted NAVs + synthetic CDI
    funds_info_path = config.SHEETS_DIR / config.APP_ASSETS["funds_info"]
    joblib.dump(navs_adjusted, funds_info_path)

    # --- Metrics (verbatim) ---
    guide_with_cdi = fund_guide.copy()
    cdi_row = {
        "Fundo de Investimento": "RENDA FIXA CDI",
        "CNPJ": config.CDI_CNPJ,
        "Categoria BTG": "Renda Fixa",
        "Subcategoria BTG": "-",
        "Suitability": "Conservador",
        "Gestor": "-",
        "Tributação": "Longo Prazo",
        "Status": "Sim",
        "Liquidez": "D+0",
    }
    guide_with_cdi.loc[len(guide_with_cdi)] = [cdi_row.get(c, "-") for c in guide_with_cdi.columns]

    result_df = calculate_fund_metrics(navs_adjusted, guide_with_cdi, cdi_series)
    metrics_path = config.SHEETS_DIR / config.APP_ASSETS["fund_metrics"]
    result_df.to_excel(metrics_path, index=False)

    logger.info("Wrote %s, %s, %s",
                metrics_path.name, funds_info_path.name, cache_path.name)
    return [metrics_path, funds_info_path, cache_path]
