"""Offline tests: incremental splice logic, compression round-trip, metric smoke."""
import sys, io, zipfile, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import config, release_io, benchmarks, funds, assets, metrics

PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✓ {name}")
    else:    FAIL += 1; print(f"  ✗ {name}  <-- FAILED")

def dates(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")

# ---------------------------------------------------------------------------
print("\n[A] release_io pkl<->zip round trip")
df = pd.DataFrame({"x": [1, 2, 3]}, index=dates(3))
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "thing.pkl"
    joblib.dump(df, p)
    zbytes = release_io._compress_pkl(p)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = zf.namelist()
        inner = joblib.load(io.BytesIO(zf.read(names[0])))
    check("inner filename is thing.pkl", names == ["thing.pkl"])
    check("round-trips identically", inner.equals(df))

# ---------------------------------------------------------------------------
print("\n[B] benchmark splice + build")
cache = pd.DataFrame({"IBOVESPA": [10, 11, 12, 13, 14],
                      "SP500":    [20, 21, 22, 23, 24]}, index=dates(5))
fresh = pd.DataFrame({"IBOVESPA": [999, 998, 50, 51],   # d4,d5 restated; d6,d7 new
                      "SP500":    [999, 998, 60, 61]}, index=dates(4, "2024-01-04"))
spliced = benchmarks._splice(cache, fresh)
check("splice length = 7 (5 cache + 2 new)", len(spliced) == 7)
check("overlap d4 overwritten by fresh", spliced.loc[dates(1,'2024-01-04')[0], "IBOVESPA"] == 999)
check("old d1 retained", spliced.loc[dates(1)[0], "IBOVESPA"] == 10)
check("new d7 appended", spliced.loc[pd.Timestamp('2024-01-07'), "IBOVESPA"] == 51)
check("index sorted", spliced.index.is_monotonic_increasing)

cdi = pd.Series([0.0004]*7, index=dates(7), name="CDI")
bdf = benchmarks.build_benchmarks_df(spliced, cdi)
check("benchmarks_df has CDI col", "CDI" in bdf.columns)
check("benchmarks_df has IBOVESPA col", "IBOVESPA" in bdf.columns)
check("benchmarks_df rows == CDI calendar (7)", len(bdf) == 7)

# ---------------------------------------------------------------------------
print("\n[C] assets incremental update")
# fake price source: returns requested tickers over [start, end] daily
_UNIVERSE = {
    "AAA": 100.0, "BBB": 200.0, "CCC": 300.0,
}
def fake_download(tickers, start, end=None, batch_size=None):
    idx = pd.date_range(pd.Timestamp(start).normalize(),
                        pd.Timestamp(end).normalize() if end else pd.Timestamp.today().normalize(),
                        freq="D")
    data = {}
    for t in tickers:
        base = _UNIVERSE.get(t, 1.0)
        # deterministic ramp so restatement is detectable
        data[t] = np.arange(len(idx), dtype=float) + base
    return pd.DataFrame(data, index=idx)

assets._download_batches = fake_download

# cache: AAA,BBB over a 10-day window ending 5 days ago (so trailing overlaps tail)
today = pd.Timestamp.today().normalize()
cache_idx = pd.date_range(today - pd.Timedelta(days=730), today - pd.Timedelta(days=3), freq="D")
asset_cache = pd.DataFrame({"AAA": np.arange(len(cache_idx)) + 1000.0,
                            "BBB": np.arange(len(cache_idx)) + 2000.0}, index=cache_idx)
assets.release_io.load_pickle = lambda *a, **k: asset_cache.copy()

# universe adds CCC (new) and KEEPS AAA,BBB
prices = assets.update_asset_prices({}, {}, ["AAA", "BBB", "CCC"], full_rebuild=False)
check("CCC backfilled into prices", "CCC" in prices.columns)
check("AAA,BBB,CCC all present", set(prices.columns) == {"AAA", "BBB", "CCC"})
check("trailing dates appended (max == today)", prices.index.max() == today)
check("old cache dates retained (cache start preserved)", cache_idx.min() in prices.index)
# overlap restatement: on overlapping dates the *fresh* (download) value wins.
overlap_day = today - pd.Timedelta(days=3)  # in both cache and trailing fetch
fresh_val = fake_download(["AAA"], today - pd.Timedelta(days=config.TRAILING_CALENDAR_DAYS), today)
check("overlap value comes from fresh, not cache",
      prices.loc[overlap_day, "AAA"] == fresh_val.loc[overlap_day, "AAA"])

# removed ticker drops out
prices2 = assets.update_asset_prices({}, {}, ["AAA", "CCC"], full_rebuild=False)
check("removed ticker BBB dropped", "BBB" not in prices2.columns)

# full rebuild ignores cache
called = {"load": False}
def _flag(*a, **k): called["load"] = True; return asset_cache.copy()
assets.release_io.load_pickle = _flag
prices3 = assets.update_asset_prices({}, {}, ["AAA", "BBB"], full_rebuild=True)
check("full_rebuild does NOT read cache", called["load"] is False)
check("full_rebuild fetches from ASSET_START",
      prices3.index.min() == pd.Timestamp(config.ASSET_START_DATE))

# ---------------------------------------------------------------------------
print("\n[D] funds helpers + incremental update")
check("_refetch_months = current + 1 prior (len 2)", len(funds._refetch_months(today)) == 2)
ml = funds._month_list(today - pd.DateOffset(months=3), today)
check("_month_list inclusive (4 months for 3-month span)", len(ml) == 4)

# dedup keeps max NR_COTST per (cnpj,date)
dd = pd.DataFrame({
    "CNPJ_FUNDO": ["X", "X", "Y"],
    "VL_PATRIM_LIQ": [1, 1, 1], "NR_COTST": [5, 9, 3],
    "VL_QUOTA": [1.0, 1.0, 1.0], "ALAVANCAGEM": [1, 1, 1], "MOVIMENTACAO": [0, 0, 0],
}, index=pd.DatetimeIndex([today, today, today], name="DT_COMPTC"))
ded = funds._dedup(dd)
check("_dedup collapses X to one row", (ded["CNPJ_FUNDO"] == "X").sum() == 1)
check("_dedup keeps max NR_COTST (9)", ded[ded.CNPJ_FUNDO == "X"]["NR_COTST"].iloc[0] == 9)

# incremental: fake CVM fetch keyed by months requested
def fake_cvm(months, cnpjs, log=print):
    months = sorted({int(m) for m in months})
    rows = []
    for m in months:
        d = pd.Timestamp(m // 100, m % 100, 15)
        for c in cnpjs:
            rows.append((d, c, 1.0, 100, 1.0, 1.0, 0.0))
    if not rows:
        from pipeline.sources import NAV_COLUMNS
        return pd.DataFrame(columns=NAV_COLUMNS, index=pd.DatetimeIndex([], name="DT_COMPTC"))
    df = pd.DataFrame(rows, columns=["DT_COMPTC", "CNPJ_FUNDO", "VL_PATRIM_LIQ",
                                     "NR_COTST", "VL_QUOTA", "ALAVANCAGEM", "MOVIMENTACAO"])
    return df.set_index("DT_COMPTC")
funds.fetch_cvm_months = fake_cvm

# cache with funds X,Y across the last 4 months incl current+prior, with a STALE
# current-month NR_COTST that the refetch should overwrite.
refetch = funds._refetch_months(today)
months4 = funds._month_list(today - pd.DateOffset(months=3), today)
cache_rows = []
for m in months4:
    d = pd.Timestamp(m // 100, m % 100, 15)
    for c in ["X", "Y"]:
        nr = 1 if m in refetch else 100  # stale low value in refetch window
        cache_rows.append((d, c, 1.0, nr, 1.0, 1.0, 0.0))
fund_cache = pd.DataFrame(cache_rows, columns=["DT_COMPTC", "CNPJ_FUNDO", "VL_PATRIM_LIQ",
                          "NR_COTST", "VL_QUOTA", "ALAVANCAGEM", "MOVIMENTACAO"]).set_index("DT_COMPTC")
funds.release_io.load_pickle = lambda *a, **k: fund_cache.copy()

# tracked adds new fund Z
navs = funds.update_fund_navs({}, {}, ["X", "Y", "Z"], full_rebuild=False)
check("Z backfilled across older months", navs[navs.CNPJ_FUNDO == "Z"].shape[0] >= 1)
# refetch-window rows replaced by fresh (NR_COTST=100 from fake_cvm, not stale 1)
cur_month = today.year * 100 + today.month
cur_rows = navs[(navs.index.year == today.year) & (navs.index.month == today.month) & (navs.CNPJ_FUNDO == "X")]
check("current-month X present", len(cur_rows) >= 1)
check("refetch overwrote stale NR_COTST (now 100)", cur_rows["NR_COTST"].iloc[0] == 100)
check("older months retained for X,Y", navs[navs.CNPJ_FUNDO == "X"].shape[0] == len(months4))

# trim drops anything older than lookback
old = pd.DataFrame([(today - pd.DateOffset(months=config.FUND_LOOKBACK_MONTHS + 6),
                     "X", 1.0, 1, 1.0, 1.0, 0.0)],
                   columns=["DT_COMPTC", "CNPJ_FUNDO", "VL_PATRIM_LIQ", "NR_COTST",
                            "VL_QUOTA", "ALAVANCAGEM", "MOVIMENTACAO"]).set_index("DT_COMPTC")
trimmed = funds._trim(pd.concat([old, navs]), today)
check("_trim removes beyond lookback window", trimmed.index.min() >= today - pd.DateOffset(months=config.FUND_LOOKBACK_MONTHS))

# ---------------------------------------------------------------------------
print("\n[E] metric functions smoke (verbatim code executes)")
pidx = pd.date_range("2021-01-01", periods=500, freq="B")
rng = np.random.default_rng(0)
sp = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 500)),
                   "BBB": 50 * np.cumprod(1 + rng.normal(0.0002, 0.012, 500))}, index=pidx)
sm = metrics.calculate_stock_metrics(sp)
check("stock metrics returns 2 rows", len(sm) == 2)
check("stock metrics has VaR(95)_D", "VaR(95)_D" in sm.columns)
check("stock metrics has Return 12M", "Return 12M" in sm.columns)

# fund NAV long-frame + guide + cdi
navrows = []
for c in ["11.111.111/1111-11", "22.222.222/2222-22"]:
    q = 1.0
    for d in pidx:
        q *= (1 + rng.normal(0.0003, 0.005))
        navrows.append((d, c, 1e6, 100, q, 1.0, 0.0))
navdf = pd.DataFrame(navrows, columns=["DT_COMPTC", "CNPJ_FUNDO", "VL_PATRIM_LIQ",
                     "NR_COTST", "VL_QUOTA", "ALAVANCAGEM", "MOVIMENTACAO"]).set_index("DT_COMPTC")
guide = pd.DataFrame({
    "Fundo de Investimento": ["Fund A", "Fund B"],
    "CNPJ": ["11.111.111/1111-11", "22.222.222/2222-22"],
    "Categoria BTG": ["Renda Fixa", "Multimercado"], "Subcategoria BTG": ["Ativo", "Macro"],
    "Suitability": ["Moderado", "Sofisticado"], "Gestor": ["G1", "G2"],
    "Tributação": ["Longo Prazo", "Isento"], "Status": ["Sim", "Sim"], "Liquidez": ["D+1", "D+30"],
})
cdi_s = pd.Series(rng.normal(0.0004, 0.0001, 500), index=pidx, name="CDI")
# gross-up no-op vs Isento handling
g = metrics.gross_up_tax_exempt_navs(navdf, guide)
check("gross_up returns same row count", len(g) == len(navdf))
fm = metrics.calculate_fund_metrics(navdf, guide, cdi_s)
check("fund metrics returns >=2 rows", len(fm) >= 2)
check("fund metrics columns uppercased", "VL_PATRIM_LIQ" in fm.columns and "RETURN_12M" in fm.columns)

# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nRESULT: {PASS} passed, {FAIL} failed\n{'='*60}")
sys.exit(1 if FAIL else 0)
