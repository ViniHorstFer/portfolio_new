"""
etf_benchmarks.py — Unified benchmark returns for the ETF system.

Historically the ETF system compared everything against VOO only. This module
merges two benchmark sources into a single {name: daily_return_series} map:

  1. ETF-price-derived benchmarks (USD): VOO, SPY, QQQ, ... taken from the ETF
     price frame (assets_prices) via pct_change.
  2. Fund benchmarks (from benchmarks_data.xlsx, already daily returns):
     CDI, IBOVESPA, SP500, GOLD, USDBRL, BITCOIN.

Optionally it can add FX-adjusted (BRL) variants of the USD benchmarks so a
Brazilian investor can compare an ETF in local-currency terms (USDBRL is one of
the fund benchmarks, so the conversion is available for free).

Every series returned is a *clean* daily-return Series on its own native index
(no injected NaNs). Consumers align on demand (intersection / reindex), which is
what the existing ETF code already does.
"""

import pandas as pd

# ETF tickers we treat as benchmarks when present in the ETF price frame.
ETF_BENCHMARK_TICKERS = ['VOO', 'SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM',
                         'GLD', 'TLT', 'HYG', 'LQD', 'VNQ', 'XLF', 'XLE', 'XLK']

# Fund benchmark columns (as published in benchmarks_data.xlsx — already returns).
FUND_BENCHMARK_COLS = ['CDI', 'IBOVESPA', 'SP500', 'GOLD', 'USDBRL', 'BITCOIN']

# USD-denominated benchmarks eligible for BRL conversion. CDI/IBOVESPA are
# already BRL; USDBRL is the FX rate itself; so those are excluded.
_USD_BENCHMARKS_FOR_FX = set(ETF_BENCHMARK_TICKERS) | {'SP500', 'GOLD', 'BITCOIN'}


def build_benchmark_returns(prices_df, fund_benchmarks_df=None, fx_adjust=False):
    """
    Build the unified benchmark-returns map.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide ETF price frame (date index x tickers).
    fund_benchmarks_df : pd.DataFrame or None
        benchmarks_data.xlsx as loaded (date index; columns include CDI, ...).
        Columns are already DAILY RETURNS.
    fx_adjust : bool
        If True and USDBRL is available, also add '<NAME> (BRL)' variants for the
        USD benchmarks.

    Returns
    -------
    dict[str, pd.Series]  (name -> clean daily-return Series, VOO first if present)
    """
    out = {}

    # 1) ETF-price-derived benchmarks (USD)
    if prices_df is not None:
        for t in ETF_BENCHMARK_TICKERS:
            if t in prices_df.columns:
                s = prices_df[t].dropna().pct_change().dropna()
                if len(s):
                    out[t] = s

    # 2) Fund benchmarks (already returns)
    if fund_benchmarks_df is not None:
        for col in FUND_BENCHMARK_COLS:
            if col in fund_benchmarks_df.columns:
                s = pd.to_numeric(fund_benchmarks_df[col], errors='coerce').dropna()
                if len(s):
                    out[col] = s

    # 3) Optional FX-adjusted (BRL) variants of USD benchmarks
    if fx_adjust and 'USDBRL' in out:
        fx = out['USDBRL']
        for name in list(out.keys()):
            if name in _USD_BENCHMARKS_FOR_FX:
                usd = out[name]
                common = usd.index.intersection(fx.index)
                if len(common) > 1:
                    brl = (1 + usd.reindex(common)) * (1 + fx.reindex(common)) - 1
                    out[f"{name} (BRL)"] = brl.dropna()

    # Order: VOO first (preserves default), then the rest in a stable order.
    ordered = {}
    if 'VOO' in out:
        ordered['VOO'] = out.pop('VOO')
    for k in list(out.keys()):
        ordered[k] = out[k]
    return ordered


def to_benchmark_frame(returns_map):
    """
    Convert the returns map into a DataFrame (outer-joined on the union index)
    for backward-compatible column access like ``benchmarks['VOO']``.

    Consumers reindex to their own return index, so the union-index NaNs are
    harmless (they are dropped on alignment).
    """
    if not returns_map:
        return pd.DataFrame()
    return pd.DataFrame(returns_map).sort_index()


def available_benchmarks(returns_map, exclude=None):
    """Ordered list of benchmark names, optionally excluding one (e.g. the asset itself)."""
    exclude = exclude or set()
    if isinstance(exclude, str):
        exclude = {exclude}
    return [k for k in returns_map.keys() if k not in exclude]
