"""
benchmarks.py — Incremental benchmark series.

Strategy: cache the RAW benchmark prices (the only expensive part). Each run
re-downloads just a trailing window and splices it in (absorbing Adj-Close
restatements); a weekly full rebuild re-anchors everything. Returns and the
CDI-calendar join are then recomputed cheaply in memory from the full series —
exactly the sequence in the original download_all_benchmarks().
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config
from . import release_io

logger = logging.getLogger("pipeline.benchmarks")


def _fetch_prices(start, end):
    """Download benchmark Close prices and rename to display names."""
    import yfinance as yf
    tickers = config.BENCHMARK_TICKERS
    raw = yf.download(
        list(tickers.keys()),
        interval="1d",
        start=start,
        end=end,
        progress=False,
    )["Close"]
    if isinstance(raw, pd.Series):  # single ticker edge case
        raw = raw.to_frame()
    raw = raw.rename(columns=tickers)
    raw = raw.replace(0, np.nan).sort_index()
    # keep only known display columns that came back
    keep = [c for c in tickers.values() if c in raw.columns]
    return raw[keep]


def _splice(cache: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Splice fresh prices over the cache; fresh wins on overlapping cells."""
    if cache is None or cache.empty:
        return fresh.sort_index()
    combined = fresh.combine_first(cache)  # caller (fresh) wins on overlap
    return combined.sort_index()


def update_benchmark_prices(cfg, release, full_rebuild: bool) -> pd.DataFrame:
    """Return an up-to-date raw benchmark price frame (and refresh the cache)."""
    today = datetime.today()
    cache = None if full_rebuild else release_io.load_pickle(
        cfg, release, config.CACHE_ASSETS["benchmark_prices"]
    )

    if cache is None or cache.empty:
        start = today - pd.DateOffset(months=config.BENCHMARK_LOOKBACK_MONTHS)
        logger.info("Benchmarks: FULL fetch from %s", start.date())
        prices = _fetch_prices(start, today)
    else:
        start = today - timedelta(days=config.TRAILING_CALENDAR_DAYS)
        logger.info("Benchmarks: trailing fetch from %s (cache through %s)",
                    start.date(), cache.index.max().date())
        fresh = _fetch_prices(start, today)
        prices = _splice(cache, fresh)

    return prices


def build_benchmarks_df(prices: pd.DataFrame, cdi_series: pd.Series) -> pd.DataFrame:
    """
    Align benchmark prices, compute returns, and join onto the CDI calendar.
    Mirrors download_all_benchmarks() steps 1–3.
    """
    # Step 1: forward/back-fill across benchmarks (previous-tick alignment)
    aligned = prices.ffill().bfill()
    # Step 2: returns on aligned prices
    benchmark_returns = aligned.pct_change().iloc[1:]
    # Step 3: left-join onto CDI dates
    cdi_df = pd.DataFrame(cdi_series).rename(columns={cdi_series.name or 0: "CDI"})
    cdi_df.columns = ["CDI"]
    benchmarks_df = cdi_df.join(benchmark_returns, how="left")
    return benchmarks_df


def run(cfg, release, cdi_series, full_rebuild: bool) -> list:
    """Execute the benchmark stage. Returns the list of output file paths."""
    logger.info("=" * 70)
    logger.info("BENCHMARKS  (full_rebuild=%s)", full_rebuild)
    logger.info("=" * 70)

    prices = update_benchmark_prices(cfg, release, full_rebuild)
    logger.info("Benchmark prices: %s, %s → %s",
                prices.shape, prices.index.min().date(), prices.index.max().date())

    benchmarks_df = build_benchmarks_df(prices, cdi_series)

    # Persist cache + app file
    cache_path = config.SHEETS_DIR / config.CACHE_ASSETS["benchmark_prices"]
    out_path = config.SHEETS_DIR / config.APP_ASSETS["benchmarks"]
    import joblib
    joblib.dump(prices, cache_path)
    benchmarks_df.to_excel(out_path)
    logger.info("Wrote %s and %s", out_path.name, cache_path.name)

    return [out_path, cache_path]
