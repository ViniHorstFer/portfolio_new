"""
assets.py — Incremental stock/ETF prices and metrics.

Strategy:
  * Cache the wide Adj-Close frame (this IS the app's assets_prices file).
  * Each run, re-download a trailing window for existing tickers and splice it
    over the cache (fresh wins on overlap → absorbs split/dividend restatements).
  * New tickers in assets_list get a full-history backfill (that ticker only).
  * Weekly full rebuild re-anchors every ticker's adjusted history.
  * Output is subset to the current ticker universe so removed tickers drop out.
Adjusted-close download semantics are preserved exactly from the notebook
(auto_adjust=False, actions=False, 'Adj Close').
"""

import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

from . import config
from . import release_io
from .metrics import calculate_stock_metrics

logger = logging.getLogger("pipeline.assets")


def _download_batches(tickers, start, end=None, batch_size=None) -> pd.DataFrame:
    """Download Adj Close for tickers in batches (notebook semantics)."""
    import yfinance as yf
    batch_size = batch_size or config.ASSET_BATCH_SIZE
    tickers = list(tickers)
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info("  assets batch %d (%d tickers)", i // batch_size + 1, len(batch))
        try:
            df = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                timeout=30,
                progress=False,
            )["Adj Close"]
            if isinstance(df, pd.Series):  # single ticker → Series
                df = df.to_frame(name=batch[0])
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            logger.warning("  batch failed %s: %s", batch[:3], e)
            continue
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index()


def _splice(cache: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if cache is None or cache.empty:
        return fresh.sort_index()
    if fresh is None or fresh.empty:
        return cache.sort_index()
    return fresh.combine_first(cache).sort_index()  # fresh wins on overlap


def update_asset_prices(cfg, release, tickers, full_rebuild: bool) -> pd.DataFrame:
    today = datetime.today()
    tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order

    cache = None if full_rebuild else release_io.load_pickle(
        cfg, release, config.APP_ASSETS["assets_prices"]
    )

    if cache is None or cache.empty:
        logger.info("Assets: FULL fetch from %s for %d tickers",
                    config.ASSET_START_DATE, len(tickers))
        prices = _download_batches(tickers, config.ASSET_START_DATE, today)
    else:
        if not isinstance(cache.index, pd.DatetimeIndex):
            cache.index = pd.to_datetime(cache.index, errors="coerce")
            cache = cache[cache.index.notna()]

        existing = [t for t in tickers if t in cache.columns]
        new = [t for t in tickers if t not in cache.columns]

        start = today - timedelta(days=config.TRAILING_CALENDAR_DAYS)
        logger.info("Assets: trailing fetch from %s for %d existing tickers; "
                    "%d new tickers to backfill",
                    start.date(), len(existing), len(new))

        fresh = _download_batches(existing, start, today) if existing else pd.DataFrame()
        prices = _splice(cache, fresh)

        if new:
            new_full = _download_batches(new, config.ASSET_START_DATE, today)
            prices = _splice(prices, new_full)

    # Restrict to the current universe (drops removed tickers, bounds the cache).
    keep = [t for t in tickers if t in prices.columns]
    prices = prices[keep]
    return prices


def run(cfg, release, assets_list: pd.DataFrame, full_rebuild: bool) -> list:
    """Execute the assets stage. Returns the list of output file paths."""
    logger.info("=" * 70)
    logger.info("ASSETS (stocks/ETFs)  (full_rebuild=%s)", full_rebuild)
    logger.info("=" * 70)

    tickers = assets_list.index.tolist()
    prices = update_asset_prices(cfg, release, tickers, full_rebuild)
    logger.info("Asset prices: %s span=%s→%s",
                prices.shape,
                prices.index.min().date() if len(prices) else "—",
                prices.index.max().date() if len(prices) else "—")

    metrics_df = calculate_stock_metrics(prices)
    df_final = metrics_df.join(assets_list, how="left")
    df_final.index.name = "TICKER"

    prices_path = config.SHEETS_DIR / config.APP_ASSETS["assets_prices"]
    metrics_path = config.SHEETS_DIR / config.APP_ASSETS["assets_metrics"]
    joblib.dump(prices, prices_path)
    df_final.to_excel(metrics_path, index=True)

    logger.info("Wrote %s and %s", metrics_path.name, prices_path.name)
    return [metrics_path, prices_path]
