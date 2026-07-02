#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_daily.py — Single entry point for the daily data refresh.

One command rebuilds (incrementally) every data file the Streamlit app consumes
and publishes them to the GitHub Release:

    benchmarks_data.xlsx        assets_metrics.xlsx
    fund_metrics.xlsx           assets_prices.zip   (pkl)
    funds_info.zip   (pkl)
  + cache_funds_info_raw.zip, cache_benchmark_prices.zip   (pipeline caches)

Incremental by default; full rebuild automatically on the configured weekday or
with --full. Each stage publishes its own outputs as soon as it finishes, so a
later failure never blocks earlier-good data.

Usage:
    python run_daily.py                  # incremental (full rebuild on Sundays)
    python run_daily.py --full           # force full rebuild of everything
    python run_daily.py --only funds     # run a single stage (repeatable)
    python run_daily.py --no-upload      # compute locally, skip publishing
"""

import sys
import time
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline.run_daily")


def parse_args():
    p = argparse.ArgumentParser(description="Daily portfolio data refresh.")
    p.add_argument("--full", action="store_true",
                   help="Force a full rebuild of every series.")
    p.add_argument("--only", action="append", choices=["benchmarks", "funds", "assets"],
                   help="Run only the given stage(s). Repeatable.")
    p.add_argument("--no-upload", action="store_true",
                   help="Compute files locally but do not publish to GitHub.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from pipeline import config
    from pipeline import release_io
    from pipeline import benchmarks as bench_stage
    from pipeline import funds as funds_stage
    from pipeline import assets as assets_stage
    from pipeline.sources import baixar_cdi

    import pandas as pd

    stages = args.only or ["benchmarks", "funds", "assets"]
    full_rebuild = args.full or config.is_full_rebuild_day()
    t0 = time.time()

    logger.info("Daily refresh — stages=%s  full_rebuild=%s  upload=%s",
                stages, full_rebuild, not args.no_upload)

    cfg = config.load_github_config()
    logger.info("GitHub: %s/%s  release='%s'", cfg["owner"], cfg["repo"], cfg["release_tag"])
    release = release_io.get_or_create_release(cfg)

    # CDI is shared by the benchmark calendar join and the synthetic CDI fund.
    cdi_series = None
    if {"benchmarks", "funds"} & set(stages):
        logger.info("Fetching CDI (BCB SGS)…")
        cdi_series = baixar_cdi(config.BENCHMARK_LOOKBACK_MONTHS)[0]
        logger.info("CDI: %d obs, %s → %s",
                    len(cdi_series), cdi_series.index.min().date(),
                    cdi_series.index.max().date())

    failures = []

    def execute(name, fn):
        try:
            outputs = fn()
            if not args.no_upload:
                logger.info("Publishing %s outputs…", name)
                release_io.publish(cfg, release, outputs)
            return True
        except Exception:
            logger.exception("Stage '%s' FAILED", name)
            failures.append(name)
            return False

    if "benchmarks" in stages:
        execute("benchmarks",
                lambda: bench_stage.run(cfg, release, cdi_series, full_rebuild))

    if "funds" in stages:
        fund_guide = pd.read_excel(config.FUND_GUIDE_PATH)
        execute("funds",
                lambda: funds_stage.run(cfg, release, fund_guide, cdi_series, full_rebuild))

    if "assets" in stages:
        assets_list = pd.read_excel(config.ASSETS_LIST_PATH, index_col="TICKER")
        execute("assets",
                lambda: assets_stage.run(cfg, release, assets_list, full_rebuild))

    elapsed = time.time() - t0
    if failures:
        logger.error("DONE with failures in: %s  (%.1fs)", failures, elapsed)
        return 1
    logger.info("DONE — all stages succeeded (%.1fs)", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
