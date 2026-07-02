"""
config.py — Central configuration for the daily data pipeline.

Credentials are read from environment variables first (GitHub Actions secrets),
then fall back to .streamlit/secrets.toml for local runs. Everything else lives
here as a tunable constant so the pipeline has a single control panel.
"""

import os
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SHEETS_DIR = REPO_ROOT / "Sheets"               # generated outputs live here
FUND_GUIDE_PATH = REPO_ROOT / "Fund_Guide.xlsx"  # committed input universe (funds)
ASSETS_LIST_PATH = REPO_ROOT / "assets_list.xlsx"  # committed input universe (stocks/ETFs)

SHEETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Incremental-update policy
# ---------------------------------------------------------------------------
# How much history to keep. Funds use a rolling month window; assets/benchmarks
# anchor to a fixed start date.
FUND_LOOKBACK_MONTHS = 72
ASSET_START_DATE = "2015-01-01"
BENCHMARK_LOOKBACK_MONTHS = 72

# Trailing window re-downloaded & spliced every run to absorb yfinance
# Adj-Close restatements (splits/dividends) between full rebuilds.
TRAILING_SESSIONS = 10
TRAILING_CALENDAR_DAYS = 18  # ~10 trading sessions, with slack for weekends/holidays

# CVM: re-pull the current month + this many prior months every run
# (current month grows daily; prior months can be revised).
CVM_REFETCH_PRIOR_MONTHS = 1

# Weekly full rebuild (Mon=0 ... Sun=6). On this weekday every series is
# rebuilt from scratch, re-anchoring all adjusted prices.
FULL_REBUILD_WEEKDAY = 6  # Sunday

# yfinance batch size for asset downloads
ASSET_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Benchmark universe (yfinance ticker -> display name)
# ---------------------------------------------------------------------------
BENCHMARK_TICKERS = {
    '^BVSP': 'IBOVESPA',
    '^GSPC': 'SP500',
    'GLD': 'GOLD',
    'BRL=X': 'USDBRL',
    'BTC-USD': 'BITCOIN',
}

# Synthetic "CDI fund" injected into the fund universe so the app can treat the
# risk-free benchmark like any other fund.
CDI_CNPJ = '00.000.000/0000-00'

# ---------------------------------------------------------------------------
# Release asset names
# ---------------------------------------------------------------------------
# Files the Streamlit app consumes (names/formats MUST NOT change).
APP_ASSETS = {
    'fund_metrics':  'fund_metrics.xlsx',
    'funds_info':    'funds_info.pkl',        # uploaded as funds_info.zip
    'benchmarks':    'benchmarks_data.xlsx',
    'assets_metrics':'assets_metrics.xlsx',
    'assets_prices': 'assets_prices.pkl',      # uploaded as assets_prices.zip
}

# Raw caches used only by this pipeline to enable incremental updates.
# (The app ignores these.) assets_prices doubles as both app asset AND cache.
CACHE_ASSETS = {
    'funds_nav_raw':      'cache_funds_info_raw.pkl',   # pre-gross-up long NAVs
    'benchmark_prices':   'cache_benchmark_prices.pkl', # raw benchmark prices
}


# ---------------------------------------------------------------------------
# GitHub configuration
# ---------------------------------------------------------------------------
def load_github_config() -> dict:
    """
    Resolve GitHub config. Priority:
      1. Environment variables (GitHub Actions):
         GH_TOKEN / GITHUB_TOKEN, GH_OWNER, GH_REPO, GH_RELEASE_TAG
         (GH_OWNER/GH_REPO may also come from GITHUB_REPOSITORY="owner/repo")
      2. .streamlit/secrets.toml  [github] token/owner/repo/release_tag
    """
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN', '')
    owner = os.environ.get('GH_OWNER', '')
    repo = os.environ.get('GH_REPO', '')
    tag = os.environ.get('GH_RELEASE_TAG', '')

    # GitHub Actions exposes "owner/repo" in GITHUB_REPOSITORY
    gh_repository = os.environ.get('GITHUB_REPOSITORY', '')
    if gh_repository and (not owner or not repo):
        parts = gh_repository.split('/', 1)
        if len(parts) == 2:
            owner = owner or parts[0]
            repo = repo or parts[1]

    # Local fallback: .streamlit/secrets.toml
    if not all([token, owner, repo]):
        secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                import toml
                gh = toml.load(secrets_path).get('github', {})
                token = token or gh.get('token', '')
                owner = owner or gh.get('owner', '')
                repo = repo or gh.get('repo', '')
                tag = tag or gh.get('release_tag', '')
            except Exception:
                pass

    cfg = {
        'token': token,
        'owner': owner,
        'repo': repo,
        'release_tag': tag or 'data',
    }

    missing = [k for k in ('token', 'owner', 'repo') if not cfg[k]]
    if missing:
        raise RuntimeError(
            "Missing GitHub config: " + ", ".join(missing) +
            ". Set GH_TOKEN/GH_OWNER/GH_REPO (or GITHUB_REPOSITORY) env vars, "
            "or provide .streamlit/secrets.toml."
        )
    return cfg


def is_full_rebuild_day(today: datetime = None) -> bool:
    today = today or datetime.today()
    return today.weekday() == FULL_REBUILD_WEEKDAY
