# Daily Data Pipeline

A single, incremental daily job that refreshes every data file the Streamlit
app consumes and publishes them to a GitHub Release. Designed to run on GitHub
Actions: each run pulls the cached history from the Release, downloads only the
missing tail, recomputes metrics, and re-uploads.

One command does everything:

```bash
python run_daily.py
```

---

## What it produces

Each run uploads these assets to the Release (tag `data` by default):

| Asset on the Release        | Format        | Consumer        |
| --------------------------- | ------------- | --------------- |
| `fund_metrics.xlsx`         | xlsx          | Streamlit app   |
| `funds_info.zip`            | zip → `.pkl`  | Streamlit app   |
| `benchmarks_data.xlsx`      | xlsx          | Streamlit app   |
| `assets_metrics.xlsx`       | xlsx          | Streamlit app   |
| `assets_prices.zip`         | zip → `.pkl`  | Streamlit app   |
| `cache_funds_info_raw.zip`  | zip → `.pkl`  | pipeline cache  |
| `cache_benchmark_prices.zip`| zip → `.pkl`  | pipeline cache  |

The first five are exactly the names and formats your app already reads — the
app, `github_releases.py`, `components.py` and `data_storage.py` need **no
changes**. The last two are private caches the pipeline uses to stay
incremental; the app ignores them. (`assets_prices` doubles as both the app
file and the asset-price cache.)

---

## Repository layout

Add these to your repo. Nothing existing is modified — the pipeline is additive.

```
your-repo/
├── run_daily.py                 # ← single entry point
├── requirements.txt             # ← pinned environment
├── Fund_Guide.xlsx              # ← committed input universe (funds)
├── assets_list.xlsx             # ← committed input universe (stocks/ETFs)
├── .gitignore
├── pipeline/
│   ├── __init__.py
│   ├── config.py                # paths, constants, credential loading
│   ├── sources.py               # CDI (BCB) + CVM month fetcher
│   ├── metrics.py               # your validated metric math (verbatim)
│   ├── release_io.py            # GitHub Release download + publish
│   ├── benchmarks.py            # incremental benchmarks
│   ├── funds.py                 # incremental fund NAVs
│   └── assets.py                # incremental stock/ETF prices
├── .github/workflows/daily.yml  # the scheduler
│
│   # --- your existing app, untouched ---
├── app_v12.py
├── github_releases.py
├── components.py
└── ...
```

`Sheets/` is created at runtime as a working directory and is git-ignored — the
Release, not the repo, holds the data.

### Inputs

* **`Fund_Guide.xlsx`** — 9 columns: `Fundo de Investimento, CNPJ, Categoria
  BTG, Subcategoria BTG, Suitability, Gestor, Tributação, Status, Liquidez`.
* **`assets_list.xlsx`** — `TICKER, Name, Class, Category`.

To track a new fund or ticker, add a row and commit. The job auto-detects it and
backfills that one instrument's full history on the next run (everything else
stays incremental — see below).

---

## GitHub Actions setup

1. **Commit everything above**, including the two `.xlsx` input files.

2. **Permissions.** The workflow already declares `permissions: contents:
   write`, which lets the built-in `GITHUB_TOKEN` create the Release and upload
   assets in this repo. Confirm repo-level Actions permissions allow it:
   *Settings → Actions → General → Workflow permissions →* **Read and write
   permissions**.

   No personal access token is needed for the job. It authenticates with the
   automatic `secrets.GITHUB_TOKEN`.

3. **Release tag (optional).** Defaults to `data`. If yours differs, set a repo
   variable *Settings → Secrets and variables → Actions → Variables →*
   `GH_RELEASE_TAG`.

4. **Your Streamlit app secrets are unchanged.** If the repo is private, the app
   keeps using its existing read token to fetch Release assets; if public, the
   assets are public. Either way the writer (this job) and the reader (the app)
   are independent.

5. **Seed the caches with a first full run.** Go to *Actions → Daily data
   refresh → Run workflow*, tick **full_rebuild = true**, and run it. This cold
   start downloads the full history (72 months of fund NAVs, stock/ETF prices
   from 2015, benchmarks) and writes all seven assets. It is the slow run;
   every subsequent run is incremental and fast.

After that, the daily schedule (`0 10 * * *`, i.e. 07:00 BRT) takes over.

---

## How "incremental" works

The Release is the source of truth for already-retrieved history, so the job is
stateless and safe on ephemeral runners.

* **Benchmarks & stock/ETF prices.** The cached price frame is reused; each run
  re-downloads only a trailing window (~10 sessions) and splices it over the
  cache, with the fresh values winning on overlap. That absorbs yfinance
  adjusted-close restatements (splits/dividends) between rebuilds.

* **Fund NAVs (CVM).** All closed months stay in the cache. Each run re-pulls
  only the **current month + 1 prior month** (current month grows daily; the
  prior month can still be revised), then splices.

* **Weekly full rebuild.** Every **Sunday** the job rebuilds every series from
  scratch, re-anchoring all adjusted history. Force it any day with `--full`.

* **New instrument auto-backfill.** A new CNPJ in `Fund_Guide.xlsx` triggers a
  full-history backfill for *that fund only*; a new ticker in `assets_list.xlsx`
  triggers a full-history download for *that ticker only*. Everything already
  cached stays incremental. Removing a ticker drops it from the next output.

Metrics are always recomputed from the full series each run — only the
*downloads* are incremental, so the numbers are identical to a from-scratch run.

---

## Running locally

Provide credentials via `.streamlit/secrets.toml` (the same file your app uses):

```toml
[github]
token = "ghp_..."        # a PAT with repo/contents access
owner = "your-username"
repo  = "your-repo"
release_tag = "data"
```

Then:

```bash
pip install -r requirements.txt

python run_daily.py                 # incremental (full rebuild on Sundays)
python run_daily.py --full          # force a full rebuild of everything
python run_daily.py --only funds    # run a single stage (repeatable)
python run_daily.py --no-upload     # compute into Sheets/ without publishing
```

Environment variables override the TOML: `GH_TOKEN`, `GH_OWNER`, `GH_REPO`,
`GH_RELEASE_TAG` (or `GITHUB_REPOSITORY="owner/repo"`).

---

## Environment pin

`requirements.txt` pins **pandas 2.2.x** to match your validated environment.
Your metric code uses the legacy `'M'` resample alias (still valid in 2.2.x,
removed in pandas 3.0) while `download_benchmarks.py` uses the newer `'ME'`
alias — only the 2.2.x line supports both, so the math runs byte-for-byte
unchanged. The `'M'` `FutureWarning`s are suppressed at startup, as in your
original scripts.

---

## What changed versus the old scripts

The pipeline consolidates the two daily drivers (`calc_benchmarks_standalone.py`
for benchmarks/funds and `model_total_v2.ipynb` for stocks/ETFs) into one
orchestrator. Two latent bugs from the standalone script were fixed in the
process:

* the GitHub upload block ran at **import time** (outside `__main__`); and
* `load_secrets` / `get_github_config` were defined **after** the code that
  called them.

The notebook's stock/ETF stage previously had **no upload step** — that gap is
closed: `assets_metrics.xlsx` and `assets_prices.zip` are now published like
everything else. All of your performance and risk metric functions are reused
verbatim.
