"""
release_io.py — GitHub Release as the cache store AND publish target.

The Release is the single source of truth for already-retrieved history, which
makes the pipeline stateless (safe for ephemeral GitHub Actions runners).

Conventions preserved from the app's github_releases.py:
  * .pkl files are uploaded as "<stem>.zip" containing the original "<stem>.pkl"
  * .xlsx files are uploaded as-is
The Streamlit app's loaders depend on these exact names, so they must not change.
"""

import io
import time
import zipfile
import logging
from pathlib import Path

import requests
import joblib
import pandas as pd

logger = logging.getLogger("pipeline.release_io")

GITHUB_API = "https://api.github.com"
GITHUB_UPLOADS = "https://uploads.github.com"


def _headers(token: str, accept: str = "application/vnd.github+json") -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ---------------------------------------------------------------------------
# Release lifecycle
# ---------------------------------------------------------------------------
def get_or_create_release(cfg: dict) -> dict:
    """Fetch the release by tag, creating it if absent."""
    owner, repo, tag = cfg["owner"], cfg["repo"], cfg["release_tag"]
    url = f"{GITHUB_API}/repos/{owner}/{repo}/releases/tags/{tag}"
    r = requests.get(url, headers=_headers(cfg["token"]), timeout=30)

    if r.status_code == 200:
        logger.info("Found release '%s'", tag)
        return r.json()
    if r.status_code != 404:
        raise RuntimeError(f"GitHub release lookup failed: {r.status_code} {r.text}")

    logger.info("Release '%s' not found — creating it", tag)
    create_url = f"{GITHUB_API}/repos/{owner}/{repo}/releases"
    payload = {
        "tag_name": tag,
        "name": f"Data Release — {tag}",
        "body": "Automated data release for the portfolio analytics platform.",
        "draft": False,
        "prerelease": False,
    }
    r = requests.post(create_url, headers=_headers(cfg["token"]), json=payload, timeout=30)
    if r.status_code != 201:
        raise RuntimeError(f"Failed to create release: {r.status_code} {r.text}")
    return r.json()


def refresh_release(cfg: dict, release: dict) -> dict:
    """Re-fetch the release to get an up-to-date asset list."""
    owner, repo = cfg["owner"], cfg["repo"]
    url = f"{GITHUB_API}/repos/{owner}/{repo}/releases/{release['id']}"
    r = requests.get(url, headers=_headers(cfg["token"]), timeout=30)
    if r.status_code == 200:
        return r.json()
    return release


def _find_asset(release: dict, name: str):
    for asset in release.get("assets", []):
        if asset["name"] == name:
            return asset
    return None


# ---------------------------------------------------------------------------
# Download (cache reads)
# ---------------------------------------------------------------------------
def _download_asset_bytes(cfg: dict, release: dict, asset_name: str):
    asset = _find_asset(release, asset_name)
    if asset is None:
        return None
    r = requests.get(
        asset["url"],
        headers=_headers(cfg["token"], accept="application/octet-stream"),
        timeout=180,
    )
    if r.status_code == 200:
        return r.content
    logger.warning("Download of %s failed: %s", asset_name, r.status_code)
    return None


def load_pickle(cfg: dict, release: dict, pkl_name: str):
    """
    Load a pickled object stored as '<stem>.zip'. `pkl_name` is the logical
    .pkl name (e.g. 'assets_prices.pkl'); we look up '<stem>.zip' on the release.
    Returns the object, or None if not present.
    """
    zip_name = Path(pkl_name).stem + ".zip"
    content = _download_asset_bytes(cfg, release, zip_name)
    if content is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            if not names:
                return None
            pkl_bytes = zf.read(names[0])
        return joblib.load(io.BytesIO(pkl_bytes))
    except Exception as e:
        logger.warning("Could not unpickle %s: %s", zip_name, e)
        return None


def load_excel(cfg: dict, release: dict, xlsx_name: str, **read_kwargs):
    content = _download_asset_bytes(cfg, release, xlsx_name)
    if content is None:
        return None
    try:
        return pd.read_excel(io.BytesIO(content), **read_kwargs)
    except Exception as e:
        logger.warning("Could not read %s: %s", xlsx_name, e)
        return None


# ---------------------------------------------------------------------------
# Upload (publish)
# ---------------------------------------------------------------------------
def _compress_pkl(path: Path) -> bytes:
    buf = io.BytesIO()
    with open(path, "rb") as f:
        content = f.read()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name, content)  # inner name stays "<stem>.pkl"
    buf.seek(0)
    return buf.read()


def _delete_asset(cfg: dict, asset_id: int):
    owner, repo = cfg["owner"], cfg["repo"]
    url = f"{GITHUB_API}/repos/{owner}/{repo}/releases/assets/{asset_id}"
    requests.delete(url, headers=_headers(cfg["token"]), timeout=30)


def upload_path(cfg: dict, release: dict, path: Path, retries: int = 2) -> str:
    """Upload one file, replacing an existing asset of the same name."""
    path = Path(path)
    if path.suffix == ".pkl":
        data = _compress_pkl(path)
        upload_name = path.stem + ".zip"
        content_type = "application/zip"
    elif path.suffix == ".xlsx":
        data = path.read_bytes()
        upload_name = path.name
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise ValueError(f"Unsupported upload type: {path.suffix}")

    # Delete existing asset of the same name (assets are immutable by name).
    existing = _find_asset(release, upload_name)
    if existing:
        _delete_asset(cfg, existing["id"])

    upload_url = release["upload_url"].replace("{?name,label}", f"?name={upload_name}")
    headers = _headers(cfg["token"])
    headers["Content-Type"] = content_type

    last_err = None
    for attempt in range(retries + 1):
        r = requests.post(upload_url, headers=headers, data=data, timeout=300)
        if r.status_code == 201:
            logger.info("  uploaded %s (%.1f KB)", upload_name, len(data) / 1024)
            return upload_name
        last_err = f"{r.status_code} {r.text[:200]}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to upload {upload_name}: {last_err}")


def publish(cfg: dict, release: dict, paths) -> list:
    """Upload a list of files, refreshing the asset list first."""
    release = refresh_release(cfg, release)
    uploaded = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning("  skip missing file %s", p)
            continue
        uploaded.append(upload_path(cfg, release, p))
        # Refresh so the next delete sees the asset we just replaced.
        release = refresh_release(cfg, release)
    return uploaded
