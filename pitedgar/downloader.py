"""Step 2: bulk download companyfacts.zip from SEC EDGAR."""

import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from pitedgar.config import PitEdgarConfig


def download_bulk(config: PitEdgarConfig, force: bool = False) -> Path:
    """Download and extract the SEC companyfacts bulk ZIP.

    Args:
        config: pipeline configuration.
        force: re-extract even if facts_dir already exists.

    Returns:
        Path to the extracted facts directory.
    """
    config.ensure_dirs()
    zip_path = config.data_dir / "companyfacts.zip"

    headers = {"User-Agent": config.edgar_identity}

    logger.info(f"Downloading {config.zip_url} …")
    with requests.get(config.zip_url, stream=True, headers=headers, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            open(zip_path, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="companyfacts.zip",
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    logger.info(f"ZIP saved: {zip_path}")

    facts_dir = config.facts_dir
    if force or not facts_dir.exists() or not any(facts_dir.iterdir()):
        logger.info(f"Extracting to {facts_dir} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc="Extracting", unit="file"):
                zf.extract(member, facts_dir)
        logger.info("Extraction complete.")
    else:
        logger.info(f"Facts dir already populated, skipping extraction (use force=True to override).")

    return facts_dir
