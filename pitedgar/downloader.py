"""Step 2: bulk download companyfacts.zip from SEC EDGAR."""

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import requests
from loguru import logger
from tqdm import tqdm

from pitedgar.config import PitEdgarConfig

def _safe_extract(zf: ZipFile, member_info: ZipInfo, dest: Path) -> None:
    """Extract a single ZIP member to *dest* after validating against zip-slip attacks.

    Raises RuntimeError for any member whose resolved target path falls outside
    *dest*, as well as for absolute paths, parent-traversal components, and
    symlink entries.
    """
    filename = member_info.filename

    # Reject absolute paths (POSIX "/" prefix or Windows drive letters like "C:").
    if filename.startswith("/") or (len(filename) >= 2 and filename[1] == ":"):
        raise RuntimeError(f"Unsafe zip member path: {filename!r}")

    # Reject paths containing parent-traversal components.
    parts = Path(filename).parts
    if ".." in parts:
        raise RuntimeError(f"Unsafe zip member path: {filename!r}")

    # Reject symlink entries: external_attr upper 16 bits hold Unix mode;
    # 0xA000 is the S_IFLNK file-type mask.
    unix_mode = (member_info.external_attr >> 16) & 0xFFFF
    if unix_mode & 0xF000 == 0xA000:
        raise RuntimeError(f"Unsafe zip member path: {filename!r}")

    # Resolve target and assert it stays inside dest.
    dest_resolved = dest.resolve()
    target = (dest / filename).resolve()
    if not (str(target).startswith(str(dest_resolved) + os.sep) or target == dest_resolved):
        raise RuntimeError(f"Unsafe zip member path: {filename!r}")

    zf.extract(member_info, dest)


_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_BACKOFF_BASE_SECONDS = 2.0


def _compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _stream_to_file(
    url: str,
    headers: dict[str, str],
    dest: Path,
    *,
    timeout: int = 120,
) -> dict[str, str]:
    """Stream an HTTP GET to *dest* via a .part sidecar, then atomically rename.

    Returns the response headers dict so callers can inspect ETag / Last-Modified.
    """
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            resp_headers = dict(resp.headers)
            # Content-Length of 0 means unknown; pass None so tqdm shows an indeterminate bar.
            total = int(resp.headers.get("Content-Length") or 0) or None
            with (
                open(tmp_path, "wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    bar.update(len(chunk))
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return resp_headers


def _download_with_retries(
    url: str,
    headers: dict[str, str],
    dest: Path,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    backoff_base: float = _DEFAULT_BACKOFF_BASE_SECONDS,
    timeout: int = 120,
) -> dict[str, str]:
    """Retry transient network failures with exponential backoff.

    Returns the response headers from the successful attempt.
    """
    attempt = 0
    while True:
        try:
            resp_headers = _stream_to_file(url, headers, dest, timeout=timeout)
            return resp_headers
        except _RETRYABLE_EXCEPTIONS as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = backoff_base**attempt
            logger.warning(
                f"Download attempt {attempt}/{max_retries} failed ({exc!r}); retrying in {sleep_s:.1f}s…"
            )
            time.sleep(sleep_s)


def _write_sidecars(zip_path: Path, resp_headers: dict[str, str]) -> None:
    """Compute SHA-256 of *zip_path* and persist sidecar files next to it."""
    sha256 = _compute_sha256(zip_path)
    sha256_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha256_path.write_text(sha256)

    content_length_raw = resp_headers.get("Content-Length") or resp_headers.get("content-length")
    content_length = int(content_length_raw) if content_length_raw else None
    meta = {
        "sha256": sha256,
        "etag": resp_headers.get("ETag") or resp_headers.get("etag"),
        "last_modified": resp_headers.get("Last-Modified") or resp_headers.get("last-modified"),
        "content_length": content_length,
    }
    meta_path = zip_path.parent / "companyfacts.zip.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"SHA-256 sidecar written: {sha256_path} ({sha256[:16]}…)")


def _verify_sidecar(zip_path: Path) -> None:
    """Compare the on-disk ZIP against its .sha256 sidecar (if present)."""
    sha256_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    if not sha256_path.exists():
        logger.warning(
            f"No SHA-256 sidecar found for {zip_path}. Skipping integrity check (legacy cache)."
        )
        return
    expected = sha256_path.read_text().strip()
    actual = _compute_sha256(zip_path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {zip_path}: expected {expected!r} but got {actual!r}. "
            "Re-run with --force to re-download."
        )
    logger.debug(f"SHA-256 verified for {zip_path}: {actual[:16]}…")


_SINGLE_FILE_LIMIT_BYTES = 10 * 1024**3  # 10 GiB — no legitimate SEC JSON comes close


def _check_zip_size(zf: zipfile.ZipFile, limit: int) -> None:
    """Reject ZIP archives whose declared decompressed size looks like a zip bomb."""
    total = 0
    for info in zf.infolist():
        if info.file_size > _SINGLE_FILE_LIMIT_BYTES:
            raise RuntimeError(
                f"ZIP member '{info.filename}' declares decompressed size "
                f"{info.file_size:,} bytes which exceeds the single-file cap "
                f"{_SINGLE_FILE_LIMIT_BYTES:,} bytes — possible zip bomb"
            )
        total += info.file_size
    if total > limit:
        raise RuntimeError(
            f"Zip decompressed size {total:,} exceeds cap {limit:,} bytes — possible zip bomb"
        )


def download_bulk(config: PitEdgarConfig, force: bool = False) -> Path:
    """Download and extract the SEC companyfacts bulk ZIP.

    The ZIP is downloaded to `companyfacts.zip.part` and atomically renamed on
    success, so a crash mid-download never leaves a corrupted ZIP behind.
    Transient network errors (ConnectionError, Timeout, ChunkedEncodingError)
    are retried with exponential backoff (2^attempt seconds, up to 4 attempts).

    On a fresh download a SHA-256 sidecar (`companyfacts.zip.sha256`) and a
    metadata file (`companyfacts.zip.meta.json`) are written next to the ZIP.
    On subsequent runs the cached ZIP is re-hashed and compared against the
    sidecar.  A mismatch raises `RuntimeError` so the caller knows to re-fetch.

    Args:
        config: pipeline configuration.
        force: re-download and re-extract even if files already exist.

    Returns:
        Path to the extracted facts directory.
    """
    config.ensure_dirs()
    zip_path = config.data_dir / "companyfacts.zip"

    headers = {"User-Agent": config.edgar_identity}

    if not force and zip_path.exists():
        logger.info(f"ZIP already exists at {zip_path}, skipping download (use force=True to override).")
        _verify_sidecar(zip_path)
    else:
        logger.info(f"Downloading {config.zip_url} …")
        resp_headers = _download_with_retries(config.zip_url, headers, zip_path)
        logger.info(f"ZIP saved: {zip_path}")
        _write_sidecars(zip_path, resp_headers)

    facts_dir = config.facts_dir
    if force or not facts_dir.exists() or not any(facts_dir.iterdir()):
        logger.info(f"Extracting to {facts_dir} …")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _check_zip_size(zf, config.max_extracted_bytes)
                members = zf.infolist()
                for member_info in tqdm(members, desc="Extracting", unit="file"):
                    _safe_extract(zf, member_info, facts_dir)
        except zipfile.BadZipFile as exc:
            # Corrupted ZIP on disk — delete it so a rerun can fetch a fresh copy.
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupt ZIP at {zip_path}: {exc}. Deleted — rerun to re-download.") from exc
        logger.info("Extraction complete.")
    else:
        logger.info("Facts dir already populated, skipping extraction (use force=True to override).")

    return facts_dir
