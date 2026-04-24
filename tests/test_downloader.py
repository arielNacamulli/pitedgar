"""Tests for downloader.py — cache-skip, retry, atomicity, corrupt-ZIP behavior."""

import io
import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from pitedgar.config import PitEdgarConfig
from pitedgar.downloader import download_bulk


@pytest.fixture
def config(tmp_path):
    facts_dir = tmp_path / "companyfacts"
    return PitEdgarConfig(
        edgar_identity="Test test@example.com",
        data_dir=tmp_path,
        facts_dir=facts_dir,
    )


def _make_zip(path):
    """Write a minimal valid ZIP at *path*."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("dummy.json", "{}")


def test_skips_download_when_zip_exists(config):
    """If companyfacts.zip exists and force=False, no HTTP request is made."""
    zip_path = config.data_dir / "companyfacts.zip"
    _make_zip(zip_path)
    config.facts_dir.mkdir(parents=True, exist_ok=True)
    # Put a file in facts_dir so extraction is also skipped.
    (config.facts_dir / "dummy.json").write_text("{}")

    with patch("pitedgar.downloader.requests.get") as mock_get:
        download_bulk(config, force=False)
        mock_get.assert_not_called()


def test_downloads_when_zip_missing(config):
    """If companyfacts.zip does not exist, the HTTP request is made."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    # Build a fake streaming response whose content is a valid ZIP.
    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.headers = {"Content-Length": str(len(zip_bytes))}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content = MagicMock(return_value=[zip_bytes])

    with patch("pitedgar.downloader.requests.get", return_value=mock_resp) as mock_get:
        download_bulk(config, force=False)
        mock_get.assert_called_once()


def test_force_redownloads_even_when_zip_exists(config):
    """force=True must trigger a new HTTP request even if ZIP is present."""
    zip_path = config.data_dir / "companyfacts.zip"
    _make_zip(zip_path)
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.headers = {"Content-Length": str(len(zip_bytes))}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content = MagicMock(return_value=[zip_bytes])

    with patch("pitedgar.downloader.requests.get", return_value=mock_resp) as mock_get:
        download_bulk(config, force=True)
        mock_get.assert_called_once()


def _make_streaming_response(payload: bytes) -> MagicMock:
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.headers = {"Content-Length": str(len(payload))}
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(return_value=[payload])
    return resp


def test_retries_on_transient_network_error(config):
    """A transient ConnectionError should be retried, then succeed."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    good_resp = _make_streaming_response(zip_bytes)

    call_count = {"n": 0}

    def flaky_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.ConnectionError("simulated flap")
        return good_resp

    with (
        patch("pitedgar.downloader.requests.get", side_effect=flaky_get),
        patch("pitedgar.downloader.time.sleep"),  # skip real backoff in tests
    ):
        download_bulk(config, force=False)

    assert call_count["n"] == 2
    assert (config.data_dir / "companyfacts.zip").exists()


def test_gives_up_after_max_retries(config):
    """After max_retries attempts the last exception should surface."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch(
            "pitedgar.downloader.requests.get",
            side_effect=requests.Timeout("always slow"),
        ),
        patch("pitedgar.downloader.time.sleep"),
        pytest.raises(requests.Timeout),
    ):
        download_bulk(config, force=False)

    # Crucially, no partial/final zip left behind.
    assert not (config.data_dir / "companyfacts.zip").exists()
    assert not (config.data_dir / "companyfacts.zip.part").exists()


def test_partial_download_cleaned_up_on_failure(config):
    """A crash during streaming must remove the .part sidecar."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    boom_resp = MagicMock()
    boom_resp.__enter__ = lambda s: s
    boom_resp.__exit__ = MagicMock(return_value=False)
    boom_resp.headers = {"Content-Length": "100"}
    boom_resp.raise_for_status = MagicMock()
    # Emit some bytes, then blow up partway through.
    boom_resp.iter_content = MagicMock(
        side_effect=requests.exceptions.ChunkedEncodingError("truncated"),
    )

    with (
        patch("pitedgar.downloader.requests.get", return_value=boom_resp),
        patch("pitedgar.downloader.time.sleep"),
        pytest.raises(requests.exceptions.ChunkedEncodingError),
    ):
        download_bulk(config, force=False)

    assert not (config.data_dir / "companyfacts.zip").exists()
    assert not (config.data_dir / "companyfacts.zip.part").exists()


def test_corrupt_zip_is_deleted_and_surfaced(config):
    """If the on-disk ZIP is corrupt, extraction must raise and delete the file."""
    zip_path = config.data_dir / "companyfacts.zip"
    # Write garbage that is NOT a valid ZIP.
    zip_path.write_bytes(b"not a zip file")
    config.facts_dir.mkdir(parents=True, exist_ok=True)
    # Do not populate facts_dir — force extraction path.

    with pytest.raises(RuntimeError, match="Corrupt ZIP"):
        download_bulk(config, force=False)

    # The corrupt zip must have been removed so the next run re-fetches.
    assert not zip_path.exists()


def test_non_retryable_error_propagates_immediately(config):
    """HTTPError (e.g. 404/500) should not be retried — surface once."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.raise_for_status = MagicMock(side_effect=requests.HTTPError("404"))

    with (
        patch("pitedgar.downloader.requests.get", return_value=resp) as mock_get,
        patch("pitedgar.downloader.time.sleep"),
        pytest.raises(requests.HTTPError),
    ):
        download_bulk(config, force=False)

    assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# SHA-256 sidecar / integrity tests
# ---------------------------------------------------------------------------

def _make_streaming_response_with_meta(payload: bytes) -> MagicMock:
    """Build a mock streaming response that includes ETag / Last-Modified headers."""
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.headers = {
        "Content-Length": str(len(payload)),
        "ETag": '"abc123"',
        "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT",
    }
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(return_value=[payload])
    return resp


def test_download_writes_sha256_sidecar(config):
    """After a fresh download both .sha256 and .meta.json sidecars must exist and be non-empty."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    mock_resp = _make_streaming_response_with_meta(zip_bytes)

    with patch("pitedgar.downloader.requests.get", return_value=mock_resp):
        download_bulk(config, force=False)

    sha256_path = config.data_dir / "companyfacts.zip.sha256"
    meta_path = config.data_dir / "companyfacts.zip.meta.json"

    assert sha256_path.exists(), ".sha256 sidecar was not written"
    assert meta_path.exists(), ".meta.json sidecar was not written"

    digest = sha256_path.read_text().strip()
    assert len(digest) == 64, "SHA-256 digest should be 64 hex chars"

    meta = json.loads(meta_path.read_text())
    assert meta["sha256"] == digest
    assert meta["etag"] == '"abc123"'
    assert meta["last_modified"] == "Wed, 01 Jan 2025 00:00:00 GMT"
    assert meta["content_length"] == len(zip_bytes)


def test_cache_hit_verifies_sha256(config):
    """Second run (no force) must pass SHA-256 check without re-downloading."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    mock_resp = _make_streaming_response_with_meta(zip_bytes)

    # First run — fresh download, writes sidecar.
    with patch("pitedgar.downloader.requests.get", return_value=mock_resp) as mock_get:
        download_bulk(config, force=False)
        assert mock_get.call_count == 1

    # Second run — cache hit; sidecar present; no re-download.
    with patch("pitedgar.downloader.requests.get") as mock_get2:
        download_bulk(config, force=False)
        mock_get2.assert_not_called()


def test_cache_hit_rejects_corrupted_zip(config):
    """After fresh download, flipping a byte in the ZIP must cause RuntimeError on next run."""
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    mock_resp = _make_streaming_response_with_meta(zip_bytes)

    # First run — writes the ZIP and its sidecar.
    with patch("pitedgar.downloader.requests.get", return_value=mock_resp):
        download_bulk(config, force=False)

    # Corrupt the ZIP on disk (flip one byte).
    zip_path = config.data_dir / "companyfacts.zip"
    data = bytearray(zip_path.read_bytes())
    data[0] ^= 0xFF
    zip_path.write_bytes(bytes(data))

    # Second run must raise RuntimeError mentioning SHA-256 mismatch.
    with pytest.raises(RuntimeError, match="(?i)sha.?256"):
        download_bulk(config, force=False)


def test_cache_hit_without_sidecar_warns_and_proceeds(config):
    """If the .sha256 sidecar is absent (legacy cache), a warning is logged and the run continues."""
    from loguru import logger as loguru_logger

    config.facts_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    _make_zip(buf)
    zip_bytes = buf.getvalue()

    mock_resp = _make_streaming_response_with_meta(zip_bytes)

    # First run — download + sidecar.
    with patch("pitedgar.downloader.requests.get", return_value=mock_resp):
        download_bulk(config, force=False)

    # Remove sidecar to simulate legacy cache.
    sha256_path = config.data_dir / "companyfacts.zip.sha256"
    sha256_path.unlink()

    # Capture loguru WARNING messages during the second run.
    captured: list[str] = []

    def _sink(message) -> None:
        if message.record["level"].name == "WARNING":
            captured.append(message)

    sink_id = loguru_logger.add(_sink, level="WARNING")
    try:
        with patch("pitedgar.downloader.requests.get") as mock_get2:
            download_bulk(config, force=False)
            mock_get2.assert_not_called()
    finally:
        loguru_logger.remove(sink_id)

    assert any(
        "sidecar" in str(m).lower() or "sha-256" in str(m).lower()
        for m in captured
    ), "Expected a loguru WARNING about the missing SHA-256 sidecar"
from pathlib import Path
from pitedgar.downloader import _safe_extract, download_bulk


# ---------------------------------------------------------------------------
# Zip-slip security tests
# ---------------------------------------------------------------------------

def _make_zip_with_member(path, member_name: str, content: bytes = b"{}") -> None:
    """Write a ZIP at *path* containing a single member with the given name."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member_name, content)


def test_extraction_rejects_parent_traversal(config, tmp_path):
    """A member named '../evil.json' must raise RuntimeError."""
    zip_path = config.data_dir / "companyfacts.zip"
    _make_zip_with_member(zip_path, "../evil.json")
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    with patch("pitedgar.downloader.requests.get"):
        with pytest.raises(RuntimeError, match="Unsafe zip member path"):
            download_bulk(config, force=False)


def test_extraction_rejects_absolute_path(config, tmp_path):
    """A member with an absolute path '/tmp/evil.json' must raise RuntimeError."""
    zip_path = config.data_dir / "companyfacts.zip"
    _make_zip_with_member(zip_path, "/tmp/evil.json")
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    with patch("pitedgar.downloader.requests.get"):
        with pytest.raises(RuntimeError, match="Unsafe zip member path"):
            download_bulk(config, force=False)


def test_extraction_rejects_symlink_entry(tmp_path):
    """A ZipInfo entry with symlink external_attr must raise RuntimeError."""
    # Build a real ZipFile so _safe_extract has a valid zf object.
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("link_target.json", "{}")
        info = zf.infolist()[0]

    # Fabricate a symlink ZipInfo: set S_IFLNK (0xA000) in the upper 16 bits.
    symlink_info = zipfile.ZipInfo(filename="link.json")
    symlink_info.external_attr = 0xA1ED0000  # S_IFLNK | 0755 permissions

    dest = tmp_path / "dest"
    dest.mkdir()

    with zipfile.ZipFile(zip_path, "r") as zf:
        with pytest.raises(RuntimeError, match="Unsafe zip member path"):
            _safe_extract(zf, symlink_info, dest)


def test_extraction_accepts_legitimate_nested_path(config):
    """A member like 'subdir/file.json' should extract without error."""
    zip_path = config.data_dir / "companyfacts.zip"
    _make_zip_with_member(zip_path, "subdir/file.json")
    config.facts_dir.mkdir(parents=True, exist_ok=True)

    with patch("pitedgar.downloader.requests.get"):
        result = download_bulk(config, force=False)

    assert (result / "subdir" / "file.json").exists()
from pitedgar.downloader import _check_zip_size, download_bulk


# ---------------------------------------------------------------------------
# Zip-bomb size-cap tests (#20)
# ---------------------------------------------------------------------------


def _make_mock_zip_with_sizes(file_sizes: list[int]) -> MagicMock:
    """Return a mock ZipFile whose infolist() reports the given declared sizes."""
    infos = []
    for i, size in enumerate(file_sizes):
        info = MagicMock(spec=zipfile.ZipInfo)
        info.filename = f"file_{i}.json"
        info.file_size = size
        infos.append(info)
    zf = MagicMock(spec=zipfile.ZipFile)
    zf.infolist.return_value = infos
    return zf


def test_extraction_rejects_oversized_zip():
    """Total declared decompressed size above the configured cap raises RuntimeError."""
    # Three members totalling 301 bytes, cap is 300 bytes.
    zf = _make_mock_zip_with_sizes([100, 100, 101])
    with pytest.raises(RuntimeError, match="Zip decompressed size"):
        _check_zip_size(zf, limit=300)


def test_extraction_accepts_reasonable_zip():
    """A ZIP whose total size is within the default cap passes without error."""
    # Build a real tiny ZIP in memory and run _check_zip_size against it.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf_write:
        zf_write.writestr("a.json", '{"key": "value"}')
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf_read:
        # Default cap is 25 GiB — a tiny ZIP must pass with no exception.
        _check_zip_size(zf_read, limit=25 * 1024**3)


def test_extraction_rejects_huge_single_file():
    """A single member declaring > 10 GiB raises RuntimeError regardless of total cap."""
    huge = 11 * 1024**3  # 11 GiB > 10 GiB single-file limit
    zf = _make_mock_zip_with_sizes([huge])
    with pytest.raises(RuntimeError, match="Zip decompressed size|single-file cap"):
        _check_zip_size(zf, limit=100 * 1024**3)  # generous total cap
