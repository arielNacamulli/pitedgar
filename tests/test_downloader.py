"""Tests for downloader.py — cache-skip behaviour."""

import zipfile
from unittest.mock import patch, MagicMock

import pytest

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
    import io

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

    import io

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
