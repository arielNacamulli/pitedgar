"""Tests for PitEdgarConfig edgar_identity validation."""

import pytest
from pydantic import ValidationError

from pitedgar.config import PitEdgarConfig


def _make_config(identity: str, tmp_path) -> PitEdgarConfig:
    return PitEdgarConfig(edgar_identity=identity, data_dir=tmp_path)


def test_config_accepts_valid_identity(tmp_path):
    cfg = _make_config("Ariel Nacamulli ariel@example.com", tmp_path)
    assert cfg.edgar_identity == "Ariel Nacamulli ariel@example.com"


def test_config_accepts_single_name_plus_email(tmp_path):
    # "Test test@example.com" — the common fixture form
    cfg = _make_config("Test test@example.com", tmp_path)
    assert cfg.edgar_identity == "Test test@example.com"


def test_config_accepts_multi_word_name(tmp_path):
    cfg = _make_config("First Middle Last first.last@sub.example.co.uk", tmp_path)
    assert cfg.edgar_identity == "First Middle Last first.last@sub.example.co.uk"


def test_config_rejects_empty(tmp_path):
    with pytest.raises(ValidationError):
        _make_config("", tmp_path)


def test_config_rejects_whitespace_only(tmp_path):
    with pytest.raises(ValidationError):
        _make_config("   ", tmp_path)


def test_config_rejects_single_word(tmp_path):
    with pytest.raises(ValidationError):
        _make_config("test", tmp_path)


def test_config_rejects_missing_email_at_sign(tmp_path):
    with pytest.raises(ValidationError):
        _make_config("Name email-without-at.com", tmp_path)


def test_config_rejects_name_at_email_no_space(tmp_path):
    # "Name@email.com" — no space-separated name before email
    with pytest.raises(ValidationError):
        _make_config("Name@email.com", tmp_path)


def test_config_error_message_points_to_sec_docs(tmp_path):
    with pytest.raises(ValidationError) as exc_info:
        _make_config("test", tmp_path)
    assert "sec.gov" in str(exc_info.value)


def test_config_error_message_empty_points_to_sec_docs(tmp_path):
    with pytest.raises(ValidationError) as exc_info:
        _make_config("", tmp_path)
    assert "sec.gov" in str(exc_info.value)
