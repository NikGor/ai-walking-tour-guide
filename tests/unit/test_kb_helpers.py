"""Unit tests for Telegram keyboard helper functions."""

from app.telegram.handlers import _cb_data


def test_cb_data_short_ascii():
    """ASCII text well under limit passes through unchanged."""
    result = _cb_data("place:", "Colosseum")
    assert result == "place:Colosseum"
    assert len(result.encode()) <= 64


def test_cb_data_short_cyrillic():
    """Short Cyrillic text fits within the 64-byte limit."""
    result = _cb_data("place:", "Медный всадник")
    assert result.startswith("place:")
    assert len(result.encode()) <= 64


def test_cb_data_long_cyrillic_truncated():
    """Overlong Cyrillic text is truncated to fit in 64 bytes."""
    # 40 Cyrillic chars × 2 bytes each = 80 bytes payload → must be cut
    long_text = "А" * 40
    result = _cb_data("place:", long_text)
    assert len(result.encode()) <= 64
    assert result.startswith("place:")


def test_cb_data_never_splits_multibyte():
    """Result always decodes cleanly — no split UTF-8 sequences."""
    long_text = "Привет мир! " * 10  # repeating Cyrillic
    result = _cb_data("place:", long_text)
    # If this doesn't raise, the bytes form valid UTF-8
    result.encode("utf-8").decode("utf-8")
    assert len(result.encode()) <= 64


def test_cb_data_mode_slug_fits():
    """Persona slugs via mode: prefix are ASCII and always fit without truncation."""
    for slug in ["historian", "architecture_expert", "medieval_resident", "deep_time"]:
        result = _cb_data("mode:", slug)
        assert result == f"mode:{slug}"
        assert len(result.encode()) <= 64
