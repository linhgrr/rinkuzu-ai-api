"""Focused tests for constant-time service-token matching."""

from api.security import service_tokens_match


def test_service_tokens_match_equal() -> None:
    assert service_tokens_match("same-token-value", "same-token-value") is True


def test_service_tokens_match_mismatch() -> None:
    assert service_tokens_match("token-aaaa", "token-bbbb") is False


def test_service_tokens_match_length_mismatch() -> None:
    assert service_tokens_match("short", "much-longer-token") is False


def test_service_tokens_match_missing_either_side() -> None:
    assert service_tokens_match(None, "configured-token") is False
    assert service_tokens_match("provided-token", None) is False
    assert service_tokens_match(None, None) is False


def test_service_tokens_match_empty_either_side() -> None:
    assert service_tokens_match("", "configured-token") is False
    assert service_tokens_match("provided-token", "") is False
    assert service_tokens_match("", "") is False


def test_service_tokens_match_utf8_equal() -> None:
    """Multibyte UTF-8 tokens match when string values are equal."""
    token = "tokén-パスワード-🔐-αβγ"
    assert service_tokens_match(token, token) is True
    # Same unicode code points → same UTF-8 bytes.
    assert service_tokens_match("café-token-value-xx", "café-token-value-xx") is True


def test_service_tokens_match_utf8_mismatch_and_byte_length() -> None:
    """Mismatch and byte-length checks use UTF-8, not Python char length.

    ``"é"`` is one Unicode character but two UTF-8 bytes (0xc3 0xa9), so a
    single-byte stand-in of the same char-count still fails length or content.
    """
    # Same character count, different code points / bytes.
    assert service_tokens_match("café", "cafe") is False
    assert service_tokens_match("パスワード", "passwordxx") is False

    # Same UTF-8 byte length, different content.
    a = "é"  # 1 char, 2 UTF-8 bytes
    b = "eX"  # 2 chars, 2 UTF-8 bytes
    assert len(a) == 1
    assert len(b) == 2
    assert len(a.encode("utf-8")) == 2
    assert len(b.encode("utf-8")) == 2
    assert service_tokens_match(a, b) is False

    # Char-length equal, UTF-8 byte-length unequal → fail closed before compare_digest.
    multi = "á"  # 2 UTF-8 bytes, 1 char
    single = "a"  # 1 UTF-8 byte, 1 char
    assert len(multi) == 1
    assert len(single) == 1
    assert len(multi.encode("utf-8")) != len(single.encode("utf-8"))
    assert service_tokens_match(multi, single) is False
