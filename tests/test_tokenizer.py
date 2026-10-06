"""Test byte-level tokenizer (REQ-004, Stage 3)."""

import pytest

from naze.token import ByteTokenizer


def test_roundtrip_ascii() -> None:
    tok = ByteTokenizer()
    text = "hello naze 2.0"
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unicode() -> None:
    tok = ByteTokenizer()
    text = "Naze bisa bahasa Indonesia — dengan emoji: 🤖✅"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_fixed() -> None:
    assert ByteTokenizer.vocab_size == 256
    ids = ByteTokenizer().encode("ab")
    assert all(0 <= i < 256 for i in ids)


def test_decode_rejects_out_of_range() -> None:
    tok = ByteTokenizer()
    with pytest.raises(ValueError):
        tok.decode([999])
