"""Byte-level tokenizer (REQ-004, Stage 3, DECISION-109).

Keputusan (resolusi OPEN DECISION-102): byte-level.
- Lossless by construction (setiap byte = 1 token), vocab size = 256 tetap.
- Tidak butuh training korpus, deterministik penuh.
- Trade-off: sequence lebih panjang vs BPE — dapat ditinjau ulang sebelum Stage 6.
"""

from __future__ import annotations


class ByteTokenizer:
    """Encode teks ke list token ID (0-255) dan sebaliknya. Lossless."""

    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        if any(not (0 <= i < 256) for i in ids):
            raise ValueError("token ID di luar rentang 0-255")
        return bytes(ids).decode("utf-8", errors="replace")
