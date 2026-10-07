"""Transformer decoder-only (REQ-011, Stage 6 / M-007) — konfigurasi foundation (M007-T001).

Scope T001: TransformerConfig + invariant check saja (M007_TASKS).
Komponen (Embedding, positional, attention, LayerNorm, FFN, block, model,
LM head) ditambahkan bertahap pada T002..T013 — modul ini sengaja belum
memuatnya agar tiap task tetap terisolasi.

Kontrak konfigurasi (M007_TECHNICAL_DESIGN §7/§9/§10):
- vocab_size FIXED 256 — byte-level (DECISION-010/015); validasi menolak selain 256.
- Invariant attention: d_model == num_heads * head_dim; head_dim = d_model // num_heads.
- Default = konfigurasi dev/test TD §9 (d_model=64, num_heads=4, num_layers=2,
  d_ff=128, max_sequence_length=64) — BUKAN batas final model produksi (OD-118 terbuka).
- dtype float64 (DECISION-007); seed wajib per instance (REQ-101, via seeded_rng).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from naze.core.numeric import seeded_rng

# Byte-level tokenizer (DECISION-010/015): vocab size 256 tetap.
VOCAB_SIZE = 256


@dataclass(frozen=True)
class TransformerConfig:
    """Konfigurasi Transformer; validasi fail-fast saat konstruksi (TD §10).

    Default = konfigurasi dev/test M007_TECHNICAL_DESIGN §9 — bukan batas final
    model produksi/training skala penuh (OD-118 terbuka, keputusan owner).
    """

    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    max_sequence_length: int = 64
    dtype: type = np.float64
    seed: int = 0

    def __post_init__(self) -> None:
        # vocab invariant (DECISION-015): byte-level, FIXED 256.
        if not isinstance(self.vocab_size, (int, np.integer)) or self.vocab_size != VOCAB_SIZE:
            raise ValueError(
                f"vocab_size harus {VOCAB_SIZE} (byte-level, DECISION-015), "
                f"dapat: {self.vocab_size!r}"
            )
        # Dimensi harus integer > 0 (TD §10: konfigurasi invalid = dimensi <= 0).
        for name in ("d_model", "num_heads", "num_layers", "d_ff", "max_sequence_length"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} harus integer > 0, dapat: {value!r}")
        # Invariant attention (TD §7): d_model == num_heads * head_dim.
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) tidak habis dibagi num_heads ({self.num_heads}); "
                "invariant: d_model == num_heads * head_dim"
            )
        # dtype harus tipe floating NumPy; default float64 (DECISION-007).
        try:
            kind = np.dtype(self.dtype).kind
        except TypeError as exc:
            raise ValueError(f"dtype tidak valid: {self.dtype!r}") from exc
        if kind != "f":
            raise ValueError(
                f"dtype harus tipe floating NumPy (default float64, DECISION-007), "
                f"dapat: {self.dtype!r}"
            )
        # Seed wajib valid (integer >= 0) — determinisme per-seed REQ-101 via seeded_rng.
        seeded_rng(self.seed)

    @property
    def head_dim(self) -> int:
        """Dimensi per attention head (Dh) — TD §7: Dh = d_model // num_heads."""
        return self.d_model // self.num_heads

    def rng(self) -> np.random.Generator:
        """Generator acak deterministik dari seed config (REQ-101, DECISION-007)."""
        return seeded_rng(self.seed)
