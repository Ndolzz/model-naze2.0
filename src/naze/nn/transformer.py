"""Transformer decoder-only (REQ-011, Stage 6 / M-007) — M007-T001 + M007-T002.

Scope (M007_TASKS): T001 TransformerConfig + invariant check; T002 Embedding
(token embedding lookup (B,T)->(B,T,D), backward scatter-add); T003
PositionalRepr (learned table (T_max, D), x + P[0:T]). Komponen berikutnya
(Q/K/V, attention, LayerNorm, FFN, block, model, LM head) ditambahkan
bertahap pada T004..T013 — agar tiap task terisolasi.

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

from naze.core.numeric import Array, as_array, seeded_rng
from naze.nn.layers import Layer

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


class Embedding(Layer):
    """Token embedding: lookup E[ids], ids (B, T) -> (B, T, d_model) (TD §5/§7).

    E: (vocab_size, d_model) — parameter tunggal layer ini (256 × D);
    init normal(0, 0.1) per-seed via seeded_rng (konvensi embedding MLPLM
    Stage 5); dtype mengikuti config (DECISION-007). Backward: scatter-add
    seperti MLPLM.E (TD §7). Mengembalikan None dari backward karena grad
    terhadap token ID tidak terdefinisi (lookup diskrit) — embedding adalah
    layer pertama, tidak disusun via Sequential.
    """

    def __init__(self, config: TransformerConfig, *, seed: int | None = None) -> None:
        super().__init__(f"embedding({config.vocab_size}x{config.d_model})")
        self.config = config
        # Seed per instance: default config.seed; kwarg seed untuk offset
        # per komponen (pola Linear/MLPLM: seed, seed+1, ...).
        rng = seeded_rng(config.seed if seed is None else seed)
        self.params["E"] = as_array(
            rng.normal(0.0, 0.1, (config.vocab_size, config.d_model)), dtype=config.dtype
        )
        self._ids: Array | None = None

    def forward(self, ids: Array) -> Array:
        """Lookup baris E pada indeks ids (TD §5 komponen 1). Validasi fail-fast."""
        raw = np.asarray(ids)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"token ID harus integer, dapat dtype {raw.dtype!r}")
        if raw.ndim != 2:
            raise ValueError(f"input harus (batch, seq) 2D, dapat shape {raw.shape}")
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError(f"batch dan panjang sekuens harus > 0, dapat shape {raw.shape}")
        if raw.min() < 0 or raw.max() >= self.config.vocab_size:
            raise ValueError(
                f"token ID di luar rentang [0, {self.config.vocab_size}) "
                f"(min={raw.min()}, max={raw.max()})"
            )
        if raw.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                f"panjang sekuens T ({raw.shape[1]}) melebihi max_sequence_length "
                f"({self.config.max_sequence_length})"
            )
        self._ids = raw.astype(np.int64, copy=False)
        return self.params["E"][self._ids]

    def backward(self, grad_out: Array) -> None:
        """Scatter-add grad_out ke dE (TD §7); tidak menyalin E."""
        if self._ids is None:
            raise RuntimeError("backward dipanggil sebelum forward")
        g = as_array(grad_out, dtype=self.params["E"].dtype)
        expected = self._ids.shape + (self.config.d_model,)
        if g.shape != expected:
            raise ValueError(f"grad_out harus berbentuk {expected}, dapat {g.shape}")
        d_e = np.zeros_like(self.params["E"])
        np.add.at(d_e, self._ids, g)
        self.grads["E"] = d_e
        return None


class PositionalRepr(Layer):
    """Positional representation learned: x + P[0:T] (TD §5 komponen 2, §9/OD-121).

    P: (max_sequence_length, d_model) — tabel learned per-seed (skema default
    learned per TD §9/OD-121; revisi ke sinusoidal tidak mengubah interface).
    Posisi implisit 0..T-1 — tidak ada position ID eksplisit pada API (TD §7),
    sehingga posisi selalu integer valid dalam [0, T_max) by construction;
    batas T dijaga validasi forward. Backward: dP = scatter-add grad per
    posisi (posisi t berulang di seluruh batch — pola Embedding), dan
    grad_input = grad_out karena representasi bersifat aditif; tidak ada
    grad terhadap position ID (diskrit).
    """

    def __init__(self, config: TransformerConfig, *, seed: int | None = None) -> None:
        super().__init__(f"positional({config.max_sequence_length}x{config.d_model})")
        self.config = config
        # Seed per instance: default config.seed; kwarg seed untuk offset
        # per komponen (pola Linear/MLPLM: seed, seed+1, ...).
        rng = seeded_rng(config.seed if seed is None else seed)
        self.params["P"] = as_array(
            rng.normal(0.0, 0.1, (config.max_sequence_length, config.d_model)),
            dtype=config.dtype,
        )
        self._shape: tuple[int, ...] | None = None

    def forward(self, x: Array) -> Array:
        """x (B, T, D) -> x + P[0:T], broadcast ke batch (TD §5). Fail-fast."""
        x = as_array(x, dtype=self.params["P"].dtype)
        if x.ndim != 3:
            raise ValueError(f"input harus (batch, seq, d_model) 3D, dapat shape {x.shape}")
        if x.shape[2] != self.config.d_model:
            raise ValueError(
                f"dimensi terakhir harus d_model ({self.config.d_model}), dapat {x.shape[2]}"
            )
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"batch dan panjang sekuens harus > 0, dapat shape {x.shape}")
        if x.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                f"panjang sekuens T ({x.shape[1]}) melebihi max_sequence_length "
                f"({self.config.max_sequence_length})"
            )
        self._shape = x.shape
        return x + self.params["P"][: x.shape[1]]

    def backward(self, grad_out: Array) -> Array:
        """dP via scatter-add per posisi (TD §7); grad_input = grad_out (aditif)."""
        if self._shape is None:
            raise RuntimeError("backward dipanggil sebelum forward")
        g = as_array(grad_out, dtype=self.params["P"].dtype)
        if g.shape != self._shape:
            raise ValueError(f"grad_out harus berbentuk {self._shape}, dapat {g.shape}")
        d_p = np.zeros_like(self.params["P"])
        # Posisi implisit 0..T-1, tiap posisi muncul B kali (batch) → scatter-add.
        np.add.at(d_p, np.broadcast_to(np.arange(self._shape[1]), self._shape[:2]), g)
        self.grads["P"] = d_p
        return g
