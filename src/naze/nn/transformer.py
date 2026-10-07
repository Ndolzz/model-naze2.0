"""Transformer decoder-only (REQ-011, Stage 6 / M-007) — M007-T001..M007-T005.

Scope (M007_TASKS): T001 TransformerConfig + invariant check; T002 Embedding
(token embedding lookup (B,T)->(B,T,D), backward scatter-add); T003
PositionalRepr (learned table (T_max, D), x + P[0:T]); T004 QKVProjection
(3 Linear D->D terpisah + reshape (B,T,D)->(B,H,T,Dh)); T005 CausalAttention
(scores QK^T/sqrt(Dh), mask kausal, softmax stabil, context = P @ V).
Komponen berikutnya (output projection, LayerNorm, FFN, block, model,
LM head) ditambahkan bertahap pada T006..T013 — agar tiap task terisolasi.

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
from naze.nn.activations import softmax
from naze.nn.layers import Layer, Linear

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


class QKVProjection(Layer):
    """Proyeksi Q/K/V: 3 Linear terpisah (D->D) + reshape (B,T,D)->(B,H,T,Dh).

    TD §5 komponen 3-4 & §7: Q/K/V projection memakai naze.nn.Linear
    (bias ya — TD §9), input (B,T,D) -> per-proyeksi (B,T,D) -> reshape
    multi-head (B,H,T,Dh) dengan layout
    q[b, h, t] = (x[b, t] @ Wq + bq)[h*Dh:(h+1)*Dh]. Tiga proyeksi
    terpisah dengan seed berturut (pola MLPLM: seed, seed+1, seed+2) agar
    Q/K/V independen; engine Linear selalu inisialisasi float64 sehingga
    parameter di-cast ke config.dtype setelah konstruksi (DECISION-007) —
    bukan metode init baru. backward menerima tuple (dq, dk, dv) (B,H,T,Dh)
    — deviasi terdokumentasi dari kontrak Layer.backward (pola
    Embedding.backward -> None) — dan mengembalikan dx (B,T,D) = jumlah
    jalur ketiga proyeksi; grad parameter tiap proyeksi dihitung oleh
    Linear-nya sendiri (jalur Q/K/V independen).
    """

    def __init__(self, config: TransformerConfig, *, seed: int | None = None) -> None:
        super().__init__(f"qkv({config.d_model}->{config.num_heads}x{config.head_dim})")
        self.config = config
        # Seed per instance: default config.seed; kwarg seed untuk offset
        # per komponen (pola Linear/MLPLM: seed, seed+1, ...).
        s = config.seed if seed is None else seed
        self.q_proj = Linear(config.d_model, config.d_model, seed=s)
        self.k_proj = Linear(config.d_model, config.d_model, seed=s + 1)
        self.v_proj = Linear(config.d_model, config.d_model, seed=s + 2)
        # Adapter dtype (DECISION-007): engine Linear selalu init float64.
        for lin in (self.q_proj, self.k_proj, self.v_proj):
            lin.params["W"] = as_array(lin.params["W"], dtype=config.dtype)
            lin.params["b"] = as_array(lin.params["b"], dtype=config.dtype)
        self._shape: tuple[int, ...] | None = None

    def parameter_count(self) -> int:
        # Pola Sequential: parameter layer = jumlah parameter sub-layer.
        return (self.q_proj.parameter_count()
                + self.k_proj.parameter_count()
                + self.v_proj.parameter_count())

    def forward(self, x: Array) -> tuple[Array, Array, Array]:
        """x (B,T,D) -> (q, k, v) masing-masing (B,H,T,Dh) (TD §5). Fail-fast."""
        x = as_array(x, dtype=self.config.dtype)
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
        b, t, d = x.shape
        h, dh = self.config.num_heads, self.config.head_dim
        flat = x.reshape(b * t, d)
        q = self.q_proj.forward(flat).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        k = self.k_proj.forward(flat).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        v = self.v_proj.forward(flat).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        return q, k, v

    def backward(self, grads: tuple[Array, Array, Array]) -> Array:
        """(dq, dk, dv) (B,H,T,Dh) -> dx (B,T,D); dW/db per proyeksi via Linear."""
        if self._shape is None:
            raise RuntimeError("backward dipanggil sebelum forward")
        if not isinstance(grads, tuple) or len(grads) != 3:
            raise ValueError(f"grads harus tuple (dq, dk, dv), dapat {type(grads).__name__}")
        b, t, d = self._shape
        h, dh = self.config.num_heads, self.config.head_dim
        expected = (b, h, t, dh)
        names = ("dq", "dk", "dv")
        flat: list[Array] = []
        for i, g in enumerate(grads):
            g = as_array(g, dtype=self.config.dtype)
            if g.shape != expected:
                raise ValueError(f"{names[i]} harus berbentuk {expected}, dapat {g.shape}")
            flat.append(g.transpose(0, 2, 1, 3).reshape(b * t, d))
        # dx = jumlah kontribusi ketiga proyeksi; tiap Linear.backward
        # menghitung dW/db proyeksinya sendiri (jalur independen).
        dx = self.q_proj.backward(flat[0])
        dx = dx + self.k_proj.backward(flat[1])
        dx = dx + self.v_proj.backward(flat[2])
        return dx.reshape(b, t, d)


class CausalAttention(Layer):
    """Causal scaled dot-product attention: (q, k, v) -> context (TD §5 5-10).

    scores = q @ k^T / sqrt(Dh) — (B,H,T,T); mask kausal: query t hanya
    melihat key 0..t (posisi = sequence index, tanpa API position ID);
    softmax stabil per baris axis=-1 (reuse naze.nn.activations.softmax —
    M007_TASKS T005); context = weights @ v — (B,H,T,Dh). Tanpa output
    projection (TD §5 komponen 13 = T006). Mask memakai sentinel finite
    relatif max baris: entri future = row_max - 1e9 — ekuivalen numerik
    dengan -inf (TD §5: bobot future = 0 eksak karena exp(-1e9)
    underflow), aman untuk softmax engine yang menolak input non-finite,
    dan robust untuk skor valid ekstrem positif/negatif (diagonal selalu
    valid sehingga row_max finite). backward: dV = P^T @ dC;
    dP = dC @ V^T; dS = P * (dP - sum(P*dP, -1)); dQ = dS @ K / sqrt(Dh);
    dK = dS^T @ Q / sqrt(Dh) — P = 0 di future sehingga grad tidak
    menembus mask (kausal terjaga); tanpa grad terhadap mask (bukan
    parameter). forward(q, k, v) & backward -> tuple (dq, dk, dv):
    deviasi terdokumentasi dari kontrak Layer satu-tensor (pola
    QKVProjection). Layer tanpa parameter — attention murni operator.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(f"causal_attention(h={config.num_heads},dh={config.head_dim})")
        self.config = config
        self._q: Array | None = None
        self._k: Array | None = None
        self._v: Array | None = None
        self._probs: Array | None = None
        self._shape: tuple[int, ...] | None = None

    def forward(self, q: Array, k: Array, v: Array) -> Array:
        """q/k/v (B,H,T,Dh) -> context (B,H,T,Dh) (TD §5 5-10). Fail-fast."""
        q = as_array(q, dtype=self.config.dtype)
        k = as_array(k, dtype=self.config.dtype)
        v = as_array(v, dtype=self.config.dtype)
        for name, arr in (("q", q), ("k", k), ("v", v)):
            if arr.ndim != 4:
                raise ValueError(f"{name} harus (B,H,T,Dh) 4D, dapat shape {arr.shape}")
        b, h, t, dh = q.shape
        for name, arr in (("k", k), ("v", v)):
            if arr.shape != q.shape:
                raise ValueError(
                    f"{name} harus berbentuk {q.shape} (sama dengan q), dapat {arr.shape}"
                )
        if h != self.config.num_heads:
            raise ValueError(f"H harus num_heads ({self.config.num_heads}), dapat {h}")
        if dh != self.config.head_dim:
            raise ValueError(f"Dh harus head_dim ({self.config.head_dim}), dapat {dh}")
        if b == 0 or t == 0:
            raise ValueError(f"batch dan panjang sekuens harus > 0, dapat shape {q.shape}")
        if t > self.config.max_sequence_length:
            raise ValueError(
                f"panjang sekuens T ({t}) melebihi max_sequence_length "
                f"({self.config.max_sequence_length})"
            )
        # Skor mentah wajib finite SEBELUM mask (TD §10); matmul overflow
        # (q . k -> inf) tertangkap fail-fast di sini, bukan NaN diam-diam.
        scale = float(np.sqrt(dh))
        scores = as_array(np.matmul(q, np.swapaxes(k, -1, -2)) / scale, dtype=self.config.dtype)
        # Mask kausal (TD §5/§8): query t hanya melihat key 0..t. Sentinel
        # finite relatif row_max = ekuivalen -inf — lihat docstring class.
        causal = np.tril(np.ones((t, t), dtype=bool))
        row_max = np.max(np.where(causal, scores, -np.inf), axis=-1, keepdims=True)
        scores = np.where(causal, scores, row_max - 1e9)
        probs = as_array(softmax(scores), dtype=self.config.dtype)
        context = np.matmul(probs, v)
        self._q, self._k, self._v, self._probs = q, k, v, probs
        self._shape = (b, h, t, dh)
        return context

    def backward(self, grad_out: Array) -> tuple[Array, Array, Array]:
        """dC (B,H,T,Dh) -> (dq, dk, dv) (TD §7); jalur kausal terjaga."""
        if self._probs is None or self._shape is None:
            raise RuntimeError("backward dipanggil sebelum forward")
        g = as_array(grad_out, dtype=self.config.dtype)
        if g.shape != self._shape:
            raise ValueError(f"grad_out harus berbentuk {self._shape}, dapat {g.shape}")
        probs, q, k, v = self._probs, self._q, self._k, self._v
        scale = float(np.sqrt(self._shape[3]))
        d_v = np.matmul(np.swapaxes(probs, -1, -2), g)
        d_p = np.matmul(g, np.swapaxes(v, -1, -2))
        # Softmax backward per baris; P = 0 di posisi future -> dS future = 0
        # sehingga grad tidak mengalir ke key/value masa depan (kausal).
        d_s = probs * (d_p - np.sum(probs * d_p, axis=-1, keepdims=True))
        d_q = np.matmul(d_s, k) / scale
        d_k = np.matmul(np.swapaxes(d_s, -1, -2), q) / scale
        return d_q, d_k, d_v
