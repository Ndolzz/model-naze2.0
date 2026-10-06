"""Numerical utilities — fondasi komputasi Naze (REQ-001).

Hanya utilitas deterministik dan helper NumPy. Tidak ada magic:
semua yang punya randomness harus melalui `seeded_rng` agar reproducible.
"""

from __future__ import annotations

import numpy as np

# Tipe dasar untuk seluruh array di Naze: float64 untuk presisi gradien
# pada tahap awal (dapat ditinjau ulang — lihat DECISION_LOG).
DType = np.float64
Array = np.ndarray


def seeded_rng(seed: int) -> np.random.Generator:
    """Generator acak deterministik (REQ-101 Reproducibility).

    Seluruh inisialisasi parameter WAJIB menggunakan generator ini,
    bukan np.random global, agar hasil reproducible per-seed.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed harus integer >= 0, dapat: {seed!r}")
    return np.random.default_rng(int(seed))


def as_array(x, dtype: np.dtype | type = DType) -> Array:
    """Konversi input menjadi ndarray float64. Menolak NaN/Inf."""
    arr = np.asarray(x, dtype=dtype)
    if not np.all(np.isfinite(arr)):
        raise ValueError("input mengandung nilai non-finite (NaN/Inf)")
    return arr
