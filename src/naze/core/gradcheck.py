"""Numerical gradient check (REQ-003 acceptance criteria, Stage 2).

Membandingkan gradien analitik vs central difference numerik.
Toleransi default 1e-5, sesuai DECISION-105 (float64).
"""

from __future__ import annotations

import numpy as np

from naze.core.numeric import Array


def numeric_grad(f, x: Array, eps: float = 1e-6) -> Array:
    """Central-difference gradien skalar f terhadap array x."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        g[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return g


def assert_close(analytic: Array, numeric: Array, tol: float = 1e-5, msg: str = "") -> None:
    if analytic.shape != numeric.shape:
        raise AssertionError(f"shape mismatch: {analytic.shape} vs {numeric.shape}")
    err = np.max(np.abs(analytic - numeric)) / (np.max(np.abs(numeric)) + 1e-12)
    if err > tol:
        raise AssertionError(f"gradient check gagal (rel err {err:.2e} > {tol:.1e}) {msg}")
