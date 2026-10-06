"""Activation functions — implementasi sendiri di atas NumPy (REQ-002, Stage 1).

Kontrak setiap aktivasi:
- forward(x) -> Array, beroperasi elementwise (kecuali softmax pada axis terakhir).
- Stateless (murni fungsi) pada tahap ini; state untuk backward menyusul di Stage 2.
"""

from __future__ import annotations

import numpy as np

from naze.core.numeric import Array, as_array


def relu(x) -> Array:
    """max(0, x) elementwise."""
    return np.maximum(as_array(x), 0.0)


def sigmoid(x) -> Array:
    """1 / (1 + exp(-x)), stabil secara numerik."""
    x = as_array(x)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def tanh(x) -> Array:
    """tanh elementwise (NumPy native, stabil)."""
    return np.tanh(as_array(x))


def softmax(x, axis: int = -1) -> Array:
    """Softmax stabil pada `axis` (default: axis batch terakhir)."""
    x = as_array(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
