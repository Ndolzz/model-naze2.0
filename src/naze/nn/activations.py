"""Activation functions + derivatives (REQ-002/REQ-003, Stage 1-2).

Kontrak:
- forward fn: f(x) -> Array (softmax hanya untuk output; grad digabung dengan CE).
- grad fn: df(x, y) -> Array, dengan y = f(x) (menghemat recomputasi).
"""

from __future__ import annotations

import numpy as np

from naze.core.numeric import Array, as_array


def relu(x) -> Array:
    return np.maximum(as_array(x), 0.0)


def relu_grad(x, y) -> Array:
    return (as_array(x) > 0).astype(as_array(x).dtype)


def sigmoid(x) -> Array:
    x = as_array(x)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sigmoid_grad(x, y) -> Array:
    return y * (1.0 - y)


def tanh(x) -> Array:
    return np.tanh(as_array(x))


def tanh_grad(x, y) -> Array:
    return 1.0 - y * y


def softmax(x, axis: int = -1) -> Array:
    x = as_array(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
