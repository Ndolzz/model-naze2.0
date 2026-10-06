"""Layers: forward + backward pass (REQ-002/REQ-003, DECISION-008, DECISION-105).

Pendekatan: backprop terstruktur per-layer (bukan graph-based autodiff).
- forward(x) meng-cache input; backward(grad_out) menghitung grad_input
  dan menyimpan grad parameter di self.grads.
- Sequential.backward = komposisi terbalik.
Konvensi shape: (batch, features).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np

from naze.core.numeric import Array, as_array, seeded_rng


class Layer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.params: dict[str, Array] = {}
        self.grads: dict[str, Array] = {}

    @abstractmethod
    def forward(self, x: Array) -> Array:
        ...

    @abstractmethod
    def backward(self, grad_out: Array) -> Array:
        ...

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.params.values()))


class Linear(Layer):
    """y = x @ W + b. backward: dW = x.T @ g, db = sum(g), dx = g @ W.T."""

    def __init__(self, in_features: int, out_features: int, *, seed: int = 0) -> None:
        super().__init__(f"linear({in_features}->{out_features})")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features dan out_features harus > 0")
        rng = seeded_rng(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.params["W"] = rng.normal(0.0, np.sqrt(2.0 / in_features), (in_features, out_features))
        self.params["b"] = np.zeros(out_features, dtype=self.params["W"].dtype)
        self._x: Array | None = None

    def forward(self, x: Array) -> Array:
        x = as_array(x)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Linear mengharapkan input (batch, {self.in_features}), dapat {x.shape}")
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out: Array) -> Array:
        if self._x is None:
            raise RuntimeError("backward dipanggil sebelum forward")
        g = as_array(grad_out)
        self.grads["W"] = self._x.T @ g
        self.grads["b"] = g.sum(axis=0)
        return g @ self.params["W"].T


class Activation(Layer):
    """Wrapper aktivasi stateless; grad_fn(x, y) -> dy/dx."""

    def __init__(self, fn: Callable[[Array], Array], grad_fn: Callable[[Array, Array], Array], name: str) -> None:
        super().__init__(name)
        self._fn = fn
        self._grad_fn = grad_fn
        self._x: Array | None = None
        self._y: Array | None = None

    def forward(self, x: Array) -> Array:
        self._x = as_array(x)
        self._y = self._fn(self._x)
        return self._y

    def backward(self, grad_out: Array) -> Array:
        if self._x is None or self._y is None:
            raise RuntimeError("backward dipanggil sebelum forward")
        return as_array(grad_out) * self._grad_fn(self._x, self._y)


class Sequential(Layer):
    def __init__(self, layers: Iterable[Layer]) -> None:
        super().__init__("sequential")
        self.layers = list(layers)

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: Array) -> Array:
        g = as_array(grad_out)
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def parameter_count(self) -> int:
        return sum(l.parameter_count() for l in self.layers)
