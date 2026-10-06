"""Layers dan model container — forward pass saja (REQ-002, Stage 1).

DESAIN (sesuai ARCHITECTURE.md Stage 1):
- `Layer` adalah abstraction minimal: forward(x) -> Array, plus daftar parameter.
- Parameter disimpan sebagai ndarray biasa (dict bernama) — tanpa framework.
- `Sequential` menyusun layer berurutan; ini cukup untuk Stage 1.
- Tidak ada backward/optimizer di stage ini (Stage 2, menunggu OPEN DECISION-105).

Konvensi shape: (batch, features). Linear mengoperasikan fitur pada axis terakhir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np

from naze.core.numeric import Array, as_array, seeded_rng


class Layer(ABC):
    """Abstraksi dasar layer. Subclass menyimpan parameter di `self.params`."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.params: dict[str, Array] = {}

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """Hitung output layer untuk input (batch, ...)."""

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.params.values()))


class Linear(Layer):
    """Affine: y = x @ W + b.

    Shape:
        x: (batch, in_features) -> y: (batch, out_features)
        W: (in_features, out_features), b: (out_features,)
    Init: He/Kaiming-style (normal * sqrt(2/fan_in)) via seeded rng.
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int = 0) -> None:
        super().__init__(f"linear({in_features}->{out_features})")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features dan out_features harus > 0")
        rng = seeded_rng(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.params["W"] = rng.normal(0.0, np.sqrt(2.0 / in_features), (in_features, out_features))
        self.params["b"] = np.zeros(out_features, dtype=self.params["W"].dtype)

    def forward(self, x: Array) -> Array:
        x = as_array(x)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Linear mengharapkan input (batch, {self.in_features}), dapat {x.shape}")
        return x @ self.params["W"] + self.params["b"]


class Activation(Layer):
    """Wrapper layer untuk fungsi aktivasi stateless."""

    def __init__(self, fn: Callable[[Array], Array], name: str) -> None:
        super().__init__(name)
        self._fn = fn

    def forward(self, x: Array) -> Array:
        return self._fn(x)


class Sequential(Layer):
    """Menyusun layer berurutan; forward = komposisi kiri-ke-kanan."""

    def __init__(self, layers: Iterable[Layer]) -> None:
        super().__init__("sequential")
        self.layers = list(layers)

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameter_count(self) -> int:
        return sum(l.parameter_count() for l in self.layers)
