"""Test layers & Sequential (REQ-002) — forward pass diverifikasi numerik
dengan nilai acuan hand-computed, sesuai acceptance criteria Stage 1.
"""

import numpy as np
import pytest

from naze.core.numeric import seeded_rng
from naze.nn.activations import relu
from naze.nn.layers import Activation, Linear, Sequential


def test_linear_forward_hand_computed() -> None:
    layer = Linear(2, 2, seed=0)
    # Ganti parameter dengan nilai tetap agar hasil dapat dihitung manual.
    layer.params["W"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.params["b"] = np.array([0.5, -0.5])
    x = np.array([[1.0, 1.0]])
    # y = x@W + b = [1+3, 2+4] + [0.5, -0.5] = [4.5, 5.5]
    assert np.allclose(layer.forward(x), [[4.5, 5.5]])


def test_linear_init_deterministic_per_seed() -> None:
    a = Linear(3, 4, seed=7)
    b = Linear(3, 4, seed=7)
    assert np.array_equal(a.params["W"], b.params["W"])
    c = Linear(3, 4, seed=8)
    assert not np.array_equal(a.params["W"], c.params["W"])


def test_linear_rejects_bad_shapes() -> None:
    layer = Linear(2, 2)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        layer.forward(np.zeros(2))  # harus 2D (batch, features)
    with pytest.raises(ValueError):
        Linear(0, 2)


def test_sequential_mlp_forward_hand_computed() -> None:
    """MLP 2->2 + ReLU + Linear 2->1, seluruh nilai acuan dihitung manual."""
    l1 = Linear(2, 2, seed=0)
    l1.params["W"] = np.array([[1.0, -1.0], [2.0, 0.5]])
    l1.params["b"] = np.array([0.0, 0.0])
    l2 = Linear(2, 1, seed=0)
    l2.params["W"] = np.array([[1.0], [1.0]])
    l2.params["b"] = np.array([0.0])
    model = Sequential([l1, Activation(relu, "relu"), l2])

    x = np.array([[1.0, 1.0]])
    # h = x@W1 = [1*1 + 1*2, 1*(-1) + 1*0.5] = [3.0, -0.5]
    # relu(h) = [3.0, 0.0]
    # out = [3.0*1 + 0.0*1] = [3.0]
    assert np.allclose(model(x), [[3.0]])


def test_sequential_parameter_count() -> None:
    model = Sequential([Linear(4, 8, seed=0), Activation(relu, "relu"), Linear(8, 2, seed=0)])
    # Linear(4,8): 32 + 8 = 40; Linear(8,2): 16 + 2 = 18; total 58
    assert model.parameter_count() == 58


def test_forward_reproducible_with_same_seed() -> None:
    def build() -> Sequential:
        return Sequential([Linear(5, 16, seed=123), Activation(relu, "relu"), Linear(16, 3, seed=124)])

    x = seeded_rng(0).normal(size=(4, 5))
    assert np.allclose(build()(x), build()(x))
