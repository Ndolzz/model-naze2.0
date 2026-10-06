"""Test backward pass + gradient check (REQ-003, Stage 2)."""

import numpy as np

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.activations import sigmoid, sigmoid_grad, tanh, tanh_grad
from naze.nn.layers import Activation, Linear, Sequential


def test_linear_gradient_check() -> None:
    layer = Linear(3, 2, seed=0)
    x = seeded_rng(1).normal(size=(4, 3))
    g = seeded_rng(2).normal(size=(4, 2))

    layer.forward(x)
    dx = layer.backward(g)

    def f() -> float:
        return float((layer.forward(x) * g).sum())

    assert_close(dx, numeric_grad(f, x))
    assert_close(layer.grads["W"], numeric_grad(f, layer.params["W"]))
    assert_close(layer.grads["b"], numeric_grad(f, layer.params["b"]))


def test_activation_grad_checks() -> None:
    x = seeded_rng(3).normal(size=(5, 4))
    for fn, gfn in [(tanh, tanh_grad), (sigmoid, sigmoid_grad)]:
        act = Activation(fn, gfn, fn.__name__)
        g = seeded_rng(4).normal(size=(5, 4))
        y = act.forward(x)
        dx = act.backward(g)
        assert_close(dx, g * gfn(x, y))


def test_sequential_backward_matches_numeric() -> None:
    model = Sequential([Linear(3, 4, seed=0), Activation(tanh, tanh_grad, "tanh"), Linear(4, 1, seed=1)])
    x = seeded_rng(5).normal(size=(6, 3))
    y = model.forward(x)
    g = np.ones_like(y)
    dx = model.backward(g)

    def f() -> float:
        return float(model.forward(x).sum())

    assert_close(dx, numeric_grad(f, x))
    l1, l3 = model.layers[0], model.layers[2]
    assert_close(l1.grads["W"], numeric_grad(f, l1.params["W"]))
    assert_close(l3.grads["W"], numeric_grad(f, l3.params["W"]))


def test_backward_before_forward_raises() -> None:
    import pytest
    from naze.nn.layers import Linear

    with pytest.raises(RuntimeError):
        Linear(2, 2).backward(np.ones((1, 2)))
