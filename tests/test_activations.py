"""Test activations (REQ-002) — nilai acuan dihitung manual."""

import numpy as np

from naze.nn.activations import relu, sigmoid, softmax, tanh


def test_relu() -> None:
    x = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(relu(x), [[0.0, 0.0, 3.0]])


def test_sigmoid_hand_computed() -> None:
    # sigmoid(0) = 0.5; sigmoid(1) ≈ 0.7310585786300049
    out = sigmoid(np.array([0.0, 1.0]))
    assert out[0] == 0.5
    assert np.isclose(out[1], 0.7310585786300049)


def test_sigmoid_extreme_values_stable() -> None:
    out = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 1.0


def test_tanh_hand_computed() -> None:
    assert np.isclose(tanh(np.array([0.0]))[0], 0.0)
    assert np.isclose(tanh(np.array([np.log(2.0)]))[0], 0.6)


def test_softmax_sums_to_one() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = softmax(x, axis=-1)
    assert np.allclose(out.sum(axis=-1), 1.0)
    # Uniform case: 1/3 masing-masing
    assert np.allclose(out[1], 1.0 / 3.0)


def test_softmax_extreme_values_stable() -> None:
    out = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.all(np.isfinite(out))
    assert np.isclose(out[0, 0], 1.0)
