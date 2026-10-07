"""Test FeedForward (M007-T008, REQ-011/REQ-003) — Linear(D->D_ff) -> tanh ->
Linear(D_ff->D) per token (TD §5 komponen 16, §7 backward). Hanya T008 —
tanpa residual/LayerNorm/block (T009+). Expected values dihitung via
sub-komponen referensi independen (fc1/act/fc2 manual).
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import (
    Embedding,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    PositionalRepr,
    TransformerConfig,
)


def _small_cfg(**over) -> TransformerConfig:
    """Config dev/test kecil; seluruh invariant T001 tetap terpenuhi."""
    params: dict = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                        max_sequence_length=8, seed=0)
    params.update(over)
    return TransformerConfig(**params)


def _x(seed: int, b: int = 2, t: int = 5, d: int = 4) -> np.ndarray:
    """Input acak deterministik (B,T,D) untuk config kecil."""
    return seeded_rng(seed).normal(size=(b, t, d))


def _param_count(cfg: TransformerConfig) -> int:
    """Formula parameter dari config (bukan hard-coded angka)."""
    d, dff = cfg.d_model, cfg.d_ff
    return (d * dff + dff) + (dff * d + d)


def test_constructor_and_parameter_count() -> None:
    """Konstruksi + shape parameter & count dari config (dev = 16.576)."""
    cfg = _small_cfg()  # D=4, D_ff=8
    ffn = FeedForward(cfg)
    assert ffn.fc1.params["W"].shape == (4, 8)
    assert ffn.fc1.params["b"].shape == (8,)
    assert ffn.fc2.params["W"].shape == (8, 4)
    assert ffn.fc2.params["b"].shape == (4,)
    assert ffn.parameter_count() == _param_count(cfg) == 76
    dev = FeedForward(TransformerConfig())  # D=64, D_ff=128
    assert dev.parameter_count() == _param_count(TransformerConfig()) == 16576


def test_hidden_dimension_uses_d_ff() -> None:
    """Hidden dimension mengikuti config.d_ff (bukan konstanta lain)."""
    cfg = _small_cfg(d_ff=16)
    ffn = FeedForward(cfg)
    assert ffn.fc1.params["W"].shape == (4, 16)
    assert ffn.fc2.params["W"].shape == (16, 4)
    assert ffn.parameter_count() == _param_count(cfg)
    assert ffn.forward(_x(0)).shape == (2, 5, 4)


def test_output_shape() -> None:
    """forward(x: (B,T,D)) -> (B,T,D) untuk dev & config kecil (TD §5)."""
    ffn = FeedForward(_small_cfg())
    assert ffn.forward(_x(1)).shape == (2, 5, 4)
    for b, t in ((1, 1), (1, 8), (3, 3)):
        assert ffn.forward(_x(2, b=b, t=t)).shape == (b, t, 4)
    dev = FeedForward(TransformerConfig())  # D=64, D_ff=128, T_max=64
    assert dev.forward(seeded_rng(3).normal(size=(2, 5, 64))).shape == (2, 5, 64)


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    ffn32 = FeedForward(TransformerConfig(d_model=8, num_heads=2, d_ff=12, dtype=np.float32))
    x = seeded_rng(4).normal(size=(1, 4, 8)).astype(np.float32)
    assert ffn32.forward(x).dtype == np.float32
    assert ffn32.forward(x.astype(np.float64)).dtype == np.float32
    ffn64 = FeedForward(_small_cfg())
    assert ffn64.forward(_x(5)).dtype == np.float64


def test_flatten_preserves_token_order() -> None:
    """Flatten row-major: token (b,t) identik pipeline token tunggal."""
    ffn = FeedForward(_small_cfg(), seed=6)
    x = _x(7)
    out = ffn.forward(x)
    for b, t in ((0, 0), (1, 4), (0, 3)):
        single = ffn.fc2.forward(
            np.tanh(ffn.fc1.forward(x[b, t].reshape(1, 4)))
        ).reshape(4)
        assert np.allclose(out[b, t], single)


def test_tanh_activation_applied() -> None:
    """Hidden layer lewat tanh (keputusan owner; bukan relu/identity)."""
    ffn = FeedForward(_small_cfg(), seed=8)
    x = _x(9)
    flat = x.reshape(10, 4)
    manual = ffn.fc2.forward(np.tanh(ffn.fc1.forward(flat))).reshape(2, 5, 4)
    assert np.allclose(ffn.forward(x), manual)
    relu_pipe = ffn.fc2.forward(
        np.maximum(ffn.fc1.forward(flat), 0.0)
    ).reshape(2, 5, 4)
    assert not np.allclose(ffn.forward(x), relu_pipe)


def test_deterministic_initialization() -> None:
    """Seed sama -> parameter identik (REQ-101); seed beda -> beda."""
    a = FeedForward(_small_cfg(), seed=10)
    b = FeedForward(_small_cfg(), seed=10)
    for layer_a, layer_b in ((a.fc1, b.fc1), (a.fc2, b.fc2)):
        assert np.array_equal(layer_a.params["W"], layer_b.params["W"])
        assert np.array_equal(layer_a.params["b"], layer_b.params["b"])
    c = FeedForward(_small_cfg(), seed=11)
    assert not np.array_equal(a.fc1.params["W"], c.fc1.params["W"])
    assert not np.array_equal(a.fc2.params["W"], c.fc2.params["W"])


def test_deterministic_forward() -> None:
    """Forward deterministik: input sama -> output identik (REQ-101)."""
    ffn = FeedForward(_small_cfg())
    x = _x(12)
    assert np.array_equal(ffn.forward(x), ffn.forward(x))
    ffn2 = FeedForward(_small_cfg())
    assert np.array_equal(ffn.forward(x), ffn2.forward(x))


def test_invalid_input_dimensions_rejected() -> None:
    """Input harus (B,T,D) 3D — 2D/4D ditolak (TD §10)."""
    ffn = FeedForward(_small_cfg())
    with pytest.raises(ValueError):
        ffn.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        ffn.forward(np.zeros((2, 5, 4, 1)))  # 4D


def test_wrong_model_dimension_rejected() -> None:
    """D harus d_model; D != d_model ditolak (TD §10)."""
    ffn = FeedForward(_small_cfg())  # D=4
    with pytest.raises(ValueError):
        ffn.forward(np.zeros((2, 5, 8)))  # D=8 != d_model


def test_empty_batch_sequence_rejected() -> None:
    """B=0 atau T=0 ditolak ValueError (TD §10)."""
    ffn = FeedForward(_small_cfg())
    with pytest.raises(ValueError):
        ffn.forward(np.zeros((0, 5, 4)))  # B=0
    with pytest.raises(ValueError):
        ffn.forward(np.zeros((2, 0, 4)))  # T=0


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    ffn = FeedForward(_small_cfg())  # T_max=8
    with pytest.raises(ValueError):
        ffn.forward(_x(13, t=9))


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    ffn = FeedForward(_small_cfg())
    x = _x(14)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ffn.forward(x)
    y = _x(15)
    y[1, 4, 3] = np.inf
    with pytest.raises(ValueError):
        ffn.forward(y)


def test_backward_shape_and_guard() -> None:
    """backward -> dx (B,T,D); guard urutan & shape grad (TD §7/§10)."""
    ffn = FeedForward(_small_cfg())
    with pytest.raises(RuntimeError):
        ffn.backward(np.zeros((2, 5, 4)))  # belum forward
    x = _x(16)
    ffn.forward(x)
    g = seeded_rng(17).normal(size=(2, 5, 4))
    dx = ffn.backward(g)
    assert dx.shape == (2, 5, 4)
    assert np.all(np.isfinite(dx))
    with pytest.raises(ValueError):
        ffn.backward(np.zeros((2, 6, 4)))  # shape grad salah


def test_gradient_flows_to_input_and_params() -> None:
    """Grad mengalir: dx != 0; dW1/db1/dW2/db2 terisi (TD §7)."""
    ffn = FeedForward(_small_cfg(), seed=18)
    x = _x(19)
    ffn.forward(x)
    g = seeded_rng(20).normal(size=(2, 5, 4))
    dx = ffn.backward(g)
    assert np.any(dx != 0.0)
    assert ffn.fc1.grads["W"].shape == (4, 8)
    assert ffn.fc2.grads["W"].shape == (8, 4)
    for layer in (ffn.fc1, ffn.fc2):
        assert np.any(layer.grads["W"] != 0.0)
        assert np.any(layer.grads["b"] != 0.0)


def test_numeric_grad_check_input() -> None:
    """Central difference vs analytic dx (engine gradcheck, REQ-003)."""
    ffn = FeedForward(_small_cfg(), seed=21)
    x = _x(22, b=2, t=3)
    g = seeded_rng(23).normal(size=(2, 3, 4))
    ffn.forward(x)
    dx = ffn.backward(g)

    def loss() -> float:
        return float((ffn.forward(x) * g).sum())

    assert_close(dx, numeric_grad(loss, x))


def test_numeric_grad_check_w1() -> None:
    """Central difference vs analytic dW1 (gradcheck, REQ-003)."""
    ffn = FeedForward(_small_cfg(), seed=24)
    x = _x(25, b=2, t=3)
    g = seeded_rng(26).normal(size=(2, 3, 4))
    ffn.forward(x)
    ffn.backward(g)  # mengisi grads
    dw1_analytic = ffn.fc1.grads["W"].copy()
    w = ffn.fc1.params["W"]

    def loss() -> float:
        return float((ffn.forward(x) * g).sum())

    assert_close(dw1_analytic, numeric_grad(loss, w))


def test_numeric_grad_check_w2() -> None:
    """Central difference vs analytic dW2 (gradcheck, REQ-003)."""
    ffn = FeedForward(_small_cfg(), seed=27)
    x = _x(28, b=2, t=3)
    g = seeded_rng(29).normal(size=(2, 3, 4))
    ffn.forward(x)
    ffn.backward(g)  # mengisi grads
    dw2_analytic = ffn.fc2.grads["W"].copy()
    w = ffn.fc2.params["W"]

    def loss() -> float:
        return float((ffn.forward(x) * g).sum())

    assert_close(dw2_analytic, numeric_grad(loss, w))


def test_regression_t001_t007_compose() -> None:
    """T001-T007 utuh + T008: ids -> Emb -> Pos -> MHA -> LN -> FFN."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2, D_ff=8
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    mha = MultiHeadAttention(cfg, seed=2)
    ln = LayerNorm(cfg)
    ffn = FeedForward(cfg, seed=3)
    ids = np.zeros((2, 5), dtype=np.int64)
    h = pos.forward(emb.forward(ids))
    out = ffn.forward(ln.forward(mha.forward(h)))
    assert out.shape == (2, 5, 4)
    assert np.all(np.isfinite(out))
    # FFN layer terpisah: bukan bagian LayerNorm/MHA (parameter murni FFN).
    assert ffn.parameter_count() == _param_count(cfg)
    assert ln.parameter_count() == 2 * 4
    assert mha.parameter_count() == 4 * (4 * 4 + 4)
