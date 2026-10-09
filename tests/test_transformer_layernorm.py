"""Test LayerNorm (M007-T007, REQ-011/REQ-003) — normalisasi per token pada
dimensi terakhir D + affine gamma/beta (TD §5 komponen 15, §7 backward).
Hanya T007 — tanpa residual/FFN/block (T009+). Expected values dihitung
via rumus referensi independen (mu/variance manual per token).
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import (
    Embedding,
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


def _nontrivial_ln(seed: int = 24) -> LayerNorm:
    """LayerNorm dgn gamma/beta non-trivial agar grad-check bermakna."""
    ln = LayerNorm(_small_cfg())
    ln.params["gamma"] = seeded_rng(seed).normal(size=4) + 2.0
    ln.params["beta"] = seeded_rng(seed + 1).normal(size=4)
    return ln


def test_constructor_and_parameter_shapes() -> None:
    """Konstruksi + shape parameter: gamma/beta (D,); total 2*D; eps."""
    cfg = _small_cfg()  # D=4
    ln = LayerNorm(cfg)
    assert ln.params["gamma"].shape == (4,)
    assert ln.params["beta"].shape == (4,)
    assert ln.parameter_count() == 2 * 4
    assert ln.eps == 1e-5  # default eksplisit (TD §5)
    dev = LayerNorm(TransformerConfig())  # D=64
    assert dev.parameter_count() == 2 * 64


def test_gamma_beta_initialization() -> None:
    """Init standar: gamma ones, beta zeros; dtype ikut config (D-007)."""
    ln = LayerNorm(_small_cfg())
    assert np.array_equal(ln.params["gamma"], np.ones(4))
    assert np.array_equal(ln.params["beta"], np.zeros(4))
    assert ln.params["gamma"].dtype == np.float64
    ln32 = LayerNorm(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    assert ln32.params["gamma"].dtype == np.float32
    assert ln32.params["beta"].dtype == np.float32


def test_eps_configurable() -> None:
    """eps configurable; <= 0 / NaN ditolak ValueError (TD §10)."""
    ln = LayerNorm(_small_cfg(), eps=1e-3)
    assert ln.eps == 1e-3
    assert ln.forward(_x(0)).shape == (2, 5, 4)  # eps valid tidak mengganggu
    with pytest.raises(ValueError):
        LayerNorm(_small_cfg(), eps=0.0)
    with pytest.raises(ValueError):
        LayerNorm(_small_cfg(), eps=-1e-5)
    with pytest.raises(ValueError):
        LayerNorm(_small_cfg(), eps=float("nan"))


def test_output_shape() -> None:
    """forward(x: (B,T,D)) -> (B,T,D) untuk dev & config kecil (TD §5)."""
    ln = LayerNorm(_small_cfg())
    assert ln.forward(_x(1)).shape == (2, 5, 4)
    for b, t in ((1, 1), (1, 8), (3, 3)):
        assert ln.forward(_x(2, b=b, t=t)).shape == (b, t, 4)
    dev = LayerNorm(TransformerConfig())  # D=64, T_max=64
    assert dev.forward(seeded_rng(3).normal(size=(2, 5, 64))).shape == (2, 5, 64)


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    ln32 = LayerNorm(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    x = seeded_rng(4).normal(size=(1, 4, 8)).astype(np.float32)
    assert ln32.forward(x).dtype == np.float32
    assert ln32.forward(x.astype(np.float64)).dtype == np.float32
    ln64 = LayerNorm(_small_cfg())
    assert ln64.forward(_x(5)).dtype == np.float64


def test_mean_and_variance_per_token() -> None:
    """Output mean ~ 0 & variance ~ 1 per token (AC §13-6; TD §11)."""
    ln = LayerNorm(_small_cfg())
    x = _x(6)
    out = ln.forward(x)
    mu = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    # gamma=1, beta=0 -> out = x_hat; variance tepat var/(var+eps).
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-12)
    assert np.allclose(out.var(axis=-1), (var / (var + ln.eps)).squeeze(-1))
    # eps kecil -> variance praktis 1 (bukti normalisasi bekerja).
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-4)


def test_affine_gamma_beta() -> None:
    """y = gamma * x_hat + beta — affine diterapkan benar (TD §7)."""
    ln = LayerNorm(_small_cfg(), eps=1e-8)
    gamma = seeded_rng(7).normal(size=4) + 2.0
    beta = seeded_rng(8).normal(size=4) - 1.0
    ln.params["gamma"] = gamma
    ln.params["beta"] = beta
    x = _x(9)
    mu = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + ln.eps)
    assert np.allclose(ln.forward(x), gamma * x_hat + beta)


def test_normalization_axis_only() -> None:
    """Normalisasi hanya axis=-1: token (0,3) tak peduli token lain."""
    ln = LayerNorm(_small_cfg())
    x = _x(10)
    full = ln.forward(x)
    sub = x[0:1, 3:4, :]  # token tunggal, batch/seq lain dibuang
    assert np.allclose(ln.forward(sub)[0, 0], full[0, 3])


def test_deterministic_construction_and_forward() -> None:
    """Deterministik per seed-agnostic: init & forward identik (REQ-101)."""
    a = LayerNorm(_small_cfg())
    b = LayerNorm(_small_cfg())
    assert np.array_equal(a.params["gamma"], b.params["gamma"])
    assert np.array_equal(a.params["beta"], b.params["beta"])
    x = _x(11)
    assert np.array_equal(a.forward(x), b.forward(x))
    assert np.array_equal(a.forward(x), a.forward(x))


def test_invalid_input_dimensions_rejected() -> None:
    """Input harus (B,T,D) 3D — 2D/4D ditolak (TD §10)."""
    ln = LayerNorm(_small_cfg())
    with pytest.raises(ValueError):
        ln.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        ln.forward(np.zeros((2, 5, 4, 1)))  # 4D


def test_wrong_model_dimension_rejected() -> None:
    """D harus d_model; D != d_model ditolak (TD §10)."""
    ln = LayerNorm(_small_cfg())  # D=4
    with pytest.raises(ValueError):
        ln.forward(np.zeros((2, 5, 8)))  # D=8 != d_model


def test_empty_batch_sequence_rejected() -> None:
    """B=0 atau T=0 ditolak ValueError (TD §10)."""
    ln = LayerNorm(_small_cfg())
    with pytest.raises(ValueError):
        ln.forward(np.zeros((0, 5, 4)))  # B=0
    with pytest.raises(ValueError):
        ln.forward(np.zeros((2, 0, 4)))  # T=0


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    ln = LayerNorm(_small_cfg())  # T_max=8
    with pytest.raises(ValueError):
        ln.forward(_x(12, t=9))


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    ln = LayerNorm(_small_cfg())
    x = _x(13)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ln.forward(x)
    y = _x(14)
    y[1, 4, 3] = np.inf
    with pytest.raises(ValueError):
        ln.forward(y)


def test_backward_shape_and_guard() -> None:
    """backward -> dx (B,T,D); guard urutan & shape grad (TD §7/§10)."""
    ln = LayerNorm(_small_cfg())
    with pytest.raises(RuntimeError):
        ln.backward(np.zeros((2, 5, 4)))  # belum forward
    x = _x(15)
    ln.forward(x)
    g = seeded_rng(16).normal(size=(2, 5, 4))
    dx = ln.backward(g)
    assert dx.shape == (2, 5, 4)
    assert np.all(np.isfinite(dx))
    with pytest.raises(ValueError):
        ln.backward(np.zeros((2, 6, 4)))  # shape grad salah


def test_grads_gamma_beta_values() -> None:
    """d_beta = sum(dY,(B,T)); d_gamma = sum(dY*x_hat,(B,T)) — manual."""
    ln = _nontrivial_ln()
    x = _x(17)
    g = seeded_rng(18).normal(size=(2, 5, 4))
    out = ln.forward(x)
    mu = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + ln.eps)
    ln.backward(g)
    assert ln.grads["gamma"].shape == (4,)
    assert ln.grads["beta"].shape == (4,)
    assert np.allclose(ln.grads["beta"], g.sum(axis=(0, 1)))
    assert np.allclose(ln.grads["gamma"], (g * x_hat).sum(axis=(0, 1)))
    assert np.any(ln.grads["gamma"] != 0.0)
    assert out.shape == (2, 5, 4)  # forward tetap dipakai (no dead value)


def test_numeric_grad_check_input() -> None:
    """Central difference vs analytic dx (engine gradcheck, REQ-003)."""
    ln = _nontrivial_ln()
    x = _x(19, b=2, t=3)
    g = seeded_rng(20).normal(size=(2, 3, 4))
    ln.forward(x)
    dx = ln.backward(g)

    def loss() -> float:
        return float((ln.forward(x) * g).sum())

    assert_close(dx, numeric_grad(loss, x))


def test_numeric_grad_check_gamma() -> None:
    """Central difference vs analytic d_gamma (gradcheck, REQ-003)."""
    ln = _nontrivial_ln()
    x = _x(21, b=2, t=3)
    g = seeded_rng(22).normal(size=(2, 3, 4))
    ln.forward(x)
    ln.backward(g)  # mengisi grads
    dgamma_analytic = ln.grads["gamma"].copy()
    w = ln.params["gamma"]

    def loss() -> float:
        return float((ln.forward(x) * g).sum())

    assert_close(dgamma_analytic, numeric_grad(loss, w))


def test_numeric_grad_check_beta() -> None:
    """Central difference vs analytic d_beta (gradcheck, REQ-003)."""
    ln = _nontrivial_ln()
    x = _x(23, b=2, t=3)
    g = seeded_rng(24).normal(size=(2, 3, 4))
    ln.forward(x)
    ln.backward(g)  # mengisi grads
    dbeta_analytic = ln.grads["beta"].copy()
    w = ln.params["beta"]

    def loss() -> float:
        return float((ln.forward(x) * g).sum())

    assert_close(dbeta_analytic, numeric_grad(loss, w))


def test_constant_and_large_input_stability() -> None:
    """§5 stabilitas: variance 0 & nilai besar tetap finite (no NaN/Inf)."""
    ln = LayerNorm(_small_cfg())
    xc = np.full((2, 3, 4), 7.5)  # konstan per token -> variance 0
    out = ln.forward(xc)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)  # x_hat = 0 -> y = beta = 0
    g = seeded_rng(25).normal(size=(2, 3, 4))
    dx = ln.backward(g)
    assert np.all(np.isfinite(dx))
    big = _x(26) * 1e8  # besar tapi finite
    outb = ln.forward(big)
    assert np.all(np.isfinite(outb))
    # Grad harus cocok dengan cache forward terakhir (big: (2, 5, 4)).
    g_big = seeded_rng(27).normal(size=(2, 5, 4))
    assert np.all(np.isfinite(ln.backward(g_big)))


def test_regression_t001_t006_compose() -> None:
    """T001-T006 utuh + T007: ids -> Emb -> Pos -> MHA -> LN (B,T,D)."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    mha = MultiHeadAttention(cfg, seed=2)
    ln = LayerNorm(cfg)
    ids = np.zeros((2, 5), dtype=np.int64)
    h = pos.forward(emb.forward(ids))
    out = ln.forward(mha.forward(h))
    assert out.shape == (2, 5, 4)
    assert np.all(np.isfinite(out))
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-4)
    # MHA tetap layer terpisah — output MHA tidak tersentuh LayerNorm.
    mha_out = mha.forward(h)
    assert np.array_equal(mha_out, mha.forward(h))
    # LN terpisah pula dari konstruksi MHA (bukan sub-layer MHA).
    assert ln.parameter_count() == 2 * 4  # hanya gamma+beta
