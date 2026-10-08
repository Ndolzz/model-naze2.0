"""Test ResidualBlock (M007-T009, REQ-011/REQ-003) — helper post-norm
x + branch(x) -> LayerNorm (TD §5 komponen 14-15; TASKS T009 "x + f(x)
-> LN"). Branch = MultiHeadAttention / FeedForward existing. Hanya T009 —
tanpa TransformerBlock/stacking (T010+). Expected values dihitung via
sub-komponen referensi independen.
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import Array, seeded_rng
from naze.nn.layers import Layer
from naze.nn.transformer import (
    Embedding,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    PositionalRepr,
    ResidualBlock,
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


class _BadShapeLayer(Layer):
    """Layer uji: output shape salah — memicu guard no-implicit-broadcast."""

    def __init__(self) -> None:
        super().__init__("bad_shape")

    def forward(self, x: Array) -> Array:
        return np.zeros((x.shape[0], x.shape[1], x.shape[2] + 1))

    def backward(self, grad_out: Array) -> Array:
        return grad_out


def test_constructor_and_sublayer_composition() -> None:
    """Konstruksi: branch = instance existing; LN internal; config shared."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=1)
    block = ResidualBlock(cfg, mha)
    assert block.branch is mha  # orchestration — bukan salinan
    assert block.config is cfg
    assert isinstance(block.norm, LayerNorm)
    assert "mha" in block.name  # nama layer mereferensikan branch


def test_output_shape() -> None:
    """forward(x: (B,T,D)) -> (B,T,D) utk branch MHA & FFN + dev (TD §5)."""
    cfg = _small_cfg()
    for branch in (MultiHeadAttention(cfg, seed=1), FeedForward(cfg, seed=2)):
        block = ResidualBlock(cfg, branch)
        assert block.forward(_x(3)).shape == (2, 5, 4)
        for b, t in ((1, 1), (1, 8), (3, 3)):
            assert block.forward(_x(4, b=b, t=t)).shape == (b, t, 4)
    dev = ResidualBlock(TransformerConfig(), MultiHeadAttention(TransformerConfig(), seed=5))
    assert dev.forward(seeded_rng(6).normal(size=(2, 5, 64))).shape == (2, 5, 64)


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    cfg32 = TransformerConfig(d_model=8, num_heads=2, d_ff=12, dtype=np.float32)
    block32 = ResidualBlock(cfg32, MultiHeadAttention(cfg32, seed=1))
    x = seeded_rng(7).normal(size=(1, 4, 8)).astype(np.float32)
    assert block32.forward(x).dtype == np.float32
    block64 = ResidualBlock(_small_cfg(), FeedForward(_small_cfg(), seed=2))
    assert block64.forward(_x(8)).dtype == np.float64


def test_residual_connection_and_operation_order() -> None:
    """y = LN(x + branch(x)) persis; AC: dengan vs tanpa residual beda."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=1)
    block = ResidualBlock(cfg, mha)
    x = _x(2)
    y = block.forward(x)
    # urutan persis TASKS T009: x + f(x) -> LN (post-norm)
    ref = block.norm.forward(x + mha.forward(x))
    assert np.allclose(y, ref)
    # AC T009: residual benar-benar digunakan — tanpa residual beda
    no_res = block.norm.forward(mha.forward(x))
    assert not np.allclose(y, no_res)
    # bukan urutan lain: x + LN(f(x)) (pre-norm-ish) berbeda
    other = x + block.norm.forward(mha.forward(x))
    assert not np.allclose(y, other)


def test_deterministic_construction_and_forward() -> None:
    """Sublayer seed sama -> block identik; forward deterministik (REQ-101)."""
    cfg = _small_cfg()
    a = ResidualBlock(cfg, MultiHeadAttention(cfg, seed=9))
    b = ResidualBlock(cfg, MultiHeadAttention(cfg, seed=9))
    assert np.array_equal(a.norm.params["gamma"], b.norm.params["gamma"])
    assert np.array_equal(a.branch.qkv.q_proj.params["W"], b.branch.qkv.q_proj.params["W"])
    x = _x(10)
    assert np.array_equal(a.forward(x), b.forward(x))
    assert np.array_equal(a.forward(x), a.forward(x))


def test_parameter_count() -> None:
    """Count = branch + norm aktual (dihitung, bukan hard-coded)."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=1)
    block = ResidualBlock(cfg, mha)
    assert block.parameter_count() == mha.parameter_count() + block.norm.parameter_count()
    assert block.parameter_count() == 4 * (4 * 4 + 4) + 2 * 4  # 88
    ffn = FeedForward(cfg, seed=2)
    block2 = ResidualBlock(cfg, ffn)
    assert block2.parameter_count() == ((4 * 8 + 8) + (8 * 4 + 4)) + 2 * 4  # 84


def test_parameter_aggregation() -> None:
    """Param tetap via mekanisme sublayer; tidak ada parameter duplikat."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=1)
    block = ResidualBlock(cfg, mha)
    assert block.params == {}  # tanpa duplikat — param hidup di sublayer
    assert mha.out_proj.params["W"].shape == (4, 4)
    assert block.norm.params["gamma"].shape == (4,)
    assert isinstance(block.norm, LayerNorm)
    assert block.norm.parameter_count() == 2 * 4


def test_invalid_input_dimensions_rejected() -> None:
    """Input harus (B,T,D) 3D — 2D/4D ditolak (TD §10)."""
    block = ResidualBlock(_small_cfg(), MultiHeadAttention(_small_cfg(), seed=1))
    with pytest.raises(ValueError):
        block.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        block.forward(np.zeros((2, 5, 4, 1)))  # 4D


def test_wrong_model_dimension_rejected() -> None:
    """D harus d_model; D != d_model ditolak (TD §10)."""
    block = ResidualBlock(_small_cfg(), FeedForward(_small_cfg(), seed=1))
    with pytest.raises(ValueError):
        block.forward(np.zeros((2, 5, 8)))  # D=8 != d_model


def test_empty_batch_sequence_rejected() -> None:
    """B=0 atau T=0 ditolak ValueError (TD §10)."""
    block = ResidualBlock(_small_cfg(), MultiHeadAttention(_small_cfg(), seed=1))
    with pytest.raises(ValueError):
        block.forward(np.zeros((0, 5, 4)))  # B=0
    with pytest.raises(ValueError):
        block.forward(np.zeros((2, 0, 4)))  # T=0


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    block = ResidualBlock(_small_cfg(), FeedForward(_small_cfg(), seed=1))
    with pytest.raises(ValueError):
        block.forward(_x(12, t=9))


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    block = ResidualBlock(_small_cfg(), MultiHeadAttention(_small_cfg(), seed=1))
    x = _x(13)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        block.forward(x)
    y = _x(14)
    y[1, 4, 3] = np.inf
    with pytest.raises(ValueError):
        block.forward(y)


def test_no_implicit_broadcasting_rejected() -> None:
    """Branch output shape != input -> ValueError (bukan broadcast senyap)."""
    block = ResidualBlock(_small_cfg(), _BadShapeLayer())
    with pytest.raises(ValueError):
        block.forward(_x(15))


def test_backward_shape_and_guard() -> None:
    """backward -> dx (B,T,D); guard urutan & shape grad (TD §7/§10)."""
    block = ResidualBlock(_small_cfg(), MultiHeadAttention(_small_cfg(), seed=1))
    with pytest.raises(RuntimeError):
        block.backward(np.zeros((2, 5, 4)))  # belum forward
    x = _x(16)
    block.forward(x)
    g = seeded_rng(17).normal(size=(2, 5, 4))
    dx = block.backward(g)
    assert dx.shape == (2, 5, 4)
    assert np.all(np.isfinite(dx))
    with pytest.raises(ValueError):
        block.backward(np.zeros((2, 6, 4)))  # shape grad salah


def test_gradient_flow_to_input_and_all_sublayers() -> None:
    """dx != 0; grad MHA + LN (block-1) dan FFN + LN (block-2) terisi."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=1)
    block = ResidualBlock(cfg, mha)
    x = _x(18)
    block.forward(x)
    g = seeded_rng(19).normal(size=(2, 5, 4))
    dx = block.backward(g)
    assert np.any(dx != 0.0)
    for proj in (mha.qkv.q_proj, mha.qkv.k_proj, mha.qkv.v_proj, mha.out_proj):
        assert np.any(proj.grads["W"] != 0.0)
        assert np.any(proj.grads["b"] != 0.0)
    assert np.any(block.norm.grads["gamma"] != 0.0)
    assert np.any(block.norm.grads["beta"] != 0.0)
    ffn = FeedForward(cfg, seed=2)
    block2 = ResidualBlock(cfg, ffn)
    block2.forward(x)
    dx2 = block2.backward(g)
    assert np.any(dx2 != 0.0)
    for lin in (ffn.fc1, ffn.fc2):
        assert np.any(lin.grads["W"] != 0.0)
        assert np.any(lin.grads["b"] != 0.0)
    assert np.any(block2.norm.grads["gamma"] != 0.0)


def test_residual_gradient_path() -> None:
    """Jalur residual: dx = d_norm + d_branch (grad tidak hilang)."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=3)
    block = ResidualBlock(cfg, mha)
    x = _x(20)
    g = seeded_rng(21).normal(size=(2, 5, 4))
    block.forward(x)
    dx = block.backward(g)
    # rekomputasi manual via sublayer (cache sama; deterministik)
    d_norm = block.norm.backward(g)
    d_branch = block.branch.backward(d_norm)
    assert np.allclose(dx, d_norm + d_branch)
    assert np.any(d_branch != 0.0)  # branch path ikut mengalir
    assert np.allclose(dx, dx)  # finite & stabil


def test_numeric_grad_check_input() -> None:
    """Central difference vs analytic dx — block MHA (REQ-003)."""
    cfg = _small_cfg()
    block = ResidualBlock(cfg, MultiHeadAttention(cfg, seed=22))
    x = _x(23, b=2, t=3)
    g = seeded_rng(24).normal(size=(2, 3, 4))
    block.forward(x)
    dx = block.backward(g)

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dx, numeric_grad(loss, x))


def test_numeric_grad_check_mha_parameter() -> None:
    """Central difference vs analytic dW_O MHA dalam block (REQ-003)."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=25)
    block = ResidualBlock(cfg, mha)
    x = _x(26, b=2, t=3)
    g = seeded_rng(27).normal(size=(2, 3, 4))
    block.forward(x)
    block.backward(g)
    dw_analytic = mha.out_proj.grads["W"].copy()
    w = mha.out_proj.params["W"]

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dw_analytic, numeric_grad(loss, w))


def test_numeric_grad_check_layernorm_gamma_beta() -> None:
    """Central difference vs analytic d_gamma/d_beta LN dalam block."""
    cfg = _small_cfg()
    block = ResidualBlock(cfg, FeedForward(cfg, seed=28))
    x = _x(29, b=2, t=3)
    g = seeded_rng(30).normal(size=(2, 3, 4))
    block.forward(x)
    block.backward(g)
    dgamma = block.norm.grads["gamma"].copy()
    dbeta = block.norm.grads["beta"].copy()

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dgamma, numeric_grad(loss, block.norm.params["gamma"]))
    assert_close(dbeta, numeric_grad(loss, block.norm.params["beta"]))


def test_numeric_grad_check_ffn_parameter() -> None:
    """Central difference vs analytic dW1 FFN dalam block (REQ-003)."""
    cfg = _small_cfg()
    ffn = FeedForward(cfg, seed=31)
    block = ResidualBlock(cfg, ffn)
    x = _x(32, b=2, t=3)
    g = seeded_rng(33).normal(size=(2, 3, 4))
    block.forward(x)
    block.backward(g)
    dw_analytic = ffn.fc1.grads["W"].copy()
    w = ffn.fc1.params["W"]

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dw_analytic, numeric_grad(loss, w))


def test_regression_t001_t008_compose() -> None:
    """T001-T008 utuh + T009: Emb -> Pos -> [MHA+res+LN] -> [FFN+res+LN]."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2, D_ff=8
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    mha = MultiHeadAttention(cfg, seed=2)
    ffn = FeedForward(cfg, seed=3)
    attn_block = ResidualBlock(cfg, mha)
    ffn_block = ResidualBlock(cfg, ffn)
    ids = np.zeros((2, 5), dtype=np.int64)
    h = pos.forward(emb.forward(ids))
    out = ffn_block.forward(attn_block.forward(h))
    assert out.shape == (2, 5, 4)
    assert np.all(np.isfinite(out))
    # Kausal terjaga melalui wrapper residual (T005/T006 utuh; §8 TD).
    ids2 = ids.copy()
    ids2[0, 4] = 1  # token masa depan diganti
    out2 = ffn_block.forward(attn_block.forward(pos.forward(emb.forward(ids2))))
    assert np.array_equal(out[:, :4], out2[:, :4])
    assert not np.array_equal(out[0, 4], out2[0, 4])
