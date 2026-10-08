"""Test TransformerBlock (M007-T010, REQ-011/REQ-003) — [MHA + res+LN]
-> [FFN + res+LN] (TD §5 komponen 17; TASKS T010). Composition layer atas
komponen T006/T008/T009 — tanpa TransformerModel/stacking (T011+).
Expected values dihitung via sub-komponen referensi independen.
"""

import numpy as np
import pytest

import naze.nn as nn
from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.layers import Layer
from naze.nn.transformer import (
    Embedding,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    PositionalRepr,
    ResidualBlock,
    TransformerBlock,
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


def test_constructor_and_sublayer_composition() -> None:
    """Konstruksi: dua ResidualBlock (branch MHA & FFN); config shared."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=1)
    assert isinstance(block, Layer)
    assert isinstance(block.attention, ResidualBlock)
    assert isinstance(block.attention.branch, MultiHeadAttention)
    assert isinstance(block.ffn, ResidualBlock)
    assert isinstance(block.ffn.branch, FeedForward)
    assert block.config is cfg
    assert block.attention.config is cfg
    assert block.ffn.config is cfg
    assert isinstance(block.attention.norm, LayerNorm)
    assert isinstance(block.ffn.norm, LayerNorm)
    assert "mha" in block.attention.name
    assert "ffn" in block.ffn.name
    assert "transformer_block" in block.name


def test_exported_api() -> None:
    """TransformerBlock diekspor via naze.nn (import + __all__)."""
    assert nn.TransformerBlock is TransformerBlock
    assert "TransformerBlock" in nn.__all__


def test_output_shape() -> None:
    """forward(x: (B,T,D)) -> (B,T,D) (TD §5 17) + dev config."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=2)
    for b, t in ((1, 1), (1, 8), (3, 3)):
        assert block.forward(_x(3, b=b, t=t)).shape == (b, t, 4)
    dev = TransformerBlock(TransformerConfig(), seed=5)
    assert dev.forward(seeded_rng(6).normal(size=(2, 5, 64))).shape == (2, 5, 64)


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    cfg32 = TransformerConfig(d_model=8, num_heads=2, d_ff=12, dtype=np.float32)
    block32 = TransformerBlock(cfg32, seed=1)
    x = seeded_rng(7).normal(size=(1, 4, 8)).astype(np.float32)
    assert block32.forward(x).dtype == np.float32
    block64 = TransformerBlock(_small_cfg(), seed=2)
    assert block64.forward(_x(8)).dtype == np.float64


def test_execution_order_and_t009_composition() -> None:
    """y = ffn_block(attn_block(x)) persis; bukan formula/urutan lain."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=1)
    x = _x(2)
    y = block.forward(x)
    # urutan persis TASKS T010: [MHA + res+LN] -> [FFN + res+LN]
    ref = block.ffn.forward(block.attention.forward(x))
    assert np.array_equal(y, ref)
    # komposisi dari komponen T009 independen (seed sama) identik
    attn = ResidualBlock(cfg, MultiHeadAttention(cfg, seed=1))
    ffn = ResidualBlock(cfg, FeedForward(cfg, seed=2))
    ref2 = ffn.forward(attn.forward(x))
    assert np.allclose(y, ref2)
    # bukan urutan terbalik: [FFN + res+LN] -> [MHA + res+LN] berbeda
    rev = block.attention.forward(block.ffn.forward(x))
    assert not np.allclose(y, rev)


def test_deterministic_construction_and_forward() -> None:
    """Seed sama -> block identik; forward deterministik (REQ-101)."""
    cfg = _small_cfg()
    a = TransformerBlock(cfg, seed=9)
    b = TransformerBlock(cfg, seed=9)
    assert np.array_equal(a.attention.branch.qkv.q_proj.params["W"],
                         b.attention.branch.qkv.q_proj.params["W"])
    assert np.array_equal(a.ffn.branch.fc1.params["W"], b.ffn.branch.fc1.params["W"])
    x = _x(10)
    assert np.array_equal(a.forward(x), b.forward(x))
    assert np.array_equal(a.forward(x), a.forward(x))
    # seed berbeda -> parameter berbeda
    c = TransformerBlock(cfg, seed=11)
    assert not np.array_equal(a.attention.branch.qkv.q_proj.params["W"],
                              c.attention.branch.qkv.q_proj.params["W"])


def test_parameter_count() -> None:
    """Count = attention_block + ffn_block aktual (dihitung, bukan hard-coded)."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=1)
    assert block.parameter_count() == (block.attention.parameter_count()
                                       + block.ffn.parameter_count())
    # D=4,H=2,D_ff=8: MHA 4*(16+4) + LN 8 + FFN (32+8)+(32+4) + LN 8
    assert block.parameter_count() == 4 * (4 * 4 + 4) + 2 * 4 + ((4 * 8 + 8) + (8 * 4 + 4)) + 2 * 4
    assert block.parameter_count() == 172


def test_parameter_aggregation() -> None:
    """Param tetap via mekanisme sublayer; tidak ada parameter duplikat."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=1)
    assert block.params == {}  # tanpa duplikat — param hidup di sub-block
    assert block.attention.branch.out_proj.params["W"].shape == (4, 4)
    assert block.attention.branch.parameter_count() == 80  # MHA (T006)
    assert block.ffn.branch.parameter_count() == 76  # FFN (T008)
    assert block.attention.norm.parameter_count() == 2 * 4
    assert block.ffn.norm.parameter_count() == 2 * 4
    assert block.attention.parameter_count() == 88  # MHA 80 + LN 8 (T009)
    assert block.ffn.parameter_count() == 84  # FFN 76 + LN 8 (T009)


def test_invalid_input_dimensions_rejected() -> None:
    """Input harus (B,T,D) 3D — 2D/4D ditolak (TD §10)."""
    block = TransformerBlock(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        block.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        block.forward(np.zeros((2, 5, 4, 1)))  # 4D


def test_wrong_model_dimension_rejected() -> None:
    """D harus d_model; D != d_model ditolak (TD §10)."""
    block = TransformerBlock(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        block.forward(np.zeros((2, 5, 8)))  # D=8 != d_model


def test_empty_batch_sequence_rejected() -> None:
    """B=0 atau T=0 ditolak ValueError (TD §10)."""
    block = TransformerBlock(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        block.forward(np.zeros((0, 5, 4)))  # B=0
    with pytest.raises(ValueError):
        block.forward(np.zeros((2, 0, 4)))  # T=0


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    block = TransformerBlock(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        block.forward(_x(12, t=9))


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    block = TransformerBlock(_small_cfg(), seed=1)
    x = _x(13)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        block.forward(x)
    y = _x(14)
    y[1, 4, 3] = np.inf
    with pytest.raises(ValueError):
        block.forward(y)


def test_backward_shape_and_guard() -> None:
    """backward -> dx (B,T,D); guard urutan & shape grad (TD §7/§10)."""
    block = TransformerBlock(_small_cfg(), seed=1)
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


def test_gradient_flow_through_attention_and_ffn_paths() -> None:
    """dx != 0; grad MHA + LN (attn sub-block) dan FFN + LN (ffn sub-block)."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=1)
    x = _x(18)
    block.forward(x)
    g = seeded_rng(19).normal(size=(2, 5, 4))
    dx = block.backward(g)
    assert np.any(dx != 0.0)
    mha = block.attention.branch
    for proj in (mha.qkv.q_proj, mha.qkv.k_proj, mha.qkv.v_proj, mha.out_proj):
        assert np.any(proj.grads["W"] != 0.0)
        assert np.any(proj.grads["b"] != 0.0)
    assert np.any(block.attention.norm.grads["gamma"] != 0.0)
    assert np.any(block.attention.norm.grads["beta"] != 0.0)
    ffn = block.ffn.branch
    for lin in (ffn.fc1, ffn.fc2):
        assert np.any(lin.grads["W"] != 0.0)
        assert np.any(lin.grads["b"] != 0.0)
    assert np.any(block.ffn.norm.grads["gamma"] != 0.0)
    assert np.any(block.ffn.norm.grads["beta"] != 0.0)


def test_residual_gradient_path_both_sub_blocks() -> None:
    """dx = attention.backward(ffn.backward(g)) via kedua sub-block (T009)."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=3)
    x = _x(20)
    g = seeded_rng(21).normal(size=(2, 5, 4))
    block.forward(x)
    dx = block.backward(g)
    # rekomputasi manual via sub-block (cache sama; deterministik)
    d_h = block.ffn.backward(g)
    expected = block.attention.backward(d_h)
    assert np.allclose(dx, expected)
    assert np.any(d_h != 0.0)  # grad mengalir melalui sub-block FFN


def test_numeric_grad_check_input() -> None:
    """Central difference vs analytic dx — end-to-end block (REQ-003)."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=22)
    x = _x(23, b=2, t=3)
    g = seeded_rng(24).normal(size=(2, 3, 4))
    block.forward(x)
    dx = block.backward(g)

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dx, numeric_grad(loss, x))


def test_numeric_grad_check_attention_parameter() -> None:
    """Central difference vs analytic dW_O MHA dalam block (REQ-003)."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=25)
    mha = block.attention.branch
    x = _x(26, b=2, t=3)
    g = seeded_rng(27).normal(size=(2, 3, 4))
    block.forward(x)
    block.backward(g)
    dw_analytic = mha.out_proj.grads["W"].copy()
    w = mha.out_proj.params["W"]

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dw_analytic, numeric_grad(loss, w))


def test_numeric_grad_check_ffn_parameter() -> None:
    """Central difference vs analytic dW1 FFN dalam block (REQ-003)."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=31)
    ffn = block.ffn.branch
    x = _x(32, b=2, t=3)
    g = seeded_rng(33).normal(size=(2, 3, 4))
    block.forward(x)
    block.backward(g)
    dw_analytic = ffn.fc1.grads["W"].copy()
    w = ffn.fc1.params["W"]

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dw_analytic, numeric_grad(loss, w))


def test_numeric_grad_check_layernorm_parameter() -> None:
    """Central difference vs analytic d_gamma/d_beta LN dalam sub-block."""
    cfg = _small_cfg()
    block = TransformerBlock(cfg, seed=28)
    x = _x(29, b=2, t=3)
    g = seeded_rng(30).normal(size=(2, 3, 4))
    block.forward(x)
    block.backward(g)
    dgamma = block.attention.norm.grads["gamma"].copy()
    dbeta = block.attention.norm.grads["beta"].copy()

    def loss() -> float:
        return float((block.forward(x) * g).sum())

    assert_close(dgamma, numeric_grad(loss, block.attention.norm.params["gamma"]))
    assert_close(dbeta, numeric_grad(loss, block.attention.norm.params["beta"]))


def test_regression_t001_t009_compose() -> None:
    """T001-T009 utuh + T010: Emb -> Pos -> [attn-res] -> [ffn-res]."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2, D_ff=8
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    block = TransformerBlock(cfg, seed=2)
    ids = np.zeros((2, 5), dtype=np.int64)
    out = block.forward(pos.forward(emb.forward(ids)))
    assert out.shape == (2, 5, 4)
    assert np.all(np.isfinite(out))
    # Kausal terjaga melalui block (T005/T006 utuh; §8 TD).
    ids2 = ids.copy()
    ids2[0, 4] = 1  # token masa depan diganti
    out2 = block.forward(pos.forward(emb.forward(ids2)))
    assert np.array_equal(out[:, :4], out2[:, :4])
    assert not np.array_equal(out[0, 4], out2[0, 4])
