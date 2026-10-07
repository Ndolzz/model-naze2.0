"""Test MultiHeadAttention (M007-T006, REQ-011/REQ-003) — QKVProjection ->
CausalAttention -> merge head (B,T,D) -> output projection W_O (TD §5
komponen 11-13, §7 backward). Hanya T006 — tanpa residual/LayerNorm/FFN/
block (T007+). Expected values dihitung via komponen referensi independen.
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import (
    CausalAttention,
    Embedding,
    MultiHeadAttention,
    PositionalRepr,
    QKVProjection,
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


def test_constructor_and_parameter_count() -> None:
    """Konstruksi + parameter: 4 proyeksi D->D (Q,K,V,O) dengan bias."""
    cfg = _small_cfg()  # D=4
    mha = MultiHeadAttention(cfg)
    assert mha.qkv.config is cfg
    assert isinstance(mha.attention, CausalAttention)
    per_proj = 4 * 4 + 4  # W (D,D) + b (D,)
    assert mha.parameter_count() == 4 * per_proj
    dev = MultiHeadAttention(TransformerConfig())  # D=64
    assert dev.parameter_count() == 4 * (64 * 64 + 64)


def test_output_shape() -> None:
    """forward(x: (B,T,D)) -> (B,T,D) untuk dev & config kecil (TD §5)."""
    mha = MultiHeadAttention(_small_cfg())
    assert mha.forward(_x(0)).shape == (2, 5, 4)
    for b, t in ((1, 1), (1, 8), (3, 3)):
        assert mha.forward(_x(1, b=b, t=t)).shape == (b, t, 4)
    dev = MultiHeadAttention(TransformerConfig())  # D=64, H=4, T_max=64
    assert dev.forward(seeded_rng(2).normal(size=(2, 5, 64))).shape == (2, 5, 64)


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    att32 = MultiHeadAttention(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    x = seeded_rng(3).normal(size=(1, 4, 8)).astype(np.float32)
    assert att32.forward(x).dtype == np.float32
    assert att32.forward(x.astype(np.float64)).dtype == np.float32
    mha64 = MultiHeadAttention(_small_cfg())
    assert mha64.forward(_x(4)).dtype == np.float64


def test_multi_head_merge_layout() -> None:
    """Merge (B,H,T,Dh) -> (B,T,D): merged[b,t,h*Dh+dh] = ctx[b,h,t,dh]."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=5)
    qkv = QKVProjection(cfg, seed=5)  # sub-komponen identik (seed sama)
    att = CausalAttention(cfg)
    x = _x(6)
    q, k, v = qkv.forward(x)
    ctx = att.forward(q, k, v)  # (B,H,T,Dh)
    b, h, t, dh = ctx.shape
    manual = np.empty((b, t, h * dh))
    for bb in range(b):
        for tt in range(t):
            for hh in range(h):
                manual[bb, tt, hh * dh:(hh + 1) * dh] = ctx[bb, hh, tt]
    assert np.allclose(mha.attention.forward(*qkv.forward(x)), ctx)
    # merge manual vs pipeline mha (sebelum W_O): bandingkan via sub-layer.
    merged_manual = manual.reshape(b * t, h * dh)
    out_manual = mha.out_proj.forward(merged_manual)
    assert np.allclose(mha.forward(x), out_manual.reshape(b, t, h * dh))


def test_output_projection_applied() -> None:
    """Output = W_O @ merged + b (bukan context mentah tanpa proyeksi)."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=7)
    x = _x(8)
    out = mha.forward(x)
    q, k, v = mha.qkv.forward(x)
    ctx = mha.attention.forward(q, k, v)
    merged = ctx.transpose(0, 2, 1, 3).reshape(2 * 5, 4)
    assert np.allclose(out, (merged @ mha.out_proj.params["W"]
                            + mha.out_proj.params["b"]).reshape(2, 5, 4))
    assert not np.allclose(out, ctx.transpose(0, 2, 1, 3).reshape(2, 5, 4))


def test_deterministic_initialization() -> None:
    """Seed sama -> parameter identik (REQ-101); seed beda -> beda."""
    a = MultiHeadAttention(_small_cfg(), seed=9)
    b = MultiHeadAttention(_small_cfg(), seed=9)
    assert np.array_equal(a.qkv.q_proj.params["W"], b.qkv.q_proj.params["W"])
    assert np.array_equal(a.qkv.k_proj.params["W"], b.qkv.k_proj.params["W"])
    assert np.array_equal(a.qkv.v_proj.params["W"], b.qkv.v_proj.params["W"])
    assert np.array_equal(a.out_proj.params["W"], b.out_proj.params["W"])
    c = MultiHeadAttention(_small_cfg(), seed=10)
    assert not np.array_equal(a.out_proj.params["W"], c.out_proj.params["W"])


def test_deterministic_forward() -> None:
    """Forward deterministik: input sama -> output identik (REQ-101)."""
    mha = MultiHeadAttention(_small_cfg())
    x = _x(11)
    assert np.array_equal(mha.forward(x), mha.forward(x))
    mha2 = MultiHeadAttention(_small_cfg())
    assert np.array_equal(mha.forward(x), mha2.forward(x))


def test_invalid_input_shape_rejected() -> None:
    """Input harus (B,T,D) 3D — 2D/4D ditolak (TD §10)."""
    mha = MultiHeadAttention(_small_cfg())
    with pytest.raises(ValueError):
        mha.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        mha.forward(np.zeros((2, 5, 4, 1)))  # 4D


def test_wrong_model_dimension_rejected() -> None:
    """D harus d_model; D != H*Dh ditolak oleh invariant T001."""
    mha = MultiHeadAttention(_small_cfg())  # D=4
    with pytest.raises(ValueError):
        mha.forward(np.zeros((2, 5, 8)))  # D=8 != d_model
    with pytest.raises(ValueError):
        MultiHeadAttention(TransformerConfig(d_model=5, num_heads=2))  # 5 % 2


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    mha = MultiHeadAttention(_small_cfg())  # T_max=8
    with pytest.raises(ValueError):
        mha.forward(_x(12, t=9))
    with pytest.raises(ValueError):
        mha.forward(_x(13, b=0, t=1))
    with pytest.raises(ValueError):
        mha.forward(_x(14, t=0))


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    mha = MultiHeadAttention(_small_cfg())
    x = _x(15)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        mha.forward(x)


def test_backward_shape_and_guard() -> None:
    """backward -> dx (B,T,D); guard urutan & shape grad (TD §7/§10)."""
    mha = MultiHeadAttention(_small_cfg())
    with pytest.raises(RuntimeError):
        mha.backward(np.zeros((2, 5, 4)))
    x = _x(16)
    mha.forward(x)
    g = seeded_rng(17).normal(size=(2, 5, 4))
    dx = mha.backward(g)
    assert dx.shape == (2, 5, 4)
    with pytest.raises(ValueError):
        mha.backward(np.zeros((2, 6, 4)))


def test_gradient_flows_to_input_and_params() -> None:
    """Grad mengalir: dx != 0; dW_O/db_O & dW/db QKV terisi (TD §7)."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=18)
    x = _x(19)
    mha.forward(x)
    g = seeded_rng(20).normal(size=(2, 5, 4))
    dx = mha.backward(g)
    assert np.any(dx != 0.0)
    assert mha.out_proj.grads["W"].shape == (4, 4)
    assert np.any(mha.out_proj.grads["W"] != 0.0)
    assert np.any(mha.out_proj.grads["b"] != 0.0)
    for proj in (mha.qkv.q_proj, mha.qkv.k_proj, mha.qkv.v_proj):
        assert np.any(proj.grads["W"] != 0.0)
        assert np.any(proj.grads["b"] != 0.0)


def test_numeric_grad_check_input() -> None:
    """Central difference vs analytic dx (engine gradcheck, REQ-003)."""
    mha = MultiHeadAttention(_small_cfg(), seed=21)
    x = _x(22)
    g = seeded_rng(23).normal(size=(2, 5, 4))
    mha.forward(x)
    dx = mha.backward(g)

    def loss() -> float:
        return float((mha.forward(x) * g).sum())

    assert_close(dx, numeric_grad(loss, x))


def test_numeric_grad_check_out_proj_weight() -> None:
    """Central difference vs analytic dW_O (engine gradcheck, REQ-003)."""
    mha = MultiHeadAttention(_small_cfg(), seed=24)
    x = _x(25)
    g = seeded_rng(26).normal(size=(2, 5, 4))
    mha.forward(x)
    mha.backward(g)  # mengisi grads sub-layer
    w = mha.out_proj.params["W"]
    dw_analytic = mha.out_proj.grads["W"].copy()

    def loss() -> float:
        return float((mha.forward(x) * g).sum())

    dw_numeric = numeric_grad(loss, w)
    assert_close(dw_analytic, dw_numeric)


def test_causal_behavior_preserved() -> None:
    """§8: ganti token masa depan -> output posisi lama identik (T005 utuh)."""
    cfg = _small_cfg()
    mha = MultiHeadAttention(cfg, seed=27)
    xa = _x(28, b=1, t=3)
    xb = xa.copy()
    xb[0, 2] += 10.0  # hanya posisi 2 (masa depan) berbeda
    ca = mha.forward(xa)
    cb = mha.forward(xb)
    assert np.array_equal(ca[0, 0], cb[0, 0])
    assert np.array_equal(ca[0, 1], cb[0, 1])
    assert not np.array_equal(ca[0, 2], cb[0, 2])  # posisi 2 terpengaruh
    # Grad kausal: loss hanya posisi <= 1 -> dx future = 0.
    mha.forward(xa)
    g = np.zeros((1, 3, 4))
    g[0, :2] = seeded_rng(29).normal(size=(2, 4))
    dx = mha.backward(g)
    assert np.all(dx[0, 2:] == 0.0)
    assert np.any(dx[0, 1] != 0.0)


def test_regression_t001_t005_compose() -> None:
    """T001-T005 utuh + T006: ids -> Emb -> Pos -> MHA -> (B,T,D)."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    mha = MultiHeadAttention(cfg, seed=2)
    ids = np.zeros((2, 5), dtype=np.int64)
    out = mha.forward(pos.forward(emb.forward(ids)))
    assert out.shape == (2, 5, 4)
    assert np.all(np.isfinite(out))
    # Causal melalui pipeline penuh: token masa depan diganti -> lama sama.
    ids2 = ids.copy()
    ids2[0, 4] = 1  # posisi 4 (masa depan) berbeda
    out2 = mha.forward(pos.forward(emb.forward(ids2)))
    assert np.array_equal(out[:, :4], out2[:, :4])
    assert not np.array_equal(out[0, 4], out2[0, 4])
    # Sub-komponen T004/T005 tetap berperilaku sama saat dipakai MHA.
    q, k, v = mha.qkv.forward(pos.forward(emb.forward(ids)))
    assert q.shape == (2, 2, 5, 2)
    assert mha.attention.forward(q, k, v).shape == (2, 2, 5, 2)
