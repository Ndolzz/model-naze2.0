"""Test CausalAttention (M007-T005, REQ-011/REQ-003) — scores QK^T/sqrt(Dh)
(B,H,T,T), causal mask (query t hanya melihat key 0..t), softmax stabil
axis=-1 (reuse naze.nn.activations), context = weights @ V (TD §5 komponen
5-10, §8 causal requirement). Hanya T005 — tanpa output projection/MHA
(T006+). Expected values dihitung via formula referensi independen.
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import (
    CausalAttention,
    Embedding,
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


def _rand_qkv(seed: int, b: int = 2, t: int = 5, h: int = 2, dh: int = 2):
    """Q/K/V acak deterministik (B,H,T,Dh) untuk config kecil."""
    r = seeded_rng(seed)
    return (r.normal(size=(b, h, t, dh)), r.normal(size=(b, h, t, dh)),
            r.normal(size=(b, h, t, dh)))


def _manual_probs(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
    """Formula referensi independen (tanpa CausalAttention/softmax engine):
    softmax kausal atas q @ k^T / scale; upper triangle = -inf."""
    t = q.shape[2]
    s = np.matmul(q, np.swapaxes(k, -1, -2)) / scale
    mask = np.tril(np.ones((t, t), dtype=bool))
    s = np.where(mask, s, -np.inf)
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def test_output_shapes() -> None:
    """context (B,H,T,Dh); dev/test D=64,H=4 -> Dh=16 (TD §5)."""
    att = CausalAttention(TransformerConfig())
    q, k, v = _rand_qkv(0, b=2, t=5, h=4, dh=16)
    assert att.forward(q, k, v).shape == (2, 4, 5, 16)
    small = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(1)
    assert small.forward(q, k, v).shape == (2, 2, 5, 2)


def test_varying_batch_size() -> None:
    """Batch bervariasi valid; hasil per-baris tak bergantung batch lain."""
    att = CausalAttention(_small_cfg())
    for b in (1, 2, 3):
        q, k, v = _rand_qkv(2, b=b, t=3)
        assert att.forward(q, k, v).shape == (b, 2, 3, 2)
    r = seeded_rng(3)
    q1, k1, v1 = (r.normal(size=(1, 2, 4, 2)) for _ in range(3))
    q2 = np.concatenate([q1, q1])
    k2 = np.concatenate([k1, k1])
    v2 = np.concatenate([v1, v1])
    c1 = att.forward(q1, k1, v1)
    c2 = att.forward(q2, k2, v2)
    assert np.array_equal(c1[0], c2[0])
    assert np.array_equal(c1[0], c2[1])


def test_varying_sequence_length() -> None:
    """T bervariasi valid untuk 1..T_max (TD §10)."""
    att = CausalAttention(_small_cfg())  # T_max=8
    for t in (1, 2, 4, 8):
        q, k, v = _rand_qkv(4, t=t)
        assert att.forward(q, k, v).shape == (2, 2, t, 2)


def test_t_equals_one_attends_only_self() -> None:
    """T=1: bobot [[1.0]] ke token sendiri; context == v (TD §5)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(5, t=1)
    c = att.forward(q, k, v)
    assert att._probs.shape == (2, 2, 1, 1)
    assert np.all(att._probs == 1.0)
    assert np.array_equal(c, v)


def test_t_equals_max_sequence_length() -> None:
    att = CausalAttention(_small_cfg())  # T_max=8
    q, k, v = _rand_qkv(6, t=8)
    assert att.forward(q, k, v).shape == (2, 2, 8, 2)


def test_t_over_max_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(7, t=9)
    with pytest.raises(ValueError):
        att.forward(q, k, v)


def test_invalid_qkv_shapes_rejected() -> None:
    """q/k/v harus (B,H,T,Dh) 4D — 3D/5D ditolak (TD §10)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(8)
    with pytest.raises(ValueError):
        att.forward(np.zeros((2, 5, 2)), k, v)  # q 3D
    with pytest.raises(ValueError):
        att.forward(q, np.zeros((2, 2, 5)), v)  # k 3D
    with pytest.raises(ValueError):
        att.forward(q, k, np.zeros((2, 5, 2)))  # v 3D
    with pytest.raises(ValueError):
        att.forward(np.zeros((2, 2, 5, 2, 1)), k, v)  # q 5D


def test_mismatched_shapes_rejected() -> None:
    """B/T/H/Dh antar q,k,v harus sama (TD §10)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(9)
    with pytest.raises(ValueError):
        att.forward(q, k[:1], v)  # B beda
    with pytest.raises(ValueError):
        att.forward(q, k[:, :, :4], v)  # T beda
    with pytest.raises(ValueError):
        att.forward(q, k[:, :1], v)  # H beda
    with pytest.raises(ValueError):
        att.forward(q, k, v[..., :1])  # Dh beda


def test_head_dim_config_mismatch_rejected() -> None:
    """H & Dh harus sesuai config meski q,k,v konsisten (TD §10)."""
    att = CausalAttention(_small_cfg())  # H=2, Dh=2
    q, k, v = _rand_qkv(10, h=1)
    with pytest.raises(ValueError):
        att.forward(q, k, v)  # H=1 != num_heads
    q, k, v = _rand_qkv(11, dh=4)
    with pytest.raises(ValueError):
        att.forward(q, k, v)  # Dh=4 != head_dim


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(12)
    q[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        att.forward(q, k, v)


def test_dtype_follows_config() -> None:
    """dtype context mengikuti config (DECISION-007; adapter cast)."""
    att32 = CausalAttention(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    r = seeded_rng(13)
    q, k, v = (r.normal(size=(1, 2, 3, 4)).astype(np.float32) for _ in range(3))
    assert att32.forward(q, k, v).dtype == np.float32
    att64 = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(14)
    assert att64.forward(q, k, v).dtype == np.float64


def test_scaling_uses_sqrt_dh() -> None:
    """Skor = QK^T/sqrt(Dh) — bukan sqrt(d_model), bukan tanpa scaling."""
    att = CausalAttention(_small_cfg())  # Dh=2, d_model=4
    r = seeded_rng(15)
    q = r.normal(size=(1, 2, 4, 2))
    k = r.normal(size=(1, 2, 4, 2))
    v = r.normal(size=(1, 2, 4, 2))
    att.forward(q, k, v)
    probs = att._probs
    assert np.allclose(probs, _manual_probs(q, k, np.sqrt(2)))
    assert not np.allclose(probs, _manual_probs(q, k, np.sqrt(4)))  # sqrt(d_model)
    assert not np.allclose(probs, _manual_probs(q, k, 1.0))  # tanpa scaling


def test_causal_mask_structure() -> None:
    """Bobot nonzero hanya pada key 0..t (segitiga bawah)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(16, t=5)
    att.forward(q, k, v)
    probs = att._probs
    for t in range(probs.shape[2]):
        assert np.all(probs[:, :, t, t + 1:] == 0.0)
        assert np.any(probs[:, :, t, : t + 1] > 0.0)


def test_masked_future_weights_exactly_zero() -> None:
    """Bobot future eksak 0.0 (ekuivalen -inf; tanpa bobot bocor)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(17, t=6)
    att.forward(q, k, v)
    upper = np.triu(np.ones((6, 6), dtype=bool), k=1)
    assert np.all(att._probs[:, :, upper] == 0.0)


def test_attention_row_sums_approx_one() -> None:
    """Jumlah bobot per baris (axis key) ≈ 1 untuk semua (B,H,T)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(18, t=5)
    att.forward(q, k, v)
    sums = att._probs.sum(axis=-1)
    assert sums.shape == (2, 2, 5)
    assert np.allclose(sums, 1.0)


def test_future_token_change_does_not_affect_past() -> None:
    """§8: ganti token masa depan -> output posisi lama tidak berubah."""
    cfg = _small_cfg()
    qkv = QKVProjection(cfg, seed=1)
    att = CausalAttention(cfg)
    xa = seeded_rng(19).normal(size=(1, 3, 4))
    xb = xa.copy()
    xb[0, 2] += 10.0  # hanya posisi 2 (masa depan) berbeda
    qa, ka, va = qkv.forward(xa)
    qb, kb, vb = qkv.forward(xb)
    ca = att.forward(qa, ka, va)
    cb = att.forward(qb, kb, vb)
    assert np.array_equal(ca[:, :, 0], cb[:, :, 0])
    assert np.array_equal(ca[:, :, 1], cb[:, :, 1])
    assert not np.array_equal(ca[:, :, 2], cb[:, :, 2])  # posisi 2 terpengaruh


def test_forward_no_nan_inf() -> None:
    """Output & bobot finite untuk input finite (TD §10)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(20, t=8)
    c = att.forward(q, k, v)
    assert np.all(np.isfinite(c))
    assert np.all(np.isfinite(att._probs))


def test_stability_large_scores() -> None:
    """Skor ekstrem (±1e6): finite, row-sum ≈ 1, future tetap 0."""
    att = CausalAttention(_small_cfg())
    t = 4
    base = np.ones((1, 2, t, 2))
    upper = np.triu(np.ones((t, t), dtype=bool), k=1)
    for q, k in ((base * 1e6, base * 1e6), (base * 1e6, base * -1e6),
                 (base * -1e6, base * 1e6)):
        v = seeded_rng(21).normal(size=(1, 2, t, 2))
        c = att.forward(q, k, v)
        assert np.all(np.isfinite(c))
        probs = att._probs
        assert np.allclose(probs.sum(axis=-1), 1.0)
        assert np.all(probs[:, :, upper] == 0.0)
        assert np.allclose(probs, _manual_probs(q, k, np.sqrt(2)))


def test_hand_computed_small() -> None:
    """AC: formula hand-computed — expected dihitung independen di test."""
    att = CausalAttention(TransformerConfig(d_model=2, num_heads=1, max_sequence_length=4))
    q = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])  # (1,1,3,2)
    k = q.copy()
    v = np.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    c = att.forward(q, k, v)
    expected_probs = _manual_probs(q, k, np.sqrt(2.0))
    assert np.allclose(att._probs, expected_probs)
    assert np.allclose(c, np.matmul(expected_probs, v))
    # Arah mask: query 0 hanya melihat key 0 -> context[0] == v[0] eksak.
    assert np.array_equal(att._probs[0, 0, 0], [1.0, 0.0, 0.0])
    assert np.array_equal(c[0, 0, 0], v[0, 0])
    # Softmax tanpa mask akan melihat semua key -> jelas berbeda.
    no_mask = np.exp(q[0, 0] @ k[0, 0].T / np.sqrt(2.0))
    no_mask = no_mask / no_mask.sum(axis=-1, keepdims=True)
    assert not np.allclose(att._probs[0, 0], no_mask)


def test_deterministic_behavior() -> None:
    """Forward deterministik: input sama -> output identik (REQ-101)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(22)
    c1 = att.forward(q, k, v)
    c2 = att.forward(q, k, v)
    assert np.array_equal(c1, c2)
    att2 = CausalAttention(_small_cfg())
    assert np.array_equal(att2.forward(q, k, v), c1)


def test_backward_before_forward_rejected() -> None:
    """Guard urutan engine (DECISION-009, TD §10): RuntimeError."""
    att = CausalAttention(_small_cfg())
    with pytest.raises(RuntimeError):
        att.backward(np.zeros((2, 2, 5, 2)))


def test_backward_grad_shapes() -> None:
    """dQ/dK/dV masing-masing (B,H,T,Dh) (TD §7)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(23)
    g = seeded_rng(24).normal(size=(2, 2, 5, 2))
    att.forward(q, k, v)
    dq, dk, dv = att.backward(g)
    assert dq.shape == dk.shape == dv.shape == (2, 2, 5, 2)
    with pytest.raises(ValueError):
        att.backward(np.zeros((2, 2, 6, 2)))  # shape grad_out salah


def test_backward_causal_gradient() -> None:
    """Loss hanya posisi <= 1 -> grad key/value future = 0 (kausal)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(25, t=4)
    g = np.zeros((2, 2, 4, 2))
    g[:, :, :2] = seeded_rng(26).normal(size=(2, 2, 2, 2))
    att.forward(q, k, v)
    dq, dk, dv = att.backward(g)
    assert np.all(dk[:, :, 2:] == 0.0)  # key future tidak menerima grad
    assert np.all(dv[:, :, 2:] == 0.0)
    assert np.all(dq[:, :, 2:] == 0.0)
    assert np.any(dk[:, :, 1] != 0.0)  # key valid tetap menerima grad


def test_numeric_grad_check_q() -> None:
    """Central difference vs analytic dQ (engine gradcheck, REQ-003)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(27)
    g = seeded_rng(28).normal(size=(2, 2, 5, 2))
    att.forward(q, k, v)
    dq, _, _ = att.backward(g)

    def loss() -> float:
        return float((att.forward(q, k, v) * g).sum())

    assert_close(dq, numeric_grad(loss, q))


def test_numeric_grad_check_k() -> None:
    """Central difference vs analytic dK (engine gradcheck, REQ-003)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(29)
    g = seeded_rng(30).normal(size=(2, 2, 5, 2))
    att.forward(q, k, v)
    _, dk, _ = att.backward(g)

    def loss() -> float:
        return float((att.forward(q, k, v) * g).sum())

    assert_close(dk, numeric_grad(loss, k))


def test_numeric_grad_check_v() -> None:
    """Central difference vs analytic dV (engine gradcheck, REQ-003)."""
    att = CausalAttention(_small_cfg())
    q, k, v = _rand_qkv(31)
    g = seeded_rng(32).normal(size=(2, 2, 5, 2))
    att.forward(q, k, v)
    _, _, dv = att.backward(g)

    def loss() -> float:
        return float((att.forward(q, k, v) * g).sum())

    assert_close(dv, numeric_grad(loss, v))


def test_regression_t001_t004_compose() -> None:
    """T001-T004 utuh: ids -> Embedding -> PositionalRepr -> QKV -> Attention."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    qkv = QKVProjection(cfg, seed=2)
    att = CausalAttention(cfg)
    ids = np.zeros((2, 5), dtype=np.int64)
    q, k, v = qkv.forward(pos.forward(emb.forward(ids)))
    c = att.forward(q, k, v)
    assert c.shape == (2, 2, 5, 2)
    assert np.all(np.isfinite(c))
    # Causal melalui pipeline penuh: token masa depan diganti -> lama sama.
    ids2 = ids.copy()
    ids2[0, 4] = 1  # posisi 4 (masa depan) berbeda
    q2, k2, v2 = qkv.forward(pos.forward(emb.forward(ids2)))
    c2 = att.forward(q2, k2, v2)
    assert np.array_equal(c[:, :, :4], c2[:, :, :4])
    assert not np.array_equal(c[0, :, 4, :], c2[0, :, 4, :])
