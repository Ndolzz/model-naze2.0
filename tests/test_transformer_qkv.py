"""Test QKVProjection (M007-T004, REQ-011/REQ-101) — 3 Linear D->D terpisah
+ reshape (B,T,D)->(B,H,T,Dh) (TD §5 komponen 3-4, §7; bias Linear — TD §9),
backward = jumlah jalur tiga proyeksi via Linear.backward engine. Hanya T004 —
tanpa test attention score/softmax/causal mask (T005+).
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import Embedding, PositionalRepr, QKVProjection, TransformerConfig


def _small_cfg(**over) -> TransformerConfig:
    """Config dev/test kecil; seluruh invariant T001 tetap terpenuhi."""
    params: dict = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                        max_sequence_length=8, seed=0)
    params.update(over)
    return TransformerConfig(**params)


def test_param_shapes_and_count() -> None:
    """W (D,D) + b (D,) per proyeksi; total 3×(D²+D) (TD §12: 12,480)."""
    qkv = QKVProjection(TransformerConfig())  # D=64
    for lin in (qkv.q_proj, qkv.k_proj, qkv.v_proj):
        assert lin.params["W"].shape == (64, 64)
        assert lin.params["b"].shape == (64,)
    assert qkv.parameter_count() == 3 * (64 * 64 + 64) == 12480
    small = QKVProjection(_small_cfg())  # D=4
    assert small.parameter_count() == 3 * (4 * 4 + 4) == 60


def test_output_shapes() -> None:
    """Q/K/V masing-masing (B,H,T,Dh); dev/test D=64,H=4 -> Dh=16 (TD §5)."""
    qkv = QKVProjection(TransformerConfig())  # T_max=64
    q, k, v = qkv.forward(np.zeros((2, 5, 64)))
    assert q.shape == k.shape == v.shape == (2, 4, 5, 16)


def test_varying_batch_and_sequence() -> None:
    """Batch dan T bervariasi valid; hasil per-baris tak bergantung batch lain."""
    qkv = QKVProjection(_small_cfg())  # D=4, H=2, Dh=2, T_max=8
    for b in (1, 2, 3):
        for t in (1, 4, 8):
            q, k, v = qkv.forward(np.zeros((b, t, 4)))
            assert q.shape == k.shape == v.shape == (b, 2, t, 2)
    x1 = seeded_rng(1).normal(size=(1, 3, 4))
    x2 = np.stack([x1[0], x1[0]])
    q1, _, _ = qkv.forward(x1)
    q2, _, _ = qkv.forward(x2)
    assert np.array_equal(q1[0], q2[0])
    assert np.array_equal(q1[0], q2[1])


def test_projection_and_layout_hand_computed() -> None:
    """AC: proyeksi = hand-computed; layout q[b,h,t] = (x[b,t]@Wq+bq)[h*Dh:(h+1)*Dh]."""
    qkv = QKVProjection(_small_cfg())  # D=4, H=2, Dh=2
    qkv.q_proj.params["W"] = np.arange(16, dtype=np.float64).reshape(4, 4)
    qkv.q_proj.params["b"] = np.zeros(4)
    x = np.zeros((1, 2, 4))
    x[0, 0, 0] = 1.0  # e0 -> x@Wq = baris 0 = [0, 1, 2, 3]
    x[0, 1, 3] = 1.0  # e3 -> x@Wq = baris 3 = [12, 13, 14, 15]
    q, _, _ = qkv.forward(x)
    # Proyeksi = hand-computed (flat x @ Wq + bq), lalu reshape multi-head.
    flat_expected = x.reshape(2, 4) @ qkv.q_proj.params["W"] + qkv.q_proj.params["b"]
    assert np.array_equal(q.transpose(0, 2, 1, 3).reshape(2, 4), flat_expected)
    assert np.array_equal(q[0, 0, 0], [0.0, 1.0])  # b0, h0, t0
    assert np.array_equal(q[0, 1, 0], [2.0, 3.0])  # b0, h1, t0
    assert np.array_equal(q[0, 0, 1], [12.0, 13.0])  # b0, h0, t1
    assert np.array_equal(q[0, 1, 1], [14.0, 15.0])  # b0, h1, t1


def test_qkv_not_identical() -> None:
    """Seed proyeksi berturut -> Q/K/V tidak identik (bukan proyeksi ganda)."""
    qkv = QKVProjection(_small_cfg())
    assert not np.array_equal(qkv.q_proj.params["W"], qkv.k_proj.params["W"])
    assert not np.array_equal(qkv.k_proj.params["W"], qkv.v_proj.params["W"])
    q, k, v = qkv.forward(seeded_rng(1).normal(size=(2, 5, 4)))
    assert not np.array_equal(q, k)
    assert not np.array_equal(q, v)
    assert not np.array_equal(k, v)


def test_invalid_shape_rejected() -> None:
    """Input harus (B,T,D) 3D, tidak kosong, D == config.d_model (TD §10)."""
    qkv = QKVProjection(_small_cfg())  # D=4, T_max=8
    with pytest.raises(ValueError):
        qkv.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        qkv.forward(np.zeros((2, 5, 4, 1)))  # 4D
    with pytest.raises(ValueError):
        qkv.forward(np.zeros((2, 5, 3)))  # D != d_model
    with pytest.raises(ValueError):
        qkv.forward(np.zeros((0, 5, 4)))  # batch kosong
    with pytest.raises(ValueError):
        qkv.forward(np.zeros((2, 0, 4)))  # sekuens kosong


def test_sequence_over_max_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    qkv = QKVProjection(_small_cfg())  # T_max=8
    with pytest.raises(ValueError):
        qkv.forward(np.zeros((2, 9, 4)))


def test_non_finite_input_rejected() -> None:
    """Invariant engine as_array: input non-finite (NaN/Inf) ditolak."""
    qkv = QKVProjection(_small_cfg())
    x = np.zeros((1, 3, 4))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        qkv.forward(x)


def test_invalid_head_divisibility_rejected() -> None:
    """Invariant T001 (TD §7): d_model % num_heads != 0 ditolak saat config."""
    with pytest.raises(ValueError):
        TransformerConfig(d_model=66, num_heads=4)


def test_dtype_follows_config() -> None:
    """dtype parameter & output mengikuti config (DECISION-007)."""
    qkv64 = QKVProjection(TransformerConfig())
    for lin in (qkv64.q_proj, qkv64.k_proj, qkv64.v_proj):
        assert lin.params["W"].dtype == np.float64
        assert lin.params["b"].dtype == np.float64
    q, k, v = qkv64.forward(np.zeros((1, 3, 64)))
    assert q.dtype == k.dtype == v.dtype == np.float64
    qkv32 = QKVProjection(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    assert qkv32.q_proj.params["W"].dtype == np.float32
    q32, _, _ = qkv32.forward(np.ones((1, 2, 8)))
    assert q32.dtype == np.float32


def test_deterministic_init_same_seed() -> None:
    """Seed/config sama -> parameter identik + forward deterministik (REQ-101)."""
    a = QKVProjection(_small_cfg(seed=7))
    b = QKVProjection(_small_cfg(seed=7))
    for la, lb in ((a.q_proj, b.q_proj), (a.k_proj, b.k_proj), (a.v_proj, b.v_proj)):
        assert np.array_equal(la.params["W"], lb.params["W"])
        assert np.array_equal(la.params["b"], lb.params["b"])
    x = seeded_rng(2).normal(size=(1, 3, 4))
    qa, ka, va = a.forward(x)
    qb, kb, vb = b.forward(x)
    assert np.array_equal(qa, qb)
    assert np.array_equal(ka, kb)
    assert np.array_equal(va, vb)


def test_different_seed_different_init() -> None:
    """Seed berbeda (config atau kwarg offset) -> parameter berbeda."""
    a = QKVProjection(_small_cfg(seed=7))
    other = QKVProjection(_small_cfg(seed=8))
    assert not np.array_equal(a.q_proj.params["W"], other.q_proj.params["W"])
    kwarg = QKVProjection(_small_cfg(seed=7), seed=9)
    assert not np.array_equal(a.q_proj.params["W"], kwarg.q_proj.params["W"])


def test_forward_no_nan_inf() -> None:
    """Output finite untuk input finite (TD §10: non-finite = failure)."""
    qkv = QKVProjection(_small_cfg())
    q, k, v = qkv.forward(seeded_rng(2).normal(size=(3, 8, 4)))
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(k))
    assert np.all(np.isfinite(v))


def test_backward_before_forward_rejected() -> None:
    """Guard urutan engine (DECISION-009, TD §10): RuntimeError."""
    qkv = QKVProjection(_small_cfg())
    with pytest.raises(RuntimeError):
        qkv.backward((np.zeros((1, 2, 3, 2)), np.zeros((1, 2, 3, 2)),
                      np.zeros((1, 2, 3, 2))))


def test_backward_grad_shapes() -> None:
    """dx (B,T,D); dW (D,D) + db (D,) per proyeksi (TD §7)."""
    qkv = QKVProjection(_small_cfg())  # D=4, H=2
    qkv.forward(np.zeros((2, 5, 4)))
    g = seeded_rng(3).normal(size=(2, 2, 5, 2))
    dx = qkv.backward((g, g, g))
    assert dx.shape == (2, 5, 4)
    for lin in (qkv.q_proj, qkv.k_proj, qkv.v_proj):
        assert lin.grads["W"].shape == (4, 4)
        assert lin.grads["b"].shape == (4,)


def test_backward_invalid_grads_rejected() -> None:
    """dq/dk/dv wajib tuple-3 berbentuk (B,H,T,Dh) sesuai forward terakhir."""
    qkv = QKVProjection(_small_cfg())  # D=4, H=2, T_max=8
    qkv.forward(np.zeros((2, 5, 4)))
    ok = np.zeros((2, 2, 5, 2))
    with pytest.raises(ValueError):
        qkv.backward((np.zeros((2, 5, 4)), ok, ok))  # dq shape salah
    with pytest.raises(ValueError):
        qkv.backward((ok, ok))  # bukan 3 elemen
    with pytest.raises(ValueError):
        qkv.backward(ok)  # bukan tuple


def test_backward_independent_paths() -> None:
    """Jalur Q/K/V independen: hanya dq -> dWk/dWv nol, dWq non-zero."""
    qkv = QKVProjection(_small_cfg())
    qkv.forward(seeded_rng(1).normal(size=(2, 5, 4)))
    zero = np.zeros((2, 2, 5, 2))
    gq = seeded_rng(2).normal(size=(2, 2, 5, 2))
    qkv.backward((gq, zero, zero))
    assert np.any(qkv.q_proj.grads["W"] != 0.0)
    assert np.all(qkv.k_proj.grads["W"] == 0.0)
    assert np.all(qkv.v_proj.grads["W"] == 0.0)
    assert np.all(qkv.k_proj.grads["b"] == 0.0)


def test_backward_layout_inverse_reshape() -> None:
    """dq (B,H,T,Dh) di-unreshape ke (B,T,D) sebelum masuk Linear.backward."""
    qkv = QKVProjection(_small_cfg())  # D=4, H=2, Dh=2
    qkv.q_proj.params["W"] = np.eye(4)
    qkv.q_proj.params["b"] = np.zeros(4)
    x = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    qkv.forward(x)
    dq = np.arange(12, dtype=np.float64).reshape(1, 2, 3, 2)
    dx = qkv.backward((dq, np.zeros_like(dq), np.zeros_like(dq)))
    assert np.array_equal(dx, dq.transpose(0, 2, 1, 3).reshape(1, 3, 4))


def test_numeric_grad_check_wq_bq() -> None:
    """Central difference vs analytic dWq/dbq (engine gradcheck, REQ-003)."""
    qkv = QKVProjection(_small_cfg(seed=3), seed=3)
    x = seeded_rng(4).normal(size=(2, 5, 4))
    gq = seeded_rng(5).normal(size=(2, 2, 5, 2))
    gk = seeded_rng(6).normal(size=(2, 2, 5, 2))
    gv = seeded_rng(7).normal(size=(2, 2, 5, 2))
    qkv.forward(x)
    qkv.backward((gq, gk, gv))

    def loss() -> float:
        q, k, v = qkv.forward(x)
        return float((q * gq).sum() + (k * gk).sum() + (v * gv).sum())

    assert_close(qkv.q_proj.grads["W"], numeric_grad(loss, qkv.q_proj.params["W"]))
    assert_close(qkv.q_proj.grads["b"], numeric_grad(loss, qkv.q_proj.params["b"]))


def test_numeric_grad_check_input() -> None:
    """Central difference vs analytic dx (gradient input)."""
    qkv = QKVProjection(_small_cfg(seed=3), seed=3)
    x = seeded_rng(4).normal(size=(2, 5, 4))
    gq = seeded_rng(5).normal(size=(2, 2, 5, 2))
    gk = seeded_rng(6).normal(size=(2, 2, 5, 2))
    gv = seeded_rng(7).normal(size=(2, 2, 5, 2))

    def loss() -> float:
        q, k, v = qkv.forward(x)
        return float((q * gq).sum() + (k * gk).sum() + (v * gv).sum())

    loss()
    dx = qkv.backward((gq, gk, gv))
    assert_close(dx, numeric_grad(loss, x))


def test_regression_t001_t003_compose() -> None:
    """T001-T003 utuh: ids -> Embedding -> PositionalRepr -> QKVProjection."""
    cfg = _small_cfg()  # T_max=8, D=4, H=2
    emb = Embedding(cfg)
    pos = PositionalRepr(cfg, seed=1)
    qkv = QKVProjection(cfg, seed=2)
    ids = np.zeros((2, 5), dtype=np.int64)
    x = pos.forward(emb.forward(ids))
    assert x.shape == (2, 5, 4)
    q, k, v = qkv.forward(x)
    assert q.shape == k.shape == v.shape == (2, 2, 5, 2)
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(k)) and np.all(np.isfinite(v))
