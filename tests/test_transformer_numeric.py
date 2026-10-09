"""Test numerical Transformer (M007-T015, REQ-003/REQ-011/REQ-101, TD 8/11).

Suite numerical TD 11: shape seluruh tahap pipeline (TD 5), nilai finite pada
logits, determinisme per-seed (REQ-101), mask correctness (TD 8: mengubah
token masa depan tidak mengubah logits posisi <= t), statistik LayerNorm,
dan gradient check central-difference (tol 1e-5, DECISION-105) untuk komponen
ber-backward (MHA, FFN, LayerNorm) plus spot-check end-to-end TransformerLM.
Catatan desain: grad bias K secara analitik ~0 (pergeseran skor konstanta
per baris diinvariansikan softmax) sehingga dites absolut, bukan relatif.
"""

from __future__ import annotations

import numpy as np

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.lm import TransformerLM
from naze.nn.transformer import (
    CausalAttention,
    Embedding,
    FeedForward,
    LanguageModelHead,
    LayerNorm,
    MultiHeadAttention,
    PositionalRepr,
    QKVProjection,
    TransformerConfig,
    TransformerModel,
)


def _config(**overrides) -> TransformerConfig:
    # Config dev kecil (TD 9 diperkecil agar numerical test cepat).
    values = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                  max_sequence_length=8, seed=0)
    values.update(overrides)
    return TransformerConfig(**values)


def test_shapes_all_stages():
    # T015 (TD 5/11): shape seluruh tahap pipeline diverifikasi eksplisit.
    cfg = _config()
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    emb = Embedding(cfg, seed=0)
    pos = PositionalRepr(cfg, seed=1)
    qkv = QKVProjection(cfg, seed=0)
    attn = CausalAttention(cfg)
    mha = MultiHeadAttention(cfg, seed=0)
    ffn = FeedForward(cfg, seed=0)
    block = TransformerBlock(cfg, seed=0)
    model = TransformerModel(cfg, seed=0)
    head = LanguageModelHead(cfg, seed=0)
    x = emb.forward(ids)  # (B, T, D) — TD 5 komponen 1
    assert x.shape == (2, 3, 4)
    x = pos.forward(x)  # komponen 2
    assert x.shape == (2, 3, 4)
    q, k, v = qkv.forward(x)  # komponen 3-4: (B, T, D) -> (B, H, T, Dh)
    assert q.shape == k.shape == v.shape == (2, 2, 3, 2)
    ctx = attn.forward(q, k, v)  # komponen 5-10; scores internal (B, H, T, T)
    assert ctx.shape == (2, 2, 3, 2)
    assert mha.forward(x).shape == (2, 3, 4)  # komponen 11-13
    assert ffn.forward(x).shape == (2, 3, 4)  # komponen 16
    assert block.forward(x).shape == (2, 3, 4)  # komponen 17
    hidden = model.forward(ids)  # komponen 18-19
    assert hidden.shape == (2, 3, 4)
    logits = head.forward(hidden)  # komponen 20-21
    assert logits.shape == (2, 3, 256)  # D-015: dimensi vocab FIXED 256


def test_default_config_pipeline_finite():
    # T015 (TD 9/11): config default dev/test 9; logits finite (TD 10).
    lm = TransformerLM(TransformerConfig(), seed=0)
    ids = (np.arange(64, dtype=np.int64) % 256).reshape(1, 64)
    logits = lm.forward(ids)
    assert logits.shape == (1, 64, 256)
    assert np.all(np.isfinite(logits))
    hidden = lm.model.forward(ids)
    assert hidden.shape == (1, 64, 64)
    assert np.all(np.isfinite(hidden))


def test_determinism_per_seed():
    # T015 (REQ-101, TD 11): seed sama -> logits identik; seed beda -> beda.
    cfg = _config()
    ids = np.array([[3, 17, 250], [99, 5, 42]], dtype=np.int64)
    first = TransformerLM(cfg, seed=0).forward(ids)
    second = TransformerLM(cfg, seed=0).forward(ids)
    np.testing.assert_array_equal(first, second)
    # forward berulang pada instance baru seed sama juga identik.
    np.testing.assert_array_equal(TransformerLM(cfg, seed=0).forward(ids), first)
    other = TransformerLM(cfg, seed=7).forward(ids)
    assert not np.array_equal(first, other)


def test_mask_correctness_logits():
    # T015 (AC, TD 8): mengubah token masa depan tidak mengubah logits
    # posisi <= t; logits pada posisi >= token yang diubah ikut berubah.
    cfg = _config()
    lm = TransformerLM(cfg, seed=0)
    ids = np.array([[3, 17, 40, 9, 100, 250],
                    [99, 5, 42, 7, 12, 200]], dtype=np.int64)
    base = lm.forward(ids)
    changed = ids.copy()
    changed[:, 4] = np.array([1, 255], dtype=np.int64)  # ubah posisi 4
    after = lm.forward(changed)
    np.testing.assert_array_equal(after[:, :4], base[:, :4])
    assert not np.allclose(after[:, 4:], base[:, 4:])
    changed2 = ids.copy()
    changed2[:, 5] = np.array([2, 3], dtype=np.int64)  # ubah posisi 5
    after2 = lm.forward(changed2)
    np.testing.assert_array_equal(after2[:, :5], base[:, :5])
    assert not np.allclose(after2[:, 5:], base[:, 5:])


def test_mask_correctness_attention():
    # T015 (TD 8): level attention multi-head — key/value masa depan tidak
    # memengaruhi konteks posisi sebelumnya (bobot future = 0 eksak).
    cfg = _config()
    attn = CausalAttention(cfg)
    rng = seeded_rng(11)
    q = rng.normal(0.0, 1.0, (2, 2, 3, 2))
    k = rng.normal(0.0, 1.0, (2, 2, 3, 2))
    v = rng.normal(0.0, 1.0, (2, 2, 3, 2))
    base = attn.forward(q, k, v)
    k2 = k.copy()
    k2[:, :, 2, :] = 50.0  # key masa depan diubah ekstrem
    v2 = v.copy()
    v2[:, :, 2, :] = -50.0  # value masa depan diubah ekstrem
    np.testing.assert_array_equal(attn.forward(q, k2, v)[:, :, :2], base[:, :, :2])
    np.testing.assert_array_equal(attn.forward(q, k, v2)[:, :, :2], base[:, :, :2])
    assert not np.allclose(attn.forward(q, k, v2)[:, :, 2], base[:, :, 2])


def test_layernorm_stats_numeric():
    # T015 (TD 11): output LN gamma=1/beta=0 -> mean ~0 dan var ~1 per token.
    ln = LayerNorm(_config())
    x = seeded_rng(3).normal(0.0, 1.0, (4, 8, 4))
    y = ln.forward(x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-9)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, atol=1e-3)
    # input konstan (variance 0) tetap finite karena eps eksplisit (TD 10).
    assert np.all(np.isfinite(ln.forward(np.full((1, 2, 4), 5.0))))


def test_gradcheck_mha():
    # T015 (REQ-003, TD 11): grad-check central difference tol 1e-5 untuk
    # seluruh parameter MHA (W/b Q/K/V/O) dan dx. Bias K pengecualian:
    # grad analitik ~0 (softmax invarian pergeseran konstanta per baris)
    # sehingga dites absolut — lihat docstring modul.
    mha = MultiHeadAttention(_config(), seed=0)
    x = seeded_rng(21).normal(0.0, 1.0, (2, 3, 4))
    coeff = seeded_rng(22).normal(0.0, 1.0, (2, 3, 4))

    def f() -> float:
        return float(np.sum(mha.forward(x) * coeff))

    f()
    dx = mha.backward(coeff)
    assert_close(dx, numeric_grad(f, x))
    lin_pairs = [
        (mha.qkv.q_proj, "q"), (mha.qkv.k_proj, "k"),
        (mha.qkv.v_proj, "v"), (mha.out_proj, "o"),
    ]
    for lin, name in lin_pairs:
        assert_close(lin.grads["W"], numeric_grad(f, lin.params["W"]))
        if name != "k":  # k.b dites absolut (invariansi softmax, bukan relatif)
            assert_close(lin.grads["b"], numeric_grad(f, lin.params["b"]))
    assert np.max(np.abs(mha.qkv.k_proj.grads["b"])) < 1e-9
    assert np.max(np.abs(numeric_grad(f, mha.qkv.k_proj.params["b"]))) < 1e-6


def test_gradcheck_ffn():
    # T015 (REQ-003, TD 11): grad-check parameter FFN (fc1/fc2 W/b) + dx.
    ffn = FeedForward(_config(), seed=0)
    x = seeded_rng(31).normal(0.0, 1.0, (2, 3, 4))
    coeff = seeded_rng(32).normal(0.0, 1.0, (2, 3, 4))

    def f() -> float:
        return float(np.sum(ffn.forward(x) * coeff))

    f()
    dx = ffn.backward(coeff)
    assert_close(dx, numeric_grad(f, x))
    for lin in (ffn.fc1, ffn.fc2):
        assert_close(lin.grads["W"], numeric_grad(f, lin.params["W"]))
        assert_close(lin.grads["b"], numeric_grad(f, lin.params["b"]))


def test_gradcheck_layernorm():
    # T015 (REQ-003, TD 11): grad-check gamma/beta non-trivial + dx.
    ln = LayerNorm(_config())
    ln.params["gamma"] = np.linspace(0.5, 2.0, 4)
    ln.params["beta"] = np.linspace(-0.1, 0.2, 4)
    x = seeded_rng(41).normal(0.0, 1.0, (2, 3, 4))
    coeff = seeded_rng(42).normal(0.0, 1.0, (2, 3, 4))

    def f() -> float:
        return float(np.sum(ln.forward(x) * coeff))

    f()
    dx = ln.backward(coeff)
    assert_close(dx, numeric_grad(f, x))
    assert_close(ln.grads["gamma"], numeric_grad(f, ln.params["gamma"]))
    assert_close(ln.grads["beta"], numeric_grad(f, ln.params["beta"]))


def test_gradcheck_transformer_lm_spot():
    # T015 (REQ-003/REQ-010, TD 11): spot-check end-to-end — backward
    # TransformerLM vs central difference pada pos.P dan head.b.
    lm = TransformerLM(_config(), seed=0)
    ids = np.array([[3, 17, 250], [99, 5, 42]], dtype=np.int64)
    targets = np.array([7, 200], dtype=np.int64)

    def f() -> float:
        return lm.loss(lm.forward(ids), targets)

    assert np.isfinite(f())
    lm.loss(lm.forward(ids), targets)
    grads = lm.backward()
    params = lm.params()
    assert_close(grads["pos.P"], numeric_grad(f, params["pos.P"]))
    assert_close(grads["head.b"], numeric_grad(f, params["head.b"]))
