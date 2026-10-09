"""Test unit komponen Transformer (M007-T014, REQ-103/REQ-011, TD §11).

Tiap komponen T002..T012 punya minimal satu test nilai acuan: acuan dihitung
ulang secara independen memakai NumPy murni (rumus ditulis ulang, bukan
memanggil metode komponen), ditambah contoh kecil hand-computed untuk
attention, QKV, LayerNorm, dan FFN. Grad-check penuh dan suite numerical
(shape/determinisme/mask) berada di test_transformer_numeric.py (T015);
pipeline integrasi di test_transformer_lm.py (T013).
"""

from __future__ import annotations

import numpy as np
import pytest

from naze.nn.transformer import (
    CausalAttention,
    Embedding,
    FeedForward,
    LanguageModelHead,
    LayerNorm,
    MultiHeadAttention,
    PositionalRepr,
    QKVProjection,
    ResidualBlock,
    TransformerBlock,
    TransformerConfig,
    TransformerModel,
)


def _config(**overrides) -> TransformerConfig:
    """Config dev kecil (TD §9 diperkecil agar test cepat)."""
    values = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                  max_sequence_length=8, seed=0)
    values.update(overrides)
    return TransformerConfig(**values)


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Referensi softmax stabil per baris, ditulis ulang independen."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def test_config_defaults_and_head_dim():
    # T001: default valid (TD §9); Dh = d_model // num_heads (invariant §7).
    cfg = TransformerConfig()
    assert cfg.vocab_size == 256
    assert cfg.d_model == 64 and cfg.num_heads == 4
    assert cfg.head_dim == 64 // 4
    assert _config().head_dim == 2


def test_config_invalid_rejected():
    # T001 (TD §10): konfigurasi invalid ditolak ValueError fail-fast.
    with pytest.raises(ValueError):
        TransformerConfig(vocab_size=128)
    with pytest.raises(ValueError):
        _config(d_model=5, num_heads=2)  # 5 tidak habis dibagi 2
    with pytest.raises(ValueError):
        _config(d_model=0)
    with pytest.raises(ValueError):
        _config(num_layers=-1)
    with pytest.raises(ValueError):
        _config(max_sequence_length=0)
    with pytest.raises(ValueError):
        TransformerConfig(dtype=np.int64)


def test_embedding_lookup_reference():
    # T002: lookup == E[ids] (nilai acuan: indexing langsung); init per-seed.
    emb = Embedding(_config(), seed=0)
    twin = Embedding(_config(), seed=0)
    np.testing.assert_array_equal(emb.params["E"], twin.params["E"])
    table = np.arange(256 * 4, dtype=np.float64).reshape(256, 4)
    emb.params["E"] = table
    ids = np.array([[0, 5, 255], [3, 3, 7]], dtype=np.int64)
    np.testing.assert_array_equal(emb.forward(ids), table[ids])


def test_embedding_backward_scatter_reference():
    # T002: dE scatter-add (nilai acuan np.add.at); backward -> None.
    emb = Embedding(_config(), seed=0)
    ids = np.array([[0, 1], [1, 0]], dtype=np.int64)
    grad = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
    emb.forward(ids)
    assert emb.backward(grad) is None
    ref = np.zeros((256, 4))
    np.add.at(ref, ids, grad)
    np.testing.assert_array_equal(emb.grads["E"], ref)
    # token 0 menerima grad dari posisi (0,0) dan (1,1); token 1 sebaliknya.
    np.testing.assert_array_equal(emb.grads["E"][0], grad[0, 0] + grad[1, 1])
    np.testing.assert_array_equal(emb.grads["E"][1], grad[0, 1] + grad[1, 0])


def test_embedding_invalid_ids_rejected():
    # T002 (TD §10): ids float, di luar vocab, T > T_max, dan backward
    # tanpa forward semuanya ditolak.
    emb = Embedding(_config(), seed=0)
    with pytest.raises(ValueError):
        emb.forward(np.array([[1.5, 2.0]]))
    with pytest.raises(ValueError):
        emb.forward(np.array([[256]], dtype=np.int64))
    with pytest.raises(ValueError):
        emb.forward(np.zeros((1, 9), dtype=np.int64))
    with pytest.raises(RuntimeError):
        emb.backward(np.zeros((1, 1, 4)))


def test_positional_forward_reference():
    # T003: output == x + P[0:T] (nilai acuan broadcast langsung).
    pos = PositionalRepr(_config(), seed=1)
    table = np.arange(8 * 4, dtype=np.float64).reshape(8, 4)
    pos.params["P"] = table
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_allclose(pos.forward(x), x + table[:3])


def test_positional_backward_reference():
    # T003: dP scatter-add per posisi; grad_input == grad_out (aditif).
    pos = PositionalRepr(_config(), seed=1)
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    pos.forward(x)
    grad = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_array_equal(pos.backward(grad), grad)
    ref = np.zeros((8, 4))
    ref[:3] = grad.sum(axis=0)  # tiap posisi muncul B kali (scatter-add)
    np.testing.assert_array_equal(pos.grads["P"], ref)


def test_positional_tmax_rejected_and_determinism():
    # T003: T > T_max ditolak; init deterministik per-seed (REQ-101).
    pos = PositionalRepr(_config(), seed=2)
    twin = PositionalRepr(_config(), seed=2)
    np.testing.assert_array_equal(pos.params["P"], twin.params["P"])
    with pytest.raises(ValueError):
        pos.forward(np.zeros((1, 9, 4)))


def test_qkv_projection_reference():
    # T004: W=I dan b konstan -> y = x + b; head h memotong slice
    # [h*Dh:(h+1)*Dh] (nilai acuan hand-computed, layout TD §5 3-4).
    cfg = _config()
    qkv = QKVProjection(cfg, seed=0)
    for lin, bias in ((qkv.q_proj, 0.5), (qkv.k_proj, -0.25), (qkv.v_proj, 0.0)):
        lin.params["W"] = np.eye(cfg.d_model)
        lin.params["b"] = np.full(cfg.d_model, bias)
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
    q, k, v = qkv.forward(x)
    assert q.shape == k.shape == v.shape == (1, 2, 1, 2)
    np.testing.assert_allclose(q[0, 0, 0], [1.5, 2.5])
    np.testing.assert_allclose(q[0, 1, 0], [3.5, 4.5])
    np.testing.assert_allclose(k[0, 0, 0], [0.75, 1.75])
    np.testing.assert_allclose(k[0, 1, 0], [2.75, 3.75])
    np.testing.assert_allclose(v[0, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(v[0, 1, 0], [3.0, 4.0])


def test_causal_attention_hand_reference():
    # T005: contoh hand (B=H=1, T=2, Dh=1): skor [[1,1],[2,2]] -> baris 0
    # bobot [1,0] -> konteks 1.0; baris 1 bobot [0.5,0.5] -> konteks 1.5.
    cfg = _config(d_model=2, num_heads=1, d_ff=4, max_sequence_length=4)
    attn = CausalAttention(cfg)
    q = np.array([[[[1.0], [2.0]]]])  # (1, 1, 2, 1)
    k = np.array([[[[1.0], [1.0]]]])
    v = np.array([[[[1.0], [2.0]]]])
    ctx = attn.forward(q, k, v)
    assert ctx.shape == (1, 1, 2, 1)
    np.testing.assert_allclose(ctx[0, 0, :, 0], [1.0, 1.5])


def test_causal_attention_future_independence():
    # T005 (§8): mengubah token masa depan tidak mengubah konteks masa lalu.
    cfg = _config(d_model=2, num_heads=1, d_ff=4, max_sequence_length=4)
    attn = CausalAttention(cfg)
    q = np.array([[[[1.0], [2.0], [3.0]]]])
    k = np.array([[[[1.0], [1.0], [1.0]]]])
    v = np.array([[[[1.0], [2.0], [4.0]]]])
    first = attn.forward(q, k, v)
    v_changed = v.copy()
    v_changed[0, 0, 2, 0] = 100.0
    second = attn.forward(q, k, v_changed)
    np.testing.assert_array_equal(second[0, 0, :2], first[0, 0, :2])
    assert not np.array_equal(second[0, 0, 2], first[0, 0, 2])


def test_causal_attention_extreme_scores_stable():
    # T005: softmax stabil (shift-max) pada skor ekstrem -> tetap finite.
    cfg = _config(d_model=2, num_heads=1, d_ff=4, max_sequence_length=4)
    attn = CausalAttention(cfg)
    big = np.array([[[[1e4], [1e4], [-1e4]]]])
    ctx = attn.forward(big, big, big)
    assert np.all(np.isfinite(ctx))


def test_mha_h1_equivalence_reference():
    # T006: kasus H=1 ekuivalen reference single-head dari params proyeksi.
    cfg = _config(d_model=2, num_heads=1, d_ff=4, max_sequence_length=4)
    mha = MultiHeadAttention(cfg, seed=0)
    x = np.arange(6, dtype=np.float64).reshape(1, 3, 2)
    out = mha.forward(x)
    assert out.shape == (1, 3, 2)
    flat = x.reshape(3, 2)
    q = flat @ mha.qkv.q_proj.params["W"] + mha.qkv.q_proj.params["b"]
    k = flat @ mha.qkv.k_proj.params["W"] + mha.qkv.k_proj.params["b"]
    v = flat @ mha.qkv.v_proj.params["W"] + mha.qkv.v_proj.params["b"]
    scores = q @ k.T / np.sqrt(2.0)
    causal = np.tril(np.ones((3, 3), dtype=bool))
    probs = _softmax_rows(np.where(causal, scores, -np.inf))
    ref = probs @ v @ mha.out_proj.params["W"] + mha.out_proj.params["b"]
    np.testing.assert_allclose(out, ref.reshape(1, 3, 2), atol=1e-12)


def test_mha_determinism_and_shape():
    # T006: H>1 bekerja (shape invarian); seed sama -> output identik.
    cfg = _config()
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    out = MultiHeadAttention(cfg, seed=0).forward(x)
    twin = MultiHeadAttention(cfg, seed=0).forward(x)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out, twin)


def test_layernorm_reference_and_stats():
    # T007: y = gamma*xhat + beta; mean ~0 dan var ~1 per token (gamma=1).
    ln = LayerNorm(_config())
    x = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    y = ln.forward(x)
    mu = x.mean(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=-1, keepdims=True)
                              + ln.eps)
    np.testing.assert_allclose(y, xhat, atol=1e-12)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-9)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3)
    # gamma/beta diubah -> y = 2*xhat + 1 (nilai acuan affine).
    ln.params["gamma"] = np.full(4, 2.0)
    ln.params["beta"] = np.full(4, 1.0)
    np.testing.assert_allclose(ln.forward(x), 2.0 * xhat + 1.0, atol=1e-12)
    # input konstan (variance 0) tetap finite karena eps eksplisit.
    assert np.all(np.isfinite(ln.forward(np.full((1, 2, 4), 5.0))))


def test_layernorm_param_grads_reference():
    # T007: dgamma = sum(g*xhat); dbeta = sum(g) atas dimensi (batch, seq).
    ln = LayerNorm(_config())
    x = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    ln.forward(x)
    g = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    dx = ln.backward(g)
    assert dx.shape == x.shape
    mu = x.mean(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=-1, keepdims=True)
                              + ln.eps)
    np.testing.assert_allclose(ln.grads["gamma"], (g * xhat).sum(axis=(0, 1)))
    np.testing.assert_allclose(ln.grads["beta"], g.sum(axis=(0, 1)))


def test_ffn_hand_reference():
    # T008: fc1=I -> tanh -> fc2=I: y = tanh(x); hand: tanh(1) ~ 0.7616.
    cfg = _config(d_model=2, num_heads=1, d_ff=2, max_sequence_length=4)
    ffn = FeedForward(cfg, seed=0)
    for lin in (ffn.fc1, ffn.fc2):
        lin.params["W"] = np.eye(2)
        lin.params["b"] = np.zeros(2)
    x = np.array([[[0.0, 1.0]]])  # (1, 1, 2)
    y = ffn.forward(x)
    assert y.shape == (1, 1, 2)
    np.testing.assert_allclose(y[0, 0], [0.0, np.tanh(1.0)], atol=1e-12)


def test_residual_block_reference_and_difference():
    # T009: y == LN(x + branch(x)); dengan vs tanpa residual berbeda.
    cfg = _config()
    block = ResidualBlock(cfg, FeedForward(cfg, seed=3))
    x = np.arange(8, dtype=np.float64).reshape(1, 2, 4)
    y = block.forward(x)
    assert y.shape == x.shape
    ref = block.norm.forward(x + block.branch.forward(x))
    np.testing.assert_array_equal(y, ref)
    without = block.norm.forward(block.branch.forward(x))
    assert not np.allclose(y, without)


def test_transformer_block_reference_chain():
    # T010: y == ffn_residual(attn_residual(x)); rantai dihitung ulang.
    cfg = _config()
    block = TransformerBlock(cfg, seed=0)
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    y = block.forward(x)
    assert y.shape == x.shape
    h = block.attention.forward(x)
    ref = block.ffn.norm.forward(h + block.ffn.branch.forward(h))
    np.testing.assert_array_equal(y, ref)


def test_model_determinism_and_stacking():
    # T011: rantai dihitung ulang; seed sama -> identik; L=2 invarian shape.
    cfg = _config()
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    model = TransformerModel(cfg, seed=0)
    hidden = model.forward(ids)
    assert hidden.shape == (2, 3, 4)
    ref = model.final_norm.forward(
        model.blocks[0].forward(model.positional.forward(
            model.embedding.forward(ids)))
    )
    np.testing.assert_array_equal(hidden, ref)
    twin = TransformerModel(cfg, seed=0).forward(ids)
    np.testing.assert_array_equal(hidden, twin)
    deep = TransformerModel(_config(num_layers=2), seed=0).forward(ids)
    assert deep.shape == (2, 3, 4)
    other = TransformerModel(cfg, seed=7).forward(ids)
    assert not np.array_equal(hidden, other)


def test_lm_head_reference():
    # T012: logits == x @ W + b per token; dimensi terakhir 256 (D-015).
    cfg = _config()
    head = LanguageModelHead(cfg, seed=0)
    weight = np.arange(4 * 256, dtype=np.float64).reshape(4, 256)
    bias = np.arange(256, dtype=np.float64)
    head.proj.params["W"] = weight
    head.proj.params["b"] = bias
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    logits = head.forward(x)
    assert logits.shape == (2, 3, 256)
    assert np.all(np.isfinite(logits))
    ref = (x.reshape(6, 4) @ weight + bias).reshape(2, 3, 256)
    np.testing.assert_allclose(logits, ref, atol=1e-9)


def test_lm_head_backward_reference():
    # T012: dx == g @ W^T; dW == x^T @ g; db == sum(g) (acuan Linear).
    cfg = _config()
    head = LanguageModelHead(cfg, seed=0)
    weight = np.arange(4 * 256, dtype=np.float64).reshape(4, 256)
    bias = np.arange(256, dtype=np.float64)
    head.proj.params["W"] = weight
    head.proj.params["b"] = bias
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    head.forward(x)
    g = np.arange(6 * 256, dtype=np.float64).reshape(2, 3, 256)
    dx = head.backward(g)
    flat_g = g.reshape(6, 256)
    np.testing.assert_allclose(dx, (flat_g @ weight.T).reshape(2, 3, 4),
                               atol=1e-8)
    np.testing.assert_allclose(head.proj.grads["W"],
                               x.reshape(6, 4).T @ flat_g, atol=1e-6)
    np.testing.assert_allclose(head.proj.grads["b"], flat_g.sum(axis=0),
                               atol=1e-6)
