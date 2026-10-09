"""Test TransformerLM — integrasi pipeline LM Transformer (M007-T013, REQ-010/REQ-011).

Meniru harness TypeScript H1-H13: komposisi adapter + seed offset head,
shape logits, loss cross-entropy flatten (B*T, 256) dengan target seragam
(TD §8), guard backward, dict datar params/grads, numeric grad-check,
training toy corpus loss terukur menurun (D-016 basis 1-2), generate
greedy/temperature deterministik per-seed, dan regresi MLPLM.
"""

from __future__ import annotations

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.data.dataset import TextWindows
from naze.lm import MLPLM, TransformerLM, generate, transformer_generate
from naze.lm.transformer_lm import transformer_generate as transformer_generate_direct
from naze.nn.transformer import LanguageModelHead, TransformerConfig, TransformerModel


def _config(**overrides) -> TransformerConfig:
    """Config kecil untuk test (konfigurasi dev TD §9 diperkecil agar cepat)."""
    values = dict(d_model=8, num_heads=2, num_layers=1, d_ff=16,
                  max_sequence_length=8, seed=0)
    values.update(overrides)
    return TransformerConfig(**values)


def _ids(seed: int, batch: int, seq: int) -> np.ndarray:
    """Token ID acak deterministik (byte-level, vocab 256)."""
    return seeded_rng(seed).integers(0, 256, size=(batch, seq), dtype=np.int64)


def _train(cfg: TransformerConfig, corpus: list[int], *, epochs: int = 3,
           lr: float = 0.08, block: int = 4, batch: int = 8,
           data_seed: int = 0) -> list[float]:
    """SGD manual (pola SGDTrainer): loss per batch dikembalikan berurutan."""
    lm = TransformerLM(cfg)
    data = TextWindows(corpus, block, batch, seed=data_seed)
    losses: list[float] = []
    for _ in range(epochs):
        for x, y in data.batches():
            logits = lm.forward(x)
            loss = lm.loss(logits, y)
            grads = lm.backward()
            for name, p in lm.params().items():
                p -= lr * grads[name]
            losses.append(loss)
    return losses


def test_constructor_composition_and_head_seed_offset():
    # H1: head memakai seed s+2+num_layers (kelanjutan skema T011); model seed s.
    cfg = _config(seed=3)
    lm = TransformerLM(cfg, seed=3)
    ref_head = LanguageModelHead(cfg, seed=3 + 2 + cfg.num_layers)
    np.testing.assert_array_equal(lm.head.proj.params["W"], ref_head.proj.params["W"])
    np.testing.assert_array_equal(lm.head.proj.params["b"], ref_head.proj.params["b"])
    ref_model = TransformerModel(cfg, seed=3)
    np.testing.assert_array_equal(lm.model.embedding.params["E"], ref_model.embedding.params["E"])


def test_forward_shape_finite_and_dtype():
    # H2: logits (B,T,256), finite, float64 (DECISION-007).
    logits = TransformerLM(_config()).forward(_ids(11, 2, 5))
    assert logits.shape == (2, 5, 256)
    assert np.all(np.isfinite(logits))
    assert logits.dtype == np.float64


def test_forward_equals_head_of_model():
    # H3: forward == head(model(ids)) eksak (komposisi, tanpa logika lain).
    lm = TransformerLM(_config())
    ids = _ids(11, 2, 5)
    hidden = lm.model.forward(ids)
    np.testing.assert_array_equal(lm.forward(ids), lm.head.forward(hidden))


def test_loss_equals_reference_cross_entropy():
    # H4: loss == CE referensi independen (softmax + target seragam per baris, §8).
    lm = TransformerLM(_config())
    ids = _ids(11, 2, 5)
    targets = np.array([97, 200], dtype=np.int64)
    logits = lm.forward(ids)
    loss = lm.loss(logits, targets)
    flat = logits.reshape(10, 256)
    shifted = flat - flat.max(axis=1, keepdims=True)
    p = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    rows = np.repeat(targets, 5)
    ref = float(-np.log(p[np.arange(10), rows] + 1e-12).mean())
    assert loss == pytest.approx(ref, rel=1e-12)
    assert np.isfinite(loss)


def test_loss_validation_rejected():
    # H5: targets 2D, panjang batch salah, dan ID di luar vocab ditolak.
    lm = TransformerLM(_config())
    logits = lm.forward(_ids(11, 2, 5))
    with pytest.raises(ValueError):
        lm.loss(logits, np.array([[97, 200], [1, 2]], dtype=np.int64))
    with pytest.raises(ValueError):
        lm.loss(logits, np.array([97], dtype=np.int64))
    with pytest.raises(ValueError):
        lm.loss(logits, np.array([97, 999], dtype=np.int64))


def test_backward_guard_before_loss():
    # H6: backward tanpa loss (cache softmax kosong) -> RuntimeError.
    with pytest.raises(RuntimeError):
        TransformerLM(_config()).backward()


def test_backward_grad_keys_shapes_and_param_count():
    # H7: dict datar 22 kunci; shapes grads == params; total == model + head.
    lm = TransformerLM(_config())
    logits = lm.forward(_ids(11, 2, 5))
    lm.loss(logits, np.array([97, 200], dtype=np.int64))
    grads = lm.backward()
    params = lm.params()
    assert sorted(grads) == sorted(params)
    assert len(params) == 22
    for name, param in params.items():
        assert grads[name].shape == param.shape
        assert np.all(np.isfinite(grads[name]))
    assert sum(p.size for p in params.values()) == (
        lm.model.parameter_count() + lm.head.parameter_count()
    )


def test_numeric_grad_check_parameters():
    # H8: central-difference vs analitik pada 10 tensor (tol 1e-5, eps 1e-6).
    lm = TransformerLM(_config(seed=5), seed=5)
    ids = _ids(42, 2, 5)
    targets = np.array([7, 250], dtype=np.int64)

    def f() -> float:
        return lm.loss(lm.forward(ids), targets)

    base = f()
    grads = lm.backward()
    names = [
        "head.W", "head.b", "pos.P", "blk0.attn.q.W", "blk0.attn.v.b",
        "blk0.ffn.fc1.W", "blk0.ffn.fc1.b", "blk0.ffn.fc2.W",
        "final.gamma", "emb.E",
    ]
    for name in names:
        assert_close(grads[name], numeric_grad(f, lm.params()[name]),
                      tol=1e-5, msg=name)
    # numeric_grad memutasi lalu merestorasi parameter — loss harus sama.
    assert f() == pytest.approx(base, abs=1e-12)


def test_training_toy_corpus_loss_decreases():
    # H9 (AC T013, D-016 basis 1-2): korpus "ab" — loss terukur menurun.
    losses = _train(_config(), list(b"ab" * 64))
    first = float(np.mean(losses[:8]))
    last = float(np.mean(losses[-8:]))
    assert last < first - 0.5


def test_generate_greedy_deterministic_capped_and_manual():
    # H10: greedy bebas seed; == rantai argmax manual; prompt panjang di-cap.
    a = transformer_generate(TransformerLM(_config()), [1, 2, 3], 6, temperature=0.0, seed=0)
    b = transformer_generate(TransformerLM(_config()), [1, 2, 3], 6, temperature=0.0, seed=999)
    assert a == b
    lm = TransformerLM(_config())
    ids = [1, 2, 3]
    for _ in range(6):
        ctx = np.array([ids[-8:]], dtype=np.int64)
        ids.append(int(np.argmax(lm.forward(ctx)[0][-1])))
    assert a == ids[3:]
    long_prompt = [i % 256 for i in range(20)]
    out = transformer_generate(TransformerLM(_config()), long_prompt, 3, temperature=0.0)
    assert len(out) == 3


def test_generate_temperature_deterministic_per_seed():
    # H11 (AC T013): sampling temperature deterministik per-seed (REQ-101).
    a = transformer_generate(TransformerLM(_config()), [5, 6], 8, temperature=1.0, seed=7)
    b = transformer_generate(TransformerLM(_config()), [5, 6], 8, temperature=1.0, seed=7)
    assert a == b
    assert len(a) == 8


def test_generate_empty_prompt_rejected():
    # H12: prompt kosong ditolak fail-fast (TD §10).
    with pytest.raises(ValueError):
        transformer_generate(TransformerLM(_config()), [], 3)


def test_training_second_corpus_loss_decreases():
    # H13: korpus berbeda + seed data berbeda — loss tetap menurun.
    losses = _train(_config(seed=3), list(b"abcdef" * 16), data_seed=7)
    assert float(np.mean(losses[-8:])) < float(np.mean(losses[:8]))


def test_exported_api():
    # API publik naze.lm: export baru tersedia, export lama tetap.
    import naze.lm as lm_mod

    assert lm_mod.TransformerLM is TransformerLM
    assert lm_mod.transformer_generate is transformer_generate_direct
    assert lm_mod.MLPLM is MLPLM
    assert lm_mod.generate is generate
    assert "TransformerLM" in lm_mod.__all__
    assert "transformer_generate" in lm_mod.__all__
    assert "MLPLM" in lm_mod.__all__


def test_mlplm_unchanged_regression():
    # AC T013: MLPLM tidak berubah perilaku — forward/loss/backward/generate.
    model = MLPLM(vocab_size=256, block_size=4, d_embed=8, d_hidden=16, seed=0)
    ids = np.tile(np.array([[97, 98, 99, 100]], dtype=np.int64), (3, 1))
    logits = model.forward(ids)
    loss = model.loss(logits, np.array([97, 98, 99], dtype=np.int64))
    assert np.isfinite(loss)
    grads = model.backward()
    assert sorted(grads) == sorted(model.params())
    for g in grads.values():
        assert np.all(np.isfinite(g))
    out1 = generate(model, [1, 2, 3, 4], 4, seed=0)
    out2 = generate(model, [1, 2, 3, 4], 4, seed=0)
    assert out1 == out2 and len(out1) == 4
