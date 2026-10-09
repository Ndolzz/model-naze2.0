"""Test integrasi pipeline Transformer (M007-T016, REQ-004/005/010/011).

Integrasi penuh TD 11: ByteTokenizer -> TextWindows -> TransformerLM ->
loss/backward, adapter generate, dan sanity evaluasi kerangka 5-basis
DECISION-016 pada toy corpus: (1) baseline uniform ln(256), (2) perbaikan
loss terukur, (3) stabilitas antar epoch, (4) loss validasi holdout,
(5) konfigurasi reproducible per-seed (REQ-101).
"""

from __future__ import annotations

import math

import numpy as np

from naze.data.dataset import TextWindows
from naze.lm import MLPLM, TransformerLM, generate, transformer_generate
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer

KORPUS = "naze kecil belajar bahasa. " * 20
HOLDOUT = "naze kecil belajar bahasa. " * 5
BASIS1_JUMLAH_EPOCH = 5
BASIS1_LR = 0.08
BASIS1_BLOCK = 6
BASIS1_BATCH = 8


def _config(**overrides) -> TransformerConfig:
    # Config kecil untuk test (TD 9 diperkecil agar cepat).
    values = dict(d_model=8, num_heads=2, num_layers=1, d_ff=16,
                  max_sequence_length=8, seed=0)
    values.update(overrides)
    return TransformerConfig(**values)


def _pipeline_ids(text: str) -> list[int]:
    # REQ-004: byte-level tokenizer lossless, vocab fixed 256.
    return ByteTokenizer().encode(text)


def _loss_rata(lm: TransformerLM, data: TextWindows) -> float:
    # Rata-rata loss tertimbang batch (untuk evaluasi basis 1/4).
    total, n = 0.0, 0
    for x, y in data.batches():
        logits = lm.forward(x)
        total += lm.loss(logits, y) * x.shape[0]
        n += x.shape[0]
    return total / n


def _train_epoch_means(cfg: TransformerConfig, ids: list[int], *, epochs: int,
                       lr: float, block: int, batch: int,
                       data_seed: int) -> list[float]:
    # SGD manual (pola T013); epoch = iterasi penuh TextWindows (order tetap
    # per-seed karena batches() memakai seed sama setiap epoch, REQ-005).
    lm = TransformerLM(cfg)
    data = TextWindows(ids, block, batch, seed=data_seed)
    means: list[float] = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for x, y in data.batches():
            loss = lm.loss(lm.forward(x), y)
            grads = lm.backward()
            for name, p in lm.params().items():
                p -= lr * grads[name]
            total += loss * x.shape[0]
            n += x.shape[0]
        means.append(total / n)
    return means


def test_tokenizer_dataset_pipeline_shapes():
    # T016 (REQ-004/005/011): encode -> windows; x (B,T), y (B,) int64,
    # seluruh ID di rentang byte 0-255.
    ids = _pipeline_ids(KORPUS)
    assert 0 < len(ids) <= 256 * 30
    data = TextWindows(ids, BASIS1_BLOCK, BASIS1_BATCH, seed=0)
    seen = 0
    for x, y in data.batches():
        assert x.shape[1] == BASIS1_BLOCK
        assert x.dtype == np.int64 and y.dtype == np.int64
        assert y.shape == (x.shape[0],)
        assert int(x.min()) >= 0 and int(x.max()) < 256
        assert int(y.min()) >= 0 and int(y.max()) < 256
        seen += x.shape[0]
    assert seen == data.n_windows


def test_tokenizer_roundtrip_unicode_emoji():
    # T016 (REQ-004): lossless roundtrip termasuk unicode dan emoji.
    tok = ByteTokenizer()
    teks = ["naze kecil", "ba\u2019in", "caf\u00e9 \u2603 \U0001f680"]
    for text in teks:
        assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 256


def test_dataset_deterministic_per_seed():
    # T016 (REQ-005/101): urutan batch identik per-seed, beda seed beda urutan.
    ids = _pipeline_ids(KORPUS)
    a1 = list(TextWindows(ids, 4, 8, seed=3).batches())
    a2 = list(TextWindows(ids, 4, 8, seed=3).batches())
    for (x1, y1), (x2, y2) in zip(a1, a2):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
    b1 = list(TextWindows(ids, 4, 8, seed=9).batches())
    assert not all(
        np.array_equal(x1, x2) for (x1, _), (x2, _) in zip(a1, b1)
    )


def test_dataset_epoch_size_matches_batches():
    # T016 (REQ-005): epoch_size() == jumlah batch aktual per iterasi penuh.
    data = TextWindows(_pipeline_ids(KORPUS), BASIS1_BLOCK, BASIS1_BATCH)
    assert data.epoch_size() == len(list(data.batches()))


def test_full_pipeline_forward_loss_finite():
    # T016 (REQ-010/011, TD 8): tokenizer -> dataset -> model -> loss finite.
    lm = TransformerLM(_config())
    data = TextWindows(_pipeline_ids(KORPUS), BASIS1_BLOCK, BASIS1_BATCH)
    x, y = next(iter(data.batches()))
    logits = lm.forward(x)
    assert logits.shape == (x.shape[0], BASIS1_BLOCK, 256)
    assert np.all(np.isfinite(logits))
    assert np.isfinite(lm.loss(logits, y))


def test_full_pipeline_backward_grads_finite():
    # T016 (REQ-010/011): backward end-to-end lewat pipeline dataset.
    lm = TransformerLM(_config())
    data = TextWindows(_pipeline_ids(KORPUS), BASIS1_BLOCK, BASIS1_BATCH)
    x, y = next(iter(data.batches()))
    lm.loss(lm.forward(x), y)
    grads = lm.backward()
    params = lm.params()
    assert sorted(grads) == sorted(params)
    for name, g in grads.items():
        assert g.shape == params[name].shape
        assert np.all(np.isfinite(g))


def test_untrained_baseline_near_uniform():
    # T016 (D-016 basis 1, TD 13): loss awal ~ uniform ln(256) karena
    # inisialisasi dekat nol; tidak ada ambang universal (DECISION-016).
    lm = TransformerLM(_config())
    data = TextWindows(_pipeline_ids(KORPUS), BASIS1_BLOCK, BASIS1_BATCH)
    baseline = _loss_rata(lm, data)
    assert abs(baseline - math.log(256)) < 0.5


def test_training_beats_uniform_baseline():
    # T016 (D-016 basis 1): setelah training, loss jauh di bawah baseline
    # uniform ln(256) — model belajar distribusi toy corpus.
    means = _train_epoch_means(_config(), _pipeline_ids(KORPUS),
                               epochs=BASIS1_JUMLAH_EPOCH, lr=BASIS1_LR,
                               block=BASIS1_BLOCK, batch=BASIS1_BATCH,
                               data_seed=0)
    assert means[-1] < math.log(256) - 1.0


def test_training_improves_loss():
    # T016 (D-016 basis 2): perbaikan loss terukur antar epoch.
    means = _train_epoch_means(_config(), _pipeline_ids(KORPUS),
                               epochs=BASIS1_JUMLAH_EPOCH, lr=BASIS1_LR,
                               block=BASIS1_BLOCK, batch=BASIS1_BATCH,
                               data_seed=0)
    assert len(means) == BASIS1_JUMLAH_EPOCH
    assert all(math.isfinite(m) for m in means)
    assert means[-1] < means[0] - 1.0


def test_training_stability():
    # T016 (D-016 basis 3): bukti stabilitas — rata-rata per epoch
    # menurun tanpa lonjakan besar setelah epoch pertama.
    means = _train_epoch_means(_config(), _pipeline_ids(KORPUS),
                               epochs=BASIS1_JUMLAH_EPOCH, lr=BASIS1_LR,
                               block=BASIS1_BLOCK, batch=BASIS1_BATCH,
                               data_seed=0)
    jumps = [abs(b - a) for a, b in zip(means[1:], means[2:])]
    assert max(jumps) < 0.5
    assert means[-1] <= means[1]


def test_holdout_validation_improves():
    # T016 (D-016 basis 4): loss pada windows holdout menurun setelah
    # training — windows eval tidak pernah dipakai untuk update parameter.
    cfg = _config()
    lm = TransformerLM(cfg)
    hold = TextWindows(_pipeline_ids(HOLDOUT), BASIS1_BLOCK, BASIS1_BATCH)
    before = _loss_rata(lm, hold)
    data = TextWindows(_pipeline_ids(KORPUS), BASIS1_BLOCK, BASIS1_BATCH)
    for _ in range(BASIS1_JUMLAH_EPOCH):
        for x, y in data.batches():
            lm.loss(lm.forward(x), y)
            grads = lm.backward()
            for name, p in lm.params().items():
                p -= BASIS1_LR * grads[name]
    after = _loss_rata(lm, hold)
    assert after < before - 1.0


def test_config_reproducible():
    # T016 (D-016 basis 5): konfigurasi evaluasi reproducible — seed sama
    # memberikan kurva loss identik; seed data beda tetap membaik.
    ids = _pipeline_ids(KORPUS)
    a = _train_epoch_means(_config(), ids, epochs=3, lr=BASIS1_LR,
                           block=BASIS1_BLOCK, batch=BASIS1_BATCH,
                           data_seed=0)
    b = _train_epoch_means(_config(), ids, epochs=3, lr=BASIS1_LR,
                           block=BASIS1_BLOCK, batch=BASIS1_BATCH,
                           data_seed=0)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    c = _train_epoch_means(_config(), ids, epochs=3, lr=BASIS1_LR,
                           block=BASIS1_BLOCK, batch=BASIS1_BATCH,
                           data_seed=5)
    assert c[-1] < c[0] - 0.5


def test_generate_adapter_from_encoded_prompt():
    # T016 (REQ-008/011, TD 8): generate dari prompt hasil tokenizer;
    # keluaran token valid byte, greedy deterministik, decode tidak error.
    tok = ByteTokenizer()
    lm = TransformerLM(_config())
    data = TextWindows(_pipeline_ids(KORPUS), BASIS1_BLOCK, BASIS1_BATCH)
    for _ in range(2):
        for x, y in data.batches():
            lm.loss(lm.forward(x), y)
            grads = lm.backward()
            for name, p in lm.params().items():
                p -= BASIS1_LR * grads[name]
    prompt = tok.encode("naze ")
    out = transformer_generate(lm, prompt, 8, temperature=0.0, seed=0)
    out2 = transformer_generate(lm, prompt, 8, temperature=0.0, seed=99)
    assert out == out2
    assert len(out) == 8
    assert all(0 <= i < 256 for i in out)
    tok.decode(out)  # tidak boleh raise
    sampled = transformer_generate(lm, prompt, 8, temperature=1.0, seed=7)
    sampled2 = transformer_generate(lm, prompt, 8, temperature=1.0, seed=7)
    assert sampled == sampled2 and len(sampled) == 8


def test_mlplm_pipeline_still_trains():
    # T016 (REQ-010): pipeline lama MLPLM tetap terintegrasi dengan
    # tokenizer -> dataset (regresi ringan; regresi penuh = T017).
    tok = ByteTokenizer()
    ids = tok.encode(KORPUS)
    model = MLPLM(vocab_size=256, block_size=6, d_embed=8, d_hidden=16,
                  seed=0)
    data = TextWindows(ids, 6, 8, seed=0)
    losses: list[float] = []
    for _ in range(2):
        for x, y in data.batches():
            loss = model.loss(model.forward(x), y)
            grads = model.backward()
            for name, p in model.params().items():
                p -= 0.08 * grads[name]
            losses.append(loss)
    assert np.isfinite(losses[-1])
    assert float(np.mean(losses[-8:])) < float(np.mean(losses[:8]))
    out = generate(model, tok.encode("naze "), 4, seed=0)
    assert len(out) == 4 and all(0 <= i < 256 for i in out)
