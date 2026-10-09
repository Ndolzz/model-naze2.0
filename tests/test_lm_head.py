"""Test LanguageModelHead (M007-T012, REQ-011/REQ-004) — Linear D->256
(default tidak weight-tied) -> logits (B,T,256) (TD §5 komponen 20-21,
§9; TASKS T012). Komponen atas engine Linear — tanpa integrasi pipeline
(T013+). Expected values dihitung via sub-komponen referensi independen.
"""

import numpy as np
import pytest

import naze.nn as nn
from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.layers import Layer, Linear
from naze.nn.transformer import (
    Embedding,
    LanguageModelHead,
    PositionalRepr,
    TransformerBlock,
    TransformerConfig,
    TransformerModel,
    VOCAB_SIZE,
)


def _small_cfg(**over) -> TransformerConfig:
    """Config dev/test kecil; seluruh invariant T001 tetap terpenuhi."""
    params: dict = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                        max_sequence_length=8, seed=0)
    params.update(over)
    return TransformerConfig(**params)


def _ids(seed: int, b: int = 2, t: int = 5) -> np.ndarray:
    """Token ID acak deterministik (B,T) int64 dalam rentang byte valid."""
    return seeded_rng(seed).integers(0, 256, size=(b, t)).astype(np.int64)


def _x(seed: int, shape: tuple = (2, 5, 4)) -> np.ndarray:
    """Input hidden states acak deterministik (B,T,D)."""
    return seeded_rng(seed).normal(size=shape)


def _g(seed: int, shape: tuple) -> np.ndarray:
    """Grad acak deterministik (pola suite T010)."""
    return seeded_rng(seed).normal(size=shape)


def test_constructor_and_projection_composition() -> None:
    """Konstruksi: satu Linear (D->256) atas Layer; config shared."""
    cfg = _small_cfg()
    head = LanguageModelHead(cfg, seed=1)
    assert isinstance(head, Layer)
    assert isinstance(head.proj, Linear)
    assert head.proj.in_features == cfg.d_model
    assert head.proj.out_features == cfg.vocab_size == VOCAB_SIZE
    assert head.config is cfg
    assert "lm_head" in head.name


def test_exported_api() -> None:
    """LanguageModelHead diekspor via naze.nn (import + __all__)."""
    assert nn.LanguageModelHead is LanguageModelHead
    assert "LanguageModelHead" in nn.__all__


def test_output_shape_vocab_dimension() -> None:
    """forward (B,T,D) -> logits (B,T,256); dimensi terakhir FIXED."""
    cfg = _small_cfg()
    head = LanguageModelHead(cfg, seed=1)
    for b, t in ((1, 1), (2, 5), (3, 8)):
        assert head.forward(_x(2, (b, t, cfg.d_model))).shape == (b, t, VOCAB_SIZE)
    dev = LanguageModelHead(TransformerConfig(), seed=3)
    assert dev.forward(_x(4, (2, 5, 64))).shape == (2, 5, 256)


def test_logits_finite() -> None:
    """Logits affine mentah selalu finite untuk input finite (TD §8)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    for s in (5, 6, 7):
        assert np.all(np.isfinite(head.forward(_x(s))))


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    cfg32 = TransformerConfig(d_model=8, num_heads=2, num_layers=1, d_ff=12,
                              dtype=np.float32)
    head32 = LanguageModelHead(cfg32, seed=1)
    assert head32.forward(_x(5, (1, 4, 8))).dtype == np.float32
    assert head32.proj.params["W"].dtype == np.float32
    head64 = LanguageModelHead(_small_cfg(), seed=2)
    assert head64.forward(_x(6)).dtype == np.float64


def test_forward_equals_reference_linear() -> None:
    """forward == Linear referensi flatten (B*T,D)->(B*T,256)->reshape."""
    cfg = _small_cfg()
    head = LanguageModelHead(cfg, seed=7)
    ref = Linear(cfg.d_model, cfg.vocab_size, seed=7)
    x = _x(13)
    expected = ref.forward(x.reshape(10, cfg.d_model)).reshape(2, 5, VOCAB_SIZE)
    assert np.array_equal(head.forward(x), expected)


def test_not_weight_tied_by_default() -> None:
    """Default TIDAK weight-tied (TD §9): proyeksi independen embedding."""
    cfg = _small_cfg()
    head = LanguageModelHead(cfg, seed=1)
    emb = Embedding(cfg, seed=1)  # seed sama pun tetap independen
    assert head.proj.params["W"].shape == (cfg.d_model, cfg.vocab_size)
    assert emb.params["E"].shape == (cfg.vocab_size, cfg.d_model)
    assert head.proj.params["W"] is not emb.params["E"]
    assert not np.array_equal(head.proj.params["W"], emb.params["E"].T)


def test_deterministic_construction_and_forward() -> None:
    """Seed sama -> head identik; seed beda -> beda (REQ-101)."""
    cfg = _small_cfg()
    a = LanguageModelHead(cfg, seed=9)
    b = LanguageModelHead(cfg, seed=9)
    assert np.array_equal(a.proj.params["W"], b.proj.params["W"])
    assert np.array_equal(a.proj.params["b"], b.proj.params["b"])
    x = _x(14)
    assert np.array_equal(a.forward(x), b.forward(x))
    c = LanguageModelHead(cfg, seed=10)
    assert not np.array_equal(a.proj.params["W"], c.proj.params["W"])
    # Default seed = config.seed (pola komponen M007).
    d = LanguageModelHead(cfg)
    e = LanguageModelHead(cfg, seed=cfg.seed)
    assert np.array_equal(d.proj.params["W"], e.proj.params["W"])


def test_parameter_count() -> None:
    """parameter_count = D*256 + 256 (proyeksi; tanpa duplikat)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    assert head.parameter_count() == 4 * 256 + 256
    dev = LanguageModelHead(TransformerConfig(), seed=2)
    assert dev.parameter_count() == 64 * 256 + 256


def test_parameter_aggregation() -> None:
    """Parameter aktual = parameter proyeksi; head.params kosong."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    assert head.params == {}
    assert head.parameter_count() == head.proj.parameter_count()


def test_invalid_input_dimensions_rejected() -> None:
    """Input bukan 3D ditolak fail-fast (TD §10)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        head.forward(_x(3, (5, 4)))
    with pytest.raises(ValueError):
        head.forward(_x(4, (2, 4)))


def test_wrong_model_dimension_rejected() -> None:
    """Dimensi terakhir != d_model ditolak (TD §10)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        head.forward(_x(5, (2, 5, 8)))


def test_empty_batch_sequence_rejected() -> None:
    """Batch/sekuens kosong ditolak (pola Embedding T002)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        head.forward(_x(6, (0, 5, 4)))
    with pytest.raises(ValueError):
        head.forward(_x(7, (2, 0, 4)))


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak (pola pipeline M007)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        head.forward(_x(8, (2, 9, 4)))


def test_non_finite_input_rejected() -> None:
    """Input NaN/Inf ditolak as_array (pola komponen M007)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    bad = _x(9)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        head.forward(bad)


def test_backward_guard_shape_and_grad_input() -> None:
    """Guard backward: urutan (RuntimeError) + shape grad (ValueError)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    with pytest.raises(RuntimeError):
        head.backward(_g(10, (2, 5, VOCAB_SIZE)))
    head.forward(_x(11))
    with pytest.raises(ValueError):
        head.backward(_g(12, (2, 5, 4)))


def test_gradient_flow_to_projection() -> None:
    """backward mengalirkan dW/db ke proyeksi + dx (B,T,D)."""
    head = LanguageModelHead(_small_cfg(), seed=1)
    head.forward(_x(13))
    dx = head.backward(_g(14, (2, 5, VOCAB_SIZE)))
    assert dx.shape == (2, 5, 4)
    assert head.proj.grads["W"].shape == (4, VOCAB_SIZE)
    assert head.proj.grads["b"].shape == (VOCAB_SIZE,)


def test_backward_composition_manual() -> None:
    """backward == referensi manual: dx = g @ W.T; dW = x.T @ g; db."""
    cfg = _small_cfg()
    head = LanguageModelHead(cfg, seed=5)
    x = _x(15)
    head.forward(x)
    g = _g(16, (2, 5, VOCAB_SIZE))
    dx = head.backward(g)
    x2 = x.reshape(10, cfg.d_model)
    g2 = g.reshape(10, VOCAB_SIZE)
    w = head.proj.params["W"]
    assert_close(dx, (g2 @ w.T).reshape(2, 5, cfg.d_model))
    assert_close(head.proj.grads["W"], x2.T @ g2)
    assert_close(head.proj.grads["b"], g2.sum(axis=0))


def test_numeric_grad_check_input() -> None:
    """Numeric grad check dx (REQ-003; central difference, tol 1e-5)."""
    head = LanguageModelHead(_small_cfg(), seed=3)
    x = _x(17, (2, 3, 4))
    head.forward(x)
    analytic = head.backward(np.ones((2, 3, VOCAB_SIZE)))

    def f() -> float:
        return float(np.sum(head.forward(x)))

    assert_close(analytic, numeric_grad(f, x))


def test_numeric_grad_check_parameters() -> None:
    """Numeric grad check dW/db proyeksi (REQ-003, tol 1e-5)."""
    head = LanguageModelHead(_small_cfg(), seed=4)
    x = _x(18, (2, 3, 4))
    head.forward(x)
    head.backward(np.ones((2, 3, VOCAB_SIZE)))

    def f() -> float:
        return float(np.sum(head.forward(x)))

    assert_close(head.proj.grads["W"], numeric_grad(f, head.proj.params["W"]))
    assert_close(head.proj.grads["b"], numeric_grad(f, head.proj.params["b"]))


def test_vocab_size_fixed_256() -> None:
    """Dimensi vocab FIXED 256 (DECISION-015); bukan konfigurabel."""
    cfg = _small_cfg()
    head = LanguageModelHead(cfg, seed=1)
    logits = head.forward(_x(19))
    assert logits.shape[-1] == VOCAB_SIZE == 256
    assert cfg.vocab_size == VOCAB_SIZE
    assert head.proj.out_features == VOCAB_SIZE


def test_causality_preserved_with_transformer_model() -> None:
    """Posisi t hanya bergantung token <= t (kausal T005; per posisi)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    head = LanguageModelHead(cfg, seed=2)
    assert isinstance(model.positional, PositionalRepr)
    assert all(isinstance(blk, TransformerBlock) for blk in model.blocks)
    ids1 = _ids(20)
    ids2 = ids1.copy()
    ids2[:, 3:] = (ids1[:, 3:] + 7) % 256  # ubah token masa depan
    l1 = head.forward(model.forward(ids1))
    l2 = head.forward(model.forward(ids2))
    assert np.array_equal(l1[:, :3], l2[:, :3])
    assert not np.array_equal(l1[:, 3:], l2[:, 3:])


def test_regression_t001_t011_compose() -> None:
    """Komposisi T001..T011 + T012: model -> head -> backward penuh."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    head = LanguageModelHead(cfg, seed=2)
    logits = head.forward(model.forward(_ids(30)))
    assert logits.shape == (2, 5, VOCAB_SIZE)
    assert np.all(np.isfinite(logits))
    d_logits = _g(32, (2, 5, VOCAB_SIZE))
    assert head.backward(d_logits).shape == (2, 5, cfg.d_model)
    assert model.backward(_g(31, (2, 5, 4))) is None
