"""Test TransformerModel (M007-T011, REQ-011/REQ-101) — Embedding ->
PositionalRepr -> TransformerBlock x num_layers -> final LayerNorm
(TD §5 komponen 18-19; TASKS T011). Container atas komponen
T002/T003/T010 — tanpa LM head/logits (T012+). Expected values
dihitung via sub-komponen referensi independen.
"""

import numpy as np
import pytest

import naze.nn as nn
from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.layers import Layer
from naze.nn.transformer import (
    Embedding,
    LayerNorm,
    PositionalRepr,
    TransformerBlock,
    TransformerConfig,
    TransformerModel,
)


def _small_cfg(**over) -> TransformerConfig:
    """Config dev/test kecil; seluruh invariant T001 tetap terpenuhi."""
    params: dict = dict(d_model=4, num_heads=2, num_layers=2, d_ff=8,
                        max_sequence_length=8, seed=0)
    params.update(over)
    return TransformerConfig(**params)


def _ids(seed: int, b: int = 2, t: int = 5) -> np.ndarray:
    """Token ID acak deterministik (B,T) int64 dalam rentang byte valid."""
    return seeded_rng(seed).integers(0, 256, size=(b, t)).astype(np.int64)


def _g(seed: int, shape: tuple) -> np.ndarray:
    """Grad acak deterministik (pola suite T010)."""
    return seeded_rng(seed).normal(size=shape)


def test_constructor_and_composition() -> None:
    """Konstruksi: emb + pos + blocks x L + final LN; config shared (T011)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=0)
    assert isinstance(model, Layer)
    assert isinstance(model.embedding, Embedding)
    assert isinstance(model.positional, PositionalRepr)
    assert len(model.blocks) == cfg.num_layers == 2
    assert all(isinstance(block, TransformerBlock) for block in model.blocks)
    assert isinstance(model.final_norm, LayerNorm)
    assert model.config is cfg
    for block in model.blocks:
        assert block.config is cfg
    assert "transformer_model" in model.name


def test_exported_api() -> None:
    """TransformerModel diekspor via naze.nn (import + __all__)."""
    assert nn.TransformerModel is TransformerModel
    assert "TransformerModel" in nn.__all__


def test_num_layers_controls_block_count_and_parameter_count() -> None:
    """num_layers mengubah jumlah block & parameter count (TD §5 18)."""
    for layers, expected in ((1, 1236), (2, 1408), (3, 1580)):
        cfg = _small_cfg(num_layers=layers)
        model = TransformerModel(cfg)
        assert len(model.blocks) == layers
        assert model.parameter_count() == expected
    a = TransformerModel(_small_cfg(num_layers=1))
    b = TransformerModel(_small_cfg(num_layers=2))
    assert len(a.blocks) != len(b.blocks)
    assert a.parameter_count() != b.parameter_count()
    # L=1 tetap berjalan; AC T011: L>1 invarian + uji kecil L=1
    assert a.forward(_ids(1)).shape == (2, 5, 4)


def test_output_shape_hidden_states() -> None:
    """forward(ids: (B,T)) -> hidden (B,T,D) — BUKAN logits (TD §5 18-19)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    for b, t in ((1, 1), (1, 8), (3, 3)):
        assert model.forward(_ids(2, b=b, t=t)).shape == (b, t, 4)
    dev = TransformerModel(TransformerConfig(), seed=3)
    assert dev.forward(_ids(4, b=2, t=5)).shape == (2, 5, 64)  # D, bukan 256


def test_output_dtype_follows_config() -> None:
    """dtype output mengikuti config (DECISION-007; adapter cast)."""
    cfg32 = TransformerConfig(d_model=8, num_heads=2, num_layers=1, d_ff=12,
                               dtype=np.float32)
    model32 = TransformerModel(cfg32, seed=1)
    assert model32.forward(_ids(5, b=1, t=4)).dtype == np.float32
    model64 = TransformerModel(_small_cfg(), seed=2)
    assert model64.forward(_ids(6)).dtype == np.float64


def test_embedding_positional_integration() -> None:
    """Komposisi emb -> pos -> block -> final LN identik dgn referensi."""
    cfg = _small_cfg(num_layers=1)
    model = TransformerModel(cfg)  # seed allocation T011: emb 0, pos 1, block 2
    ids = _ids(7)
    y = model.forward(ids)
    emb = Embedding(cfg, seed=0)
    pos = PositionalRepr(cfg, seed=1)
    block = TransformerBlock(cfg, seed=2)
    ref = model.final_norm.forward(block.forward(pos.forward(emb.forward(ids))))
    assert np.array_equal(y, ref)
    # tanpa pos (emb saja) berbeda — P benar-benar dipakai
    ref_nopos = model.final_norm.forward(block.forward(emb.forward(ids)))
    assert not np.allclose(y, ref_nopos)


def test_sequential_block_execution() -> None:
    """Stacking berurutan b0 -> b1; bukan paralel/last-only (TD §5 18)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    ids = _ids(8)
    y = model.forward(ids)
    x0 = model.embedding.forward(ids)
    x1 = model.positional.forward(x0)
    h0 = model.blocks[0].forward(x1)
    h1 = model.blocks[1].forward(h0)
    assert np.array_equal(y, model.final_norm.forward(h1))
    # bukan hanya block terakhir / pertama yang dipakai
    last_only = model.final_norm.forward(model.blocks[1].forward(x1))
    first_only = model.final_norm.forward(model.blocks[0].forward(x1))
    assert not np.allclose(y, last_only)
    assert not np.allclose(y, first_only)
    # output tiap block berubah (bukan input awal utk semua block)
    assert not np.allclose(h0, x1)
    assert not np.allclose(h1, h0)


def test_parameter_count() -> None:
    """Count = emb + pos + blocks + final LN aktual; bukan hard-coded."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    assert model.parameter_count() == (model.embedding.parameter_count()
                                       + model.positional.parameter_count()
                                       + sum(b.parameter_count() for b in model.blocks)
                                       + model.final_norm.parameter_count())
    # D=4,H=2,d_ff=8,L=2: E 256x4 + P 8x4 + 2 x 172 + LN 2x4
    assert model.embedding.parameter_count() == 1024
    assert model.positional.parameter_count() == 32
    assert model.blocks[0].parameter_count() == 172
    assert model.final_norm.parameter_count() == 8
    assert model.parameter_count() == 1024 + 32 + 2 * 172 + 8
    assert model.parameter_count() == 1408


def test_parameter_aggregation() -> None:
    """Param via sub-komponen; tanpa duplikat (model.params kosong)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    assert model.params == {}
    assert model.embedding.params["E"].shape == (256, 4)
    assert model.positional.params["P"].shape == (8, 4)
    assert model.final_norm.params["gamma"].shape == (4,)
    for block in model.blocks:
        assert block.params == {}  # pola T010 utuh


def test_no_accidental_parameter_sharing() -> None:
    """Block berbeda tidak berbagi parameter (objek & nilai) antar layer."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    w0 = model.blocks[0].attention.branch.qkv.q_proj.params["W"]
    w1 = model.blocks[1].attention.branch.qkv.q_proj.params["W"]
    assert w0 is not w1
    assert not np.array_equal(w0, w1)
    f0 = model.blocks[0].ffn.branch.fc1.params["W"]
    f1 = model.blocks[1].ffn.branch.fc1.params["W"]
    assert f0 is not f1
    assert not np.array_equal(f0, f1)
    assert model.embedding.params["E"] is not model.positional.params["P"]


def test_deterministic_construction_and_forward() -> None:
    """Seed sama -> parameter & output identik; seed beda -> beda (REQ-101)."""
    cfg = _small_cfg()
    a = TransformerModel(cfg, seed=4)
    b = TransformerModel(cfg, seed=4)
    assert np.array_equal(a.embedding.params["E"], b.embedding.params["E"])
    assert np.array_equal(a.positional.params["P"], b.positional.params["P"])
    for i in range(cfg.num_layers):
        assert np.array_equal(a.blocks[i].attention.branch.qkv.q_proj.params["W"],
                              b.blocks[i].attention.branch.qkv.q_proj.params["W"])
    ids = _ids(9)
    assert np.array_equal(a.forward(ids), b.forward(ids))
    assert np.array_equal(a.forward(ids), a.forward(ids))
    c = TransformerModel(cfg, seed=5)
    assert not np.array_equal(a.embedding.params["E"], c.embedding.params["E"])
    assert not np.allclose(a.forward(ids), c.forward(ids))


def test_invalid_input_shape_rejected() -> None:
    """ids harus (B,T) 2D — 1D/3D ditolak (TD §10; kontrak T002)."""
    model = TransformerModel(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        model.forward(np.zeros(5, dtype=np.int64))  # 1D
    with pytest.raises(ValueError):
        model.forward(np.zeros((2, 5, 1), dtype=np.int64))  # 3D


def test_non_integer_and_out_of_range_ids_rejected() -> None:
    """Token ID harus integer dalam rentang byte valid (TD §10; T002)."""
    model = TransformerModel(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        model.forward(np.zeros((2, 5)))  # float
    ids = _ids(10)
    ids[0, 0] = -1
    with pytest.raises(ValueError):
        model.forward(ids)
    ids2 = _ids(11)
    ids2[1, 4] = 256
    with pytest.raises(ValueError):
        model.forward(ids2)


def test_empty_batch_sequence_rejected() -> None:
    """B=0 atau T=0 ditolak ValueError (TD §10)."""
    model = TransformerModel(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        model.forward(np.zeros((0, 5), dtype=np.int64))
    with pytest.raises(ValueError):
        model.forward(np.zeros((2, 0), dtype=np.int64))


def test_sequence_length_overflow_rejected() -> None:
    """T > max_sequence_length ditolak ValueError (TD §8/§10)."""
    model = TransformerModel(_small_cfg(), seed=1)
    with pytest.raises(ValueError):
        model.forward(_ids(12, t=9))


def test_backward_guard_shape_and_none_return() -> None:
    """Guard urutan & shape grad; backward -> None (pola Embedding)."""
    model = TransformerModel(_small_cfg(), seed=1)
    with pytest.raises(RuntimeError):
        model.backward(np.zeros((2, 5, 4)))  # belum forward
    model.forward(_ids(13))
    g = _g(14, (2, 5, 4))
    assert model.backward(g) is None  # ids diskrit: grad input tak terdefinisi
    with pytest.raises(ValueError):
        model.backward(np.zeros((2, 6, 4)))  # shape grad salah


def test_gradient_flow_to_all_components() -> None:
    """Grad mengalir ke emb, pos, SEMUA block, dan final LN (REQ-003)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    model.forward(_ids(15))
    model.backward(_g(16, (2, 5, 4)))
    assert np.any(model.embedding.grads["E"] != 0.0)
    assert np.any(model.positional.grads["P"] != 0.0)
    assert np.any(model.final_norm.grads["gamma"] != 0.0)
    assert np.any(model.final_norm.grads["beta"] != 0.0)
    for block in model.blocks:
        mha = block.attention.branch
        for proj in (mha.qkv.q_proj, mha.qkv.k_proj, mha.qkv.v_proj, mha.out_proj):
            assert np.any(proj.grads["W"] != 0.0)
            assert np.any(proj.grads["b"] != 0.0)
        ffn = block.ffn.branch
        for lin in (ffn.fc1, ffn.fc2):
            assert np.any(lin.grads["W"] != 0.0)
            assert np.any(lin.grads["b"] != 0.0)
        for norm in (block.attention.norm, block.ffn.norm):
            assert np.any(norm.grads["gamma"] != 0.0)
            assert np.any(norm.grads["beta"] != 0.0)


def test_gradient_not_shared_between_layers() -> None:
    """Grad block pertama & terakhir tidak alias (tidak tertukar)."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    model.forward(_ids(17))
    model.backward(_g(18, (2, 5, 4)))
    g0 = model.blocks[0].attention.branch.qkv.q_proj.grads["W"]
    g1 = model.blocks[1].attention.branch.qkv.q_proj.grads["W"]
    assert g0 is not g1
    assert not np.array_equal(g0, g1)
    assert np.any(g0 != 0.0)
    assert np.any(g1 != 0.0)


def test_backward_composition_manual() -> None:
    """dx chain terbalik: final LN -> blocks (L-1..0) -> pos -> emb."""
    cfg = _small_cfg()
    model = TransformerModel(cfg, seed=1)
    model.forward(_ids(19))
    g = _g(20, (2, 5, 4))
    model.backward(g)
    d_e = model.embedding.grads["E"].copy()
    # recompute manual via sub-komponen (cache sama; deterministik)
    d_h = model.final_norm.backward(g)
    for block in reversed(model.blocks):
        d_h = block.backward(d_h)
    d_h = model.positional.backward(d_h)
    model.embedding.backward(d_h)
    assert np.array_equal(d_e, model.embedding.grads["E"])


def test_numeric_grad_check_embedding_and_positional() -> None:
    """Central difference dE (embedding path) & dP (REQ-003; tol 1e-5)."""
    cfg = _small_cfg(num_layers=1)
    model = TransformerModel(cfg, seed=21)
    ids = _ids(22, b=1, t=2)
    g = _g(23, (1, 2, 4))
    model.forward(ids)
    model.backward(g)

    def loss() -> float:
        return float((model.forward(ids) * g).sum())

    w = model.embedding.params["E"]
    assert_close(model.embedding.grads["E"], numeric_grad(loss, w))
    p = model.positional.params["P"]
    assert_close(model.positional.grads["P"], numeric_grad(loss, p))


def test_numeric_grad_check_first_last_block_and_final_norm() -> None:
    """Central difference block pertama, terakhir, final LN (REQ-003)."""
    cfg = _small_cfg()  # L=2
    model = TransformerModel(cfg, seed=24)
    ids = _ids(25, b=2, t=3)
    g = _g(26, (2, 3, 4))
    model.forward(ids)
    model.backward(g)

    def loss() -> float:
        return float((model.forward(ids) * g).sum())

    first = model.blocks[0].ffn.branch.fc1
    assert_close(first.grads["W"], numeric_grad(loss, first.params["W"]))
    last = model.blocks[1].attention.branch.out_proj
    assert_close(last.grads["W"], numeric_grad(loss, last.params["W"]))
    norm = model.final_norm
    assert_close(norm.grads["gamma"], numeric_grad(loss, norm.params["gamma"]))
    assert_close(norm.grads["beta"], numeric_grad(loss, norm.params["beta"]))


def test_causality_preserved_through_stack() -> None:
    """Ubah token masa depan -> hidden posisi sebelumnya tetap (TD §8)."""
    cfg = _small_cfg()  # L=2: mask kausal tetap bekerja saat stacked
    model = TransformerModel(cfg, seed=1)
    ids = _ids(27)
    out = model.forward(ids)
    ids2 = ids.copy()
    ids2[0, 4] = (int(ids2[0, 4]) + 1) % 256  # ubah token masa depan
    out2 = model.forward(ids2)
    assert np.array_equal(out[:, :4], out2[:, :4])  # posisi <= 3 tak berubah
    assert not np.array_equal(out[0, 4], out2[0, 4])  # posisi 4 berubah
    assert np.all(np.isfinite(out))  # finite (TD §10)


def test_regression_t001_t010() -> None:
    """T001-T010 utuh: komposisi & validasi config tidak berubah."""
    # T001: config invalid tetap ditolak (regresi invariant)
    with pytest.raises(ValueError):
        TransformerConfig(d_model=4, num_heads=3)  # 4 % 3 != 0
    with pytest.raises(ValueError):
        _small_cfg(num_layers=0)
    # T002+T003+T010 via instance independen identik dgn model L=1
    cfg = _small_cfg(num_layers=1)
    model = TransformerModel(cfg)
    ids = _ids(28, b=3, t=7)
    y = model.forward(ids)
    emb = Embedding(cfg, seed=0)
    pos = PositionalRepr(cfg, seed=1)
    block = TransformerBlock(cfg, seed=2)
    ref = model.final_norm.forward(block.forward(pos.forward(emb.forward(ids))))
    assert np.array_equal(y, ref)
    # TransformerBlock sendiri tetap bekerja (regresi T010)
    x = _g(29, (2, 5, 4))
    assert block.forward(x).shape == (2, 5, 4)
