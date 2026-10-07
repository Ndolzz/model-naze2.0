"""Test Embedding (M007-T002, REQ-011) — lookup (B,T)->(B,T,D), backward
scatter-add, validasi token ID & shape (TD §5/§7/§10). Hanya T002 — tanpa
test positional representation atau attention (T003+).
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import Embedding, TransformerConfig


def _small_cfg(**over) -> TransformerConfig:
    """Config dev/test kecil; seluruh invariant T001 tetap terpenuhi."""
    params: dict = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                        max_sequence_length=8, seed=0)
    params.update(over)
    return TransformerConfig(**params)


def test_param_shape_and_count() -> None:
    """E = (vocab_size, d_model); jumlah parameter eksplisit 256×64 = 16384 (TD §12)."""
    emb = Embedding(TransformerConfig())
    assert emb.params["E"].shape == (256, 64)
    assert emb.parameter_count() == 256 * 64 == 16384


def test_output_shape_bt_d() -> None:
    """Output forward: (B, T, d_model) (TD §5 komponen 1)."""
    emb = Embedding(_small_cfg())
    out = emb.forward(np.zeros((2, 5), dtype=np.int64))
    assert out.shape == (2, 5, 4)


def test_different_batch_sizes() -> None:
    """Baris output untuk token sama tidak bergantung pada batch size."""
    emb = Embedding(_small_cfg())
    a = emb.forward(np.array([[1, 2], [1, 2]], dtype=np.int64))  # B=2
    b = emb.forward(np.array([[1, 2]], dtype=np.int64))  # B=1
    c = emb.forward(np.array([[5], [5], [5]], dtype=np.int64))  # B=3, T=1
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[0])
    assert np.array_equal(c[0, 0], emb.params["E"][5])


def test_different_sequence_lengths() -> None:
    """Panjang sekuens berbeda valid selama T <= max_sequence_length."""
    emb = Embedding(_small_cfg())  # T_max=8
    for t in (1, 4, 8):
        out = emb.forward(np.zeros((2, t), dtype=np.int64))
        assert out.shape == (2, t, 4)


def test_token_ids_0_and_255_valid() -> None:
    """Batas bawah & atas vocab byte-level valid (DECISION-015)."""
    emb = Embedding(TransformerConfig(d_model=4, num_heads=2))
    out = emb.forward(np.array([[0, 255, 0, 255]], dtype=np.int64))
    assert np.array_equal(out[0, 0], emb.params["E"][0])
    assert np.array_equal(out[0, 1], emb.params["E"][255])


def test_invalid_token_ids_rejected() -> None:
    """Token ID di luar [0, 256) ditolak ValueError dengan pesan jelas (TD §10)."""
    emb = Embedding(_small_cfg())
    with pytest.raises(ValueError):
        emb.forward(np.array([[256]], dtype=np.int64))
    with pytest.raises(ValueError):
        emb.forward(np.array([[-1]], dtype=np.int64))
    with pytest.raises(ValueError):
        emb.forward(np.array([[0, 300]], dtype=np.int64))


def test_non_integer_token_ids_rejected() -> None:
    """Token non-integer (float/string) ditolak — meski nilainya bulat."""
    emb = Embedding(_small_cfg())
    with pytest.raises(ValueError):
        emb.forward(np.array([[1.5, 2.0]]))
    with pytest.raises(ValueError):
        emb.forward(np.array([["1", "2"]]))


def test_invalid_input_shape_rejected() -> None:
    """Input harus (batch, seq) 2D dan tidak kosong (TD §10)."""
    emb = Embedding(_small_cfg())
    with pytest.raises(ValueError):
        emb.forward(np.zeros(5, dtype=np.int64))  # 1D
    with pytest.raises(ValueError):
        emb.forward(np.zeros((2, 5, 1), dtype=np.int64))  # 3D
    with pytest.raises(ValueError):
        emb.forward(np.zeros((0, 5), dtype=np.int64))  # batch kosong
    with pytest.raises(ValueError):
        emb.forward(np.zeros((2, 0), dtype=np.int64))  # sekuens kosong


def test_sequence_length_over_max_rejected() -> None:
    """TD §8/§10: T > max_sequence_length ditolak; T == max valid."""
    emb = Embedding(_small_cfg())  # T_max=8
    with pytest.raises(ValueError):
        emb.forward(np.zeros((2, 9), dtype=np.int64))
    assert emb.forward(np.zeros((2, 8), dtype=np.int64)).shape == (2, 8, 4)


def test_dtype_follows_config() -> None:
    """dtype E dan output mengikuti config (DECISION-007; default float64)."""
    emb64 = Embedding(TransformerConfig())
    assert emb64.params["E"].dtype == np.float64
    assert emb64.forward(np.zeros((1, 3), dtype=np.int64)).dtype == np.float64
    emb32 = Embedding(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    assert emb32.params["E"].dtype == np.float32
    assert emb32.forward(np.zeros((1, 3), dtype=np.int64)).dtype == np.float32


def test_deterministic_init_same_seed() -> None:
    """Seed sama → E identik (REQ-101); kwarg seed per-instance juga deterministik."""
    assert np.array_equal(Embedding(_small_cfg(seed=7)).params["E"],
                          Embedding(_small_cfg(seed=7)).params["E"])
    assert np.array_equal(Embedding(_small_cfg(seed=7), seed=9).params["E"],
                          Embedding(_small_cfg(seed=7), seed=9).params["E"])


def test_different_seed_different_init() -> None:
    """Seed berbeda → inisialisasi berbeda (stream RNG independen per-seed)."""
    a = Embedding(_small_cfg(seed=7))
    b = Embedding(_small_cfg(seed=8))
    assert not np.array_equal(a.params["E"], b.params["E"])
    assert not np.array_equal(Embedding(_small_cfg(seed=7), seed=9).params["E"],
                              a.params["E"])


def test_forward_lookup_values() -> None:
    """Lookup benar: output = baris E pada indeks ids (hand-computed)."""
    emb = Embedding(_small_cfg())
    emb.params["E"] = np.arange(256 * 4, dtype=np.float64).reshape(256, 4)
    ids = np.array([[0, 255], [3, 0]], dtype=np.int64)
    out = emb.forward(ids)
    assert np.array_equal(out[0, 0], np.arange(4, dtype=np.float64))
    assert np.array_equal(out[0, 1], np.arange(1020.0, 1024.0))
    assert np.array_equal(out[1, 0], np.arange(12.0, 16.0))
    assert np.array_equal(out[1, 1], np.arange(4, dtype=np.float64))


def test_forward_no_nan_inf() -> None:
    """Output forward finite (TD §10: nilai non-finite = failure, bukan di-silent)."""
    emb = Embedding(_small_cfg())
    ids = seeded_rng(1).integers(0, 256, size=(3, 8))
    assert np.all(np.isfinite(emb.forward(ids)))


def test_backward_scatter_add_hand_computed() -> None:
    """Scatter-add: grad token terakumulasi; token tak terpakai grad 0 (TD §7)."""
    emb = Embedding(TransformerConfig(d_model=2, num_heads=1, max_sequence_length=4))
    emb.forward(np.array([[0, 1], [1, 0]], dtype=np.int64))
    g = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    assert emb.backward(g) is None
    d_e = emb.grads["E"]
    assert d_e.shape == (256, 2)
    assert np.array_equal(d_e[0], [8.0, 10.0])  # g[0,0] + g[1,1]
    assert np.array_equal(d_e[1], [8.0, 10.0])  # g[0,1] + g[1,0]
    assert np.all(d_e[2:] == 0.0)


def test_embedding_grad_check_numeric() -> None:
    """Grad scatter-add vs central difference — engine gradcheck (REQ-003, AC T002)."""
    emb = Embedding(_small_cfg(seed=3), seed=3)
    ids = np.array([[0, 2], [3, 1]], dtype=np.int64)
    g = seeded_rng(5).normal(size=(2, 2, 4))
    emb.forward(ids)
    emb.backward(g)

    def loss() -> float:
        return float((emb.params["E"][ids] * g).sum())

    assert_close(emb.grads["E"], numeric_grad(loss, emb.params["E"]))


def test_backward_guard_and_grad_shape() -> None:
    """Guard urutan (RuntimeError) dan validasi shape/finite grad_out (TD §10)."""
    emb = Embedding(_small_cfg())
    with pytest.raises(RuntimeError):
        emb.backward(np.zeros((1, 2, 4)))  # backward sebelum forward
    emb.forward(np.array([[1, 2]], dtype=np.int64))
    with pytest.raises(ValueError):
        emb.backward(np.zeros((2, 2, 4)))  # shape grad_out salah
    with pytest.raises(ValueError):
        emb.backward(np.array([[[np.nan] * 4, [0.0] * 4]]))  # non-finite ditolak
