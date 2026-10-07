"""Test PositionalRepr (M007-T003, REQ-011/REQ-101) — learned table
(T_max, d_model), x + P[0:T] (TD §5/§7/§9, OD-121), backward scatter-add
per posisi + grad_input = grad_out. Hanya T003 — tanpa test attention/FFN
(T004+). Posisi implisit 0..T-1 (TD §7): selalu integer valid by
construction; uji batas via panjang sekuens T.
"""

import numpy as np
import pytest

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import seeded_rng
from naze.nn.transformer import PositionalRepr, TransformerConfig


def _small_cfg(**over) -> TransformerConfig:
    """Config dev/test kecil; seluruh invariant T001 tetap terpenuhi."""
    params: dict = dict(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                        max_sequence_length=8, seed=0)
    params.update(over)
    return TransformerConfig(**params)


def test_param_shape_and_count() -> None:
    """P = (T_max, d_model); default dev/test 64×64 = 4096 param (TD §12)."""
    pos = PositionalRepr(TransformerConfig())
    assert pos.params["P"].shape == (64, 64)
    assert pos.parameter_count() == 64 * 64 == 4096
    small = PositionalRepr(_small_cfg())
    assert small.params["P"].shape == (8, 4)
    assert small.parameter_count() == 8 * 4


def test_output_shape_and_addition() -> None:
    """Output (B, T, D) = x + P[0:T] (aditif, broadcast ke batch)."""
    pos = PositionalRepr(_small_cfg())
    x = np.ones((2, 5, 4))
    out = pos.forward(x)
    assert out.shape == (2, 5, 4)
    assert np.allclose(out[0, 2], 1.0 + pos.params["P"][2])
    assert np.allclose(out[1, 4], 1.0 + pos.params["P"][4])


def test_first_and_last_position_valid() -> None:
    """Posisi pertama (0) dan terakhir (T-1) memakai baris P yang benar."""
    pos = PositionalRepr(_small_cfg())
    x = np.zeros((1, 8, 4))
    out = pos.forward(x)
    assert np.array_equal(out[0, 0], pos.params["P"][0])
    assert np.array_equal(out[0, 7], pos.params["P"][7])


def test_varying_batch_size() -> None:
    """P[0:T] sama untuk semua batch; output hanya bergantung x per baris."""
    pos = PositionalRepr(_small_cfg())
    x1 = np.zeros((1, 3, 4))
    x4 = np.zeros((4, 3, 4))
    assert np.array_equal(pos.forward(x1)[0], pos.forward(x4)[0])


def test_varying_sequence_length() -> None:
    """T berbeda valid untuk 1..T_max; baris P sesuai posisi."""
    pos = PositionalRepr(_small_cfg())  # T_max=8
    for t in (1, 2, 4, 8):
        out = pos.forward(np.zeros((2, t, 4)))
        assert out.shape == (2, t, 4)
        assert np.array_equal(out[0, t - 1], pos.params["P"][t - 1])


def test_sequence_length_max_valid() -> None:
    pos = PositionalRepr(_small_cfg())  # T_max=8
    assert pos.forward(np.zeros((2, 8, 4))).shape == (2, 8, 4)


def test_sequence_length_over_max_rejected() -> None:
    """TD §8/§10: T > max_sequence_length ditolak ValueError."""
    pos = PositionalRepr(_small_cfg())  # T_max=8
    with pytest.raises(ValueError):
        pos.forward(np.zeros((2, 9, 4)))


def test_invalid_shape_rejected() -> None:
    """Input harus (B, T, D) 3D, tidak kosong, D == config.d_model (TD §10)."""
    pos = PositionalRepr(_small_cfg())  # d_model=4
    with pytest.raises(ValueError):
        pos.forward(np.zeros((5, 4)))  # 2D
    with pytest.raises(ValueError):
        pos.forward(np.zeros((2, 5, 4, 1)))  # 4D
    with pytest.raises(ValueError):
        pos.forward(np.zeros((2, 5, 3)))  # D != d_model
    with pytest.raises(ValueError):
        pos.forward(np.zeros((0, 5, 4)))  # batch kosong
    with pytest.raises(ValueError):
        pos.forward(np.zeros((2, 0, 4)))  # sekuens kosong


def test_dtype_follows_config() -> None:
    """dtype P dan output mengikuti config (DECISION-007; default float64)."""
    pos64 = PositionalRepr(TransformerConfig())
    assert pos64.params["P"].dtype == np.float64
    assert pos64.forward(np.zeros((1, 3, 64))).dtype == np.float64
    pos32 = PositionalRepr(TransformerConfig(d_model=8, num_heads=2, dtype=np.float32))
    assert pos32.params["P"].dtype == np.float32
    assert pos32.forward(np.ones((1, 3, 8))).dtype == np.float32


def test_deterministic_init_same_seed() -> None:
    """Seed/config sama → P identik (REQ-101); kwarg seed juga deterministik."""
    assert np.array_equal(PositionalRepr(_small_cfg(seed=7)).params["P"],
                          PositionalRepr(_small_cfg(seed=7)).params["P"])
    assert np.array_equal(PositionalRepr(_small_cfg(seed=7), seed=9).params["P"],
                          PositionalRepr(_small_cfg(seed=7), seed=9).params["P"])


def test_different_seed_different_init() -> None:
    """Seed berbeda → P berbeda (stream RNG independen per-seed)."""
    a = PositionalRepr(_small_cfg(seed=7))
    assert not np.array_equal(a.params["P"], PositionalRepr(_small_cfg(seed=8)).params["P"])
    assert not np.array_equal(a.params["P"], PositionalRepr(_small_cfg(seed=7), seed=9).params["P"])


def test_forward_no_nan_inf() -> None:
    """Output finite untuk input finite (TD §10: non-finite = failure)."""
    pos = PositionalRepr(_small_cfg())
    x = seeded_rng(2).normal(size=(3, 8, 4))
    assert np.all(np.isfinite(pos.forward(x)))


def test_backward_before_forward_rejected() -> None:
    """Guard urutan engine (DECISION-009, TD §10): RuntimeError."""
    pos = PositionalRepr(_small_cfg())
    with pytest.raises(RuntimeError):
        pos.backward(np.zeros((2, 3, 4)))


def test_backward_grad_shapes_and_pass_through() -> None:
    """grads["P"] berbentuk (T_max, d_model); grad_input = grad_out (aditif)."""
    pos = PositionalRepr(_small_cfg())  # T_max=8, D=4
    g = seeded_rng(3).normal(size=(2, 5, 4))
    pos.forward(np.zeros((2, 5, 4)))
    dx = pos.backward(g)
    assert pos.grads["P"].shape == (8, 4)
    assert dx.shape == (2, 5, 4)
    assert np.array_equal(dx, g)  # representasi aditif: d output/d x = identitas
    with pytest.raises(ValueError):
        pos.backward(np.zeros((2, 6, 4)))  # shape grad_out salah


def test_backward_scatter_add_hand_computed() -> None:
    """Posisi t berulang di batch → dP[t] = jumlah grad seluruh batch (scatter)."""
    pos = PositionalRepr(TransformerConfig(d_model=2, num_heads=1, max_sequence_length=4))
    pos.forward(np.zeros((2, 2, 2)))  # T=2, posisi 0..1 dipakai
    g = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    pos.backward(g)
    d_p = pos.grads["P"]
    assert d_p.shape == (4, 2)
    # dP[0] = g[0,0] + g[1,0] = [1+5, 2+6]
    assert np.array_equal(d_p[0], [6.0, 8.0])
    # dP[1] = g[0,1] + g[1,1] = [3+7, 4+8]
    assert np.array_equal(d_p[1], [10.0, 12.0])
    # posisi 2..T_max-1 tidak dipakai → grad nol
    assert np.all(d_p[2:] == 0.0)


def test_positional_grad_check_numeric() -> None:
    """dP scatter vs central difference — engine gradcheck (REQ-003, TD §11)."""
    pos = PositionalRepr(_small_cfg(seed=3), seed=3)  # P: (8, 4)
    x = seeded_rng(4).normal(size=(2, 5, 4))
    g = seeded_rng(5).normal(size=(2, 5, 4))
    pos.forward(x)
    pos.backward(g)

    def loss() -> float:
        return float((pos.forward(x) * g).sum())

    assert_close(pos.grads["P"], numeric_grad(loss, pos.params["P"]))
