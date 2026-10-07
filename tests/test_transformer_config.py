"""Test TransformerConfig (M007-T001, REQ-011) — validasi konfigurasi sesuai
M007_TECHNICAL_DESIGN §9/§10: vocab 256 fixed (DECISION-015), invariant
d_model == num_heads × head_dim, dimensi > 0, T_max > 0, dtype floating,
seed deterministik (REQ-101). Hanya T001 — tanpa test komponen (T002+).
"""

import numpy as np
import pytest

from naze.nn.transformer import VOCAB_SIZE, TransformerConfig


def test_default_config_valid() -> None:
    """Default = konfigurasi dev/test TD §9; seluruh invariant terpenuhi."""
    cfg = TransformerConfig()
    assert cfg.vocab_size == 256
    assert (cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.d_ff, cfg.max_sequence_length) == (
        64, 4, 2, 128, 64,
    )
    assert cfg.dtype == np.float64
    assert cfg.seed == 0
    # Invariant attention: 64 == 4 × 16 (TD §9).
    assert cfg.head_dim == 16
    assert cfg.d_model == cfg.num_heads * cfg.head_dim
    assert VOCAB_SIZE == 256


def test_vocab_size_fixed_256() -> None:
    """DECISION-015: vocab byte-level FIXED 256 — selain itu ditolak."""
    for bad in (255, 257, 0, -256, 1024, 256.0, "256"):
        with pytest.raises(ValueError):
            TransformerConfig(vocab_size=bad)


def test_d_model_heads_invariant() -> None:
    """Invariant d_model == num_heads * head_dim; tidak habis dibagi ditolak."""
    cfg = TransformerConfig(d_model=32, num_heads=8)
    assert cfg.d_model == cfg.num_heads * cfg.head_dim
    with pytest.raises(ValueError):
        TransformerConfig(d_model=66, num_heads=4)  # 66 % 4 != 0
    with pytest.raises(ValueError):
        TransformerConfig(d_model=7, num_heads=3)  # 7 % 3 != 0


def test_invalid_dimensions_rejected() -> None:
    """TD §10: dimensi <= 0 (atau bukan integer) ditolak ValueError."""
    for bad in (0, -1, -64, 2.5, "64", None):
        for name in ("d_model", "num_heads", "num_layers", "d_ff", "max_sequence_length"):
            with pytest.raises(ValueError):
                TransformerConfig(**{name: bad})


def test_invalid_sequence_length_rejected() -> None:
    """max_sequence_length harus integer > 0 (TD §10)."""
    with pytest.raises(ValueError):
        TransformerConfig(max_sequence_length=0)
    with pytest.raises(ValueError):
        TransformerConfig(max_sequence_length=-8)


def test_dtype_configuration_valid() -> None:
    """dtype harus tipe floating NumPy; default float64 (DECISION-007)."""
    assert TransformerConfig(dtype=np.float64).dtype == np.float64
    assert TransformerConfig(dtype=np.float32).dtype == np.float32
    with pytest.raises(ValueError):
        TransformerConfig(dtype=np.int64)
    with pytest.raises(ValueError):
        TransformerConfig(dtype="bukan-dtype")


def test_seed_stored_and_deterministic() -> None:
    """Seed tersimpan; dua config seed sama → stream RNG identik (REQ-101)."""
    cfg = TransformerConfig(seed=42)
    assert cfg.seed == 42
    a = cfg.rng().normal(size=(2, 3))
    b = TransformerConfig(seed=42).rng().normal(size=(2, 3))
    assert np.array_equal(a, b)
    c = TransformerConfig(seed=43).rng().normal(size=(2, 3))
    assert not np.array_equal(a, c)


def test_seed_validation() -> None:
    """Seed invalid (negatif / bukan integer) ditolak (konsisten seeded_rng)."""
    for bad in (-1, -42, 1.5, "7", None):
        with pytest.raises(ValueError):
            TransformerConfig(seed=bad)


def test_valid_non_default_config_accepted() -> None:
    """Konfigurasi valid non-default diterima (dimensi configurable, REQ-011)."""
    cfg = TransformerConfig(d_model=8, num_heads=2, num_layers=1, d_ff=16,
                            max_sequence_length=16, dtype=np.float32, seed=7)
    assert cfg.head_dim == 4
    assert cfg.d_model == cfg.num_heads * cfg.head_dim
    assert cfg.rng() is not None
