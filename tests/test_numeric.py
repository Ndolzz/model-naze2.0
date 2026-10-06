"""Test numeric core (REQ-001) — determinisme dan validasi input."""

import numpy as np
import pytest

from naze.core.numeric import as_array, seeded_rng


def test_seeded_rng_deterministic() -> None:
    a = seeded_rng(42).normal(size=5)
    b = seeded_rng(42).normal(size=5)
    assert np.array_equal(a, b)


def test_seeded_rng_different_seeds_differ() -> None:
    a = seeded_rng(1).normal(size=5)
    b = seeded_rng(2).normal(size=5)
    assert not np.array_equal(a, b)


def test_seeded_rng_rejects_bad_seed() -> None:
    with pytest.raises(ValueError):
        seeded_rng(-1)
    with pytest.raises(ValueError):
        seeded_rng("x")  # type: ignore[arg-type]


def test_as_array_converts_and_validates() -> None:
    out = as_array([1, 2, 3])
    assert out.dtype == np.float64
    with pytest.raises(ValueError):
        as_array([1.0, np.nan])
    with pytest.raises(ValueError):
        as_array([1.0, np.inf])
