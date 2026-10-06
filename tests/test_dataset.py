"""Test dataset pipeline (REQ-005, Stage 4)."""

import numpy as np
import pytest

from naze.data import TextWindows


def _ids() -> list[int]:
    return list(range(40))


def test_shapes() -> None:
    ds = TextWindows(_ids(), block_size=4, batch_size=8, seed=0)
    for x, y in ds.batches():
        assert x.shape[1] == 4
        assert x.shape[0] == y.shape[0]
        # target = token tepat setelah window
        for row in range(x.shape[0]):
            assert y[row] == (x[row, -1] + 1) % 40 or y[row] == x[row, -1] + 1


def test_deterministic_same_seed() -> None:
    a = list(TextWindows(_ids(), 4, 8, seed=7).batches())
    b = list(TextWindows(_ids(), 4, 8, seed=7).batches())
    assert len(a) == len(b)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)


def test_different_seed_differs() -> None:
    a = next(TextWindows(_ids(), 4, 8, seed=1).batches())
    b = next(TextWindows(_ids(), 4, 8, seed=2).batches())
    assert not np.array_equal(a[0], b[0])


def test_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        TextWindows([1, 2], block_size=4, batch_size=1)
    with pytest.raises(ValueError):
        TextWindows(list(range(10)), block_size=0, batch_size=1)
