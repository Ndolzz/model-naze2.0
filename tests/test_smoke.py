"""Smoke test: memverifikasi tooling dan struktur proyek berjalan (MILESTONE-001).

Bukan test AI — tidak ada implementasi neural pada milestone ini.
"""

import naze


def test_package_importable() -> None:
    assert naze.__version__ == "0.0.2"


def test_numpy_available() -> None:
    import numpy as np

    # Sanity dasar: operasi NumPy tersedia sebagai fondasi numerik (REQ-203).
    result = np.arange(4).reshape(2, 2).sum()
    assert result == 6
