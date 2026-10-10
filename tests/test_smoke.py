"""Smoke test: memverifikasi tooling dan struktur proyek berjalan (MILESTONE-001).

Bukan test AI — tidak ada implementasi neural pada milestone ini.
"""

import naze


def test_package_importable() -> None:
    assert naze.__version__ == "1.0.0"  # T001: versi final Naze 1.0 (sinkron pyproject)


def test_numpy_available() -> None:
    import numpy as np

    # Sanity dasar: operasi NumPy tersedia sebagai fondasi numerik (REQ-203).
    result = np.arange(4).reshape(2, 2).sum()
    assert result == 6
