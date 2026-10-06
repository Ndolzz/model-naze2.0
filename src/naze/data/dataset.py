"""Dataset pipeline (REQ-005, Stage 4).

TextWindows: sliding window (x = ids[i:i+T], y = ids[i+T]) untuk next-token LM.
- Deterministik per-seed (shuffle via seeded_rng).
- Memori terkendali: hanya index, tidak menyalin korpus.
Resolusi OPEN DECISION-103: dataset berupa teks apa pun (string/path file);
korpus final untuk training Naze tetap OPEN (menunggu project owner).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from naze.core.numeric import seeded_rng


class TextWindows:
    """Batch window next-token dari sequence token ID."""

    def __init__(self, ids: list[int], block_size: int, batch_size: int, *, seed: int = 0) -> None:
        if block_size <= 0 or batch_size <= 0:
            raise ValueError("block_size dan batch_size harus > 0")
        if len(ids) < block_size + 1:
            raise ValueError("sequence terlalu pendek untuk block_size")
        self.ids = np.asarray(ids, dtype=np.int64)
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.n_windows = len(ids) - block_size

    def batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (x, y): x (B, T) int64, y (B,) int64. Urutan deterministik per-seed."""
        rng = seeded_rng(self.seed)
        order = rng.permutation(self.n_windows)
        for start in range(0, self.n_windows, self.batch_size):
            idx = order[start : start + self.batch_size]
            x = np.stack([self.ids[i : i + self.block_size] for i in idx])
            y = self.ids[idx + self.block_size]
            yield x, y

    def epoch_size(self) -> int:
        return (self.n_windows + self.batch_size - 1) // self.batch_size
