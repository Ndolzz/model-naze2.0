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

    def batches(self, *, seed: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (x, y): x (B, T) int64, y (B,) int64. Urutan deterministik per-seed.

        seed None berarti memakai seed instance (perilaku lama tidak berubah;
        M-008 menambahkan kwarg opsional agar training loop dapat mengatur
        seed per-epoch tanpa membuat objek baru).
        """
        rng = seeded_rng(self.seed if seed is None else seed)
        order = rng.permutation(self.n_windows)
        for start in range(0, self.n_windows, self.batch_size):
            idx = order[start : start + self.batch_size]
            x = np.stack([self.ids[i : i + self.block_size] for i in idx])
            y = self.ids[idx + self.block_size]
            yield x, y

    def batches_pos(self, *, seed: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (x, y) per-posisi: x (B, T), y (B, T) = x digeser 1 (M-008).

        Window per posisi: x = ids[i:i+T], y = ids[i+1:i+T+1] — target beda
        tiap posisi (untuk loss_pos TransformerLM). Butuh >= block_size + 2
        token. Urutan deterministik per-seed (pola batches).
        """
        t = self.block_size
        n_pos = len(self.ids) - t - 1
        if n_pos <= 0:
            raise ValueError("sequence terlalu pendek untuk batches_pos (butuh block_size + 2)")
        rng = seeded_rng(self.seed if seed is None else seed)
        order = rng.permutation(n_pos)
        for start in range(0, n_pos, self.batch_size):
            idx = order[start : start + self.batch_size]
            x = np.stack([self.ids[i : i + t] for i in idx])
            y = np.stack([self.ids[i + 1 : i + t + 1] for i in idx])
            yield x, y

    def epoch_size(self) -> int:
        return (self.n_windows + self.batch_size - 1) // self.batch_size
