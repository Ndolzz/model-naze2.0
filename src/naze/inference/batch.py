"""InferenceBatch: padding, mask, chunking untuk batch inference (M-009-T001, REQ-008).

InferenceBatch mengelompokkan token sequences dengan panjang bervariasi
menjadi batch (B, T) dengan padding dan mask validasi. Urutan deterministik
(tidak di-shuffle, untuk inference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from naze.core.numeric import Array


class InferenceBatch:
    """Batch inference dengan padding dan mask.

    Atribut:
        x: (B, T) int64, token IDs dengan padding (0) di akhir.
        mask: (B, T) bool, True = token valid (bukan padding).
        lengths: list[int], panjang asli tiap sequence.
    """

    def __init__(
        self,
        token_ids: list[list[int]],
        max_length: int,
        *,
        pad_id: int = 0,
    ) -> None:
        """Buat batch dari token sequences.

        Args:
            token_ids: List of token sequences (panjang boleh bervariasi).
            max_length: Panjang maksimum sequence (T).
            pad_id: ID token untuk padding (default 0).

        Raises:
            ValueError: token_ids kosong, sequence terlalu panjang, atau token ID invalid.
        """
        if not token_ids:
            raise ValueError("token_ids tidak boleh kosong")
        if max_length <= 0:
            raise ValueError("max_length harus > 0")

        self.pad_id = pad_id
        self.max_length = max_length

        # Validasi token IDs
        self.lengths = []
        for seq in token_ids:
            if not isinstance(seq, list):
                raise ValueError(f"tiap item token_ids harus list, dapat {type(seq).__name__}")
            if len(seq) > max_length:
                raise ValueError(
                    f"sequence terlalu panjang: {len(seq)} > max_length={max_length}"
                )
            for tok in seq:
                if not isinstance(tok, int) or tok < 0 or tok >= 256:
                    raise ValueError(
                        f"token ID harus integer di [0, 256), dapat {tok!r}"
                    )
            self.lengths.append(len(seq))

        # Build x (B, T) dengan padding
        b = len(token_ids)
        t = max_length
        self.x = np.full((b, t), pad_id, dtype=np.int64)
        for i, seq in enumerate(token_ids):
            self.x[i, : len(seq)] = np.asarray(seq, dtype=np.int64)

        # Build mask (B, T): True = valid token
        self.mask = np.zeros((b, t), dtype=bool)
        for i, length in enumerate(self.lengths):
            self.mask[i, :length] = True

    def __repr__(self) -> str:
        return (
            f"InferenceBatch(batch_size={len(self.lengths)}, "
            f"max_length={self.max_length}, "
            f"lengths={self.lengths})"
        )
