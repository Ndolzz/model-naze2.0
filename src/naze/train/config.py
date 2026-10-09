"""Konfigurasi training lengkap (M008-T001, REQ-007/REQ-101).

TrainConfig: satu sumber kebenaran konfigurasi training M-008 — model,
data, optimisasi, evaluasi berkala, checkpoint, dan dimensi arsitektur.
Default dimensi transformer mengikuti DECISION-018 (D=64, H=4, L=2,
d_ff=128, T_max=128; ~108k parameter, checkpoint <=1 MB float64).
Hash konfigurasi (SHA-256 JSON kanonik) membuat tiap run dapat
direproduksi & dilacak ke konfigurasi eksak (REQ-101).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

_MODELS = ("transformer", "mlplm")


@dataclass(frozen=True)
class TrainConfig:
    """Konfigurasi training; validasi fail-fast saat konstruksi."""

    model: str = "transformer"
    corpus_path: str = "korpus.txt"
    holdout_path: str | None = None
    block_size: int = 8
    batch_size: int = 8
    lr: float = 0.08
    epochs: int = 3
    seed: int = 0
    eval_every: int = 1
    checkpoint_path: str | None = None
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    max_sequence_length: int = 128
    d_embed: int = 8
    d_hidden: int = 16

    def __post_init__(self) -> None:
        if self.model not in _MODELS:
            raise ValueError(f"model harus salah satu dari {_MODELS}, dapat {self.model!r}")
        if self.block_size <= 0 or self.batch_size <= 0:
            raise ValueError("block_size dan batch_size harus > 0")
        if self.epochs <= 0:
            raise ValueError("epochs harus > 0")
        if self.eval_every <= 0:
            raise ValueError("eval_every harus > 0")
        if self.lr <= 0.0:
            raise ValueError(f"lr harus > 0, dapat {self.lr}")
        if self.seed < 0:
            raise ValueError(f"seed harus >= 0, dapat {self.seed}")
        if self.model == "transformer":
            for name in ("d_model", "num_heads", "num_layers", "d_ff", "max_sequence_length"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} harus > 0, dapat {getattr(self, name)}")
            if self.d_model % self.num_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) harus habis dibagi num_heads ({self.num_heads})"
                )
            if self.block_size > self.max_sequence_length:
                raise ValueError(
                    f"block_size ({self.block_size}) tidak boleh melebihi "
                    f"max_sequence_length ({self.max_sequence_length})"
                )
        else:
            if self.d_embed <= 0 or self.d_hidden <= 0:
                raise ValueError("d_embed dan d_hidden harus > 0")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrainConfig":
        return cls(**json.loads(text))

    def config_hash(self) -> str:
        """SHA-256 JSON kanonik — identitas konfigurasi (REQ-101)."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
