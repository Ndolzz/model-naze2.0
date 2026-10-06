"""Minimal training system (REQ-006, REQ-007, Stage 7 minimal).

Resolusi OPEN DECISION-106: SGD saja untuk saat ini (Adam/AdamW ditunda
sampai ada kebutuhan nyata — no overengineering).
Checkpoint: npz berisi seluruh params + step; save/load deterministik (DECISION-111).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from naze.lm.mlp_lm import MLPLM


class SGDTrainer:
    def __init__(self, model: MLPLM, lr: float = 0.1) -> None:
        if lr <= 0:
            raise ValueError("lr harus > 0")
        self.model = model
        self.lr = lr
        self.step_count = 0

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        logits = self.model.forward(x)
        loss = self.model.loss(logits, y)
        grads = self.model.backward()
        for name, p in self.model.params().items():
            p -= self.lr * grads[name]
        self.step_count += 1
        return loss


def save_checkpoint(path: str | Path, model: MLPLM, step: int) -> None:
    np.savez(str(path), **model.params(), __step=np.int64(step))


def load_checkpoint(path: str | Path, model: MLPLM) -> int:
    data = np.load(str(path))
    params = model.params()
    for name in params:
        if name not in data:
            raise KeyError(f"checkpoint tidak memuat param {name!r}")
        if data[name].shape != params[name].shape:
            raise ValueError(f"shape {name!r} tidak cocok: {data[name].shape} vs {params[name].shape}")
    for name, p in params.items():
        p[...] = data[name]
    return int(data["__step"])
