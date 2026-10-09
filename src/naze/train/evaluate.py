"""Evaluasi berkala (M008-T003, REQ-103): loss rata-rata + perplexity."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_EXP = 709.0  # batas aman math.exp float64; di atasnya -> inf


@dataclass(frozen=True)
class EvalResult:
    mean_loss: float
    perplexity: float
    n_batches: int
    n_tokens: int


def evaluate(model, data, *, per_pos: bool = False, seed: int | None = None) -> EvalResult:
    """Rata-rata loss (mean of batch means) + perplexity = exp(loss).

    per_pos=True memakai data.batches_pos + model.loss_pos (transformer);
    per_pos=False memakai data.batches + model.loss (mlplm).
    """
    batches = data.batches_pos(seed=seed) if per_pos else data.batches(seed=seed)
    total = 0.0
    n_batches = 0
    n_tokens = 0
    for x, y in batches:
        logits = model.forward(x)
        loss = model.loss_pos(logits, y) if per_pos else model.loss(logits, y)
        total += float(loss)
        n_batches += 1
        n_tokens += int(x.shape[0]) * (int(x.shape[1]) if per_pos else 1)
    if n_batches == 0:
        raise ValueError("dataset evaluasi kosong (tidak ada batch)")
    mean_loss = total / n_batches
    perplexity = math.exp(mean_loss) if mean_loss <= _MAX_EXP else math.inf
    return EvalResult(mean_loss=mean_loss, perplexity=perplexity,
                      n_batches=n_batches, n_tokens=n_tokens)
