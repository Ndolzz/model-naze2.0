"""Naze 1.0 — AI model built from scratch (spec-driven, no pretrained core).

Milestones: M-001..M-009 selesai (engine, autodiff, tokenizer, dataset,
MLP + Transformer LM, training system penuh, inference engine); M-010
(Naze 1.0 final) berjalan. Spesifikasi: docs/spec/ (PROJECT_SPEC).
"""

__version__ = "1.0.0"

from naze import core, data, export, inference, lm, nn, token, train  # noqa: F401

__all__ = [
    "core", "data", "export", "inference", "lm", "nn", "token", "train", "__version__",
]
