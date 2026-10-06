"""Naze 2.0 — AI model built from scratch.

Spec-driven; see docs/spec/. Milestones: M-001 foundation, M-002 engine,
M-003 autodiff, M-004 tokenizer, M-005 dataset, M-006 first LM + training minimal.
"""

__version__ = "0.0.2"

from naze import core, data, lm, nn, token, train  # noqa: F401

__all__ = ["core", "data", "lm", "nn", "token", "train", "__version__"]
