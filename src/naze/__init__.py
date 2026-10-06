"""Naze 2.0 — AI model built from scratch.

Core components (neural engine, autodiff, tokenizer, training, inference)
are implemented per milestone, spec-driven. See docs/spec/.
"""

__version__ = "0.0.1"

from naze import core, nn  # noqa: E402,F401

__all__ = ["core", "nn", "__version__"]
