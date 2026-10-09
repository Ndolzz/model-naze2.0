"""Inference Engine (M-009, Stage 8, REQ-008).

Forward-only production path untuk MLPLM dan TransformerLM dengan batching,
benchmark memori/latensi, dan integrasi penuh dengan pipeline yang sudah ada.
"""

from naze.inference.batch import InferenceBatch
from naze.inference.benchmark import BenchmarkResult, benchmark_model
from naze.inference.engine import InferenceConfig, InferenceEngine, InferenceOutput

__all__ = [
    "InferenceBatch",
    "InferenceOutput",
    "InferenceEngine",
    "InferenceConfig",
    "BenchmarkResult",
    "benchmark_model",
]
