"""Benchmark: metrik latensi & memori untuk inference (M-009-T005/T006, REQ-008/REQ-102).

BenchmarkResult: hasil pengukuran tokens_per_second dan peak_rss_kb.
measure_latency: ukur waktu forward model.
benchmark_model: ukur performa untuk berbagai batch_size dan sequence_length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from naze.core.numeric import Array

if TYPE_CHECKING:
    from naze.inference.engine import InferenceEngine


@dataclass(frozen=True)
class BenchmarkResult:
    """Hasil benchmark inference (M-009-T005).

    Atribut:
        tokens_per_second: Token yang diproses per detik.
        peak_rss_kb: Peak RSS proses dalam KB (None jika tidak tersedia).
        input_tokens: Jumlah token input.
        output_tokens: Jumlah token output (logits).
        batch_size: Ukuran batch (B).
        sequence_length: Panjang sequence (T).
    """

    tokens_per_second: float
    peak_rss_kb: int | None
    input_tokens: int
    output_tokens: int
    batch_size: int
    sequence_length: int


def measure_peak_rss() -> int | None:
    """Peak RSS proses dalam KB (ru_maxrss); None bila platform tak dukung."""
    try:
        import resource
    except ImportError:
        return None
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except AttributeError:
        return None


def measure_latency(
    engine: "InferenceEngine",
    batch: "InferenceBatch",
    *,
    warmup: int = 3,
    runs: int = 5,
) -> tuple[float, list[float]]:
    """Ukur latensi forward engine untuk batch (M-009-T005).

    Args:
        engine: InferenceEngine yang akan di-benchmark.
        batch: InferenceBatch input.
        warmup: Jumlah warmup runs (default 3).
        runs: Jumlah pengukuran (default 5).

    Returns:
        (mean_latency, all_times): rata-rata latensi (detik) dan list semua waktu.
    """
    # Warmup
    for _ in range(warmup):
        engine.forward(batch, return_probs=False)

    # Measure
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        engine.forward(batch, return_probs=False)
        end = time.perf_counter()
        times.append(end - start)

    return float(np.mean(times)), times


def benchmark_model(
    engine: "InferenceEngine",
    batch_sizes: list[int],
    seq_lengths: list[int],
    *,
    warmup: int = 3,
    runs: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark model untuk kombinasi batch_size dan sequence_length (M-009-T006).

    Args:
        engine: InferenceEngine yang akan di-benchmark.
        batch_sizes: Daftar ukuran batch yang akan diuji.
        seq_lengths: Daftar panjang sequence yang akan diuji.
        warmup: Jumlah warmup runs per kombinasi.
        runs: Jumlah pengukuran per kombinasi.

    Returns:
        List BenchmarkResult untuk semua kombinasi.
    """
    results = []
    peak_rss = measure_peak_rss()  # Ukur sekali (global)

    for bs in batch_sizes:
        for sl in seq_lengths:
            # Buat dummy batch
            from naze.inference.batch import InferenceBatch

            token_ids = [[0] * sl for _ in range(bs)]
            batch = InferenceBatch(token_ids, max_length=sl)

            # Ukur latensi
            mean_latency, _ = measure_latency(engine, batch, warmup=warmup, runs=runs)

            # Hitung tokens
            input_tokens = bs * sl
            output_tokens = bs * sl * 256  # logits (B, T, 256)

            # tokens_per_second
            tokens_per_second = input_tokens / mean_latency if mean_latency > 0 else 0.0

            results.append(
                BenchmarkResult(
                    tokens_per_second=tokens_per_second,
                    peak_rss_kb=peak_rss,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    batch_size=bs,
                    sequence_length=sl,
                )
            )

    return results
