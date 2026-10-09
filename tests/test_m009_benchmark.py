"""Benchmark tests untuk M-009 Inference Engine (T013).

Test: tokens_per_second > 0, peak_rss_kb < 2 GB.
"""

import pytest

from naze.inference.benchmark import benchmark_model, BenchmarkResult
from naze.inference.engine import InferenceEngine
from naze.lm.mlp_lm import MLPLM
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer


class TestBenchmarkMLPLM:
    """Benchmark tests untuk MLPLM (M-009-T013)."""

    @pytest.fixture
    def mlp_model(self):
        """MLPLM kecil untuk pengujian."""
        tok = ByteTokenizer()
        return MLPLM(
            vocab_size=tok.vocab_size,
            block_size=16,
            d_embed=8,
            d_hidden=16,
            seed=0,
        )

    def test_benchmark_single(self, mlp_model):
        """Benchmark model untuk batch_size=1, T=16."""
        engine = InferenceEngine(mlp_model)

        results = benchmark_model(
            engine,
            batch_sizes=[1],
            seq_lengths=[16],
            warmup=2,
            runs=3,
        )

        assert len(results) == 1
        result = results[0]

        assert isinstance(result, BenchmarkResult)
        assert result.batch_size == 1
        assert result.sequence_length == 16
        assert result.tokens_per_second > 0

    def test_benchmark_multiple_batch_sizes(self, mlp_model):
        """Benchmark untuk berbagai batch_size."""
        engine = InferenceEngine(mlp_model)

        results = benchmark_model(
            engine,
            batch_sizes=[1, 2, 4],
            seq_lengths=[8],
            warmup=1,
            runs=2,
        )

        assert len(results) == 3  # 3 batch sizes

        for result in results:
            assert result.tokens_per_second > 0
            assert result.batch_size in [1, 2, 4]
            assert result.sequence_length == 8

    def test_benchmark_multiple_seq_lengths(self, mlp_model):
        """Benchmark untuk berbagai sequence_length."""
        engine = InferenceEngine(mlp_model)

        results = benchmark_model(
            engine,
            batch_sizes=[1],
            seq_lengths=[4, 8, 16],
            warmup=1,
            runs=2,
        )

        assert len(results) == 3  # 3 sequence lengths

        for result in results:
            assert result.tokens_per_second > 0
            assert result.batch_size == 1
            assert result.sequence_length in [4, 8, 16]

    def test_benchmark_peak_rss_under_2gb(self, mlp_model):
        """Peak RSS < 2 GB (DECISION-021)."""
        engine = InferenceEngine(mlp_model)

        results = benchmark_model(
            engine,
            batch_sizes=[1, 8],
            seq_lengths=[16, 32],
            warmup=1,
            runs=2,
        )

        for result in results:
            if result.peak_rss_kb is not None:
                # 2 GB = 2 * 1024 * 1024 KB
                assert result.peak_rss_kb < 2 * 1024 * 1024


class TestBenchmarkTransformer:
    """Benchmark tests untuk TransformerLM (M-009-T013)."""

    @pytest.fixture
    def transformer_model(self):
        """TransformerLM kecil untuk pengujian (config D-018 scaled down)."""
        config = TransformerConfig(
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            max_sequence_length=16,
            vocab_size=256,
            seed=0,
        )
        return TransformerLM(config)

    def test_benchmark_single(self, transformer_model):
        """Benchmark TransformerLM untuk batch_size=1, T=16."""
        engine = InferenceEngine(transformer_model)

        results = benchmark_model(
            engine,
            batch_sizes=[1],
            seq_lengths=[16],
            warmup=2,
            runs=3,
        )

        assert len(results) == 1
        result = results[0]

        assert isinstance(result, BenchmarkResult)
        assert result.batch_size == 1
        assert result.sequence_length == 16
        assert result.tokens_per_second > 0

    def test_benchmark_config_d018(self):
        """Benchmark untuk config D-018 (D=64, H=4, L=2, d_ff=128, T_max=128)."""
        config = TransformerConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            max_sequence_length=128,
            vocab_size=256,
            seed=0,
        )
        model = TransformerLM(config)
        engine = InferenceEngine(model)

        results = benchmark_model(
            engine,
            batch_sizes=[1, 4],
            seq_lengths=[16, 64],
            warmup=1,
            runs=2,
        )

        assert len(results) == 4  # 2 batch * 2 seq

        for result in results:
            assert result.tokens_per_second > 0
            # Check peak_rss_kb < 2 GB
            if result.peak_rss_kb is not None:
                assert result.peak_rss_kb < 2 * 1024 * 1024

    def test_benchmark_tokens_per_second_scalable(self):
        """tokens_per_second meningkat dengan batch_size."""
        config = TransformerConfig(
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            max_sequence_length=16,
            vocab_size=256,
            seed=0,
        )
        model = TransformerLM(config)
        engine = InferenceEngine(model)

        results = benchmark_model(
            engine,
            batch_sizes=[1, 2, 4],
            seq_lengths=[16],
            warmup=1,
            runs=2,
        )

        # Urutkan oleh batch_size
        results_sorted = sorted(results, key=lambda r: r.batch_size)

        # tokens_per_second seharusnya meningkat (atau setidaknya tidak menurun drastis)
        # untuk batch_size yang lebih besar
        for i in range(len(results_sorted) - 1):
            tps_current = results_sorted[i].tokens_per_second
            tps_next = results_sorted[i + 1].tokens_per_second
            # Tolak penurunan yang terlalu drastis (> 50%)
            assert tps_next >= tps_current * 0.5


class TestBenchmarkConsistency:
    """Test konsistensi benchmark."""

    def test_benchmark_deterministik(self):
        """Benchmark dengan input yang sama memberi hasil yang konsisten."""
        tok = ByteTokenizer()
        model = MLPLM(
            vocab_size=tok.vocab_size,
            block_size=16,
            d_embed=8,
            d_hidden=16,
            seed=0,
        )
        engine = InferenceEngine(model)

        # Jalankan benchmark dua kali
        results1 = benchmark_model(
            engine,
            batch_sizes=[1],
            seq_lengths=[8],
            warmup=1,
            runs=2,
        )
        results2 = benchmark_model(
            engine,
            batch_sizes=[1],
            seq_lengths=[8],
            warmup=1,
            runs=2,
        )

        # tokens_per_second seharusnya dalam rentang yang wajar
        # (tidak identik karena timing, tapi tidak terlalu jauh)
        assert len(results1) == len(results2) == 1
        tps1 = results1[0].tokens_per_second
        tps2 = results2[0].tokens_per_second
        # Tolak perbedaan > 10x (terlalu tidak konsisten)
        assert tps1 > 0 and tps2 > 0
        assert max(tps1, tps2) / min(tps1, tps2) < 10.0
