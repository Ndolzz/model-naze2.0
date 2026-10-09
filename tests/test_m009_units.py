"""Unit tests untuk M-009 Inference Engine (T010).

Test: InferenceBatch, InferenceOutput, InferenceConfig, BenchmarkResult.
"""

import numpy as np
import pytest

from naze.core.numeric import as_array
from naze.inference.batch import InferenceBatch
from naze.inference.benchmark import BenchmarkResult, measure_peak_rss
from naze.inference.engine import InferenceConfig, InferenceOutput


class TestInferenceBatch:
    """Test InferenceBatch (M-009-T001)."""

    def test_padding_correct(self):
        """Padding benar: x[i, :lengths[i]] == input asli."""
        token_ids = [[1, 2, 3], [4, 5], [6]]
        batch = InferenceBatch(token_ids, max_length=5, pad_id=0)

        assert batch.x.shape == (3, 5)
        assert batch.lengths == [3, 2, 1]

        # Check padding
        assert list(batch.x[0, :3]) == [1, 2, 3]
        assert list(batch.x[1, :2]) == [4, 5]
        assert list(batch.x[2, :1]) == [6]
        # Padding area
        assert list(batch.x[0, 3:]) == [0, 0]
        assert list(batch.x[1, 2:]) == [0, 0, 0]
        assert list(batch.x[2, 1:]) == [0, 0, 0, 0]

    def test_mask_correct(self):
        """Mask benar: mask[i, :lengths[i]] == True."""
        token_ids = [[1, 2], [3]]
        batch = InferenceBatch(token_ids, max_length=4, pad_id=0)

        assert batch.mask.shape == (2, 4)
        # Baris 0: [True, True, False, False]
        assert list(batch.mask[0]) == [True, True, False, False]
        # Baris 1: [True, False, False, False]
        assert list(batch.mask[1]) == [True, False, False, False]

    def test_deterministik(self):
        """InferenceBatch deterministik."""
        token_ids = [[1, 2, 3], [4, 5]]
        batch1 = InferenceBatch(token_ids, max_length=5)
        batch2 = InferenceBatch(token_ids, max_length=5)

        np.testing.assert_array_equal(batch1.x, batch2.x)
        np.testing.assert_array_equal(batch1.mask, batch2.mask)
        assert batch1.lengths == batch2.lengths

    def test_empty_input_raises(self):
        """Input kosong -> ValueError."""
        with pytest.raises(ValueError, match="token_ids tidak boleh kosong"):
            InferenceBatch([], max_length=10)

    def test_too_long_sequence_raises(self):
        """Sequence terlalu panjang -> ValueError."""
        with pytest.raises(ValueError, match="sequence terlalu panjang"):
            InferenceBatch([[1, 2, 3, 4, 5, 6]], max_length=5)

    def test_invalid_token_id_raises(self):
        """Token ID invalid -> ValueError."""
        with pytest.raises(ValueError, match="token ID harus integer"):
            InferenceBatch([[1, 2, 256]], max_length=5)  # 256 invalid (vocab 256: 0-255)

    def test_negative_token_id_raises(self):
        """Token ID negatif -> ValueError."""
        with pytest.raises(ValueError, match="token ID harus integer"):
            InferenceBatch([[1, -1, 3]], max_length=5)

    def test_custom_pad_id(self):
        """Pad ID custom berfungsi."""
        token_ids = [[1], [2, 3]]
        batch = InferenceBatch(token_ids, max_length=3, pad_id=255)

        np.testing.assert_array_equal(batch.x[0, 1:], [255, 255])  # Padding dengan 255
        assert batch.x[1, 2] == 255

    def test_repr(self):
        """Repr mengandung info yang berguna."""
        batch = InferenceBatch([[1, 2]], max_length=5)
        repr_str = repr(batch)
        assert "InferenceBatch" in repr_str
        assert "batch_size=1" in repr_str
        assert "max_length=5" in repr_str


class TestInferenceOutput:
    """Test InferenceOutput (M-009-T002)."""

    def test_frozen_dataclass(self):
        """InferenceOutput immutable."""
        output = InferenceOutput(
            logits=np.zeros((2, 3, 256)),
            probs=np.zeros((2, 3, 256)),
            top_k=5,
        )
        with pytest.raises(AttributeError):
            output.logits = np.ones((2, 3, 256))

    def test_probs_optional(self):
        """probs boleh None."""
        output = InferenceOutput(
            logits=np.zeros((2, 3, 256)),
            probs=None,
            top_k=5,
        )
        assert output.probs is None

    def test_default_top_k(self):
        """top_k default = 5."""
        output = InferenceOutput(logits=np.zeros((1, 1, 256)))
        assert output.top_k == 5


class TestInferenceConfig:
    """Test InferenceConfig (M-009-T008)."""

    def test_default_values(self):
        """Default values benar."""
        config = InferenceConfig()
        assert config.max_sequence_length == 128
        assert config.pad_id == 0
        assert config.return_probs == False
        assert config.temperature == 1.0
        assert config.top_k == 5
        assert config.seed == 0
        assert config.warmup_runs == 3
        assert config.benchmark_runs == 5

    def test_validasi_max_sequence_length(self):
        """max_sequence_length <= 0 -> ValueError."""
        with pytest.raises(ValueError, match="max_sequence_length harus > 0"):
            InferenceConfig(max_sequence_length=0)

    def test_validasi_pad_id(self):
        """pad_id di luar [0, 256) -> ValueError."""
        with pytest.raises(ValueError, match="pad_id harus di \\[0, 256\\)"):
            InferenceConfig(pad_id=256)
        with pytest.raises(ValueError, match="pad_id harus di \\[0, 256\\)"):
            InferenceConfig(pad_id=-1)

    def test_validasi_temperature(self):
        """temperature < 0 -> ValueError."""
        with pytest.raises(ValueError, match="temperature harus >= 0"):
            InferenceConfig(temperature=-0.1)

    def test_validasi_top_k(self):
        """top_k <= 0 -> ValueError."""
        with pytest.raises(ValueError, match="top_k harus > 0"):
            InferenceConfig(top_k=0)

    def test_validasi_seed(self):
        """seed < 0 -> ValueError."""
        with pytest.raises(ValueError, match="seed harus >= 0"):
            InferenceConfig(seed=-1)

    def test_validasi_warmup_runs(self):
        """warmup_runs < 0 -> ValueError."""
        with pytest.raises(ValueError, match="warmup_runs harus >= 0"):
            InferenceConfig(warmup_runs=-1)

    def test_validasi_benchmark_runs(self):
        """benchmark_runs < 1 -> ValueError."""
        with pytest.raises(ValueError, match="benchmark_runs harus >= 1"):
            InferenceConfig(benchmark_runs=0)

    def test_frozen_dataclass(self):
        """InferenceConfig immutable."""
        config = InferenceConfig()
        with pytest.raises(AttributeError):
            config.max_sequence_length = 64


class TestBenchmarkResult:
    """Test BenchmarkResult (M-009-T005)."""

    def test_frozen_dataclass(self):
        """BenchmarkResult immutable."""
        result = BenchmarkResult(
            tokens_per_second=100.0,
            peak_rss_kb=1024,
            input_tokens=100,
            output_tokens=25600,
            batch_size=8,
            sequence_length=32,
        )
        with pytest.raises(AttributeError):
            result.tokens_per_second = 200.0

    def test_all_fields(self):
        """Semua field tersedia."""
        result = BenchmarkResult(
            tokens_per_second=100.0,
            peak_rss_kb=1024,
            input_tokens=100,
            output_tokens=25600,
            batch_size=8,
            sequence_length=32,
        )
        assert result.tokens_per_second == 100.0
        assert result.peak_rss_kb == 1024
        assert result.input_tokens == 100
        assert result.output_tokens == 25600
        assert result.batch_size == 8
        assert result.sequence_length == 32


class TestMeasurePeakRss:
    """Test measure_peak_rss (M-009-T005)."""

    def test_returns_int_or_none(self):
        """measure_peak_rss mengembalikan int atau None."""
        result = measure_peak_rss()
        assert result is None or isinstance(result, int)

    def test_non_negative(self):
        """Jika int, nilainya >= 0."""
        result = measure_peak_rss()
        if result is not None:
            assert result >= 0
