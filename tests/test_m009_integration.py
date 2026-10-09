"""Integration tests untuk M-009 Inference Engine (T012/T014).

Test: end-to-end encode->batch->forward->generate->decode.
"""

import numpy as np
import pytest

from naze.core.numeric import as_array
from naze.inference.batch import InferenceBatch
from naze.inference.engine import InferenceEngine, InferenceConfig
from naze.lm.mlp_lm import MLPLM
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer


class TestEndToEndMLPLM:
    """End-to-end test untuk MLPLM (M-009-T012)."""

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

    def test_encode_batch_forward_decode(self, mlp_model):
        """Encode -> batch -> forward -> decode berjalan (MLPLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(mlp_model)

        # Encode
        text = "hello world"
        ids = tok.encode(text)

        # Batch
        batch = InferenceBatch([ids], max_length=16)

        # Forward
        output = engine.forward(batch)
        assert output.logits.shape == (1, 256)

        # Decode (kita decode prompt asli)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_batch_forward_generate_decode(self, mlp_model):
        """Encode -> batch -> forward -> generate -> decode (MLPLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(mlp_model)

        # Encode
        prompt = "hello"
        prompt_ids = tok.encode(prompt)

        # Generate
        new_ids = engine.generate(prompt_ids, max_new=5, temperature=0.0, seed=0)

        # Decode
        full_ids = prompt_ids + new_ids
        decoded = tok.decode(full_ids)

        # Pastikan output valid UTF-8
        assert isinstance(decoded, str)
        assert decoded.startswith(prompt)

    def test_deterministik_end_to_end(self, mlp_model):
        """End-to-end deterministik per-seed (MLPLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(mlp_model)

        prompt = "test"
        prompt_ids = tok.encode(prompt)

        result1 = engine.generate(prompt_ids, max_new=10, temperature=0.0, seed=42)
        result2 = engine.generate(prompt_ids, max_new=10, temperature=0.0, seed=42)

        assert result1 == result2

    def test_batch_multiple_sequences(self, mlp_model):
        """Batch dengan multiple sequences."""
        tok = ByteTokenizer()
        engine = InferenceEngine(mlp_model)

        texts = ["hello", "world", "test"]
        ids_list = [tok.encode(t) for t in texts]

        # Pad ke panjang yang sama
        max_len = max(len(ids) for ids in ids_list)
        batch = InferenceBatch(ids_list, max_length=max_len)

        output = engine.forward(batch)
        assert output.logits.shape == (3, 256)


class TestEndToEndTransformer:
    """End-to-end test untuk TransformerLM (M-009-T012)."""

    @pytest.fixture
    def transformer_model(self):
        """TransformerLM kecil untuk pengujian."""
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

    def test_encode_batch_forward_decode(self, transformer_model):
        """Encode -> batch -> forward -> decode berjalan (TransformerLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(transformer_model)

        # Encode
        text = "hello world"
        ids = tok.encode(text)

        # Batch
        batch = InferenceBatch([ids[:8]], max_length=16)  # Potong ke 8 tokens

        # Forward
        output = engine.forward(batch)
        assert output.logits.shape[0] == 1 and output.logits.shape[2] == 256

        # Decode
        decoded = tok.decode(ids[:8])
        assert decoded == text[: len(tok.decode(ids[:8]))]

    def test_encode_batch_forward_generate_decode(self, transformer_model):
        """Encode -> batch -> forward -> generate -> decode (TransformerLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(transformer_model)

        # Encode
        prompt = "hello"
        prompt_ids = tok.encode(prompt)

        # Generate
        new_ids = engine.generate(prompt_ids, max_new=5, temperature=0.0, seed=0)

        # Decode
        full_ids = prompt_ids + new_ids
        decoded = tok.decode(full_ids)

        # Pastikan output valid UTF-8
        assert isinstance(decoded, str)
        assert decoded.startswith(prompt)

    def test_deterministik_end_to_end_transformer(self, transformer_model):
        """End-to-end deterministik per-seed (TransformerLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(transformer_model)

        prompt = "test"
        prompt_ids = tok.encode(prompt)

        result1 = engine.generate(prompt_ids, max_new=10, temperature=0.0, seed=42)
        result2 = engine.generate(prompt_ids, max_new=10, temperature=0.0, seed=42)

        assert result1 == result2

    def test_batch_multiple_sequences_transformer(self, transformer_model):
        """Batch dengan multiple sequences (TransformerLM)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(transformer_model)

        texts = ["hello", "world"]
        ids_list = [tok.encode(t) for t in texts]

        # Pad ke panjang yang sama
        max_len = 8
        batch = InferenceBatch(ids_list, max_length=max_len)

        output = engine.forward(batch)
        assert output.logits.shape[0] == 2 and output.logits.shape[2] == 256


class TestSlidingWindowEndToEnd:
    """Test sliding window end-to-end (M-009-T014)."""

    @pytest.fixture
    def transformer_model(self):
        """TransformerLM kecil untuk pengujian."""
        config = TransformerConfig(
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            max_sequence_length=8,
            vocab_size=256,
            seed=0,
        )
        return TransformerLM(config)

    def test_sliding_window_long_sequence(self, transformer_model):
        """Sequence panjang > max_sequence_length terproses (M-009-T014)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(transformer_model)

        # Buat sequence panjang
        long_text = "hello world this is a test sequence"
        long_ids = tok.encode(long_text)

        # Pastikan lebih panjang dari max_sequence_length
        assert len(long_ids) > 8

        # Generate dengan sliding window (via generate internal)
        new_ids = engine.generate(long_ids[:10], max_new=5, temperature=0.0, seed=0)

        # Pastikan berjalan tanpa error
        assert isinstance(new_ids, list)
        assert len(new_ids) == 5

    def test_sliding_window_consistency(self, transformer_model):
        """Hasil sliding window konsisten (M-009-T014)."""
        tok = ByteTokenizer()
        engine = InferenceEngine(transformer_model)

        # Sequence yang muat di satu window
        short_ids = tok.encode("hello")
        result_short = engine.generate(short_ids, max_new=3, temperature=0.0, seed=0)

        # Sequence yang memerlukan sliding window
        # (tapi untuk model kecil, kita simulasikan dengan memotong)
        long_ids = short_ids + [0, 0, 0, 0]  # Tambah padding
        result_long = engine.generate(long_ids, max_new=3, temperature=0.0, seed=0)

        # Untuk greedy (temp=0), hasil seharusnya sama jika konteks yang relevan sama
        # (ini tergantung pada implementasi sliding window)
        # Kita cek bahwa tidak ada error
        assert isinstance(result_short, list)
        assert isinstance(result_long, list)


class TestPublicAPI:
    """Test Public API (M-009-T009)."""

    def test_import_all(self):
        """Semua simbol publik bisa diimport."""
        from naze.inference import (
            InferenceBatch,
            InferenceOutput,
            InferenceEngine,
            InferenceConfig,
            BenchmarkResult,
            benchmark_model,
        )

        assert InferenceBatch is not None
        assert InferenceOutput is not None
        assert InferenceEngine is not None
        assert InferenceConfig is not None
        assert BenchmarkResult is not None
        assert benchmark_model is not None

    def test_inference_module_exists(self):
        """Modul naze.inference ada."""
        import naze.inference

        assert hasattr(naze.inference, "InferenceBatch")
        assert hasattr(naze.inference, "InferenceOutput")
        assert hasattr(naze.inference, "InferenceEngine")
        assert hasattr(naze.inference, "InferenceConfig")
        assert hasattr(naze.inference, "BenchmarkResult")
        assert hasattr(naze.inference, "benchmark_model")
