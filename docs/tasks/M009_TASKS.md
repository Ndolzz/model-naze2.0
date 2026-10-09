# M009_TASKS  Inference Engine (Stage 8)

**Status:** PLANNED (menunggu persetujuan owner untuk M009_TECHNICAL_DESIGN.md)
**Referensi:** docs/architecture/M009_TECHNICAL_DESIGN.md
**Requirements:** REQ-008 (Inference Engine), REQ-101 (Reproducibility), REQ-102 (Resource Efficiency), REQ-103 (Testability)

**Aturan:**
- Test lama TIDAK dimodifikasi (REQ-302: no overengineering).
- Perubahan bersifat aditif (modul inference baru, tidak mengubah lm/nn/core).
- Kode <= 100 kolom; docstring Bahasa Indonesia.
- Setiap task mereferensikan Req ID + AC dari TD.

---

## T001  InferenceBatch (src/naze/inference/batch.py)
Implementasi InferenceBatch: padding input token IDs ke (B,T), mask validasi, lengths asli.
**Input:** list[list[int]] (token sequences, panjang bervariasi).
**Output:** x (B,T) int64, mask (B,T) bool, lengths list[int].
**AC:** [ ] padding benar (x[i, :lengths[i]] == input asli); [ ] mask benar (mask[i, :lengths[i]] == True); [ ] deterministik.
**Req:** REQ-008, REQ-101

## T002  InferenceOutput (src/naze/inference/engine.py)
Frozen dataclass InferenceOutput: logits (B,T,256), probs (B,T,256) optional, top_k int.
**AC:** [ ] field immutable; [ ] probs None diperbolehkan.
**Req:** REQ-008

## T003  InferenceEngine.forward (src/naze/inference/engine.py)
InferenceEngine.forward(batch: InferenceBatch, *, return_probs=False) -> InferenceOutput.
**AC:** [ ] output shape = (B, T, 256) untuk TransformerLM; [ ] logits identik dengan model.forward langsung (tanpa batch) untuk input tunggal; [ ] probs = softmax(logits) jika return_probs=True.
**Req:** REQ-008, REQ-101

## T004  InferenceEngine.generate (src/naze/inference/engine.py)
InferenceEngine.generate(prompt_ids: list[int], max_new: int, *, temperature=1.0, top_k=5, seed=0) -> list[int].
**Pola:** transformer_generate (M-007) dengan sliding window konteks (max_sequence_length).
**AC:** [ ] deterministik per-seed; [ ] output valid (token IDs di [0, 256)); [ ] panjang output = max_new.
**Req:** REQ-008, REQ-101

## T005  BenchmarkResult + measure (src/naze/inference/benchmark.py)
BenchmarkResult frozen dataclass: tokens_per_second, peak_rss_kb, input_tokens, output_tokens, batch_size, sequence_length.
**Fungsi:** measure_latency(model, input, *, warmup=3, runs=5) -> float (detik per forward).
measure_peak_rss() -> int | None (KB, dari resource.getrusage).
**AC:** [ ] tokens_per_second > 0; [ ] peak_rss_kb stabil (tidak NaN).
**Req:** REQ-008, REQ-102

## T006  benchmark_model (src/naze/inference/benchmark.py)
benchmark_model(model, batch_sizes: list[int], seq_lengths: list[int], *, warmup=3, runs=5) -> list[BenchmarkResult].
**AC:** [ ] hasil untuk semua kombinasi batch/seq; [ ] tokens_per_second > 0 untuk semua.
**Req:** REQ-008, REQ-102

## T007  Sliding Window (src/naze/inference/engine.py, ADITIF)
InferenceEngine._slide_window(token_ids: list[int], max_seq_len: int) -> Iterator[tuple[Array, int, int]]
  # Yield (chunk (1,T), start_pos, end_pos) untuk sequence panjang.
**AC:** [ ] chunk panjang <= max_seq_len; [ ] coverage penuh input asli; [ ] overlap=0 (default).
**Req:** REQ-008, REQ-102

## T008  InferenceConfig (src/naze/inference/engine.py)
Frozen dataclass InferenceConfig: max_sequence_length, pad_id, return_probs, temperature, top_k, seed, warmup_runs, benchmark_runs.
**AC:** [ ] default konsisten (max_sequence_length=128, pad_id=0, seed=0); [ ] validasi fail-fast (max_sequence_length > 0).
**Req:** REQ-008

## T009  Public API (src/naze/inference/__init__.py)
Eksport: InferenceEngine, InferenceBatch, InferenceOutput, InferenceConfig, BenchmarkResult, benchmark_model.
**AC:** [ ] import naze.inference.* berjalan; [ ] semua simbol publik tersedia.
**Req:** REQ-008, REQ-104

## T010  Unit Tests (tests/test_m009_units.py)
Unit test: InferenceBatch padding/mask; InferenceEngine.forward shape; InferenceEngine.generate determinisme; BenchmarkResult field; InferenceConfig validasi.
**AC:** [ ] semua test lulus; [ ] test lama tidak dimodifikasi.
**Req:** REQ-103

## T011  Numerical Tests (tests/test_m009_numeric.py)
Numerical: logits InferenceEngine.forward == model.forward langsung (tanpa batch) untuk input tunggal; probs = softmax(logits).
**AC:** [ ] array_equal untuk logits; [ ] allclose untuk probs (tol 1e-10).
**Req:** REQ-008, REQ-103

## T012  Integration Tests (tests/test_m009_integration.py)
Integration: encode -> InferenceBatch -> forward -> generate -> decode (end-to-end).
**AC:** [ ] end-to-end berjalan untuk MLPLM dan TransformerLM; [ ] output valid UTF-8; [ ] deterministik per-seed.
**Req:** REQ-008, REQ-010, REQ-101, REQ-103

## T013  Benchmark Tests (tests/test_m009_benchmark.py)
Benchmark: ukur latensi untuk batch_size=1,8,16 dan T=16,64,128; verifikasi tokens_per_second > 0; peak_rss_kb < 2 GB.
**AC:** [ ] benchmark_model berjalan untuk config D-018; [ ] tokens_per_second > 0; [ ] peak_rss_kb < 2 GB (DECISION-021).
**Req:** REQ-008, REQ-102, REQ-103

## T014  Sliding Window Tests (tests/test_m009_integration.py, ADITIF)
Sliding window: sequence panjang > max_sequence_length terproses tanpa error; output konsisten.
**AC:** [ ] generate dengan prompt > max_sequence_length berjalan; [ ] output konsisten dengan chunking manual.
**Req:** REQ-008, REQ-103

## T015  CI Coverage (D-019)
.github/workflows/tests.yml: pytest --cov=naze --cov-fail-under=80 (PYTHONPATH=src).
**AC:** [ ] coverage >= 80% baris src/naze (termasuk modul inference baru).
**Req:** REQ-103

## T016  Dokumentasi Pasca-Implementasi
Update: ROADMAP.md (M-009 DONE), TRACEABILITY.md (kolom M-009), SPEC_REVIEW.md (status M-009), REQUIREMENTS.md (REQ-008 DONE).
**AC:** [ ] dokumen konsisten dengan implementasi; [ ] commit CI hijau.
**Req:** REQ-105
