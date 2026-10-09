# M009 TECHNICAL DESIGN  Inference Engine (Stage 8)

**Milestone:** M-009  Inference Engine
**Stage:** Stage 8
**Status:** DRAFT  awaiting project owner approval
**Version:** 0.1.0
**Source of truth:** PROJECT_SPEC v0.4.0, REQUIREMENTS v0.5.0 (REQ-008), DECISION-001..022, ARCHITECTURE Stage 8, M008_TECHNICAL_DESIGN.md
---

## 1. Purpose

Mendefinisikan desain teknis Inference Engine Naze (Stage 8, REQ-008) untuk melengkapi pipeline end-to-end: forward-only production path, batching efisien, benchmark memori/latensi, dan integrasi penuh dengan komponen yang sudah ada (TransformerLM, MLPLM, tokenizer, dataset). Dokumen ini adalah acuan tunggal implementasi M-009 agar inference berjalan optimal di hardware terbatas (CPU, consumer-grade).

## 2. Scope

- **Inference Engine:** forward-only path untuk MLPLM dan TransformerLM dengan batching (B > 1), konteks sliding window, dan output logits/probabilitas/token.
- **Batching:** pengelompokan input efisien (padding minimal, memori terkendali).
- **Benchmark:** metrik latensi (token/second) dan memori (peak RSS) untuk ukuran batch dan sequence length yang bervariasi.
- **Integrasi:** adapter untuk model yang sudah ada (MLPLM, TransformerLM) tanpa modifikasi core engine.
- **API Inference:** interface sederhana untuk generate teks (greedy/sampling) dengan konteks panjang.
- **Test:** unit test per komponen, integration test end-to-end, benchmark test untuk validasi performa.

## 3. Non-goals

- Training loop atau optimizer baru (sudah selesai di M-008).
- Model baru atau arsitektur neural (sudah selesai di M-006/M-007).
- Distributed inference atau multi-GPU (di luar hardware target).
- Quantization (float32/float16)  ditunda sampai terbukti diperlukan (REQ-302: no overengineering).
- Streaming inference (real-time)  ditunda sampai ada kebutuhan terukur.
- API HTTP/REST untuk inference  di luar scope milestone ini (bisa ditambahkan di M-010).

## 4. Requirements

| Req | Relevan untuk M-009 |
|---|---|
| REQ-008 Inference Engine | Inti milestone: forward-only path, batching, benchmark memori/latensi |
| REQ-006 Training | Terhubung ke pipeline training (generate deterministik sudah terimplementasi di M-006/M-007) |
| REQ-010 First LM | Integrasi dengan MLPLM dan TransformerLM yang sudah ada |
| REQ-004 Tokenizer | Input token ID dari ByteTokenizer, vocab 256 fixed (DECISION-015) |
| REQ-101 Reproducibility | Deterministik per-seed (seeded_rng) untuk generate |
| REQ-102 Resource Efficiency | Benchmark memori (peak RSS) dan latensi (token/sec) terukur |
| REQ-103 Testability | Unit test + integration test + benchmark test |
| REQ-104 Modularity | Modul inference terpisah dari train, nn, core |
| REQ-201/202/203 | Tanpa pretrained / API LLM / framework DL  permanen |
| REQ-301/302 | Spec-driven; no overengineering |

## 5. Architecture

Pipeline inference end-to-end (forward-only):

```
Input teks (string) 
  -> ByteTokenizer.encode -> token IDs (list[int])
  -> InferenceBatch (padding + mask + chunking) -> batches (B, T)
  -> Model.forward (MLPLM | TransformerLM) -> logits (B, T, 256)
  -> InferenceOutput (logits, probs, top_k tokens)
  -> (optional) generate autoregressive (greedy/sampling) -> token IDs
  -> ByteTokenizer.decode -> teks output (string)
```

**Kunci desain:**
- **Forward-only:** tidak ada backward, tidak ada gradien, tidak ada update parameter.
- **Batching:** input dengan panjang bervariasi dikumpulkan ke batch dengan padding (T = max_length di batch).
- **Sliding window:** untuk sequence panjang > max_sequence_length, gunakan jendela geser (window size = max_sequence_length, stride = max_sequence_length - overlap).
- **Efisiensi:** cache aktivasi tidak diperlukan (forward-only), memori dominan = params + input + output.

**Modul baru:**
- `naze/inference/batch.py`: InferenceBatch (padding, mask, chunking)
- `naze/inference/engine.py`: InferenceEngine (forward-only wrapper untuk MLPLM/TransformerLM)
- `naze/inference/benchmark.py`: BenchmarkResult + measure_latency/peak_rss
- `naze/inference/__init__.py`: public API

## 6. Module Boundaries

| Module | Responsibility | Baru/ubah | Dependency | Input | Output |
|---|---|---|---|---|---|
| naze/inference/batch.py | Padding, mask, chunking input untuk batch | baru | naze.core.numeric | list[int] | (B,T) ndarray + mask |
| naze/inference/engine.py | Forward-only wrapper model | baru | naze.lm, naze.nn | (B,T) | InferenceOutput |
| naze/inference/benchmark.py | Metrik latensi & memori | baru | naze.core.numeric, resource | model, input | BenchmarkResult |
| naze/inference/__init__.py | Public API (InferenceEngine, InferenceBatch, benchmark) | baru | - | - | - |
| naze/lm/mlp_lm.py | Tidak diubah (API forward tetap) | tetap | - | - | - |
| naze/lm/transformer_lm.py | Tidak diubah (API forward tetap) | tetap | - | - | - |

**Boundary rule:** inference -> lm -> nn -> core; inference tidak diimpor oleh lm/nn/train.

## 7. Interfaces

```python
# InferenceBatch: pengelompokan input untuk batch inference
class InferenceBatch:
    def __init__(self, token_ids: list[list[int]], max_length: int, *, pad_id: int = 0) -> None:
        """token_ids: list of token sequences (lengths may vary)."""
        pass
    
    @property
    def x(self) -> Array:  # (B, T) int64, padded
        pass
    
    @property  
    def mask(self) -> Array:  # (B, T) bool, True = valid token
        pass
    
    @property
    def lengths(self) -> list[int]:  # panjang asli tiap sequence
        pass


# InferenceOutput: hasil forward model
@dataclass(frozen=True)
class InferenceOutput:
    logits: Array  # (B, T, 256)
    probs: Array   # (B, T, 256), optional (bisa None untuk efisiensi)
    top_k: int = 5  # default top-k tokens untuk output


# InferenceEngine: wrapper forward-only
class InferenceEngine:
    def __init__(self, model: MLPLM | TransformerLM) -> None:
        self.model = model
    
    def forward(self, batch: InferenceBatch, *, return_probs: bool = False) -> InferenceOutput:
        """Forward pass batch -> logits. return_probs=True menghitung softmax."""
        pass
    
    def generate(self, prompt_ids: list[int], max_new: int, *,
                temperature: float = 1.0, top_k: int = 5, seed: int = 0) -> list[int]:
        """Autoregressive generate (pola transformer_generate, deterministik per-seed)."""
        pass


# Benchmark
@dataclass(frozen=True)
class BenchmarkResult:
    tokens_per_second: float
    peak_rss_kb: int | None
    input_tokens: int
    output_tokens: int
    batch_size: int
    sequence_length: int


def benchmark_model(model, batch_sizes: list[int], seq_lengths: list[int], 
                     *, warmup: int = 3, runs: int = 5) -> list[BenchmarkResult]:
    """Benchmark latensi dan memori untuk kombinasi batch/sequence."""
    pass
```

## 8. Data Flow

1. Input teks -> ByteTokenizer.encode -> list[int] token IDs.
2. Token IDs -> InferenceBatch (padding ke T = max_length, buat mask).
3. InferenceBatch.x -> InferenceEngine.forward -> model.forward -> logits (B,T,256).
4. (Opsional) softmax(logits) -> probs (jika return_probs=True).
5. (Opsional) generate: loop autoregressive dengan konteks = last max_sequence_length tokens.
6. Output token IDs -> ByteTokenizer.decode -> teks.

**Sliding window untuk sequence panjang:**
- Input: [t0, t1, ..., tN] dengan N > max_sequence_length
- Chunk: [t0..T], [tT-overlap..2T-overlap], ... dengan T = max_sequence_length
- Forward tiap chunk, gabung logits (overlap diabaikan untuk output akhir).

## 9. Configuration

```python
# InferenceConfig: konfigurasi inference (opsional, bisa default)
@dataclass(frozen=True)
class InferenceConfig:
    max_sequence_length: int = 128  # default dari DECISION-018
    pad_id: int = 0
    return_probs: bool = False
    temperature: float = 1.0
    top_k: int = 5
    seed: int = 0
    warmup_runs: int = 3
    benchmark_runs: int = 5
```

**Default:**
- max_sequence_length = 128 (konsisten dengan DECISION-018)
- pad_id = 0 (byte 0 = padding, sesuai ByteTokenizer)
- dtype = float64 (konsisten DECISION-007)

## 10. Error Handling

- Input kosong -> ValueError (fail-fast).
- Token ID di luar rentang [0, 256) -> ValueError (fail-fast).
- Sequence terlalu panjang (> max_sequence_length * 2) -> ValueError (untuk sliding window sederhana).
- Model tidak mendukung forward -> TypeError (fail-fast).
- Benchmark gagal (misal resource tidak tersedia) -> return None untuk metrik yang gagal (tidak fail-fast, catat warning).

## 11. Testing Strategy

- **Unit:** InferenceBatch padding/mask benar; InferenceEngine.forward output shape benar; BenchmarkResult field lengkap.
- **Numerical:** logits InferenceEngine == model.forward langsung (tanpa batch) untuk kasus sederhana.
- **Integration:** encode -> batch -> forward -> generate -> decode (end-to-end, deterministik per-seed).
- **Benchmark:** ukur latensi untuk batch_size=1,8,16 dan T=16,64,128; verifikasi tokens_per_second > 0; peak_rss_kb stabil.
- **Regression:** test lama (M-006..M-008) tetap hijau; tidak ada modifikasi pada modul lama.

## 12. Resource Constraints

- **Hardware target:** CPU consumer-grade (laptop/PC).
- **Metrik formal:**
  - Latensi: tokens_per_second > 0 (terukur, bukan ambang mutlak).
  - Memori: peak_rss_kb < 2 GB (konsisten DECISION-021 untuk training; inference seharusnya lebih rendah).
- **Estimasi:**
  - Transformer config D-018 (D=64, H=4, L=2, d_ff=128): ~108k params * 8B = 0.87 MB (float64).
  - Aktivasi forward-only: ~1.7 MB (estimasi M007-TD  aktivasi training ~4 MB/layer x 2).
  - Total memori inference: params + input + output + aktivasi < 10 MB untuk batch kecil.
- **Batas praktis:**
  - Batch size maksimum: terbatas oleh memori tersedia (dihitung otomatis di InferenceBatch).
  - Sequence length maksimum: max_sequence_length (128 default, bisa dikonfigurasi).

## 13. Acceptance Criteria (objective, verifiable)

1. InferenceBatch: padding benar (x[i, :lengths[i]] == input asli), mask benar (mask[i, :lengths[i]] == True).
2. InferenceEngine.forward: output shape = (B, T, 256) untuk TransformerLM; (B, 256) untuk MLPLM (kompatibel).
3. InferenceEngine.forward: logits identik dengan model.forward langsung (tanpa batch) untuk input tunggal.
4. InferenceEngine.generate: deterministik per-seed (pola transformer_generate).
5. End-to-end: encode -> batch -> forward -> generate -> decode berjalan tanpa error; output valid UTF-8.
6. Benchmark: tokens_per_second > 0 untuk semua konfigurasi; peak_rss_kb < 2 GB (DECISION-021).
7. Benchmark: latensi skalabel (tokens_per_second meningkat dengan batch_size untuk B <= 8).
8. Sliding window: sequence panjang > max_sequence_length terproses tanpa error; output konsisten dengan chunking manual.
9. Test lama (M-006..M-008) tetap hijau (361 test + test M-009 baru).
10. Coverage >= 80% baris src/naze (DECISION-019, diukur via pytest-cov).
11. Tidak ada pretrained/API LLM/framework DL (REQ-201/202/203).
12. Dokumentasi: docstring Bahasa Indonesia untuk semua fungsi/kelas publik.

## 14. Risks

1. **Sliding window untuk Transformer:** attention mask kausal harus konsisten lintas chunk (posisi global vs lokal). Mitigasi: gunakan positional offset (i - start_chunk) untuk tiap chunk; mask kausal lokal di tiap chunk.
2. **Batching dengan padding:** padding bisa mendominasi batch untuk sequence panjang bervariasi. Mitigasi: sorting input oleh panjang (descending) sebelum batching (mengurangi padding rata-rata).
3. **Memori aktivasi:** forward-only Transformer tetap butuh memori untuk aktivasi intermediate. Mitigasi: ukur peak_rss_kb untuk config D-018; optimalkan batch_size otomatis.
4. **Benchmark tidak stabil:** latensi bisa bervariasi karena sistem. Mitigasi: warmup runs + rata-rata multi-run; catat variansi.
5. **Kompleksitas sliding window:** untuk overlap > 0, perlu menggabungkan logits. Mitigasi: implementasi sederhana dengan overlap=0 (no overlap) untuk M-009; overlap bisa ditambahkan di M-010.

## 15. Open Decisions

- **OD-122 (batch strategy):** Sorting oleh panjang sebelum batching (mengurangi padding) vs urutan asli. Default: urutan asli (sederhana); sorting bisa ditambahkan di M-010.
- **OD-123 (sliding window overlap):** Overlap > 0 untuk kontinuitas konteks. Default: overlap=0 (tidak ada overlap) untuk M-009.
- **OD-124 (probs output):** Selalu hitung probs (softmax) vs optional (untuk efisiensi). Default: optional (return_probs=False).
- **OD-125 (benchmark scope):** Ukur latensi per-token vs per-batch. Default: per-batch (sederhana).
- **OD-126 (float32 inference):** Gunakan float32 untuk inference (memori 1/2 float64). Default: float64 (konsisten DECISION-007); float32 bisa ditambahkan di M-010.
