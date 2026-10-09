# M010 TECHNICAL DESIGN  Naze 1.0 (Stage 9)

**Milestone:** M-010  Naze 1.0
**Stage:** Stage 9
**Status:** DRAFT  awaiting project owner approval (blocked by OPEN DECISION-107)
**Version:** 0.1.0
**Source of truth:** PROJECT_SPEC v0.4.0, REQUIREMENTS v0.5.0, DECISION-001..022, ARCHITECTURE Stage 9
**Dependencies:** M-001..M-009 DONE (semua milestone sebelumnya selesai)
---

## 1. Purpose

Mendefinisikan desain teknis untuk rilis **Naze 1.0** (Stage 9) sebagai versi produksi pertama dari proyek model AI dari nol. Dokumen ini menstandardisasi seluruh pipeline end-to-end: data, training, inference, evaluasi, packaging, dan dokumentasi untuk rilis publik. Tujuannya adalah menghasilkan paket Python yang siap dipakai (`pip install naze`) dengan model terlatih, API stabil, dan metrik kualitas terukur.

## 2. Scope

- **Paket Produksi:** `naze` v1.0.0 di PyPI dengan struktur stabil dan API publik terdokumentasi.
- **Model Terlatih:** Bobot Transformer (config D-018: D=64, H=4, L=2, d_ff=128, T_max=128) terlatih pada korpus final (OD-112).
- **Pipeline Lengkap:** Tokenize -> Train -> Evaluate -> Infer -> Generate, seluruhnya deterministik dan teruji.
- **Dokumentasi:** API reference, usage examples, tutorial, changelog.
- **CI/CD Produksi:** GitHub Actions untuk test, coverage, lint, dan build wheel.
- **Evaluasi Resmi:** Metrik sukses sesuai OD-107 (menunggu keputusan owner).

## 3. Non-goals

- Fitur eksperimental yang belum teruji (misal: multi-modal, physical AI).
- Optimasi performa lanjutan (quantization, pruning)  ditunda ke M-011+.
- Distributed training atau inference  di luar hardware target (CPU consumer-grade).
- Framework deep learning eksternal (PyTorch, TensorFlow, JAX)  REQ-203 permanen.
- Pretrained model atau API LLM eksternal sebagai core  REQ-201/202 permanen.

## 4. Requirements

| Req | Relevan untuk M-010 | Status |
|---|---|---|
| REQ-001..011 | Semua requirement fungsional | DONE (M-001..M-009) |
| REQ-101 | Reproducibility | DONE (seeded_rng, deterministik) |
| REQ-102 | Resource Efficiency | DONE (DECISION-018/021) |
| REQ-103 | Testability | DONE (coverage >=80%, CI hijau) |
| REQ-104 | Modularity | DONE (modul terpisah) |
| REQ-105 | Documentation | Parsial (API docs belum lengkap) |
| REQ-201/202/203 | Technical Constraints | DONE (permanen) |
| REQ-301/302 | SDD + No Overengineering | DONE |
| **REQ-107** | **Definisi sukses Naze 1.0** | **BLOCKING (OD-107)** |

**Catatan:** Semua requirement teknis terpenuhi. Hanya **OD-107** (definisi sukses) yang memblokir M-010.

## 5. Architecture

Pipeline Naze 1.0 end-to-end (produksi):

```
=== PACKAGING ===
PyPI Package (naze v1.0.0)
  |
  +-- src/naze/ (code)
  +-- naze/__init__.py (public API)
  +-- pyproject.toml (metadata + dependencies)
  +-- README.md (usage)
  +-- docs/ (dokumentasi lengkap)
  +-- models/ (bobot terlatih, optional)

=== DATA PIPELINE ===
Korpus Final (OD-112)
  -> ByteTokenizer.encode (vocab 256 fixed, D-015)
  -> TextWindows (sliding window, deterministik per-seed)
  -> batches_pos (per-posisi, M-008)

=== TRAINING PIPELINE ===
TextWindows + TransformerLM (config D-018)
  -> TrainingRun (M-008: SGD, checkpoint v2, resume, logging)
  -> Evaluasi (loss/perplexity, M-008)
  -> Checkpoint final (params.npz + meta.json + SHA-256, D-018/021)

=== INFERENCE PIPELINE ===
Checkpoint final
  -> InferenceEngine (M-009: forward-only, batching, sliding window)
  -> Generate (greedy/temperature, deterministik)
  -> Benchmark (tokens_per_second, peak_rss_kb)

=== EVALUASI ===
Holdout Dataset (dari korpus final)
  -> Loss/Perplexity (M-008)
  -> Akurasi perintah (untuk nazeio, opsional)
  -> Metrik sukses (OD-107: menunggu definisi)
```

**Kunci desain:**
- **Deterministik:** Semua operasi reproduksi per-seed (REQ-101).
- **Modular:** Setiap komponen (tokenize, train, inference) terpisah dan testable (REQ-104).
- **Efisien:** Memori dan waktu terukur (REQ-102, DECISION-018/021).
- **Produksi-ready:** API stabil, dokumentasi lengkap, packaging standar.

## 6. Module Boundaries

| Module | Responsibility | Baru/ubah | Dependency |
|---|---|---|---|
| `naze/` | Root package | - | - |
| `naze/__init__.py` | Public API (versi 1.0) | ubah | Semua submodul |
| `naze/core/` | Numerical core | tetap | - |
| `naze/nn/` | Neural engine | tetap | core |
| `naze/token/` | Tokenizer | tetap | - |
| `naze/data/` | Dataset pipeline | ubah (batches_pos) | token, core |
| `naze/lm/` | Language models | ubah (transformer_lm) | nn, data |
| `naze/train/` | Training system | ubah (M-008) | lm, data |
| `naze/inference/` | Inference engine | baru (M-009) | lm, data |
| `naze/export.py` | Ekspor bobot | baru (M-009) | lm, train |
| `models/` | Bobot terlatih | baru | - |

**Boundary rule:**
- `naze/__init__.py` mengeksport **hanya API publik** (tidak ada implementasi detail).
- Tidak ada circular dependency.
- Modul lama tidak dimodifikasi kecuali untuk backward compatibility.

## 7. Interfaces

### Public API (naze v1.0.0)
```python
# ==== Tokenizer ====
from naze import ByteTokenizer
tok = ByteTokenizer()
ids = tok.encode("hello world")
text = tok.decode(ids)

# ==== Dataset ====
from naze import TextWindows
windows = TextWindows(ids, block_size=16, batch_size=8, seed=0)
for x, y in windows.batches_pos(seed=42):
    pass  # x: (B,T), y: (B,T)

# ==== Models ====
from naze import MLPLM, TransformerLM, TransformerConfig

# MLPLM (Bengio-style)
mlp = MLPLM(vocab_size=256, block_size=16, d_embed=8, d_hidden=16, seed=0)

# Transformer (config D-018)
config = TransformerConfig(
    d_model=64, num_heads=4, num_layers=2,
    d_ff=128, max_sequence_length=128, vocab_size=256, seed=0
)
model = TransformerLM(config)

# ==== Training ====
from naze import SGDTrainer, TrainConfig, TrainingRun
from naze.train import save_checkpoint_v2, load_checkpoint_v2

trainer = SGDTrainer(model, lr=0.08)
trainer.train_step(x, y)

# Training loop penuh (M-008)
config = TrainConfig(model="transformer", corpus_path="korpus.txt", 
                     epochs=10, lr=0.08, seed=0)
run = TrainingRun(config, model, windows)
summary = run.run()

# ==== Inference ====
from naze import InferenceEngine, InferenceBatch
engine = InferenceEngine(model)
batch = InferenceBatch([[1, 2, 3, 4]], max_length=16)
output = engine.forward(batch)
new_ids = engine.generate([1, 2, 3], max_new=10, temperature=0.8, seed=0)

# ==== Export/Import ====
from naze.export import export_model_weights, load_model_weights
export_model_weights(model, "naze_v1_weights.npz", version="1.0.0")
params, metadata = load_model_weights("naze_v1_weights.npz")

# ==== Benchmark ====
from naze import benchmark_model
results = benchmark_model(engine, batch_sizes=[1, 8], seq_lengths=[16, 64])
for r in results:
    print(f"{r.batch_size}x{r.sequence_length}: {r.tokens_per_second:.2f} tok/s")
```

### File Structure (naze v1.0.0)
```
naze/
├── __init__.py          # Public API v1.0
├── pyproject.toml        # Package metadata (version 1.0.0)
├── README.md             # Usage + examples
├── LICENSE               # Lisensi (MIT?)
├── src/
│   └── naze/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── numeric.py
│       │   └── gradcheck.py
│       ├── nn/
│       │   ├── __init__.py
│       │   ├── activations.py
│       │   ├── layers.py
│       │   └── transformer.py
│       ├── token/
│       │   ├── __init__.py
│       │   └── byte_tokenizer.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── lm/
│       │   ├── __init__.py
│       │   ├── mlp_lm.py
│       │   └── transformer_lm.py
│       ├── train/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── checkpoint.py
│       │   ├── evaluate.py
│       │   ├── runlog.py
│       │   ├── loop.py
│       │   └── trainer.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── batch.py
│       │   ├── engine.py
│       │   └── benchmark.py
│       └── export.py
├── tests/                # 423+ tests
├── docs/                 # Dokumentasi SDD
│   ├── spec/
│   ├── architecture/
│   └── tasks/
└── models/               # Bobot terlatih (optional)
    └── naze_v1_weights.npz
```

## 8. Data Flow

### Training Flow (Naze 1.0)
1. **Persiapan Data:**
   - Korpus final (OD-112) -> ByteTokenizer.encode -> list[int] token IDs
   - TextWindows(ids, block_size, batch_size) -> batches_pos per-epoch
   - Validasi: dataset tidak kosong, token IDs valid

2. **Training:**
   - TransformerLM(config=D-018) -> params inisialisasi
   - TrainingRun(config, model, data) -> loop epoch
   - Per epoch: batches_pos(seed=seed+epoch) -> forward -> loss_pos -> backward_pos -> SGD update
   - Per eval_every: evaluate(holdout) -> loss/perplexity -> JSONL log
   - Akhir: save_checkpoint_v2 + meta.json + checksum

3. **Output:**
   - `params.npz` (float64, ~0.87 MB)
   - `params.npz.sha256` (checksum)
   - `meta.json` (metadata: step, epoch, config, checksum, naze_version="1.0.0")

### Inference Flow (Naze 1.0)
1. **Load Model:**
   - `load_model_weights("naze_v1_weights.npz")` -> params
   - Build TransformerLM(config=D-018) -> load params

2. **Inference:**
   - InferenceEngine(model) -> forward-only wrapper
   - InferenceBatch(token_ids, max_length=128) -> padding + mask
   - engine.forward(batch) -> logits (B,T,256)
   - engine.generate(prompt_ids, max_new, temperature, seed) -> new_ids

3. **Output:**
   - Token IDs -> ByteTokenizer.decode -> teks
   - Benchmark: tokens_per_second, peak_rss_kb

### Evaluasi Flow (Naze 1.0)
1. **Holdout Dataset:**
   - Separate dari korpus training (OD-112)
   - TextWindows(holdout_ids, block_size, batch_size)

2. **Metrik:**
   - Loss (cross-entropy mean)
   - Perplexity (exp(loss))
   - **Metrik sukses (OD-107):** Menunggu definisi owner

## 9. Configuration

### pyproject.toml (v1.0.0)
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "naze"
version = "1.0.0"
description = "Naze 1.0: AI model built from scratch (spec-driven, no pretrained core)."
requires-python = ">=3.10"
authors = [{name = "ndolzz", email = "..."}]
license = {text = "MIT"}
readme = "README.md"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py310"
```

### Default Config (D-018)
- **Model:** Transformer (D=64, H=4, L=2, d_ff=128, T_max=128)
- **Vocab:** 256 (byte-level, fixed)
- **Training:** SGD, lr=0.08, batch_size=8, epochs=10
- **Checkpoint:** <= 1 MB (float64), <= 0.43 MB (float32)
- **Inference:** float64 (training), float32 (inference untuk nazeio)

## 10. Error Handling

| Error | Handling | Type |
|---|---|---|
| Input kosong | ValueError (fail-fast) | Runtime |
| Token ID invalid | ValueError (fail-fast) | Runtime |
| Sequence terlalu panjang | ValueError (fail-fast) | Runtime |
| Checkpoint korup | ValueError (checksum mismatch) | Runtime |
| Model tidak kompatibel | ValueError (shape mismatch) | Runtime |
| File tidak ditemukan | FileNotFoundError | Runtime |
| NumPy error (NaN/Inf) | ValueError (as_array) | Runtime |

**Prinsip:** Fail-fast dengan pesan error yang jelas (REQ-302: no overengineering).

## 11. Testing Strategy

### Unit Tests
- Semua fungsi publik di `naze/__init__.py` memiliki unit test.
- Test validasi input (edge cases, invalid values).
- Test determinisme (seed yang sama -> hasil yang sama).

### Numerical Tests
- Gradient check (tol 1e-5) untuk semua komponen backward.
- Kesetaraan numerik antara implementasi yang berbeda.
- Toleransi float32 vs float64 (untuk nazeio).

### Integration Tests
- End-to-end: encode -> batch -> forward -> generate -> decode.
- Training: loss turun, checkpoint save/load, resume.
- Inference: benchmark tokens_per_second > 0, peak_rss_kb < 2 GB.

### Regression Tests
- Semua test lama (M-001..M-009) tetap hijau.
- Coverage >= 80% (DECISION-019).

### Acceptance Tests (OD-107)
- **Menunggu definisi owner:**
  - Metrik sukses (loss, perplexity, akurasi, dll.)
  - Dataset evaluasi (korpus + holdout)
  - Target performa (waktu, memori)

## 12. Resource Constraints

| Resource | Batas | Status |
|---|---|---|
| Checkpoint size | <= 1 MB (float64) | ✅ DECISION-018 |
| RAM training | < 2 GB | ✅ DECISION-021 |
| RAM inference | < 2 GB | ✅ DECISION-021 |
| Model params | ~108k (D-018) | ✅ |
| Vocab size | 256 (fixed) | ✅ DECISION-015 |
| Sequence length | <= 128 | ✅ DECISION-018 |

**Estimasi:**
- Training (config D-018): params+grads ~1.7 MB, aktivasi ~4 MB/layer x 2 = 8 MB, total < 20 MB.
- Inference (config D-018): params ~0.87 MB (float64), ~0.43 MB (float32).
- APK nazeio: model ~0.43 MB + code ~5 MB = < 10 MB (target).

## 13. Acceptance Criteria (Objective, Verifiable)

### Wajib (Must)
1. ✅ Semua milestone sebelumnya (M-001..M-009) DONE.
2. ✅ 423+ tests passed (semua hijau).
3. ✅ Coverage >= 80% baris `src/naze` (DECISION-019).
4. ✅ CI hijau (pytest + coverage + lint).
5. ✅ Paket `naze` v1.0.0 bisa di-install via `pip install naze`.
6. ✅ API publik stabil dan terdokumentasi.
7. ✅ Model Transformer (config D-018) terlatih dan bobot tersedia.
8. ✅ Checkpoint final: params.npz + meta.json + checksum (D-018/021).
9. ✅ Inference Engine berfungsi (forward, generate, benchmark).
10. ✅ Deterministik per-seed (REQ-101).

### Terukur (Measurable - Menunggu OD-107)
11. ⏳ **Loss training** < [target OD-107] (terukur, tanpa ambang universal per DECISION-016).
12. ⏳ **Perplexity** < [target OD-107] (pada holdout dataset).
13. ⏳ **Akurasi perintah** >= [target OD-107] (untuk nazeio, opsional).
14. ⏳ **Waktu inference** < [target OD-107] detik (ARMv7).
15. ⏳ **Ukuran APK nazeio** < [target OD-107] MB (dengan model).

### Dokumentasi
16. ✅ README.md: usage + examples.
17. ⏳ API reference (dokumentasi lengkap).
18. ⏳ Tutorial: train model dari nol.
19. ⏳ Changelog (v1.0.0).

## 14. Risks

| Risiko | Dampak | Mitigasi | Status |
|---|---|---|---|
| OD-107 tidak diputuskan | Blocker M-010 | Menunggu owner | ⚠️ **ACTIVE** |
| OD-112 tidak tersedia | Tidak ada korpus final | Gunakan korpus toy sementara | ⚠️ |
| Training lambat (CPU) | Waktu lama | Gunakan config kecil (D-018), batch kecil | ✅ |
| Float32 vs float64 drift | Hasil tidak konsisten | Test golden + toleransi | ✅ |
| ARMv7 performa rendah | Inference lambat | Model sengaja kecil, benchmark dini | ✅ |
| PyPI publish gagal | Paket tidak tersedia | Test lokal dengan `pip install -e .` | ⏳ |

## 15. Open Decisions

| ID | Deskripsi | Status | Dampak |
|---|---|---|---|
| **OD-107** | Definisi sukses Naze 1.0 (metrik, target, dataset) | **BLOCKING** | M-010 tidak bisa dimulai |
| OD-108 | Roadmap multimodal/physical AI (Stage 10) | OPEN | Tidak blocking M-010 |
| OD-112 | Korpus final training Naze | OPEN | Dapat menggunakan korpus toy |
| OD-119 | Versioning & release policy | OPEN | Dapat menggunakan SemVer standar |

**Catatan:** Hanya **OD-107** yang **blocking** untuk M-010. Open decisions lain dapat diselesaikan selama implementasi.
