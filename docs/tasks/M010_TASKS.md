# M010_TASKS  Naze 1.0 (Stage 9)

**Status:** PLANNED (OD-107 RESOLVED via DECISION-023, blocked by OD-112)
**Referensi:** docs/architecture/M010_TECHNICAL_DESIGN.md (DECISION-023)
**Dependencies:** M-001..M-009 DONE, OD-107 RESOLVED (PROVISIONAL), **OD-112 BLOCKING**
**Requirements:** REQ-001..011, REQ-101..105, REQ-201..203, REQ-301..304

**Aturan:**
- Tidak mengubah arsitektur model (DECISION-018: D=64, H=4, L=2, d_ff=128, T_max=128).
- Tidak memulai training sampai OD-112 (korpus final) diselesaikan.
- Tidak menerbitkan rilis PyPI sampai semua acceptance criteria terpenuhi.
- Setiap task mereferensikan AC dari DECISION-023 dan M010_TECHNICAL_DESIGN.md.
- Semua task wajib memiliki acceptance criteria terukur dan metode verifikasi.

---

## Kategori: Finalisasi Kontrak dan Metadata Rilis

### T001  Finalisasi pyproject.toml v1.0.0
**Deskripsi:** Perbarui `pyproject.toml` untuk rilis produksi Naze 1.0.
**Input:** Template di M010_TECHNICAL_DESIGN.md  9.
**Output:** `pyproject.toml` dengan version="1.0.0", metadata lengkap.
**AC:**
- [ ] `name = "naze"` (provisional, verifikasi ketersediaan PyPI)
- [ ] `version = "1.0.0"`
- [ ] `requires-python = ">=3.10"`
- [ ] `dependencies = ["numpy>=1.24"]`
- [ ] `authors` terisi
- [ ] `license = {text = "MIT"}`
- [ ] `readme = "README.md"`
- [ ] `build-system` terkonfigurasi
- [ ] `tool.pytest.ini_options` terkonfigurasi
- [ ] `tool.ruff` terkonfigurasi
**Modul:** `pyproject.toml`
**Dependensi:** -
**Verifikasi:** `python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb')))"` berjalan tanpa error.
**Req:** REQ-105, REQ-203

### T002  Verifikasi nama paket PyPI "naze"
**Deskripsi:** Verifikasi ketersediaan nama paket "naze" di PyPI.
**Input:** Nama paket "naze".
**Output:** Dokumentasi ketersediaan atau alternatif.
**AC:**
- [ ] Cek `pip search naze` atau `https://pypi.org/project/naze/`
- [ ] Jika tidak tersedia: dokumentasikan di DECISION_LOG.md
- [ ] Jika tersedia: catat alternatif (misal: `naze-ai`, `naze2`, `ndolzz-naze`)
- [ ] Keputusan owner wajib dicatat sebelum mengubah metadata
**Modul:** Dokumen (DECISION_LOG.md)
**Dependensi:** T001 (jika nama tidak tersedia)
**Verifikasi:** Catatan di DECISION_LOG.md atau confirmasi owner.
**Req:** REQ-105, REQ-303
**Catatan:** **TIDAK MELAKUKAN PUBLISH**, hanya verifikasi.

---

## Kategori: Integrasi dan Validasi Training Corpus (OD-112)

### T003  Dokumentasi kebijakan dataset OD-112
**Deskripsi:** Dokumentasikan kebijakan dataset untuk OD-112 (korpus final).
**Input:** Keputusan owner untuk OD-112.
**Output:** Dokumentasi di M010_TECHNICAL_DESIGN.md atau dokumen terpisah.
**AC:**
- [ ] Sumber data tercatat (URL, lisensi, tanggal unduh)
- [ ] Lisensi data kompatibel dengan rilis publik (MIT, Apache, CC, dll.)
- [ ] Preprocessing terdefinisi (cleanup, deduplikasi, normalisasi)
- [ ] Pembagian dataset: training/validation/holdout terpisah
- [ ] Reproduktibilitas: seed/konfigurasi untuk pembagian dataset
- [ ] Holdout **TIDAK BOLEH** digunakan untuk training atau hyperparameter tuning
**Modul:** `docs/architecture/M010_TECHNICAL_DESIGN.md` (update section 8)
**Dependensi:** OD-112 (keputusan owner)
**Verifikasi:** Dokumen terdokumentasi, tidak ada fiktif.
**Req:** REQ-005, REQ-105
**Bloker:** **OD-112 WAJIB diselesaikan sebelum task ini.**

### T004  Integrasi corpus final ke pipeline
**Deskripsi:** Integrasikan corpus final (OD-112) ke pipeline training.
**Input:** Corpus final (file teks), path terkonfigurasi.
**Output:** Pipeline siap: encode -> batch -> train.
**AC:**
- [ ] Corpus dapat di-load oleh `ByteTokenizer.encode()`
- [ ] Token IDs valid (semua di [0, 256))
- [ ] `TextWindows` berjalan tanpa error
- [ ] Dataset tidak kosong
- [ ] Pembagian training/validation/holdout terpisah
**Modul:** `src/naze/data/dataset.py` (jika perlu adaptasi)
**Dependensi:** T003 (OD-112 resolved)
**Verifikasi:** `python -c "from naze.token import ByteTokenizer; from naze.data import TextWindows; tok = ByteTokenizer(); ids = tok.encode(open('corpus.txt').read()); print(f'Tokens: {len(ids)}')"` berjalan.
**Req:** REQ-005, REQ-004, REQ-101
**Bloker:** **OD-112 WAJIB diselesaikan.**

---

## Kategori: Training dan Checkpoint Management

### T005  Training model final dengan config D-018
**Deskripsi:** Lakukan training model Transformer (config D-018) pada corpus final.
**Input:** Corpus final (T004), config D-018.
**Output:** Checkpoint final (params.npz + meta.json + checksum).
**AC:**
- [ ] Training berjalan tanpa error
- [ ] Loss training **<= 2.5** (DECISION-023, WAJIB release gate)
- [ ] Loss menurun (konvergensi terlihat)
- [ ] Checkpoint tersimpan: params.npz + meta.json + .sha256
- [ ] Checkpoint size <= 1 MB (DECISION-018)
- [ ] Deterministik: seed yang sama -> hasil yang sama
**Modul:** `src/naze/train/loop.py` (gunakan existing TrainingRun)
**Dependensi:** T004 (corpus terintegrasi)
**Verifikasi:**
- Log training menunjukkan loss <= 2.5
- `ls -lh params.npz` <= 1 MB
- Checksum valid
**Req:** REQ-006, REQ-007, REQ-101, REQ-102, REQ-103
**Bloker:** **TIDAK BOLEH dimulai sampai OD-112 resolved (T004 selesai).**

### T006  Evaluasi validation dan holdout
**Deskripsi:** Evaluasi model terlatih pada validation dan holdout dataset.
**Input:** Model terlatih (T005), validation/holdout dataset (T004).
**Output:** Metrik evaluasi: loss, perplexity.
**AC:**
- [ ] Validation perplexity **<= 35** (DECISION-023, WAJIB release gate)
- [ ] Holdout perplexity terukur (dokumentasikan nilai)
- [ ] Evaluasi deterministik
- [ ] Metode: `naze.train.evaluate()` atau `EvalResult`
**Modul:** `src/naze/train/evaluate.py` (gunakan existing)
**Dependensi:** T005 (model terlatih), T004 (dataset terpisah)
**Verifikasi:**
- `perplexity <= 35` (validation)
- Nilai holdout terdokumentasi
**Req:** REQ-009, REQ-101, REQ-103
**Bloker:** T005 (training selesai)

---

## Kategori: NazeIO Command Benchmark

### T007  Persiapan command benchmark untuk nazeio
**Deskripsi:** Siapkan dataset command benchmark dengan input, expected output, aturan penilaian.
**Input:** Command list dari nazeio (jika tersedia) atau definisi eksplisit.
**Output:** File benchmark: `data/benchmark/commands.json` atau sejenisnya.
**AC:**
- [ ] Setiap command memiliki: input (teks), expected_output (teks/aksi)
- [ ] Aturan penilaian eksplisit (exact match, similarity, dll.)
- [ ] Dataset command terpisah dari training corpus
- [ ] Format terdokumentasi
**Modul:** `data/benchmark/commands.json` (file baru)
**Dependensi:** OD-112 (korpus final, jika command dari corpus)
**Verifikasi:** File JSON valid, dapat di-load.
**Req:** REQ-008, REQ-105

### T008  Pengujian NazeIO command accuracy
**Deskripsi:** Ukur akurasi command menggunakan InferenceEngine (M-009).
**Input:** Model terlatih (T005), command benchmark (T007).
**Output:** Akurasi command >= 90% (DECISION-023, WAJIB release gate).
**AC:**
- [ ] Setiap command di-infer menggunakan `InferenceEngine.generate()`
- [ ] Output dibandingkan dengan expected (aturan penilaian T007)
- [ ] **Akurasi >= 90%** (DECISION-023, WAJIB release gate)
- [ ] Hasil terdokumentasi (berapa command benar/salah)
**Modul:** Skrip benchmark: `scripts/benchmark_nazeio.py` (baru)
**Dependensi:** T005 (model terlatih), T007 (command benchmark)
**Verifikasi:**
- `accuracy = (benar / total) * 100 >= 90`
- Log benchmark terdokumentasi
**Req:** REQ-008, REQ-103
**Bloker:** T005, T007

---

## Kategori: Inference ARMv7 dan APK

### T009  Pengukuran inference latency pada ARMv7
**Deskripsi:** Ukur latency inference model pada ARMv7 (Raspberry Pi 3 referensi).
**Input:** Model terlatih (T005), config D-018, float32 precision.
**Output:** Rata-rata latency per token.
**AC:**
- [ ] Konfigurasi referensi: Raspberry Pi 3 (ARMv7, 4-core @1.2GHz)
- [ ] Model loaded dengan float32 precision
- [ ] Config: D=64, H=4, L=2, d_ff=128, T_max=128
- [ ] Metode: `naze.inference.benchmark_model()` dengan warmup=3, runs=5
- [ ] **Target: <= 2 detik/token** (DECISION-023, target optimasi)
- [ ] Pembedaan: prefill vs generation time dilaporkan terpisah
- [ ] Batch size: 1 (single prompt)
**Modul:** Skrip benchmark: `scripts/benchmark_armv7.py` (baru)
**Dependensi:** T005 (model terlatih)
**Verifikasi:**
- Latency terukur dan terdokumentasi
- Jika > 2 detik/token: dokumentasikan keterbatasan
**Req:** REQ-008, REQ-102
**Bloker:** Akses ke perangkat ARMv7 (jika tidak tersedia, dokumentasikan)
**Catatan:** **Target optimasi, bukan release gate.** Jika tidak terpenuhi, rilis tetap bisa dilakukan.

### T010  Integrasi model ke APK nazeio (opsional M-010)
**Deskripsi:** Integrasikan model terlatih ke APK nazeio untuk pengukuran ukuran.
**Input:** Model terlatih (T005), export float32 (M-009 export.py).
**Output:** APK dengan model terintegrasi.
**AC:**
- [ ] Model diekspor dengan `export_model_weights(..., float32=True)`
- [ ] Model terintegrasi ke nazeio (kolaborasi dengan proyek nazeio)
- [ ] **Target: Ukuran APK <= 50 MB** (DECISION-023, target optimasi)
- [ ] Ukuran model saja: ~0.43 MB (D-018, float32)
**Modul:** Kolaborasi dengan nazeio
**Dependensi:** T005 (model terlatih), nazeio (proyek eksternal)
**Verifikasi:**
- Ukuran APK terukur dan terdokumentasi
- Jika > 50 MB: dokumentasikan komponen penyumbang ukuran
**Req:** REQ-008
**Bloker:** Kolaborasi nazeio
**Catatan:** **Target optimasi, bukan release gate.** Target berlaku untuk integrasi Android, bukan paket PyPI.

---

## Kategori: Dokumentasi

### T011  API Reference Documentation
**Deskripsi:** Buat dokumentasi API reference untuk paket naze v1.0.0.
**Input:** Public API di `src/naze/__init__.py`.
**Output:** `docs/api.md` atau `docs/api/` dengan dokumentasi lengkap.
**AC:**
- [ ] Semua simbol publik di `naze/__init__.py` terdokumentasi
- [ ] Contoh penggunaan untuk setiap kelas/fungsi utama
- [ ] Type hints terlihat
- [ ] Parameter, return value, exception terdokumentasi
- [ ] Format: Markdown (untuk README.md dan docs/)
**Modul:** `docs/api.md` (file baru)
**Dependensi:** T001 (pyproject.toml final)
**Verifikasi:** Dokumen dapat dibaca, contoh berjalan.
**Req:** REQ-105

### T012  Tutorial: Train model dari nol
**Deskripsi:** Buat tutorial langkah-demi-langkah untuk training model.
**Input:** Pipeline training (T005), config D-018.
**Output:** `docs/tutorial_train.md` (tutorial lengkap).
**AC:**
- [ ] Langkah: instalasi, persiapan corpus, konfigurasi, training
- [ ] Contoh kode yang berjalan
- [ ] Penjelasan parameter utama
- [ ] Troubleshooting umum
**Modul:** `docs/tutorial_train.md` (file baru)
**Dependensi:** T005 (training validated)
**Verifikasi:** Tutorial dapat diikuti, kode berjalan.
**Req:** REQ-105

### T013  Changelog v1.0.0
**Deskripsi:** Buat changelog untuk rilis v1.0.0.
**Input:** Semua milestone M-001..M-010.
**Output:** `CHANGELOG.md` atau `docs/changelog.md`.
**AC:**
- [ ] Format: cronological, per-version
- [ ] v1.0.0: ringkasan fitur, bug fix, breaking changes
- [ ] Referensi ke milestone dan task
- [ ] Tanggal rilis (jika sudah dipublikasi)
**Modul:** `CHANGELOG.md` (file baru)
**Dependensi:** Semua task M-010 selesai
**Verifikasi:** Changelog lengkap dan akurat.
**Req:** REQ-105, REQ-304

### T014  Update README.md untuk v1.0.0
**Deskripsi:** Perbarui README.md dengan informasi rilis v1.0.0.
**Input:** README.md existing, API docs (T011), tutorial (T012).
**Output:** README.md terupdate.
**AC:**
- [ ] Usage examples untuk fitur utama
- [ ] Installasi: `pip install naze`
- [ ] Link ke dokumentasi (API, tutorial)
- [ ] Badges: CI status, coverage, version
- [ ] License terlihat
**Modul:** `README.md`
**Dependensi:** T001, T011, T012
**Verifikasi:** README.md informatif dan up-to-date.
**Req:** REQ-105

---

## Kategori: Validasi Package dan Kesiapan Rilis

### T015  Validasi paket lokal
**Deskripsi:** Validasi paket dapat di-install dan berfungsi lokal.
**Input:** `pyproject.toml` (T001), struktur paket.
**Output:** Paket terinstall lokal, test berjalan.
**AC:**
- [ ] `pip install -e .` berjalan tanpa error
- [ ] `python -c "import naze; print(naze.__version__)"` berjalan
- [ ] Semua test lama (M-001..M-009) tetap hijau
- [ ] Coverage >= 80% (DECISION-019)
**Modul:** Root repository
**Dependensi:** T001 (pyproject.toml)
**Verifikasi:**
- `pip install -e .` sukses
- `pytest` hijau
- `pytest --cov=naze --cov-fail-under=80` hijau
**Req:** REQ-103, REQ-105, REQ-203

### T016  Validasi acceptance criteria DECISION-023
**Deskripsi:** Verifikasi semua acceptance criteria DECISION-023 terpenuhi.
**Input:** Hasil T005, T006, T008, T009, T010.
**Output:** Laporan validasi.
**AC:**
- [ ] **WAJIB:** Training loss <= 2.5 (T005)
- [ ] **WAJIB:** Validation perplexity <= 35 (T006)
- [ ] **WAJIB:** NazeIO command accuracy >= 90% (T008)
- [ ] Target: ARMv7 latency <= 2 detik/token (T009)
- [ ] Target: APK size <= 50 MB (T010)
- [ ] Semua metrik terdokumentasi
**Modul:** `docs/validation_report.md` (file baru)
**Dependensi:** T005, T006, T008, T009, T010
**Verifikasi:** Laporan validasi lengkap.
**Req:** REQ-105, DECISION-023
**Bloker:** Semua task dependensi

### T017  CI/CD produksi
**Deskripsi:** Pastikan CI/CD siap untuk produksi.
**Input:** `.github/workflows/tests.yml` existing.
**Output:** CI hijau untuk semua check.
**AC:**
- [ ] pytest wajib hijau (DECISION-020)
- [ ] coverage >= 80% (DECISION-019)
- [ ] lint (ruff) hijau (jika diaktifkan)
- [ ] Branch protection: required status check pytest
**Modul:** `.github/workflows/`
**Dependensi:** T015 (paket valid)
**Verifikasi:** CI hijau di GitHub Actions.
**Req:** REQ-103, REQ-201

---

## Kategori: Dokumentasi Pasca-Implementasi

### T018  Update ROADMAP.md
**Deskripsi:** Update ROADMAP.md dengan status M-010.
**Input:** ROADMAP.md existing.
**Output:** ROADMAP.md terupdate.
**AC:**
- [ ] M-010: status DONE
- [ ] Tanggal selesai
- [ ] Ringkasan perubahan
**Modul:** `docs/tasks/ROADMAP.md`
**Dependensi:** Semua task M-010 selesai
**Verifikasi:** ROADMAP konsisten.
**Req:** REQ-105, REQ-304

### T019  Update TRACEABILITY.md
**Deskripsi:** Update TRACEABILITY.md dengan M-010.
**Input:** TRACEABILITY.md existing.
**Output:** TRACEABILITY.md terupdate.
**AC:**
- [ ] M-010 ter-mapped ke REQ-008/REQ-105
- [ ] DECISION-023 tercatat
- [ ] OD-107 status RESOLVED
- [ ] OD-112 status masih OPEN/BLOCKING
**Modul:** `docs/spec/TRACEABILITY.md`
**Dependensi:** Semua task M-010 selesai
**Verifikasi:** TRACEABILITY konsisten.
**Req:** REQ-105

### T020  Update SPEC_REVIEW.md
**Deskripsi:** Update SPEC_REVIEW.md dengan status M-010.
**Input:** SPEC_REVIEW.md existing.
**Output:** SPEC_REVIEW.md terupdate.
**AC:**
- [ ] M-010: IMPLEMENTED
- [ ] OD-107: RESOLVED
- [ ] Issues/blockers terdokumentasi
**Modul:** `docs/spec/SPEC_REVIEW.md`
**Dependensi:** Semua task M-010 selesai
**Verifikasi:** SPEC_REVIEW konsisten.
**Req:** REQ-105

---

## Ringkasan Dependensi Task

```
T001 -> T002 (jika nama tidak tersedia)
T001 -> T011, T012, T014, T015
T003 <- OD-112 (BLOCKER)
T004 <- T003 (OD-112 resolved)
T005 <- T004 (corpus terintegrasi)
T006 <- T005 (model terlatih)
T007 <- OD-112 (opsional, jika command dari corpus)
T008 <- T005, T007
T009 <- T005 (model terlatih)
T010 <- T005 (model terlatih), nazeio
T011 <- T001
T012 <- T005
T013 <- Semua task selesai
T014 <- T001, T011, T012
T015 <- T001
T016 <- T005, T006, T008, T009, T010
T017 <- T015
T018 <- Semua task selesai
T019 <- Semua task selesai
T020 <- Semua task selesai
```

## Blocker Utama

1. **OD-112: Korpus final training Naze**  **BLOCKING** untuk T003, T004, T005, T006, T007, T008.
   - **Status:** OPEN, menunggu keputusan owner.
   - **Impact:** Training **TIDAK BOLEH** dimulai tanpa corpus final.

2. **OD-119: Versioning & release policy**  Target optimasi, tidak blocking implementasi.

3. **Akses ARMv7:** Jika tidak tersedia, T009 dokumentasikan keterbatasan.

4. **Kolaborasi nazeio:** Jika tidak tersedia, T007, T008, T010 dokumentasikan keterbatasan.

## Task yang Bisa Dikerjakan Sekarang (Tidak Bloking OD-112)

- [ ] T001: Finalisasi pyproject.toml v1.0.0
- [ ] T002: Verifikasi nama paket PyPI "naze"
- [ ] T011: API Reference Documentation
- [ ] T014: Update README.md untuk v1.0.0
- [ ] T015: Validasi paket lokal (setelah T001)
- [ ] T017: CI/CD produksi (setelah T015)

## Task yang Membutuhkan OD-112

- [ ] T003: Dokumentasi kebijakan dataset OD-112
- [ ] T004: Integrasi corpus final ke pipeline
- [ ] T005: Training model final
- [ ] T006: Evaluasi validation dan holdout
- [ ] T007: Persiapan command benchmark (jika command dari corpus)
- [ ] T008: Pengujian NazeIO command accuracy
- [ ] T009: Pengukuran inference latency ARMv7 (butuh model terlatih)
- [ ] T010: Integrasi APK nazeio (butuh model terlatih)
- [ ] T012: Tutorial: Train model dari nol (butuh training validated)
- [ ] T013: Changelog v1.0.0 (butuh semua task selesai)
- [ ] T016: Validasi acceptance criteria (butuh semua metrik)
- [ ] T018, T019, T020: Dokumentasi pasca-implementasi

---

## Definition of Done untuk M-010

M-010 **DONE** jika:
1. [ ] **WAJIB:** Training loss <= 2.5 (DECISION-023)
2. [ ] **WAJIB:** Validation perplexity <= 35 (DECISION-023)
3. [ ] **WAJIB:** NazeIO command accuracy >= 90% (DECISION-023)
4. [ ] Semua task T001..T020 selesai atau terdokumentasi keterbatasan
5. [ ] CI hijau (pytest + coverage + lint)
6. [ ] Paket `naze` v1.0.0 dapat di-install lokal
7. [ ] Dokumentasi lengkap (API, tutorial, changelog, README)
8. [ ] TRACEABILITY.md, ROADMAP.md, SPEC_REVIEW.md terupdate
9. [ ] **TIDAK ADA** commit/push tanpa otorisasi
10. [ ] **TIDAK ADA** training/publikasi tanpa validasi metrik

---

## Catatan Penting

- **TIDAK BOLEH** mengubah arsitektur model (DECISION-018 fixed).
- **TIDAK BOLEH** memulai training sampai OD-112 resolved.
- **TIDAK BOLEH** menerbitkan ke PyPI tanpa validasi semua acceptance criteria.
- **TIDAK BOLEH** commit/push otomatis tanpa review.
- Target DECISION-023 adalah **PROVISIONAL** dan dapat direvisi owner.
- OD-112 adalah **SATU-SATUNYA BLOCKER** untuk training M-010.
