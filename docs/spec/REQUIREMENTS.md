# REQUIREMENTS — Naze 2.0

**Version:** 0.2.0
**Status:** Phase 1 approved; M-002 in review

**Priority legend:** MUST / SHOULD / DEFERRED
**Status legend:** PROPOSED / APPROVED / IN-PROGRESS / DONE

> Requirement bertanda DEFERRED adalah placeholder fondasi untuk milestone masa depan — TIDAK diimplementasikan sekarang; detail didefinisikan pada milestone terkait.

---

## A. Functional Requirements

### REQ-001 — Tensor / Numerical Core
- **Description:** Operasi numerik dasar (vector/matrix ops, broadcasting, utilitas) di atas NumPy sebagai fondasi seluruh modul.
- **Priority:** MUST (Stage 0–1)
- **Status:** IN-PROGRESS (M-002) — `naze.core.numeric` (seeded_rng, as_array, dtype konvensi)
- **Acceptance Criteria:**
  - [x] Operasi numerik dasar terdefinisi dengan interface jelas.
  - [x] Unit test lulus (determinisme, validasi input).
  - [ ] Dokumentasi API tersedia (dilengkapi saat M-002 disetujui).

### REQ-002 — Neural Network Engine
- **Description:** Engine neural network modular (layers, activations, forward pass) tanpa framework DL eksternal.
- **Priority:** MUST (Stage 1)
- **Status:** IN-PROGRESS (M-002) — `naze.nn` (Layer, Linear, Activation, Sequential; relu/sigmoid/tanh/softmax)
- **Acceptance Criteria:**
  - [x] Layer dasar dapat disusun menjadi model sederhana (Sequential).
  - [x] Forward pass teruji dengan nilai numerik acuan hand-computed.

### REQ-003 — Automatic Differentiation
- **Description:** Backward pass / gradien (pendekatan = OPEN DECISION-105).
- **Priority:** MUST (Stage 2)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Gradient check numerik lulus dengan toleransi ditetapkan; diuji di CI.

### REQ-004 — Tokenizer
- **Description:** Tokenizer teks buatan sendiri (jenis = OPEN DECISION-102).
- **Priority:** MUST (Stage 3)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Roundtrip lossless; vocab size configurable; deterministik.

### REQ-005 — Dataset Pipeline
- **Description:** Loading, cleaning, batching, konfigurasi sesuai resource, deterministik.
- **Priority:** MUST (Stage 4)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Batch reproducible per-seed; memori terkendali.

### REQ-006 — Training System
- **Description:** Training loop, optimizer (SGD minimal; lainnya OPEN DECISION-106), loss tracking, logging.
- **Priority:** MUST (Stage 5–7)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Trainable, stoppable, resumable via checkpoint; loss menurun sesuai ambang spec milestone.

### REQ-007 — Checkpointing
- **Description:** Save/load state model, optimizer, step, konfigurasi.
- **Priority:** MUST (DEFERRED bersama training)
- **Status:** PROPOSED
- **Acceptance Criteria:** Save/load deterministik; integritas terverifikasi.

### REQ-008 — Inference Engine
- **Description:** Menjalankan model terlatih forward-only, efisien memori.
- **Priority:** MUST (Stage 8)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Konsisten antar-run (seed + checkpoint sama); benchmark memori/latensi.

### REQ-009 — Evaluation
- **Description:** Metrik (loss/perplexity untuk LM) dan harness evaluasi.
- **Priority:** MUST (DEFERRED)
- **Status:** PROPOSED
- **Acceptance Criteria:** Evaluasi runnable dari checkpoint; hasil dalam format terstruktur (OPEN DECISION).

### REQ-010 — First Language Model
- **Description:** Model bahasa pertama sederhana (arsitektur = OPEN DECISION-104) sebelum Transformer.
- **Priority:** MUST (Stage 5)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Trainable end-to-end; menghasilkan teks dari checkpoint.

### REQ-011 — Transformer (Own Implementation)
- **Description:** Transformer dari nol (attention, positional encoding, blok).
- **Priority:** MUST (Stage 6)
- **Status:** PROPOSED (DEFERRED)
- **Acceptance Criteria:** Unit test per komponen lulus; dimensi dapat diskalakan turun.

---

## B. Non-Functional Requirements

### REQ-101 — Reproducibility
- **Description:** Seed, konfigurasi, versi dependency tercatat; hasil reproducible.
- **Priority:** MUST
- **Status:** IN-PROGRESS — DECISION-007 (float64 + seeded RNG) diterapkan pada core; test determinisme ada.
- **Acceptance Criteria:** [ ] Dokumen reproduksi; [ ] config+seed+versi tercatat per run (saat training stage).

### REQ-102 — Resource Efficiency
- **Description:** Memori terkendali, parameter configurable, model kecil di tahap awal.
- **Priority:** MUST
- **Status:** PROPOSED
- **Acceptance Criteria:** [ ] Batas memori configurable/diuji per modul heavy.

### REQ-103 — Testability
- **Description:** Unit + integration test; numerical correctness (gradient check, roundtrip, determinism).
- **Priority:** MUST
- **Status:** IN-PROGRESS — pytest + ruff (DECISION-006); 4 file test aktif.
- **Acceptance Criteria:** [x] Test suite satu perintah (`pytest`). [ ] Coverage target: OPEN DECISION.

### REQ-104 — Modularity
- **Description:** Komponen terpisah dengan interface jelas.
- **Priority:** MUST
- **Status:** IN-PROGRESS — `core` vs `nn` terpisah; dependency terdokumentasi di ARCHITECTURE.md.
- **Acceptance Criteria:** [x] Dependency antar modul terdokumentasi.

### REQ-105 — Documentation
- **Description:** Dokumentasi hidup selaras dengan kode.
- **Priority:** MUST
- **Status:** IN-PROGRESS — README + docs SDD; API docs menyusul saat M-002 disetujui.
- **Acceptance Criteria:** [ ] Setiap milestone selesai disertai pembaruan dokumentasi.

---

## C. Technical Constraints

### REQ-201 — No Pretrained Core — **MUST (Permanen) / APPROVED**
### REQ-202 — No External LLM API Core — **MUST (Permanen) / APPROVED**
### REQ-203 — Python + NumPy Foundation (framework DL dilarang untuk core) — **MUST / APPROVED**

---

## D. Development Constraints

### REQ-301 — Spec-Driven Only — **MUST / APPROVED**
### REQ-302 — No Overengineering — **MUST / APPROVED**
### REQ-303 — Open Decisions Escalated — **MUST / APPROVED**
### REQ-304 — Change Log for Requirement Changes — **MUST / APPROVED**
- **Acceptance Criteria:** [x] Changelog requirement up-to-date (lihat bawah).

---

## Requirement Change Log

| Date | Req ID | Change | Reason | Version |
|---|---|---|---|---|
| 2026-10-06 | ALL | Initial draft | Phase 1 — Specification Only | 0.1.0 |
| 2026-10-06 | REQ-103 | Referensi pytest (DECISION-006) | M-001 tooling | 0.1.0 |
| 2026-10-06 | REQ-001, REQ-002, REQ-101, REQ-103, REQ-104, REQ-105 | Status → IN-PROGRESS; hasil implementasi dicatat | Implementasi M-002 (Stage 1) | 0.2.0 |
