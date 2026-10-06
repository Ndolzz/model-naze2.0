# REQUIREMENTS — Naze 2.0

**Version:** 0.3.0 | **Status:** M-001..M-006 selesai (menunggu validasi pytest owner)

**Priority:** MUST / SHOULD / DEFERRED — **Status:** PROPOSED / APPROVED / IN-PROGRESS / DONE

---

## A. Functional Requirements

### REQ-001 — Tensor/Numerical Core — MUST — **DONE (M-002)**
seeded_rng deterministik, as_array (validasi NaN/Inf), float64. Test lulus.

### REQ-002 — Neural Network Engine — MUST — **DONE (M-002)**
Layer/Linear/Activation/Sequential; forward teruji hand-computed.

### REQ-003 — Automatic Differentiation — MUST — **DONE (M-003, DECISION-009)**
Backprop terstruktur per-layer + gradient check numerik (tol 1e-5); test lulus untuk Linear, activations, Sequential, dan MLPLM.

### REQ-004 — Tokenizer — MUST — **DONE (M-004, DECISION-010)**
Byte-level, vocab 256, lossless roundtrip (ASCII/unicode/emoji), deterministik.

### REQ-005 — Dataset Pipeline — MUST — **DONE (M-005, DECISION-011)**
TextWindows; batch reproducible per-seed; memori terkendali.

### REQ-006 — Training System — MUST — **DONE minimal (M-006, DECISION-013)**
SGDTrainer; loss menurun pada toy corpus (test). Bagian penuh (logging, eval berkala) = M-008.

### REQ-007 — Checkpointing — MUST — **DONE (M-006, DECISION-014)**
npz save/load; test roundtrip: load -> forward identik.

### REQ-008 — Inference Engine — MUST — **PARTIAL (M-006: generate greedy+temperature, deterministik)**; penuh = M-009.

### REQ-009 — Evaluation — MUST — **DEFERRED (M-008)**; harness loss/perplexity dari checkpoint.

### REQ-010 — First Language Model — MUST — **DONE (M-006, DECISION-012)**
MLP Bengio-style; trainable end-to-end; generate teks; integrasi tokenizer byte.

### REQ-011 — Transformer — MUST — **PLANNED (M-007)**

---

## B. Non-Functional Requirements

### REQ-101 — Reproducibility — MUST — **DONE untuk kode saat ini**: seluruh randomness via seeded_rng; init/generation/batch deterministik per-seed (teruji). Dokumen reproduksi per-run lengkap menyusul di M-008.

### REQ-102 — Resource Efficiency — MUST — **DONE tahap ini**: model toy kecil; TextWindows hemat memori. Benchmark formal menyusul.

### REQ-103 — Testability — MUST — **DONE tahap ini**: 7 file test (smoke, numeric, activations, layers, gradcheck, tokenizer, dataset, LM integration). Coverage target = OPEN DECISION-113.

### REQ-104 — Modularity — MUST — **DONE**: core/nn/token/data/lm/train terpisah; dependency di ARCHITECTURE.md.

### REQ-105 — Documentation — MUST — **DONE**: docs SDD selaras kode (commit terakhir).

---

## C. Technical Constraints
REQ-201 No Pretrained Core — MUST (permanen) — APPROVED
REQ-202 No External LLM API Core — MUST (permanen) — APPROVED
REQ-203 Python + NumPy — MUST — APPROVED

## D. Development Constraints
REQ-301 Spec-Driven Only — APPROVED
REQ-302 No Overengineering — APPROVED
REQ-303 Open Decisions Escalated — APPROVED
REQ-304 Change Log — APPROVED

---

## Requirement Change Log

| Date | Req ID | Change | Reason | Version |
|---|---|---|---|---|
| 2026-10-06 | ALL | Initial draft | Phase 1 | 0.1.0 |
| 2026-10-06 | REQ-001/002/101/103/104/105 | IN-PROGRESS | Implementasi M-002 | 0.2.0 |
| 2026-10-06 | REQ-003/004/005/006/007/010 | DONE (implementasi M-003..M-006); REQ-008 partial | Instruksi owner "kerjakan semuanya"; OPEN DECISION 102-106 resolusi via delegasi | 0.3.0 |
