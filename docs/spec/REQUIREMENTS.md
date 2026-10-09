# REQUIREMENTS — Naze 2.0

> **[RECONSTRUCTED GOVERNANCE — 2026-10-07]**
> Dokumen ini adalah RECONSTRUCTED GOVERNANCE, bukan pemulihan file asli. Versi lokal REQUIREMENTS v0.4.0 hilang dan tidak pernah di-commit.
> **Basis rekonstruksi:** versi GitHub commit `b53259b` (v0.3.0 — verbatim dari git history) + revisi AC sesuai DECISION-015/016 yang terdokumentasi di TRACEABILITY v1.3.0, M007_TECHNICAL_DESIGN, dan SPEC_REVIEW v1.1.0.
> Bagian tanpa penanda direkonstruksi dari v0.3.0 (akurat); bagian revisi governance ditandai **[RECONSTRUCTED]**. Wording revisi BUKAN teks asli yang hilang.

**Version:** 0.5.0 | **Status:** M-001..M-007 DONE (implementasi Transformer selesai 2026-10-09; suite penuh via CI); M-008+ PLANNED

**Priority:** MUST / SHOULD / DEFERRED — **Status:** PROPOSED / APPROVED / IN-PROGRESS / DONE

**Requirement count:** 23 REQ ID unik (REQ-001..011, REQ-101..105, REQ-201..203, REQ-301..304). Klaim "26 requirement" pada dokumen review/traceability lama adalah kesalahan hitung pra-existing; dikoreksi menjadi 23 per keputusan owner 2026-10-07 — tanpa membuat REQ baru.

---

## A. Functional Requirements

### REQ-001 — Tensor/Numerical Core — MUST — **DONE (M-002)**
seeded_rng deterministik, as_array (validasi NaN/Inf), float64. Test lulus.

### REQ-002 — Neural Network Engine — MUST — **DONE (M-002)**
Layer/Linear/Activation/Sequential; forward teruji hand-computed.

### REQ-003 — Automatic Differentiation — MUST — **DONE (M-003, DECISION-009)**
Backprop terstruktur per-layer + gradient check numerik (tol 1e-5); test lulus untuk Linear, activations, Sequential, MLPLM, dan komponen Transformer backward (M-007: MHA/FFN/LayerNorm/spot-check end-to-end).

### REQ-004 — Tokenizer — MUST — **DONE (M-004, DECISION-010, DECISION-015)** [RECONSTRUCTED — AC direvisi]
Byte-level; vocab **FIXED 256** (DECISION-015 — "configurable" tidak berlaku untuk vocab size); lossless roundtrip (ASCII/unicode/emoji); deterministik.

### REQ-005 — Dataset Pipeline — MUST — **DONE (M-005, DECISION-011)**
TextWindows; batch reproducible per-seed; memori terkendali.

### REQ-006 — Training System — MUST — **DONE minimal (M-006, DECISION-013, DECISION-016)** [RECONSTRUCTED — AC direvisi]
SGDTrainer; kualitas training dievaluasi via kerangka 5-basis (DECISION-016): (1) baseline comparison, (2) perbaikan loss terukur, (3) bukti konvergensi/stabilitas, (4) hasil validasi, (5) konfigurasi evaluasi reproducible — **TANPA ambang loss numerik universal**. Bagian penuh (logging, eval berkala) = M-008.

### REQ-007 — Checkpointing — MUST — **DONE (M-006, DECISION-014)**
npz save/load; test roundtrip: load -> forward identik.

### REQ-008 — Inference Engine — MUST — **PARTIAL (M-006: generate greedy+temperature, deterministik; M-007: adapter transformer_generate greedy+temperature)**; penuh = M-009.

### REQ-009 — Evaluation — MUST — **DEFERRED (M-008)** [RECONSTRUCTED — AC + mapping direvisi]
Harness loss/perplexity dari checkpoint; evaluasi mengikuti kerangka 5-basis (DECISION-016) — tanpa ambang numerik universal. Mapping stage (klarifikasi ISSUE-012): evaluation harness = Stage 7 (dipakai oleh training); benchmark inference = Stage 8.

### REQ-010 — First Language Model — MUST — **DONE (M-006, DECISION-012, DECISION-016)** [RECONSTRUCTED — AC direvisi]
MLP Bengio-style; trainable end-to-end (dievaluasi per basis 1–3 kerangka DECISION-016); generate teks deterministik; integrasi tokenizer byte (vocab fixed 256, DECISION-015).

### REQ-011 — Transformer — MUST — **DONE (M-007, 2026-10-09)** [RECONSTRUCTED — status diperbarui]
Decoder-only causal Transformer sesuai `docs/architecture/M007_TECHNICAL_DESIGN.md` (format DECISION-017); unit test per komponen; dimensi configurable & scalable turun (vocab 256 fixed — DECISION-015); evaluasi sesuai DECISION-016. Implementasi T001..T018 selesai; suite unit/numeric/integration + pytest penuh via CI (`.github/workflows/tests.yml`).

---

## B. Non-Functional Requirements

### REQ-101 — Reproducibility — MUST — **DONE untuk kode saat ini**: seluruh randomness via seeded_rng; init/generation/batch deterministik per-seed (teruji, termasuk Transformer M-007: logits/attention/generate). Dokumen reproduksi per-run lengkap menyusul di M-008.

### REQ-102 — Resource Efficiency — MUST — **DONE tahap ini**: model toy kecil; TextWindows hemat memori. Benchmark formal menyusul (metrik formal = OD-117). Estimasi resource Transformer terdokumentasi di M007 TD §12.

### REQ-103 — Testability — MUST — **DONE tahap ini**: 24 file test (termasuk 3 suite Transformer baru M-007: unit/numeric/integration). Suite penuh dieksekusi via CI workflow (T017). Coverage target = OPEN DECISION-113.

### REQ-104 — Modularity — MUST — **DONE**: core/nn/token/data/lm/train terpisah; dependency di ARCHITECTURE.md. Boundary rule Transformer (M007 TD §6): train/lm → nn → core.

### REQ-105 — Documentation — MUST — **DONE**: docs SDD selaras kode (M-007: TRACEABILITY v1.4.0, ROADMAP, REQUIREMENTS v0.5.0, SPEC_REVIEW v1.3.0, M007_TASKS status); maintenance policy governance di PROJECT_SPEC §7 (v0.4.0).

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
| 2026-10-06 | REQ-001, REQ-002, REQ-101, REQ-103, REQ-104, REQ-105 | Status → IN-PROGRESS; hasil implementasi dicatat | Implementasi M-002 (Stage 1) | 0.2.0 |
| [HISTORICAL DETAIL UNAVAILABLE] | REQ-004, REQ-006, REQ-009, REQ-010 | AC direvisi: vocab fixed 256 (DECISION-015); evaluasi 5-basis tanpa ambang loss numerik universal (DECISION-016); mapping REQ-009 diperjelas (ISSUE-012). REQ-011 → TECHNICAL DESIGN READY. | Governance M-007: resolusi ISSUE-004 & ISSUE-006/OD-116; DECISION-017 membuka tahap Technical Design | 0.4.0 [RECONSTRUCTED] |
| 2026-10-09 | REQ-011, REQ-003, REQ-008, REQ-101, REQ-103, REQ-105 | REQ-011 → DONE (M-007 implementasi T001..T018); REQ-003 mencakup grad-check Transformer; REQ-008 partial mencakup transformer_generate; REQ-101/103/105 diperluas hasil M-007; CI pytest ditambahkan (T017) | M007-T018 documentation update | 0.5.0 |
