# TRACEABILITY MATRIX — Naze 2.0

**Version:** 1.3.0 (updated for M-007 Technical Design, 2026-10-06)
Chain of traceability (per DECISION-017):
**Requirement → Decision → Architecture → Milestone → Technical Design → Task → Acceptance Criteria**

> UNMAPPED = requirement belum punya mapping. Tidak boleh ada requirement UNMAPPED.
> TD = Technical Design (docs/architecture/M###_TECHNICAL_DESIGN.md, DECISION-017).

---

## Functional Requirements

| Requirement | Decision | Architecture | Milestone | Technical Design | Task | Acceptance Criteria | Status |
|---|---|---|---|---|---|---|---|
| REQ-001 Numerical Core | D-007 | Stage 1 | M-002 (DONE) | — (pra-D-017) | TASK-002-* | Operasi terdefinisi; unit test; API docs (pending ISSUE-002) | MAPPED |
| REQ-002 Neural Engine | D-007, D-008 | Stage 1 | M-002 (DONE) | — | TASK-002-* | Layer tersusun; forward hand-computed | MAPPED |
| REQ-003 Autodiff | D-009 | Stage 2 | M-003 (DONE) | — | TASK-002-*; M007-T006/007/008/015 | Gradient check (1e-5); CI = OD-114 | MAPPED |
| REQ-004 Tokenizer | D-010, D-015 | Stage 3 | M-004 (DONE) | — (M007 TD §9: vocab 256 fixed) | M007-T001/012/013/016 | Roundtrip lossless; deterministik; vocab fixed 256 | MAPPED — ISSUE-004 RESOLVED |
| REQ-005 Dataset | D-011 | Stage 4 | M-005 (DONE) | — | M007-T013/016 | Batch reproducible; memori terkendali; cleaning = OD-115 | MAPPED |
| REQ-006 Training | D-013, D-016 | Stage 5–7 | M-006 (DONE minimal) → M-008 | M008_TD (upcoming) | M007-T013/016 (sanity); M008 tasks | AC evaluasi 5-basis (D-016); resume penuh = M-008 | MAPPED — ISSUE-006 RESOLVED |
| REQ-007 Checkpointing | D-014 | Stage 7 | M-006 (DONE minimal) → M-008 | M008_TD (upcoming) | M008 tasks | Save/load deterministik; checksum+rng-state = M-008 | MAPPED |
| REQ-008 Inference | — | Stage 8 | M-006 (partial) → M-009 | M009_TD (upcoming) | M009 tasks | Konsisten antar-run; benchmark (M-009) | MAPPED |
| REQ-009 Evaluation | D-016 | Stage 7 (harness; benchmark = Stage 8) | M-008 (PLANNED) | M008_TD (upcoming) | M008 tasks; M007-T016 sanity | 5-basis evaluasi (D-016); format hasil OPEN di TD M-008 | MAPPED |
| REQ-010 First LM | D-012, D-016 | Stage 5 | M-006 (DONE) | M007 TD §8 (integrasi pipeline) | M007-T013/016 | Trainable end-to-end (basis 1-2-3); generate deterministik; generate-from-checkpoint = M-008 | MAPPED |
| REQ-011 Transformer | D-009 (backward per-layer), D-015 (vocab 256), D-016 (evaluasi) | Stage 6 | M-007 (TECHNICAL DESIGN READY) | **M007_TECHNICAL_DESIGN.md** | **M007-T001..T018** | TD §13: 17 AC objective (shapes, causal mask, MHA, residual, LayerNorm, FFN, stacking, logits 256, pipeline, from-scratch, regression, determinisme, resource docs); batas ukuran = OD-118 | MAPPED |

## Non-Functional Requirements

| Requirement | Decision | Architecture | Milestone | Technical Design | Task | Acceptance Criteria | Status |
|---|---|---|---|---|---|---|---|
| REQ-101 Reproducibility | D-007 | Cross-stage | M-002..M-006 parsial → M-008 | M007 TD §9/§11 (determinisme per-seed) | M007-T015 | Metadata per-run = M-008; determinisme diuji | MAPPED |
| REQ-102 Resource Efficiency | D-007 | Cross-stage | M-006 parsial → M-008 | **M007 TD §12 (estimasi param/memori)** | M007-T001 (config) | Metrik memori = OD-117; estimasi resource terdokumentasi | MAPPED |
| REQ-103 Testability | D-006 | Cross-stage | M-001..M-006 parsial | M007 TD §11 | M007-T014..T017 | Coverage target = OD-113 (DEFERRED ke implementasi) | MAPPED |
| REQ-104 Modularity | D-008 | Cross-stage | M-001/M-002 (DONE) | **M007 TD §6 (module boundaries)** | semua M007 task | Dependency direction: train/lm → nn → core | MAPPED |
| REQ-105 Documentation | D-017 | Cross-stage | M-001..M-006 (DONE) | M007 TD (dokumen itu sendiri) | M007-T018 | Docs selaras per milestone; maintenance policy PROJECT_SPEC §7 | MAPPED |

## Technical & Development Constraints

| Requirement | Decision | Architecture | Milestone | Technical Design | Task | Acceptance Criteria | Status |
|---|---|---|---|---|---|---|---|
| REQ-201 No Pretrained Core | D-004 | Cross-stage (permanen) | Semua | M007 TD §5/§13 (AC 11) | semua | Enforce di Review | MAPPED |
| REQ-202 No External LLM API | D-004 | Cross-stage (permanen) | Semua | M007 TD §13 (AC 12) | semua | idem | MAPPED |
| REQ-203 Python + NumPy | D-002, D-003 | Cross-stage | Semua | M007 TD §5/§13 (AC 13) | semua | idem | MAPPED |
| REQ-301 Spec-Driven Only | D-001 | Process | Semua | M007 TD dibuat sebelum task/impl | M007-T001.. | Workflow §7 | MAPPED |
| REQ-302 No Overengineering | D-005 | Process | Semua | M007 TD §3 Non-goals (dropout, attention efisien, LM head penuh ditunda) | — | Fitur = milestone aktif | MAPPED |
| REQ-303 Open Decisions Escalated | D-001 | Process | Semua | M007 TD §15 (OD-118, OD-113, OD-114, OD-121) | — | OD tercatat | MAPPED |
| REQ-304 Change Log | D-001 | Process | Semua | — | M007-T018 | Changelog up-to-date | MAPPED |

---

## Ringkasan Traceability (v1.3.0)

- **Total requirement:** 26 — **UNMAPPED: 0** — kontradiksi: 0
- **M-007 kini terlacak penuh:** REQ-011 → D-009/015/016 → Stage 6 → M-007 → M007_TECHNICAL_DESIGN.md → M007-T001..T018 → 17 AC objective.
- OD baru dari TD M-007: OD-121 (skema positional; default learned).
- Maintenance policy dipatuhi: TD baru → TRACEABILITY diperbarui (PROJECT_SPEC §7).
