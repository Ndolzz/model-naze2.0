# TRACEABILITY MATRIX — Naze 2.0

> Perubahan vs GitHub (`81153d4`, v1.3.0): reconstruction notice di bagian atas + koreksi ringkasan "Total requirement: 26" → **23** (keputusan owner 2026-10-07). Seluruh baris matriks tidak berubah.

> **[RECONSTRUCTION NOTICE — v1.3.1, 2026-10-07]**
> DECISION_LOG (DECISION-015/016/017), REQUIREMENTS v0.4.0, PROJECT_SPEC v0.4.0, dan SPEC_REVIEW v1.2.0 yang dirujuk matriks ini adalah **RECONSTRUCTED GOVERNANCE** (direkonstruksi 2026-10-07 setelah versi asli hilang dan tidak pernah di-commit). Substansi keputusan konsisten dengan matriks ini; wording dokumen tersebut bukan teks asli. Baris matriks identik dengan v1.3.0 (commit `81153d4`).
> **Koreksi jumlah requirement (owner decision, 2026-10-07):** ringkasan v1.3.0 menyebut "Total requirement: 26" — kesalahan hitung pra-existing (sejak v1.1.0). Verifikasi git history (REQUIREMENTS v0.1.0 & v0.3.0) membuktikan jumlah REQ ID unik = **23** (REQ-001..011, REQ-101..105, REQ-201..203, REQ-301..304), sama dengan jumlah baris matriks. Tidak ada REQ baru yang dibuat untuk mencapai angka lama.

**Version:** 1.4.0 (M-007 implementasi DONE: status milestone/task diperbarui, 2026-10-09; baris matriks = v1.3.0; reconstruction notice = v1.3.1, 2026-10-07)
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
| REQ-007 Checkpointing | D-014 | Stage 7 | M-006 (DONE minimal) → M-008 | **M008_TECHNICAL_DESIGN.md** | **M008-T001..T002 (DONE)** | Save/load deterministik; checksum+rng-state = M-008; checkpoint v2 | MAPPED |
| REQ-008 Inference | — | Stage 8 | M-006 (partial) → M-009 | **M009_TECHNICAL_DESIGN.md** | **M009-T001..T016 (DONE)** | Konsisten antar-run; benchmark (M-009); forward-only path, batching, sliding window | MAPPED |
| REQ-009 Evaluation | D-016 | Stage 7 (harness; benchmark = Stage 8) | M-008 (DONE) | **M008_TECHNICAL_DESIGN.md** | **M008-T001..T010 (DONE)** | 5-basis evaluasi (D-016); format hasil di M008-T003; benchmark inference = M-009 | MAPPED |
| REQ-010 First LM | D-012, D-016 | Stage 5 | M-006 (DONE) | M007 TD §8 (integrasi pipeline) | M007-T013/016 | Trainable end-to-end (basis 1-2-3); generate deterministik; generate-from-checkpoint = M-008 | MAPPED |
| REQ-011 Transformer | D-009 (backward per-layer), D-015 (vocab 256), D-016 (evaluasi) | Stage 6 | M-007 (**DONE 2026-10-09**, T001..T018) | **M007_TECHNICAL_DESIGN.md** | **M007-T001..T018 (selesai)** | TD §13: 17 AC objective — tervalidasi via unit/numeric/integration tests; pytest penuh dieksekusi via CI (T017, `.github/workflows/tests.yml`); batas ukuran = OD-118 | MAPPED — IMPLEMENTED |

## Non-Functional Requirements

| Requirement | Decision | Architecture | Milestone | Technical Design | Task | Acceptance Criteria | Status |
|---|---|---|---|---|---|---|---|
| REQ-101 Reproducibility | D-007 | Cross-stage | M-002..M-006 parsial → M-008 | M007 TD §9/§11 (determinisme per-seed) | M007-T015 (teruji: determinisme logits/attention/generate per-seed) | Metadata per-run = M-008; determinisme diuji | MAPPED |
| REQ-102 Resource Efficiency | D-007 | Cross-stage | M-006 parsial → M-008 | **M007 TD §12 (estimasi param/memori)** | M007-T001 (config) | Metrik memori = OD-117; estimasi resource terdokumentasi | MAPPED |
| REQ-103 Testability | D-006 | Cross-stage | M-001..M-007 (DONE) | M007 TD §11 | M007-T014..T017 (selesai; CI workflow ditambah T017) | Coverage target = OD-113 (DEFERRED ke implementasi; suite penuh kini via CI) | MAPPED |
| REQ-104 Modularity | D-008 | Cross-stage | M-001/M-002 (DONE) | **M007 TD §6 (module boundaries)** | semua M007 task | Dependency direction: train/lm → nn → core | MAPPED |
| REQ-105 Documentation | D-017 | Cross-stage | M-001..M-006 (DONE) | M007 TD (dokumen itu sendiri) | M007-T018 (selesai 2026-10-09) | Docs selaras per milestone; maintenance policy PROJECT_SPEC §7 | MAPPED |

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

## Ringkasan Traceability

- **Total requirement: 23** *(dikoreksi dari "26" — kesalahan hitung pra-existing; keputusan owner 2026-10-07)* — **23/23 MAPPED** — **UNMAPPED: 0** — kontradiksi: 0
- **M-007 IMPLEMENTED (2026-10-09):** REQ-011 → D-009/015/016 → Stage 6 → M-007 (DONE) → M007_TECHNICAL_DESIGN.md → M007-T001..T018 (selesai) → 17 AC objective tervalidasi via 3 suite test baru + CI pytest penuh.
- **M-008 DONE (2026-10-09):** REQ-007/009 → D-014/016/018/019/020/021 → Stage 7 → M-008 (DONE) → M008_TECHNICAL_DESIGN.md → M008-T001..T010 (selesai) → coverage 97.75%, CI hijau.
- **M-009 DONE (2026-10-09):** REQ-008 → Stage 8 → M-009 (DONE) → M009_TECHNICAL_DESIGN.md → M009-T001..T016 (selesai) → forward-only path, batching, benchmark, coverage 93.40%.
- OD tersisa pasca-M-009: OD-101 (accelerasi), OD-108 (multimodal), OD-115 (cleaning), OD-119 (release policy), OD-122..126 (M-009 open decisions). OD-107 **RESOLVED** (DECISION-023, ACCEPTED). OD-112 **RESOLVED** (DECISION-024, APPROVED).
- Maintenance policy dipatuhi: implementasi selesai → TRACEABILITY diperbarui (PROJECT_SPEC §7).

| **REQ-107 Definisi Sukses Naze 1.0** | **D-023, D-024** | **Stage 9** | **M-010 (IMPLEMENTING)** | **M010_TECHNICAL_DESIGN.md** | **M010-T001..T020** | Target: training loss <= 2.5, validation perplexity <= 35, NazeIO accuracy >= 90%. Corpus hybrid ~7 MB (DECISION-024). | MAPPED |
| **REQ-105 Documentation (M-010)** | **D-017** | **Stage 9** | **M-010 (IMPLEMENTING)** | **M010_TECHNICAL_DESIGN.md** | **M010-T011..T014, T018..T020** | API reference, tutorial, changelog, README update | MAPPED |

