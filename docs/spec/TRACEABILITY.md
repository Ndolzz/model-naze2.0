# TRACEABILITY MATRIX — Naze 2.0

**Version:** 1.0.0 (hasil SPEC_REVIEW, 2026-10-06)
Format: Requirement | Architecture | Milestone | Acceptance Criteria | Status

> UNMAPPED = requirement belum punya mapping architecture/milestone (gap — wajib ditindaklanjuti owner).

---

## Functional Requirements

| Requirement | Architecture | Milestone | Acceptance Criteria | Status |
|---|---|---|---|---|
| REQ-001 Tensor/Numerical Core | Stage 1 | M-002 (DONE) | Operasi terdefinisi; unit test lulus; API docs | MAPPED — API docs belum lengkap (ISSUE-002) |
| REQ-002 Neural Network Engine | Stage 1 | M-002 (DONE) | Layer tersusun; forward teruji hand-computed | MAPPED |
| REQ-003 Automatic Differentiation | Stage 2 | M-003 (DONE) | Gradient check lulus; toleransi ditetapkan (1e-5); diuji di CI | MAPPED — CI belum ada (ISSUE-003) |
| REQ-004 Tokenizer | Stage 3 | M-004 (DONE) | Roundtrip lossless; vocab configurable; deterministik | MAPPED — **kontradiksi**: byte-level vocab fixed 256, "vocab configurable" tidak terpenuhi (ISSUE-004) |
| REQ-005 Dataset Pipeline | Stage 4 | M-005 (DONE) | Batch reproducible per-seed; memori terkendali | MAPPED — "cleaning" belum didefinisikan/diimplementasikan (ISSUE-005) |
| REQ-006 Training System | Stage 5–7 | M-006 (DONE minimal) | Trainable/stoppable/resumable; loss menurun sesuai ambang | MAPPED — **ambang loss belum numerik** (ISSUE-006); resume formal belum lengkap (ISSUE-007) |
| REQ-007 Checkpointing | Stage 7 | M-006 (DONE minimal) | Save/load deterministik; **integritas terverifikasi** | MAPPED — verifikasi integritas (hash) belum ada (ISSUE-008) |
| REQ-008 Inference Engine | Stage 8 | M-006 (partial: generate) | Konsisten antar-run; benchmark memori/latensi | MAPPED — benchmark belum ada; Stage 8 penuh = M-009 (PLANNED) |
| REQ-009 Evaluation | Stage 7 | M-008 (PLANNED) | Evaluasi dari checkpoint; format terstruktur | MAPPED (PLANNED) — format hasil masih OPEN |
| REQ-010 First Language Model | Stage 5 | M-006 (DONE) | Trainable end-to-end; generate dari checkpoint | MAPPED — generate dari **checkpoint** belum teruji eksplisit (ISSUE-009) |
| REQ-011 Transformer | Stage 6 | M-007 (PLANNED) | Unit test per komponen; dimensi scalable | MAPPED (PLANNED) |

## Non-Functional Requirements

| Requirement | Architecture | Milestone | Acceptance Criteria | Status |
|---|---|---|---|---|
| REQ-101 Reproducibility | Cross-stage | M-002..M-006 (DONE parsial) | Dokumen reproduksi; config+seed+versi tercatat per run | PARTIAL MAPPED — metadata per-run belum ada (ISSUE-010) |
| REQ-102 Resource Efficiency | Cross-stage | M-006 (DONE parsial) | Batas memori configurable/diuji per modul heavy | PARTIAL MAPPED — metrik memori belum terdefinisi (ISSUE-011) |
| REQ-103 Testability | Cross-stage | M-001..M-006 (DONE parsial) | Suite satu perintah; coverage target | MAPPED — coverage target UNDEFINED (OPEN DECISION-113) |
| REQ-104 Modularity | Cross-stage | M-001/M-002 (DONE) | Dependency antar modul terdokumentasi | MAPPED |
| REQ-105 Documentation | Cross-stage | M-001..M-006 (DONE) | Setiap milestone disertai pembaruan docs | MAPPED |

## Technical & Development Constraints

| Requirement | Architecture | Milestone | Acceptance Criteria | Status |
|---|---|---|---|---|
| REQ-201 No Pretrained Core | Cross-stage (permanen) | Semua | Tidak ada bobot pretrained di jalur core | MAPPED (constraint; enforce via review) |
| REQ-202 No External LLM API Core | Cross-stage (permanen) | Semua | Tidak ada API LLM eksternal di core | MAPPED (constraint) |
| REQ-203 Python + NumPy Foundation | Cross-stage | Semua | Core tanpa framework DL | MAPPED (constraint) |
| REQ-301 Spec-Driven Only | Process | Semua | Tidak ada implementasi sebelum spec disetujui | MAPPED (process) |
| REQ-302 No Overengineering | Process | Semua | Fitur hanya untuk milestone aktif | MAPPED (process) |
| REQ-303 Open Decisions Escalated | Process | Semua | OPEN DECISION tercatat di decision log | MAPPED (process) |
| REQ-304 Change Log | Process | Semua | Changelog requirement up-to-date | MAPPED (process) |

---

## Ringkasan Traceability

- **Total requirement:** 26
- **MAPPED penuh:** 18
- **MAPPED parsial (gap tercatat sebagai ISSUE):** 8
- **UNMAPPED:** 0
- **Kontradiksi requirement-vs-decision:** 1 (REQ-004 ↔ DECISION-010, ISSUE-004)
- Tidak ada milestone tanpa requirement; tidak ada requirement tanpa milestone.
