# TRACEABILITY MATRIX — Naze 2.0

**Version:** 1.1.0 (SPEC_REVIEW tahap kedua, 2026-10-06)
Format: Requirement | Architecture | Milestone | Acceptance Criteria | Status

> UNMAPPED = requirement belum punya mapping architecture/milestone (gap — wajib ditindaklanjuti owner).

---

## Functional Requirements

| Requirement | Architecture | Milestone | Acceptance Criteria | Status |
|---|---|---|---|---|
| REQ-001 Tensor/Numerical Core | Stage 1 | M-002 (DONE) | Operasi terdefinisi; unit test lulus; API docs | MAPPED — API docs belum ada (ISSUE-002) |
| REQ-002 Neural Network Engine | Stage 1 | M-002 (DONE) | Layer tersusun; forward teruji hand-computed | MAPPED |
| REQ-003 Automatic Differentiation | Stage 2 | M-003 (DONE) | Gradient check lulus; toleransi 1e-5; diuji di CI | MAPPED — CI tidak terdefinisi (ISSUE-003) |
| REQ-004 Tokenizer | Stage 3 | M-004 (DONE) | Roundtrip lossless; vocab configurable; deterministik | MAPPED — **KONTRADIKSI** dengan DECISION-010 (ISSUE-004, MAJOR, belum diputuskan owner) |
| REQ-005 Dataset Pipeline | Stage 4 | M-005 (DONE) | Batch reproducible per-seed; memori terkendali | MAPPED — "cleaning" tidak terdefinisi (ISSUE-005 / OD-115) |
| REQ-006 Training System | Stage 5–7 | M-006 (DONE minimal) | Trainable/stoppable/resumable; loss menurun sesuai ambang | MAPPED — ambang loss tidak numerik (ISSUE-006, MAJOR / OD-116); resume formal belum lengkap (ISSUE-007) |
| REQ-007 Checkpointing | Stage 7 | M-006 (DONE minimal) | Save/load deterministik; integritas terverifikasi | MAPPED — checksum belum ada (ISSUE-008) |
| REQ-008 Inference Engine | Stage 8 | M-006 (partial) → M-009 (PLANNED) | Konsisten antar-run; benchmark memori/latensi | MAPPED — benchmark belum ada |
| REQ-009 Evaluation | Stage 7 (ARCHITECTURE menyebut eval berkala di Stage 7; harness penuh dekat Stage 8) | M-008 (PLANNED) | Evaluasi dari checkpoint; format terstruktur | MAPPED — **ambiguitas mapping stage** (ISSUE-012); format hasil OPEN |
| REQ-010 First Language Model | Stage 5 | M-006 (DONE) | Trainable end-to-end; generate dari checkpoint | MAPPED — generate-from-checkpoint belum teruji eksplisit (ISSUE-009) |
| REQ-011 Transformer | Stage 6 | M-007 (PLANNED) | Unit test per komponen; dimensi scalable | MAPPED (PLANNED) — batas ukuran model belum ada (OD-118) |

## Non-Functional Requirements

| Requirement | Architecture | Milestone | Acceptance Criteria | Status |
|---|---|---|---|---|
| REQ-101 Reproducibility | Cross-stage | M-002..M-006 (parsial) | Dokumen reproduksi; metadata per run | PARTIAL MAPPED — metadata per-run belum ada (ISSUE-010) |
| REQ-102 Resource Efficiency | Cross-stage | M-006 (parsial) | Batas memori configurable/diuji | PARTIAL MAPPED — metrik tidak terdefinisi (ISSUE-011 / OD-117) |
| REQ-103 Testability | Cross-stage | M-001..M-006 (parsial) | Suite satu perintah; coverage target | MAPPED — coverage target OPEN (OD-113) |
| REQ-104 Modularity | Cross-stage | M-001/M-002 (DONE) | Dependency antar modul terdokumentasi | MAPPED |
| REQ-105 Documentation | Cross-stage | M-001..M-006 (DONE) | Docs selaras per milestone | MAPPED |

## Technical & Development Constraints

| Requirement | Architecture | Milestone | Acceptance Criteria | Status |
|---|---|---|---|---|
| REQ-201 No Pretrained Core | Cross-stage (permanen) | Semua | Bebas bobot pretrained di core | MAPPED (constraint) |
| REQ-202 No External LLM API Core | Cross-stage (permanen) | Semua | Bebas API LLM eksternal | MAPPED (constraint) |
| REQ-203 Python + NumPy Foundation | Cross-stage | Semua | Core tanpa framework DL | MAPPED (constraint) |
| REQ-301 Spec-Driven Only | Process | Semua | Implementasi setelah spec disetujui | MAPPED (process) |
| REQ-302 No Overengineering | Process | Semua | Fitur hanya milestone aktif | MAPPED (process) |
| REQ-303 Open Decisions Escalated | Process | Semua | OPEN DECISION tercatat | MAPPED (process) |
| REQ-304 Change Log | Process | Semua | Changelog up-to-date | MAPPED (process) |

---

## Ringkasan Traceability (v1.1.0)

- **Total requirement:** 26 — **ID semua unik** ✅ — semua punya priority, status, dan AC ✅
- **MAPPED penuh:** 18 | **MAPPED parsial:** 8 | **UNMAPPED:** 0
- **Kontradiksi requirement-vs-decision:** 1 (REQ-004 ↔ DECISION-010) — **belum diputuskan owner**
- **Ambiguitas mapping stage:** 1 (REQ-009 ↔ ARCHITECTURE, ISSUE-012 — baru di review ini)
- Tidak ada milestone tanpa requirement; tidak ada requirement tanpa milestone.
