# ROADMAP — Naze 2.0

**Format per milestone:**
```
MILESTONE-XXX
Name:
Objective:
Dependencies:
Requirements:
Deliverables:
Acceptance Criteria:
Status:
```
**Status legend:** PLANNED / ACTIVE / IN REVIEW / DONE / BLOCKED

> Milestone didefinisikan hanya saat dibuka, sesuai prinsip no overengineering. Outline bawah hanya indikatif.

---

## MILESTONE-001 — Project Foundation ✅
- **Objective:** Struktur repo, tooling (pytest + ruff, DECISION-006 ACCEPTED), dokumentasi SDD. Tanpa kode AI.
- **Status:** DONE (disetujui project owner, 2026-10-06)

---

## MILESTONE-002
- **Name:** Neural Network Engine (Stage 1)
- **Objective:** Fondasi komputasi neural: numerical core deterministik, layer modular, aktivasi, dan forward pass teruji — tanpa backprop (Stage 2).
- **Dependencies:** MILESTONE-001 (DONE).
- **Requirements:** REQ-001, REQ-002, REQ-101, REQ-103, REQ-104
- **Deliverables:**
  - `naze.core.numeric` — seeded RNG deterministik, validasi array, konvensi dtype (DECISION-007).
  - `naze.nn.layers` — `Layer`, `Linear`, `Activation`, `Sequential` (forward-only, DECISION-008).
  - `naze.nn.activations` — relu, sigmoid stabil, tanh, softmax stabil.
  - Test suite: determinisme per-seed, nilai acuan hand-computed, validasi shape/input.
- **Acceptance Criteria:**
  - [x] Semua operasi teruji; unit test lulus (numeric, activations, layers).
  - [x] Forward pass model susunan-layer sederhana terverifikasi numerik (hand-computed).
  - [x] Inisialisasi parameter deterministik per-seed (reproducible).
  - [ ] Test suite dijalankan & lulus di lingkungan project owner (`pytest`).
- **Status:** IN REVIEW — menunggu verifikasi owner (jalankan `pytest` + `ruff check .`).

---

## Outline Indikatif (belum didefinisikan detail — jangan dikerjakan)

| Milestone (indikatif) | Kaitan Stage | Catatan |
|---|---|---|
| M-003 | Stage 2 | Automatic Differentiation — menunggu OPEN DECISION-105 |
| M-004 | Stage 3 | Tokenizer — menunggu OPEN DECISION-102 |
| M-005 | Stage 4 | Dataset Pipeline — menunggu OPEN DECISION-103 |
| M-006 | Stage 5 | First Language Model — menunggu OPEN DECISION-104 |
| M-007 | Stage 6 | Transformer |
| M-008 | Stage 7 | Training System |
| M-009 | Stage 8 | Inference Engine |
| M-010 | Stage 9 | Naze 1.0 — menunggu OPEN DECISION-107 |
| M-011+ | Stage 10 | Multimodal / Physical AI — OPEN DECISION-108 |

## Roadmap Change Log

| Date | Milestone | Change | Reason |
|---|---|---|---|
| 2026-10-06 | ALL | Initial draft | Phase 1 — Specification Only |
| 2026-10-06 | M-001 | Deliverables dibuat; status → IN REVIEW | Lanjut ke Define Tasks + Implement (M-001, non-AI) |
| 2026-10-06 | M-001 | Status → DONE | Persetujuan project owner ("buat"), DECISION-006 ACCEPTED |
| 2026-10-06 | M-002 | Milestone dibuka & diimplementasikan; status → IN REVIEW | Instruksi project owner ("buat") |
