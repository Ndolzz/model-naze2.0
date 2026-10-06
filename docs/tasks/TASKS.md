# TASKS — Naze 2.0

Task didefinisikan per milestone aktif. Sesuai prinsip no overengineering, hanya milestone aktif yang memiliki task detail.

**Format:** `TASK-<milestone-nomor>` | Status: PLANNED / IN-PROGRESS / DONE / BLOCKED
Setiap commit wajib mereferensikan ID task/requirement terkait.

---

## MILESTONE-001 — Project Foundation ✅

| Task | Description | Requirement | Status |
|---|---|---|---|
| TASK-001-01 | Finalisasi struktur direktori proyek (`src/naze/`, `tests/`) | REQ-104 | DONE |
| TASK-001-02 | `pyproject.toml`: metadata + config pytest & ruff | REQ-103, DECISION-006 | DONE |
| TASK-001-03 | Smoke test tooling (bukan test AI) | REQ-103 | DONE |
| TASK-001-04 | README proyek | REQ-105 | DONE |
| TASK-001-05 | `.gitignore` + konvensi commit | REQ-304 | DONE |
| TASK-001-06 | Update ROADMAP + changelog | REQ-304 | DONE |

## MILESTONE-002 — Neural Network Engine (Stage 1)

| Task | Description | Requirement | Status |
|---|---|---|---|
| TASK-002-01 | Numerical core: `seeded_rng`, `as_array`, dtype konvensi | REQ-001 | DONE |
| TASK-002-02 | Layer abstraction: `Layer`, `Linear`, `Activation`, `Sequential` (forward-only) | REQ-002 | DONE |
| TASK-002-03 | Aktivasi: relu, sigmoid (stabil), tanh, softmax (stabil) | REQ-002 | DONE |
| TASK-002-04 | Test suite: determinisme, nilai acuan hand-computed, validasi shape | REQ-103, REQ-101 | DONE |
| TASK-002-05 | Update dokumentasi (decisions, roadmap, changelog) | REQ-105, REQ-304 | DONE |

**Catatan:** Backward pass/gradien TIDAK termasuk M-002 (Stage 2, menunggu OPEN DECISION-105).
