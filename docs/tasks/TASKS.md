# TASKS — Naze 2.0

**Format:** TASK-<milestone-nomor> | Status: PLANNED / IN-PROGRESS / DONE / BLOCKED

---

## MILESTONE-001 — Project Foundation ✅
TASK-001-01 struktur proyek — DONE
TASK-001-02 pyproject + tooling — DONE
TASK-001-03 smoke test — DONE
TASK-001-04 README — DONE
TASK-001-05 .gitignore + konvensi commit — DONE
TASK-001-06 update roadmap — DONE

## MILESTONE-002 — Neural Network Engine (Stage 1) ✅
TASK-002-01 numerical core — DONE
TASK-002-02 layer abstraction — DONE
TASK-002-03 activations — DONE
TASK-002-04 test suite — DONE
TASK-002-05 update docs — DONE

## MILESTONE-003 — Automatic Differentiation (Stage 2) ✅
TASK-003-01 backward per-layer (Linear/Activation/Sequential) [REQ-003] — DONE
TASK-003-02 core.gradcheck (numeric central difference) [REQ-003] — DONE
TASK-003-03 test gradient check semua operasi [REQ-103] — DONE

## MILESTONE-004 — Tokenizer (Stage 3) ✅
TASK-004-01 ByteTokenizer encode/decode [REQ-004] — DONE
TASK-004-02 test roundtrip ASCII/unicode/emoji [REQ-004] — DONE

## MILESTONE-005 — Dataset Pipeline (Stage 4) ✅
TASK-005-01 TextWindows sliding-window batch [REQ-005] — DONE
TASK-005-02 test determinisme per-seed + validasi [REQ-005] — DONE

## MILESTONE-006 — First LM + Training Minimal ✅
TASK-006-01 MLPLM (embedding + MLP + softmax-CE) [REQ-010] — DONE
TASK-006-02 backward LM + gradient check [REQ-003, REQ-010] — DONE
TASK-006-03 SGDTrainer [REQ-006] — DONE
TASK-006-04 checkpoint save/load [REQ-007] — DONE
TASK-006-05 generate (greedy + temperature) [REQ-010] — DONE
TASK-006-06 integration test end-to-end [REQ-103] — DONE
TASK-006-07 update seluruh docs SDD [REQ-105, REQ-304] — DONE

**Catatan:** Stage 6 (Transformer) ke atas belum dikerjakan — menunggu persetujuan owner (M-007).
