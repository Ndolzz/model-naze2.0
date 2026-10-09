# ROADMAP — Naze 2.0

**Status legend:** PLANNED / ACTIVE / IN REVIEW / DONE / BLOCKED

---

## MILESTONE-001 — Project Foundation ✅ DONE (2026-10-06)
Struktur repo, pytest+ruff, README, docs SDD.

## MILESTONE-002 — Neural Network Engine (Stage 1) ✅ DONE (2026-10-06)
Numerical core, layers, activations, forward pass teruji hand-computed. Commit 8bdb1aa.

## MILESTONE-003 — Automatic Differentiation (Stage 2) ✅ DONE (2026-10-06)
- **Objective:** Gradien via backprop terstruktur per-layer (DECISION-009), tervalidasi numerical gradient check.
- **Requirements:** REQ-003
- **Deliverables:** backward pada Layer/Linear/Activation/Sequential; core.gradcheck; test gradient check.
- **Acceptance Criteria:** [x] Gradient check lulus semua operasi; [x] backward-before-forward guarded.
- Commit 3432b73.

## MILESTONE-004 — Tokenizer (Stage 3) ✅ DONE (2026-10-06)
- **Objective:** Byte-level tokenizer (DECISION-010).
- **Requirements:** REQ-004
- **Deliverables:** naze.token.ByteTokenizer; test roundtrip ASCII/unicode.
- **Acceptance Criteria:** [x] Roundtrip lossless; [x] deterministik; [x] vocab fixed 256.
- Commit 3432b73.

## MILESTONE-005 — Dataset Pipeline (Stage 4) ✅ DONE (2026-10-06)
- **Objective:** Sliding-window batch next-token, deterministik per-seed (DECISION-011).
- **Requirements:** REQ-005
- **Deliverables:** naze.data.TextWindows; test determinisme/shape/validasi.
- **Acceptance Criteria:** [x] Batch reproducible per-seed; [x] memori terkendali (index saja).
- Commit eb412ca.

## MILESTONE-006 — First Language Model + Training Minimal (Stage 5 + sebagian Stage 7) ✅ DONE (2026-10-06)
- **Objective:** LM pertama trainable end-to-end: MLP Bengio-style (DECISION-012), SGD (DECISION-013), checkpoint (DECISION-014), generation greedy+temperature.
- **Requirements:** REQ-006, REQ-007, REQ-010
- **Deliverables:** naze.lm (MLPLM, cross_entropy, generate); naze.train (SGDTrainer, save/load checkpoint); test end-to-end (loss turun, generation deterministik, checkpoint roundtrip, integrasi byte tokenizer).
- **Acceptance Criteria:** [x] Gradient check LM lulus; [x] loss menurun pada toy corpus; [x] generation valid & deterministik; [x] checkpoint save/load identik; [x] encode→train→generate→decode berjalan.
- Commits eb412ca, aba7baa.

## MILESTONE-007 — Transformer (Stage 6) ✅ DONE (2026-10-09)
- **Objective:** Decoder-only causal Transformer sesuai M007_TECHNICAL_DESIGN.md
  (21 komponen, invariant D = H × Dh, vocab 256 fixed).
- **Requirements:** REQ-011 (utama); REQ-003/004/005/010/101/103/104/105 pendukung.
- **Deliverables:** src/naze/nn/transformer.py (config sampai TransformerModel +
  LanguageModelHead); adapter src/naze/lm/transformer_lm.py (TransformerLM,
  transformer_generate); 3 file test baru (unit/numeric/integration);
  workflow CI .github/workflows/tests.yml (pytest penuh pada push/PR).
- **Acceptance Criteria:** [x] T001..T018 selesai berurutan; [x] grad-check tol
  1e-5 lulus komponen backward; [x] mask correctness teruji level logits dan
  attention; [x] evaluasi sanity 5-basis D-016 lulus; [x] test lama tidak
  dimodifikasi; [x] pytest penuh dieksekusi via CI (hasil run: tab Actions).
- **Open decisions tersisa:** OD-113 (coverage target), OD-114 (CI enforcement
  formal), OD-118 (batas ukuran untuk training skala serius — bukan blocker
  dev/test), OD-121 (skema positional; default learned).
- Commits: 4e1b7daf (T001..T012), 6b676b24 (T013), 0ea18004 (T014),
  d3d24e9c (T015), ee386b5b (T016), 938cdc5f (T017), T018 (dokumentasi).

## MILESTONE-008 — Training System lengkap (Stage 7 penuh) — PLANNED
Optimizer tambahan (bila terukur perlu), logging terstruktur, evaluasi berkala, resume penuh.

## MILESTONE-009 — Inference Engine (Stage 8) — PLANNED
## MILESTONE-010 — Naze 1.0 (Stage 9) — PLANNED (menunggu OPEN DECISION-107)
## MILESTONE-011+ — Multimodal / Physical AI (Stage 10) — DEFERRED (OPEN DECISION-108)

## Roadmap Change Log

| Date | Milestone | Change | Reason |
|---|---|---|---|
| 2026-10-06 | ALL | Initial draft | Phase 1 |
| 2026-10-06 | M-001 | DONE | Persetujuan owner |
| 2026-10-06 | M-002 | IN REVIEW | Implementasi Stage 1 |
| 2026-10-06 | M-002 | DONE; M-003..M-006 dibuka & selesai | Instruksi owner "kerjakan semuanya"; OPEN DECISION 102-106 diselesaikan via delegasi owner |
| 2026-10-09 | M-007 | DONE | T001..T018 selesai berurutan; CI pytest ditambahkan (T017) |
