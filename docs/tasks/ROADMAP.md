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

## MILESTONE-008 — Training System lengkap (Stage 7 penuh) — DONE (2026-10-09)
- **Objective:** Training loop penuh sesuai M008_TECHNICAL_DESIGN.md (disetujui
  owner): TrainConfig, checkpoint v2 (checksum + meta, D-018), evaluasi
  berkala (loss/perplexity), run log JSONL, resume penuh, metrik memori
  (D-021), coverage CI >=80% (D-019). Optimizer tetap SGD (no overengineering).
- **Requirements:** REQ-007, REQ-101, REQ-102, REQ-103.
- **Deliverables:** src/naze/train/{config,checkpoint,evaluate,runlog,loop}.py;
  batches_pos (aditif); loss_pos/backward_pos (aditif); CI coverage;
  tests/test_m008_units.py + test_m008_integration.py.
- **Acceptance Criteria:** [x] T001..T010 selesai berurutan; [x] loss train
  menurun; [x] resume eksak deterministik; [x] checkpoint <=1 MB; [x] test
  lama tidak dimodifikasi; [x] CI hijau dengan coverage >=80% (97.75%).

## MILESTONE-009 — Inference Engine (Stage 8) — DONE (2026-10-09)
- **Objective:** Forward-only production path, batching efisien, benchmark memori/latensi untuk MLPLM dan TransformerLM.
- **Requirements:** REQ-008, REQ-101, REQ-102, REQ-103, REQ-104.
- **Deliverables:** src/naze/inference/{batch,engine,benchmark,__init__}.py; tests/test_m009_*.py (unit/numeric/integration/benchmark).
- **Acceptance Criteria:** [x] T001..T016 selesai berurutan; [x] end-to-end encode->batch->forward->generate->decode berjalan; [x] benchmark tokens_per_second > 0 dan peak_rss_kb < 2 GB; [x] test lama tidak dimodifikasi; [x] coverage >=80% (93.40%).
## MILESTONE-010 — Naze 1.0 (Stage 9) — DONE (2026-10-13)
- **Objective:** Training final + verifikasi release gate DECISION-023.
- **Deliverables:** Training 3 epoch pada korpus hybrid (train loss 0.5186, val ppl 1.6915);
  benchmark NazeIO (`scripts/benchmark_nazeio.py` + workflow CI); fine-tuning command dengan
  rehearsal (`scripts/finetune_nazeio.py`: perintah x25 + rehearsal teks umum, block 64, 8 epoch)
  setelah fine-tuning murni terbukti memicu catastrophic forgetting (ppl umum 7.614 pada 500 epoch);
  workflow CI training/benchmark/finetune dengan artifact.
- **Release gate (DECISION-023), model final:** [x] train loss 0.5186 <= 2.5;
  [x] holdout perplexity 1.747 <= 35; [x] NazeIO command accuracy 100% (20/20) >= 90%.
- **Acceptance tambahan:** [x] checkpoint 876 KB <= 1 MB (D-018); CI hijau.
## ID-BASIC-001 — Kemampuan Dasar Bahasa Indonesia — ACTIVE (2026-10-13)
- **Objective:** Meningkatkan kemampuan dasar bahasa Indonesia (kosakata, tata bahasa, struktur kalimat, tanda baca, makna kontekstual) yang terukur, tanpa merusak kapabilitas eksisting (NazeIO 100%, holdout ppl 1.747).
- **Requirements:** REQ-401 (PROPOSED); DECISION-025 (dataset eksternal berlisensi + fallback).
- **Deliverables (rencana):** dataset berlabel 5 kategori + manifest + validator; benchmark tetap; baseline; fine-tuning rehearsal ke checkpoint eksperimen; evaluasi pra/pasca identik.
- **Branch:** feature/id-basic-001 | **TD:** docs/architecture/IDBASIC001_TECHNICAL_DESIGN.md | **Tasks:** docs/tasks/IDBASIC001_TASKS.md
- **Status:** Tahap 0 audit DONE (temuan C1-C8, perbaikan menyusul di branch); Tahap 1 spesifikasi DONE.

## MILESTONE-011+ — Multimodal / Physical AI (Stage 10) — DEFERRED (OPEN DECISION-108)

## Roadmap Change Log

| Date | Milestone | Change | Reason |
|---|---|---|---|
| 2026-10-06 | ALL | Initial draft | Phase 1 |
| 2026-10-06 | M-001 | DONE | Persetujuan owner |
| 2026-10-06 | M-002 | IN REVIEW | Implementasi Stage 1 |
| 2026-10-06 | M-002 | DONE; M-003..M-006 dibuka & selesai | Instruksi owner "kerjakan semuanya"; OPEN DECISION 102-106 diselesaikan via delegasi owner |
| 2026-10-09 | M-007 | DONE | T001..T018 selesai berurutan; CI pytest ditambahkan (T017) |
| 2026-10-09 | M-008 | DONE | T001..T010 selesai berurutan; coverage 97.75%; CI hijau |
| 2026-10-09 | M-009 | PLANNED | TD + task breakdown T001..T016 siap, menunggu persetujuan owner |
| 2026-10-10 | M-010 | ACTIVE; T001..T004 DONE; T005 workflow training dibuat | OD-107/OD-112 resolved (DECISION-023/024); korpus + splits siap |
| 2026-10-13 | M-010 | DONE | Seluruh gate DECISION-023 terpenuhi pada model final: train loss 0.5186, holdout ppl 1.747, akurasi NazeIO 100% (20/20) |
| 2026-10-13 | ID-BASIC-001 | ACTIVE — Tahap 0 audit + Tahap 1 spesifikasi selesai (branch feature/id-basic-001) | Persetujuan owner: REQ-401; dataset eksternal berlisensi (DECISION-025); perbaikan metadata korpus C1-C8 |
