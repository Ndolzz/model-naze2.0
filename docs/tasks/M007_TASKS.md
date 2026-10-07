# M007 TASKS — Transformer Implementation Breakdown

**Milestone:** M-007 (Stage 6) | **Status:** PLANNED — menunggu persetujuan owner atas M007_TECHNICAL_DESIGN.md
**Rule:** task dikerjakan berurutan; setiap commit mereferensikan Task ID + Req ID. **Belum ada implementasi pada tahap ini.**

---

## M007-T001 — Foundation / Configuration
- **Objective:** TransformerConfig (dataclass) dengan validasi: vocab_size==256 (D-015), d_model==num_heads×Dh, dimensi>0, T_max>0, seed.
- **Scope:** konfigurasi + invariant check saja.
- **Dependencies:** — (M-002 engine ada).
- **Files:** src/naze/nn/transformer.py.
- **Notes:** Dh dihitung (d_model // num_heads); default §9 TD.
- **AC:** config invalid ditolak ValueError; valid diterima.
- **Req:** REQ-011, REQ-201..203. **TD:** §9, §10.

## M007-T002 — Embedding
- **Objective:** Embedding layer: lookup (B,T)→(B,T,D); backward scatter-add (seperti MLPLM.E).
- **Dependencies:** T001.
- **Files:** src/naze/nn/transformer.py.
- **AC:** unit: lookup benar; grad-check scatter; ids invalid → ValueError.
- **Req:** REQ-011. **TD:** §5, §7.

## M007-T003 — Positional Representation
- **Objective:** PositionalRepr (default learned table (T_max, D); skema final = OD-121).
- **Dependencies:** T001, T002.
- **Files:** idem.
- **AC:** unit: shape; determinisme per-seed; T > T_max ditolak.
- **Req:** REQ-011, REQ-101. **TD:** §5, §9, OD-121.

## M007-T004 — Q/K/V Projection
- **Objective:** proyeksi Q/K/V memakai Linear (D→D) + reshape (B,T,D)→(B,H,T,Dh).
- **Dependencies:** T001.
- **Files:** idem.
- **AC:** unit: reshape benar; proyeksi = hand-computed kecil.
- **Req:** REQ-011. **TD:** §5 komponen 3-4.

## M007-T005 — Causal Attention
- **Objective:** Attention: scores=QKᵀ/√Dh (B,H,T,T); causal mask (−inf atas diagonal); softmax stabil; context=weights@V.
- **Dependencies:** T004.
- **Files:** idem (+ reuse softmax di activations).
- **AC:** unit: formula hand-computed kecil; mask: kolom masa depan tidak berpengaruh; ekstrem stabil.
- **Req:** REQ-011. **TD:** §5 komponen 5-9, §8 causal requirement.

## M007-T006 — Multi-Head Attention
- **Objective:** MultiHeadAttention: Q/K/V proj → Attention → concat (B,T,D) → output proj W_O; backward + grad-check.
- **Dependencies:** T004, T005.
- **Files:** idem.
- **AC:** unit: H>1 bekerja; kasus uji H=1 ekuivalensi; grad-check (tol 1e-5) pass.
- **Req:** REQ-011, REQ-003. **TD:** §5 komponen 10-13, §11.

## M007-T007 — LayerNorm
- **Objective:** LayerNorm per posisi (gamma, beta); backward + grad-check.
- **Dependencies:** T001.
- **Files:** idem.
- **AC:** unit: mean≈0 var≈1; grad-check pass.
- **Req:** REQ-011, REQ-003. **TD:** §5 komponen 15, §7.

## M007-T008 — Feed-Forward Network
- **Objective:** FeedForward: Linear(D→d_ff) → aktivasi → Linear(d_ff→D); backward via engine.
- **Dependencies:** T001.
- **Files:** idem.
- **AC:** unit: shape; grad-check pass; nilai hand-computed kecil.
- **Req:** REQ-011, REQ-003. **TD:** §5 komponen 16, §9.

## M007-T009 — Residual Block (sub-komponen)
- **Objective:** helper residual + LN: x + f(x) → LN; terverifikasi terpisah.
- **Dependencies:** T006/T007/T008.
- **Files:** idem.
- **AC:** unit: dengan vs tanpa residual berbeda; shape invarian.
- **Req:** REQ-011. **TD:** §5 komponen 14-15.

## M007-T010 — TransformerBlock
- **Objective:** TransformerBlock = [MHA + res+LN] → [FFN + res+LN]; backward menyusul dari komponen.
- **Dependencies:** T006, T008, T009.
- **Files:** idem.
- **AC:** unit: shape invarian (B,T,D); grad-check end-to-end block (jika backward penuh); forward deterministik.
- **Req:** REQ-011. **TD:** §5 komponen 17, §14 risiko-1.

## M007-T011 — Stacked Transformer
- **Objective:** TransformerModel: embedding → pos → blocks × num_layers → final LN; container sesuai engine.
- **Dependencies:** T002, T003, T010.
- **Files:** idem.
- **AC:** unit: L>1 invarian; determinisme per-seed.
- **Req:** REQ-011, REQ-101. **TD:** §5 komponen 18-19.

## M007-T012 — Language-Model Head
- **Objective:** LanguageModelHead: Linear(D→256) (default tidak weight-tied) → logits (B,T,256).
- **Dependencies:** T011.
- **Files:** idem.
- **AC:** unit: dimensi terakhir == 256 (D-015); finite.
- **Req:** REQ-011, REQ-004 (via D-015). **TD:** §5 komponen 20-21, §9.

## M007-T013 — Integration (pipeline LM)
- **Objective:** integrasi: ByteTokenizer → TextWindows → TransformerModel → cross_entropy (adapter reshape (B·T, 256)); adapter generate greedy/temperature.
- **Dependencies:** T012.
- **Files:** src/naze/nn/transformer.py, adapter minimal di naze/lm/ (tanpa merusak MLPLM).
- **AC:** integration: loss finite & terukur menurun pada toy corpus (evaluasi D-016 basis 1-2); generate deterministik per-seed; MLPLM + test lama tidak berubah perilaku.
- **Req:** REQ-010, REQ-006 (parsial), REQ-011. **TD:** §8, §11 integration.

## M007-T014 — Unit Tests
- **Objective:** seluruh unit test §11 (per komponen T002..T012).
- **Dependencies:** per komponen (paralel dengan T002-T012).
- **Files:** tests/test_transformer_units.py.
- **AC:** pytest pass; tiap komponen punya ≥1 test nilai acuan.
- **Req:** REQ-103, REQ-011. **TD:** §11.

## M007-T015 — Numerical Tests
- **Objective:** shape/finiteness/determinisme/mask-correctness/grad-check suite.
- **Dependencies:** T011.
- **Files:** tests/test_transformer_numeric.py.
- **AC:** §11 numerical tests pass (mask test: ubah token masa depan → logits ≤ t tak berubah).
- **Req:** REQ-003, REQ-011, REQ-101. **TD:** §8, §11.

## M007-T016 — Integration Tests
- **Objective:** tokenizer→dataset→model→loss; generate adapter; sanity evaluasi 5-basis D-016 pada toy corpus.
- **Dependencies:** T013.
- **Files:** tests/test_transformer_integration.py.
- **AC:** §11 integration pass.
- **Req:** REQ-004, REQ-005, REQ-010, REQ-011. **TD:** §8, §11, §13.

## M007-T017 — Regression Tests
- **Objective:** verifikasi seluruh suite M-001..M-006 tetap pass; tidak ada perubahan perilaku modul lama.
- **Dependencies:** T013.
- **Files:** tanpa file baru (jalankan pytest penuh); perbaiki bila ada break.
- **AC:** pytest penuh hijau; test lama tidak dimodifikasi (kecuali adapter yang di-spec di T013).
- **Req:** REQ-103, REQ-104. **TD:** §11 regression.

## M007-T018 — Documentation Update
- **Objective:** update TRACEABILITY (status M-007), ROADMAP (M-007 → DONE saat selesai), REQUIREMENTS changelog, SPEC_REVIEW issue status (per PROJECT_SPEC §7 maintenance policy).
- **Dependencies:** T014..T017 selesai.
- **Files:** docs (spec/tasks).
- **AC:** seluruh dokumen selaras; traceability M-007 lengkap.
- **Req:** REQ-105, REQ-304. **TD:** — (PROJECT_SPEC §7).

---

**Urutan kritis:** T001 → {T002, T004} → T003/T005 → T006/T007/T008 → T009 → T010 → T011 → T012 → T013 → {T014, T015, T016, T017} → T018.
