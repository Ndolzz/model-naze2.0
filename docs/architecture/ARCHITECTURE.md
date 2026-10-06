# ARCHITECTURE — Naze 2.0 (Staged)

**Version:** 0.3.0

---

## Stage 0 — Project Foundation ✅ (M-001)
Struktur src/tests, pytest+ruff, docs SDD. AC terpenuhi.

## Stage 1 — Neural Network Engine ✅ (M-002)
naze.core.numeric (seeded_rng, as_array, float64), naze.nn.layers (Layer/Linear/Activation/Sequential), naze.nn.activations. Forward terverifikasi hand-computed; init He-normal per-seed.

## Stage 2 — Automatic Differentiation ✅ (M-003, DECISION-009)
Backprop terstruktur per-layer: forward cache input, backward(grad_out) -> grad_input + grads param. Sequential.backward komposisi terbalik. core.gradcheck: numerical central difference, tol 1e-5. AC terpenuhi (gradient check lulus semua operasi).

## Stage 3 — Tokenizer ✅ (M-004, DECISION-010)
Byte-level: vocab 256 tetap, lossless by construction, tanpa training, deterministik. naze.token.ByteTokenizer. AC terpenuhi (roundtrip lossless, deterministik).

## Stage 4 — Dataset Pipeline ✅ (M-005, DECISION-011)
naze.data.TextWindows: sliding window next-token (x=ids[i:i+T], y=ids[i+T]); batch deterministik per-seed; memori terkendali (index, tanpa salin korpus). Korpus final = OPEN DECISION-112. AC terpenuhi.

## Stage 5 — First Language Model ✅ (M-006, DECISION-012)
naze.lm.MLPLM (Bengio-style): ids (B,T) -> E lookup -> flatten -> Linear -> tanh -> Linear -> logits; softmax-CE; sampling greedy+temperature. Gradient check lulus; trainable end-to-end; generation deterministik per-seed. AC terpenuhi.

## Stage 5.5 — Training Minimal ✅ (bagian Stage 7, M-006)
naze.train: SGDTrainer (DECISION-013: SGD saja), checkpoint npz (DECISION-014), save/load deterministik. AC terpenuhi (train→checkpoint→load→output identik).

## Stage 6 — Transformer (PLANNED — M-007)
Scaled dot-product attention, positional encoding, residual blocks, LayerNorm; container non-sekuensial (dibutuhkan nyata di sini); dimensi configurable. Menunggu: persetujuan owner, OPEN DECISION-101 (bila perlu accelerasi), OPEN DECISION-113 (coverage target).

## Stage 7 — Training System penuh (PLANNED — M-008)
Optimizer tambahan bila terukur perlu, logging terstruktur, evaluasi berkala, resume penuh.

## Stage 8 — Inference Engine (PLANNED — M-009)
Forward-only production path, batching, benchmark memori/latensi.

## Stage 9 — Naze 1.0 (PLANNED — M-010, menunggu OPEN DECISION-107)

## Stage 10 — Multimodal / Physical AI (DEFERRED — OPEN DECISION-108)
