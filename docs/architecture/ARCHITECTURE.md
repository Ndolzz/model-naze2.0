# ARCHITECTURE — Naze 2.0 (Staged)

**Version:** 0.2.0
**Note:** Setiap stage hanya dibuka setelah stage sebelumnya lulus acceptance criteria dan disetujui project owner.

---

## Stage 0 — Project Foundation ✅ (M-001, DONE)
- **Purpose:** Repo, tooling, quality gate, konvensi SDD.
- **Outputs:** Struktur `src/naze/`, `tests/`, pytest+ruff (DECISION-006), README, docs SDD.
- **Acceptance Criteria:** [x] Tooling berjalan satu perintah; [x] Docs SDD lengkap & disetujui.

## Stage 1 — Neural Network Engine 🔶 (M-002, IN REVIEW)
- **Purpose:** Fondasi komputasi neural: numerical core deterministik, layer, aktivasi, forward pass.
- **Inputs:** NumPy (REQ-001).
- **Outputs:** `naze.core.numeric` (seeded_rng, as_array, float64), `naze.nn.layers` (Layer/Linear/Activation/Sequential), `naze.nn.activations` (relu/sigmoid/tanh/softmax stabil).
- **Dependencies:** Stage 0.
- **Interfaces:** `Layer.forward(x)`; `Sequential` container; `params: dict[str, Array]`; konvensi shape `(batch, features)`; konvensi init He-normal per-seed (DECISION-007, DECISION-008).
- **Constraints:** NumPy only; forward-only (backward = Stage 2); efisien memori; input divalidasi (tolak NaN/Inf/bad shape).
- **Acceptance Criteria:** [x] Operasi teruji; [x] Forward MLP sederhana terverifikasi hand-computed; [x] Init deterministik per-seed; [ ] pytest lulus di lingkungan owner.

## Stage 2 — Automatic Differentiation (PLANNED)
- **Purpose:** Gradien (backprop) agar network dapat dilatih.
- **Inputs:** Struktur operasi dari Stage 1.
- **Outputs:** Gradien per-parameter, tervalidasi numerical gradient check.
- **Dependencies:** Stage 1.
- **Interfaces:** Backward pass per operasi/layer (spec).
- **Constraints:** Toleransi error gradient check ditetapkan; pendekatan = OPEN DECISION-105 (paling mendesak).
- **Acceptance Criteria:** Gradient check lulus untuk seluruh operasi yang didukung.

## Stage 3 — Tokenizer (PLANNED — OPEN DECISION-102)
- **Purpose:** Teks ↔ token ID, dibuat sendiri. Roundtrip lossless, vocab configurable, deterministik. Dependencies: Stage 0/core.

## Stage 4 — Dataset Pipeline (PLANNED — OPEN DECISION-103)
- **Purpose:** Batch data terkontrol, reproducible per-seed, memori terkendali (streaming bila perlu). Dependencies: Stage 3.

## Stage 5 — First Language Model (PLANNED — OPEN DECISION-104)
- **Purpose:** LM sederhana end-to-end (train → generate) memvalidasi seluruh pipeline. Dependencies: Stage 1, 2, 4. Model kecil (jutaan parameter).

## Stage 6 — Transformer (PLANNED)
- **Purpose:** Transformer dari nol (scaled dot-product attention, positional encoding, residual blocks). Dependencies: Stage 2, 4. Dimensi configurable, dapat diskalakan turun. Attention efisien konteks panjang: OPEN DECISION (ditangguhkan). Container non-sekuensial (residual) ditambahkan di sini, bukan sekarang.

## Stage 7 — Training System (PLANNED — OPEN DECISION-106)
- **Purpose:** Optimizer (SGD minimal), checkpoint, resume deterministik, logging, evaluasi berkala. Dependencies: Stage 5–6.

## Stage 8 — Inference Engine (PLANNED)
- **Purpose:** Forward-only: sampling (minimal greedy + temperature), batching, benchmark memori/latensi. Dependencies: Stage 6, 7.

## Stage 9 — Naze 1.0 (PLANNED — OPEN DECISION-107)
- **Purpose:** Integrasi rilis pertama: model terlatih, inference, evaluasi, packaging, public API stabil.

## Stage 10 — Future Multimodal / Physical AI (DEFERRED — OPEN DECISION-108)
- **Purpose:** Ekspansi modality (vision, audio, kontrol robotik). Tidak didesain detail sekarang; tidak ada pekerjaan aktif.
