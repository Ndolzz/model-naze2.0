# DECISION LOG — Naze 2.0

**Status legend:** ACCEPTED / PROPOSED / OPEN DECISION / SUPERSEDED

---

## Keputusan Diterima (ACCEPTED)

## DECISION-001 — SDD sebagai workflow resmi — ACCEPTED
Spesifikasi = source of truth. Define Spec → Review → Architecture → Tasks → Implement → Test → Validate → Review → Commit → Update Spec.
Alternatif: ad-hoc (tidak traceable), waterfall (kaku). Konsekuensi: perubahan lewat proses spec.

## DECISION-002 — Python sebagai bahasa utama — ACCEPTED
Alternatif C++/Rust ditolak tahap ini. Performa bergantung NumPy.

## DECISION-003 — NumPy fondasi numerik; framework DL dilarang untuk core — ACCEPTED
Kontrol penuh; stabil untuk model kecil di hardware terbatas.

## DECISION-004 — Larangan pretrained model & API LLM eksternal sebagai core (permanen) — ACCEPTED
Prinsip project owner. Development dari nol lebih lama; model awal kecil.

## DECISION-005 — Arsitektur bertahap Stage 0-10, dibuka berurutan — ACCEPTED
Modular, tervalidasi bertahap. Stage 10 tidak didesain sekarang.

## DECISION-006 — Tooling: pytest + ruff — ACCEPTED (owner, 2026-10-06)

## DECISION-007 — Dtype float64 + seluruh randomness via seeded_rng (deterministik); validasi NaN/Inf — ACCEPTED (M-002)
Presisi gradien + reproducibility. float32 dapat ditinjau untuk inference.

## DECISION-008 — Abstraksi Layer + Sequential, He-normal init, forward-only (Stage 1) — ACCEPTED (M-002)

## DECISION-009 — [RESOLUSI OPEN DECISION-105] Backprop terstruktur per-layer — ACCEPTED (owner delegation, 2026-10-06)
- **Context:** Trade-off graph-based reverse-mode autodiff vs manual backprop terstruktur.
- **Decision:** Backprop terstruktur per-layer: forward meng-cache input; backward(grad_out) menghitung grad_input + grad parameter; Sequential.backward komposisi terbalik. Validasi via numerical gradient check (tol 1e-5, float64).
- **Reason:** Transparan penuh (visi project), sederhana, memori efisien (tanpa graph), cukup untuk arsitektur saat ini. Graph-based autodiff ditambah hanya jika kompleksitas model menuntut.
- **Alternatives:** graph-based reverse-mode (ditunda — overengineering sekarang); finite-difference only (terlalu lambat untuk training).
- **Consequences:** Setiap layer baru wajib mengimplementasikan backward + lulus gradient check.

## DECISION-010 — [RESOLUSI OPEN DECISION-102] Tokenizer: byte-level — ACCEPTED (owner delegation, 2026-10-06)
- **Decision:** Byte-level tokenizer (vocab 256 tetap, lossless by construction, tanpa training, deterministik penuh).
- **Reason:** Paling sederhana yang memenuhi REQ-004; tidak ada keputusan korpus yang menunggu. Lossless dijamin konstruksi.
- **Alternatives:** BPE (lebih pendek sequence tapi butuh training + keputusan vocab size), word-level (OOV problem), hybrid.
- **Consequences:** Sequence lebih panjang (trade-off diketahui); dapat direvisi ke BPE sebelum Stage 6 bila owner memutuskan.

## DECISION-011 — [RESOLUSI OPEN DECISION-103] Dataset: text configurable — ACCEPTED (owner delegation, 2026-10-06)
- **Decision:** Dataset berupa token ID dari teks apa pun (string); TextWindows sliding-window, batch deterministik per-seed, memori terkendali. Korpus final training Naze tetap OPEN (belum diputuskan owner).
- **Consequences:** Pipeline siap dipakai dengan korpus apa pun saat owner memutuskan.

## DECISION-012 — [RESOLUSI OPEN DECISION-104] First LM: MLP over context window (Bengio-style) — ACCEPTED (owner delegation, 2026-10-06)
- **Decision:** ids (B,T) -> embedding lookup -> flatten -> Linear -> tanh -> Linear -> logits. Loss softmax-CE. Sampling greedy + temperature.
- **Reason:** Paling sederhana yang memvalidasi seluruh pipeline end-to-end (REQ-010), sebelum Transformer (Stage 6).
- **Alternatives:** bigram (terlalu lemah untuk validasi pipeline), RNN (backprop through time — kompleksitas belum dibutuhkan).
- **Consequences:** Terbukti trainable end-to-end (loss turun, generation jalan). Kapasitas terbatas — digantikan Transformer di Stage 6.

## DECISION-013 — [RESOLUSI OPEN DECISION-106] Optimizer: SGD saja untuk saat ini — ACCEPTED (owner delegation, 2026-10-06)
- **Reason:** No overengineering. Adam/AdamW ditambah bila SGD terbukti tidak cukup (diukum, bukan dikarang).
- **Consequences:** Konvergensi lebih lambat; acceptable untuk model kecil.

## DECISION-014 — Checkpoint: npz (params + step), save/load deterministik — ACCEPTED (M-006)
- **Reason:** Format stdlib-level, sederhana, cukup untuk REQ-007 saat ini. Format lain (safetensors dsb.) tidak dibutuhkan.

---

## OPEN DECISIONS (menunggu project owner)

## OPEN DECISION-101 — Strategi accelerasi (GPU / native extension)
Dibutuhkan sebelum: Stage 6+ skala besar. Alternatif diketahui: NumPy+CPU; CUDA custom; Rust/C++ hot-path.

## OPEN DECISION-107 — Definisi sukses Naze 1.0 (metrik, target, dataset evaluasi)
Dibutuhkan sebelum: Stage 9.

## OPEN DECISION-108 — Roadmap multimodal / physical AI (Stage 10)
Dibutuhkan sebelum: Stage 10. Jauh di masa depan.

## OPEN DECISION-112 — Korpus final training Naze (bahasa, ukuran, sumber, lisensi)
Dibutuhkan sebelum: training skala serius (pasca-M-006). Pipeline sudah siap menerima korpus apa pun.

## OPEN DECISION-113 — Coverage target test suite
Dibutuhkan sebelum: M-007 (Transformer) — menetapkan ambang coverage formal.

## Riwayat Resolusi

| Open Decision | Resolusi | Tanggal | Cara |
|---|---|---|---|
| OD-105 (autodiff) | DECISION-009: backprop terstruktur per-layer | 2026-10-06 | Delegasi owner ("saya izinkan") |
| OD-102 (tokenizer) | DECISION-010: byte-level | 2026-10-06 | Delegasi owner |
| OD-103 (dataset) | DECISION-011: text configurable (korpus final tetap OPEN-112) | 2026-10-06 | Delegasi owner |
| OD-104 (first LM) | DECISION-012: MLP Bengio-style | 2026-10-06 | Delegasi owner |
| OD-106 (optimizer) | DECISION-013: SGD (Adam ditunda) | 2026-10-06 | Delegasi owner |
