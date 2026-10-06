# DECISION LOG — Naze 2.0

Format per keputusan:
```
DECISION-XXX
Title:
Status:
Context:
Decision:
Reason:
Alternatives:
Consequences:
```

**Status legend:** ACCEPTED / PROPOSED / OPEN DECISION / SUPERSEDED

---

## DECISION-001
- **Title:** Mengadopsi Spec-Driven Development (SDD) sebagai workflow resmi
- **Status:** ACCEPTED
- **Context:** Proyek jangka panjang, multi-stage, dikembangkan pada hardware terbatas; risiko scope creep tinggi.
- **Decision:** Spesifikasi adalah source of truth. Workflow resmi: Define Spec → Review → Architecture → Tasks → Implement → Test → Validate vs Spec → Review → Commit → Update Spec if changed.
- **Reason:** Menjaga fokus, traceability, dan mencegah overengineering.
- **Alternatives:** Agile ad-hoc tanpa spec (ditolak: tidak traceable); waterfall penuh (ditolak: terlalu kaku untuk riset).
- **Consequences:** Perubahan requirement harus melalui proses spec; kecepatan awal sedikit lebih lambat.

## DECISION-002
- **Title:** Python sebagai bahasa utama
- **Status:** ACCEPTED
- **Context:** Perlu bahasa yang cepat dikembangkan, ekosistem kuat, cocok riset ML.
- **Decision:** Python untuk seluruh proyek (core, tooling, test).
- **Reason:** Produktivitas tinggi, ekosistem NumPy/test matang.
- **Alternatives:** C++/Rust (ditolak untuk tahap ini: biaya dev tinggi; dapat dipertimbangkan ulang — lihat OPEN DECISION-101).
- **Consequences:** Performa bergantung NumPy; optimasi level rendah ditunda.

## DECISION-003
- **Title:** NumPy sebagai fondasi komputasi numerik tahap awal
- **Status:** ACCEPTED
- **Context:** Core harus dibuat sendiri; butuh fondasi tensor-matrix yang stabil.
- **Decision:** Semua operasi numerik tahap awal dibangun di atas NumPy. Framework DL eksternal dilarang untuk core.
- **Reason:** Kontrol penuh atas core; NumPy stabil dan cukup cepat untuk model kecil di hardware terbatas.
- **Alternatives:** Pure Python (terlalu lambat); framework DL (melanggar prinsip from-scratch).
- **Consequences:** Batas performa pada model besar; accelerasi = OPEN DECISION-101.

## DECISION-004
- **Title:** Larangan pretrained model & API LLM eksternal sebagai core Naze
- **Status:** ACCEPTED
- **Context:** Visi "core intelligence milik sendiri" yang transparan end-to-end.
- **Decision:** Pretrained model dan API LLM eksternal tidak boleh menjadi core Naze, secara permanen.
- **Reason:** Prinsip project owner.
- **Alternatives:** Fine-tune model open-source (ditolak: melanggar visi).
- **Consequences:** Development dari nol lebih lama; model awal lebih kecil.

## DECISION-005
- **Title:** Arsitektur berkembang bertahap melalui 11 stage (Stage 0–10)
- **Status:** ACCEPTED
- **Context:** Proyek jangka panjang dengan hardware terbatas.
- **Decision:** Arsitektur dibagi Stage 0–10 dengan acceptance criteria per stage; stage hanya dibuka berurutan.
- **Reason:** Modular, dapat divalidasi bertahap, sesuai resource.
- **Alternatives:** Big-bang architecture (ditolak: berisiko dan overengineering).
- **Consequences:** Stage 10 tidak didesain detail sekarang.

## DECISION-006
- **Title:** Tooling dasar: pytest (test runner) + ruff (linter/formater)
- **Status:** ACCEPTED (disetujui project owner, 2026-10-06)
- **Context:** M-001 membutuhkan quality gate minimal (REQ-103).
- **Decision:** pytest untuk test, ruff untuk lint+format, dikonfigurasi via `pyproject.toml`.
- **Reason:** Standar de facto Python, ringan, satu tool untuk lint+format.
- **Alternatives:** unittest; mypy+flake8+black; no-tooling (melanggar REQ-103).
- **Consequences:** Konfigurasi terpusat di `pyproject.toml`; mudah diganti bila perlu.

## DECISION-007
- **Title:** Konvensi dtype float64 dan RNG deterministik per-seed
- **Status:** ACCEPTED (implementasi M-002, 2026-10-06)
- **Context:** REQ-101 Reproducibility; gradient check (Stage 2) sensitif terhadap presisi.
- **Decision:** Seluruh array core menggunakan float64; seluruh randomness WAJIB melalui `seeded_rng(seed)` (np.random.default_rng), tidak melalui np.random global. Input divalidasi menolak NaN/Inf.
- **Reason:** Presisi gradien + determinisme penuh antar-run; validasi input mencegah bug senyap.
- **Alternatives:** float32 (hemat memori, tapi risiko akurasi gradient check; dapat ditinjau ulang untuk inference Stage 8); np.random global (tidak reproducible aman).
- **Consequences:** Memori 2x vs float32 — dapat dikonfigurasi `as_array(x, dtype)` bila milestone inference membutuhkan; tidak ada hidden state global.

## DECISION-008
- **Title:** Abstraksi engine Stage 1: `Layer` + `Sequential`, init He-normal, forward-only
- **Status:** ACCEPTED (implementasi M-002, 2026-10-06)
- **Context:** ARCHITECTURE.md Stage 1 meminta Layer-like abstraction; backward pass ditunda ke Stage 2 (OPEN DECISION-105).
- **Decision:** (1) `Layer` ABC minimal: `forward(x)`, `params: dict[str, Array]`, `parameter_count()`. (2) `Sequential` sebagai container berurutan (cukup untuk Stage 1; container lain menunggu kebutuhan nyata). (3) `Linear` init He/Kaiming-normal: N(0, sqrt(2/fan_in)) via seeded RNG; bias nol. (4) Shape konvensi: `(batch, features)`, Linear mengoperasikan axis terakhir.
- **Reason:** Minimal, modular, tidak overengineered; He-init cocok untuk aktivasi ReLU-like; params sebagai dict sederhana memudahkan checkpointing (REQ-007) nanti.
- **Alternatives:** NamedTuple/Tensor-class penuh (overengineering sebelum Stage 2); init Xavier (kurang cocok untuk ReLU); bias random (tidak perlu).
- **Consequences:** `Sequential` tidak menangani branching/residual — ditambah saat Transformer (Stage 6) benar-benar membutuhkan; interface backward belum ada (sengaja).

---

## OPEN DECISIONS

> Keputusan berikut BELUM ditentukan dan menunggu keputusan project owner. Jangan mengimplementasikan area terkait sebelum diputuskan.

## OPEN DECISION-101
- **Title:** Strategi accelerasi di masa depan (GPU / native extension)
- **Context:** NumPy membatasi performa untuk model lebih besar.
- **Decision:** — (menunggu project owner)
- **Alternatif yang diketahui:** tetap NumPy+CPU; backend CUDA custom; rewrite hot-path di Rust/C++.
- **Dibutuhkan sebelum:** Stage 6+ untuk skala lebih besar.

## OPEN DECISION-102
- **Title:** Jenis tokenizer (byte-level / BPE / word-level / hybrid)
- **Decision:** — (menunggu project owner)
- **Dibutuhkan sebelum:** Stage 3.

## OPEN DECISION-103
- **Title:** Dataset & korpus untuk training (bahasa, ukuran, sumber)
- **Decision:** — (menunggu project owner)
- **Dibutuhkan sebelum:** Stage 3–4.

## OPEN DECISION-104
- **Title:** Arsitektur first language model (Stage 5)
- **Context:** Kandidat: MLP over token context, bigram/trigram-style, RNN sederhana.
- **Decision:** — (menunggu project owner)
- **Dibutuhkan sebelum:** Stage 5.

## OPEN DECISION-105
- **Title:** Pendekatan autodiff (graph-based reverse-mode vs manual backprop terstruktur)
- **Context:** Trade-off kompleksitas implementasi vs fleksibilitas. **Paling mendesak berikutnya.**
- **Decision:** — (menunggu project owner)
- **Dibutuhkan sebelum:** Stage 2 (M-003).

## OPEN DECISION-106
- **Title:** Optimizer (selain SGD) — Adam/AdamW dsb.
- **Decision:** — (menunggu project owner)
- **Dibutuhkan sebelum:** Stage 7.

## OPEN DECISION-107
- **Title:** Definisi sukses Naze 1.0 (metrik, target, dataset evaluasi)
- **Decision:** — (menunggu project owner)
- **Dibutuhkan sebelum:** Stage 9.

## OPEN DECISION-108
- **Title:** Roadmap multimodal / physical AI (Stage 10)
- **Decision:** — (menunggu project owner; jauh di masa depan)
