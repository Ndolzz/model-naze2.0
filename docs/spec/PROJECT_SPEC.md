# PROJECT SPEC — Naze 2.0

**Version:** 0.1.0
**Status:** Approved by Project Owner (Phase 1) — Milestone 2 in review
**Repository:** model-naze2.0

---

## 1. Project Overview

Naze 2.0 adalah proyek membangun model AI dari nol (from scratch), tanpa pretrained model dan tanpa API LLM eksternal sebagai core. Seluruh komponen inti — neural network engine, automatic differentiation, tokenizer, dataset pipeline, training system, dan inference engine — dibuat sendiri menggunakan Python dan NumPy sebagai fondasi komputasi numerik pada tahap awal.

Proyek ini dikelola dengan pendekatan **Spec-Driven Development (SDD)**: spesifikasi adalah source of truth.

## 2. Vision

Membangun core intelligence milik sendiri: model AI yang transparan end-to-end, dipahami penuh dari level numerik hingga arsitektur, berkembang bertahap dari fondasi komputasi neural → model bahasa → multimodal → physical AI / robotics.

## 3. Long-Term Objective

1. **Near term:** Fondasi komputasi neural (numerical core, engine, autodiff) yang solid dan teruji.
2. **Mid term:** Language model pertama yang dapat dilatih dan diinferensikan end-to-end.
3. **Long term:** Transformer buatan sendiri, training system lengkap, rilis Naze 1.0.
4. **Far future:** Multimodal dan physical AI / robotics.

## 4. Current Scope

- Phase 1 (DONE): Spesifikasi proyek disetujui.
- MILESTONE-001 (DONE): Project Foundation — tooling, struktur, README.
- MILESTONE-002 (IN REVIEW): Neural Network Engine (Stage 1) — numerical core, layers, activations, forward pass teruji. Tanpa backprop.

## 5. Out of Scope (Saat Ini)

- Backprop/autodiff (Stage 2 — menunggu OPEN DECISION-105), tokenizer, dataset, training, LM, Transformer, chatbot.
- Pretrained model sebagai core (dilarang permanen). API LLM eksternal sebagai core (dilarang permanen).
- Multimodal / physical AI (ditangguhkan). Optimasi GPU / distribusi training (ditangguhkan).

## 6. Development Philosophy

1. **Specification first.** 2. **No invention** (OPEN DECISION untuk hal yang belum jelas). 3. **Modular dan bertahap.** 4. **Resource efficiency.** 5. **No overengineering.** 6. **Traceability** (changelog + decision log).

## 7. SDD Workflow (Official)

1. Define Specification → 2. Review Specification → 3. Define Architecture → 4. Define Tasks → 5. Implement → 6. Test → 7. Validate Against Specification → 8. Review → 9. Commit → 10. Update Specification if requirements change.

Setiap perubahan requirement wajib dicatat (ID, tanggal, alasan, revisi).

## 8. Technology Constraints

| Constraint | Value |
|---|---|
| Bahasa utama | Python |
| Komputasi numerik tahap awal | NumPy |
| Core model | Dibuat sendiri (from scratch) |
| Pretrained model sebagai core | **Dilarang** (permanen) |
| API LLM eksternal sebagai core | **Dilarang** (permanen) |
| Framework DL eksternal | **Dilarang** untuk core |
| Test runner / linter | pytest + ruff (DECISION-006, ACCEPTED) |
| Dtype / RNG | float64 + seeded RNG (DECISION-007, ACCEPTED) |

## 9. Hardware / Resource Constraints

- Hardware terbatas (consumer-grade); model tahap awal kecil (jutaan parameter).
- Batch size, dataset size, dimensi embedding dapat dikonfigurasi.
- Training harus mendukung checkpoint & resume.
- Benchmark memori/waktu termasuk acceptance criteria stage terkait.

## 10. Milestone Strategy

Milestone kecil, berurutan, dibuka satu per satu setelah milestone sebelumnya disetujui. Detail lihat `docs/tasks/ROADMAP.md` dan `docs/tasks/TASKS.md`.
