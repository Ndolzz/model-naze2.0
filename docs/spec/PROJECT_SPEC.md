# PROJECT SPEC — Naze 2.0

> **[RECONSTRUCTED GOVERNANCE — 2026-10-07]**
> Dokumen ini adalah RECONSTRUCTED GOVERNANCE, bukan pemulihan file asli. Versi lokal PROJECT_SPEC v0.4.0 hilang dan tidak pernah di-commit.
> **Basis rekonstruksi:** versi GitHub commit `b53259b` (v0.3.0 — verbatim dari git history) + pembaruan governance (workflow 9 tahap eksplisit + maintenance policy) sesuai DECISION-017 dan TRACEABILITY v1.3.0 / M007_TASKS (T018).
> Bagian revisi ditandai **[RECONSTRUCTED]**. Wording revisi BUKAN teks asli yang hilang.

**Version:** 0.4.0 | **Status:** Milestones 1-6 complete; M-007 technical design + task breakdown complete — READY FOR IMPLEMENTATION (menunggu persetujuan owner)

## 1. Project Overview
Model AI dari nol: tanpa pretrained model, tanpa API LLM eksternal sebagai core. Seluruh komponen (engine, autodiff, tokenizer, dataset, training, inference) dibuat sendiri dengan Python + NumPy. SDD: spesifikasi = source of truth.

## 2. Vision
Core intelligence milik sendiri, transparan end-to-end: fondasi neural -> model bahasa -> multimodal -> physical AI/robotics.

## 3. Long-Term Objective
1. Near: fondasi komputasi neural solid (TERCAPAI M-001..M-003).
2. Mid: LM pertama trainable end-to-end (TERCAPAI M-006, versi MLP kecil).
3. Long: Transformer sendiri + training lengkap + rilis Naze 1.0.
4. Far: multimodal & physical AI.

## 4. Current Scope [RECONSTRUCTED — diperbarui untuk governance M-007]
- Pipeline END-TO-END berfungsi: encode teks (byte tokenizer) -> batch dataset -> train MLP LM (SGD) -> checkpoint -> generate teks (greedy/temperature), seluruhnya deterministik per-seed dan teruji.
- Governance M-007 selesai: DECISION-015 (vocab fixed 256), DECISION-016 (evaluasi 5-basis), DECISION-017 (format Technical Design) — ISSUE-004, ISSUE-006/OD-116, ISSUE-013/OD-120 resolved.
- Technical Design M-007 selesai: `docs/architecture/M007_TECHNICAL_DESIGN.md` (15 section sesuai DECISION-017). Task Breakdown M-007 selesai: `docs/tasks/M007_TASKS.md` (M007-T001..T018).
- M-007 DONE: Transformer (Stage 6) terimplementasi penuh.
- M-008 DONE: Training System lengkap (Stage 7) terimplementasi penuh.
- M-009 DONE: Inference Engine (Stage 8) terimplementasi penuh.

## 5. Out of Scope Saat Ini [RECONSTRUCTED — Transformer keluar dari out-of-scope]
Naze 1.0 (M-010), multimodal (M-011+). Larangan permanen: pretrained core, API LLM core.

## 6. Development Philosophy
Spec first; no invention (OPEN DECISION); modular bertahap; resource efficiency; no overengineering; traceability.

## 7. SDD Workflow (Official — 9 tahap eksplisit) [RECONSTRUCTED — direvisi per DECISION-017; menutup ISSUE-013 & ISSUE-014]

1. **Specification** — REQUIREMENTS.md = source of truth; setiap perubahan lewat Requirement Change Log (REQ-304).
2. **Spec Review** — SPEC_REVIEW.md; isu MAJOR/blocking menjadi kondisi wajib sebelum tahap berikutnya.
3. **Technical Design** — satu dokumen per milestone: `docs/architecture/M###_TECHNICAL_DESIGN.md`, 15 section wajib (DECISION-017): Purpose, Scope, Non-goals, Requirements, Architecture, Module Boundaries, Interfaces, Data Flow, Configuration, Error Handling, Testing Strategy, Resource Constraints, Acceptance Criteria, Risks, Open Decisions.
4. **Task Breakdown** — `docs/tasks/M###_TASKS.md`; urutan kritis eksplisit; setiap commit implementasi mereferensikan Task ID + Req ID.
5. **Implementation** — hanya per task yang didefinisikan; tanpa task, tanpa kode.
6. **Testing** — unit/numerical/integration/regression sesuai Testing Strategy TD (§11).
7. **Validation** — hasil vs Acceptance Criteria TD (§13); evaluasi kualitas training via kerangka 5-basis (DECISION-016) — tanpa ambang loss numerik universal.
8. **Review** — konsistensi requirement ↔ implementasi; larangan permanen REQ-201/202/203 diverifikasi.
9. **Commit** — pesan commit mereferensikan Task ID + Req ID; pembaruan dokumentasi terjadi pada tahap ini.

**Governance maintenance policy (menutup ISSUE-014):**
- TRACEABILITY.md wajib diperbarui setiap kali REQUIREMENTS berubah dan setiap kali Technical Design baru dibuat/selesai.
- SPEC_REVIEW.md wajib diperbarui setiap kali status isu / open decision berubah.
- DECISION_LOG.md menerima setiap keputusan baru (accepted maupun resolusi open decision).
- Requirement Change Log REQUIREMENTS.md wajib untuk setiap perubahan requirement.

## 8. Technology Constraints
Python; NumPy; core from scratch (pretrained/API LLM dilarang permanen; framework DL dilarang); pytest+ruff; float64+seeded RNG; backprop per-layer (D-009); tokenizer byte-level vocab fixed 256 (D-010, D-015); SGD (D-013); evaluasi 5-basis tanpa ambang numerik universal (D-016); Technical Design per milestone `docs/architecture/M###_TECHNICAL_DESIGN.md` (D-017). [RECONSTRUCTED — daftar diperluas]

## 9. Hardware / Resource Constraints
Consumer-grade; model kecil (toy: ratusan ribu parameter); batch/dimensi configurable (vocab 256 fixed — D-015); checkpoint+resume; benchmark memori/waktu di AC stage terkait.

## 10. Milestone Strategy
Milestone kecil berurutan, dibuka setelah milestone sebelumnya disetujui. M-001..M-006 DONE. M-007 (Transformer): Technical Design + Task Breakdown DONE — READY FOR IMPLEMENTATION, menunggu persetujuan owner.
Milestone kecil, berurutan, dibuka satu per satu setelah milestone sebelumnya disetujui. Detail lihat `docs/tasks/ROADMAP.md` dan `docs/tasks/TASKS.md`.
