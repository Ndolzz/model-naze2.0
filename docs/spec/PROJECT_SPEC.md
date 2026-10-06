# PROJECT SPEC — Naze 2.0

**Version:** 0.3.0 | **Status:** Milestones 1-6 complete (awaiting owner pytest validation)

## 1. Project Overview
Model AI dari nol: tanpa pretrained model, tanpa API LLM eksternal sebagai core. Seluruh komponen (engine, autodiff, tokenizer, dataset, training, inference) dibuat sendiri dengan Python + NumPy. SDD: spesifikasi = source of truth.

## 2. Vision
Core intelligence milik sendiri, transparan end-to-end: fondasi neural -> model bahasa -> multimodal -> physical AI/robotics.

## 3. Long-Term Objective
1. Near: fondasi komputasi neural solid (TERCAPAI M-001..M-003).
2. Mid: LM pertama trainable end-to-end (TERCAPAI M-006, versi MLP kecil).
3. Long: Transformer sendiri + training lengkap + rilis Naze 1.0.
4. Far: multimodal & physical AI.

## 4. Current Scope (setelah M-006)
- Pipeline END-TO-END berfungsi: encode teks (byte tokenizer) -> batch dataset -> train MLP LM (SGD) -> checkpoint -> generate teks (greedy/temperature), seluruhnya deterministik per-seed dan teruji.
- Menunggu persetujuan owner untuk M-007 (Transformer, Stage 6).

## 5. Out of Scope Saat Ini
Transformer (M-007), training system penuh (M-008), inference engine penuh (M-009), Naze 1.0 (M-010), multimodal (M-011+). Larangan permanen: pretrained core, API LLM core.

## 6. Development Philosophy
Spec first; no invention (OPEN DECISION); modular bertahap; resource efficiency; no overengineering; traceability.

## 7. SDD Workflow (Official)
1. Define Spec -> 2. Review -> 3. Architecture -> 4. Tasks -> 5. Implement -> 6. Test -> 7. Validate vs Spec -> 8. Review -> 9. Commit -> 10. Update Spec if changed. Perubahan requirement tercatat di changelog REQUIREMENTS.md.

## 8. Technology Constraints
Python; NumPy; core from scratch (pretrained/API LLM dilarang permanen; framework DL dilarang); pytest+ruff; float64+seeded RNG; backprop per-layer (D-009); tokenizer byte-level (D-010); SGD (D-013).

## 9. Hardware / Resource Constraints
Consumer-grade; model kecil (toy: ratusan ribu parameter); batch/dimensi configurable; checkpoint+resume; benchmark memori/waktu di AC stage terkait.

## 10. Milestone Strategy
Milestone kecil berurutan, dibuka setelah milestone sebelumnya disetujui. M-001..M-006 DONE. M-007 (Transformer) berikutnya — menunggu persetujuan owner.
