# ID-BASIC-001 TASKS — Kemampuan Dasar Bahasa Indonesia

**Status legend:** PLANNED / IN-PROGRESS / DONE / BLOCKED | **Branch:** feature/id-basic-001
**TD:** docs/architecture/IDBASIC001_TECHNICAL_DESIGN.md | **REQ:** REQ-401 | **D-025**

| ID | Task | Status |
|---|---|---|
| IDB-T001 | Tahap 0 audit repositori aktual (bukti: laporan audit 2026-10-13; temuan C1-C8) | DONE |
| IDB-T002 | Tahap 1 spesifikasi: TD 15 section, REQ-401, DECISION-025, TRACEABILITY/ROADMAP/TASKS, perbaikan metadata korpus C1-C8 | DONE |
| IDB-T003 | Akuisisi + verifikasi lisensi sumber eksternal (kandidat: Wordnet Bahasa MIT via OMW; IndoNLU MIT; verifikasi Kateglo) dan derivasi contoh lima kategori + fallback buatan sendiri; berkas data/idbasic/*.txt + manifest provenance | PLANNED |
| IDB-T004 | Validator scripts/validate_idbasic.py (UTF-8, label, duplikat, leakage, distribusi; hitung sha256 + jumlah aktual ke manifest) + unit test | PLANNED |
| IDB-T005 | Benchmark tetap data/idbasic/benchmark/*.json (terpisah dari training; disusun SETELAH dataset agar nol leakage) | PLANNED |
| IDB-T006 | Evaluator scripts/benchmark_idbasic.py + unit test | PLANNED |
| IDB-T007 | Workflow CI idbasic.yml + baseline dijalankan pada checkpoint produksi + laporan + penetapan ambang (OD-127) | PLANNED |
| IDB-T008 | Fine-tuning rehearsal scripts/finetune_idbasic.py + smoke test kecil + eksperimen training via CI | PLANNED |
| IDB-T009 | Evaluasi pascatraining (benchmark identik) + tabel perbandingan + regresi NazeIO/holdout + analisis kesalahan | PLANNED |
| IDB-T010 | Dokumentasi final: CHANGELOG, ROADMAP, TRACEABILITY, SPEC_REVIEW, laporan akhir (status PARTIAL/BLOCKED bila ada sisa) | PLANNED |

Urutan kritis: T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010. Gerbang: training tidak dimulai sebelum T001-T007 selesai.
