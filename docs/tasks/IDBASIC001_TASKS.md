# ID-BASIC-001 TASKS — Kemampuan Dasar Bahasa Indonesia

**Status legend:** PLANNED / IN-PROGRESS / DONE / BLOCKED | **Branch:** feature/id-basic-001
**TD:** docs/architecture/IDBASIC001_TECHNICAL_DESIGN.md | **REQ:** REQ-401 | **D-025**

| ID | Task | Status |
|---|---|---|
| IDB-T001 | Tahap 0 audit repositori aktual (bukti: laporan audit 2026-10-13; temuan C1-C8) | DONE |
| IDB-T002 | Tahap 1 spesifikasi: TD 15 section, REQ-401, DECISION-025, TRACEABILITY/ROADMAP/TASKS, perbaikan metadata korpus C1-C8 | DONE |
| IDB-T003 | Dataset lima kategori contoh buatan sendiri (482 contoh train) + manifest + PROVENANCE.md; tiga sumber eksternal diinvestigasi dan dikecualikan dengan bukti (IndoNLU, Kateglo, Wordnet Bahasa) | DONE |
| IDB-T004 | Validator scripts/validate_idbasic.py (stdlib-only) + unit test tests/test_idbasic_validator.py. Kode selesai; eksekusi NOT RUN pada lingkungan kerja lokal (Python tidak tersedia) — menunggu CI | IN-PROGRESS |
| IDB-T005 | Benchmark tetap data/idbasic/benchmark/*.json (135 contoh, exact_match; prompt benchmark diverifikasi tidak muncul pada training) | DONE |
| IDB-T006 | Evaluator scripts/benchmark_idbasic.py + unit test | PLANNED |
| IDB-T007 | Workflow CI idbasic.yml + baseline dijalankan pada checkpoint produksi + laporan + penetapan ambang (OD-127) | PLANNED |
| IDB-T008 | Fine-tuning rehearsal scripts/finetune_idbasic.py + smoke test kecil + eksperimen training via CI | PLANNED |
| IDB-T009 | Evaluasi pascatraining (benchmark identik) + tabel perbandingan + regresi NazeIO/holdout + analisis kesalahan | PLANNED |
| IDB-T010 | Dokumentasi final: CHANGELOG, ROADMAP, TRACEABILITY, SPEC_REVIEW, laporan akhir (status PARTIAL/BLOCKED bila ada sisa) | PLANNED |

Urutan kritis: T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010. Gerbang: training tidak dimulai sebelum T001-T007 selesai.

## Catatan Tahap 2 (2026-10-13)

- Dataset 100% contoh buatan sendiri sesuai fallback DECISION-025; tidak ada data pihak ketiga yang diimpor.
- Sumber eksternal diinvestigasi langsung dan dikecualikan: IndoNLU (lisensi data tidak terverifikasi tegas), Kateglo (CC-BY-NC-SA 3.0), Wordnet Bahasa (berkas lema melebihi batas pengambilan; mirror tidak kompatibel wn30/wn31). Bukti: data/idbasic/PROVENANCE.md.
- Manifest data/idbasic/manifest.json memuat jumlah contoh aktual dan SHA-256 yang dihitung dari berkas akhir.
- Anti-kebocoran: seluruh prompt benchmark diverifikasi programatik tidak muncul pada prompt training.
- Batas contoh maksimum 88 byte (batas validasi 96 byte) agar kompatibel dengan rencana block_size 96.
- Validator dan unit test belum dieksekusi lokal (lingkungan kerja tanpa Python); status eksekusi menunggu CI — tidak ada klaim PASS tanpa bukti run.
