# ID-BASIC-001 TASKS — Kemampuan Dasar Bahasa Indonesia

**Status legend:** PLANNED / IN-PROGRESS / DONE / BLOCKED | **Branch:** feature/id-basic-001
**TD:** docs/architecture/IDBASIC001_TECHNICAL_DESIGN.md | **REQ:** REQ-401 | **D-025**

| ID | Task | Status |
|---|---|---|
| IDB-T001 | Tahap 0 audit repositori aktual (bukti: laporan audit 2026-10-13; temuan C1-C8) | DONE |
| IDB-T002 | Tahap 1 spesifikasi: TD 15 section, REQ-401, DECISION-025, TRACEABILITY/ROADMAP/TASKS, perbaikan metadata korpus C1-C8 | DONE |
| IDB-T003 | Dataset lima kategori contoh buatan sendiri (482 contoh train) + manifest + PROVENANCE.md; tiga sumber eksternal diinvestigasi dan dikecualikan dengan bukti (IndoNLU, Kateglo, Wordnet Bahasa) | DONE |
| IDB-T004 | Validator scripts/validate_idbasic.py (stdlib-only) + unit test tests/test_idbasic_validator.py. Kode selesai; eksekusi NOT RUN pada lingkungan kerja lokal (Python tidak tersedia); dieksekusi oleh job validate workflow idbasic.yml saat workflow dipicu | IN-PROGRESS |
| IDB-T005 | Benchmark tetap data/idbasic/benchmark/*.json (135 contoh, exact_match; prompt benchmark diverifikasi tidak muncul pada training) | DONE |
| IDB-T006 | Evaluator scripts/benchmark_idbasic.py (exact-match ketat tanpa normalisasi; kegagalan data/model/prediksi = exit 2 tanpa skor; kategori dibaca dari manifest) + unit test tests/test_idbasic_evaluator.py. Kode selesai; eksekusi NOT RUN lokal — dieksekusi oleh workflow idbasic.yml | IN-PROGRESS |
| IDB-T007 | Workflow CI .github/workflows/idbasic.yml (validate → baseline → finetune opsional; verifikasi checksum checkpoint sebelum evaluasi; artifact idbasic-baseline-*/idbasic-finetune-*). Kode selesai. Baseline BLOCKED pada lingkungan kerja (artifact run 38376902867 tidak dapat diunduh dari sini) — menunggu pemtriggeran workflow oleh owner. Ambang OD-127 masih OPEN (usulan tercatat, menunggu persetujuan) | IN-PROGRESS (baseline: BLOCKED) |
| IDB-T008 | Fine-tuning rehearsal scripts/finetune_idbasic.py (korpus campuran: idbasic ×12 + perintah NazeIO ×25 + rehearsal umum 256 KB/epoch, block 96; guard anti-penimpaan checkpoint produksi; validasi dataset wajib sebelum training) + smoke test tests/test_idbasic_finetune_smoke.py. Kode selesai; rehearsal NOT RUN — hanya boleh berjalan setelah baseline (workflow menjaga urutan needs: baseline) | IN-PROGRESS (eksekusi: NOT RUN) |
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

## Catatan Tahap 3 (2026-10-14)

Audit dan integritas:

- Audit ulang branch: tip `f7b0a08` (5 commit Tahap 0-2 di atas `3129d8e`); `main` tetap `576d1b0`, tidak tersentuh.
- Verifikasi programatik ulang seluruh dataset pada `f7b0a08`: SHA-256, ukuran byte, dan jumlah contoh seluruh 10 berkas cocok dengan manifest; train 482 / benchmark 135; kebocoran prompt benchmark→training = 0; duplikat = 0; seluruh konten ASCII murni; contoh terpanjang 86 byte (≤ 88 manifest, ≤ 96 validator). SHA-256 manifest.json = `48a575df1710979f7a4a261d639ea9c29d8b907ab3f2908f13c6e0edba24e6f2`.

Checkpoint sumber baseline:

- Run artifact `38376902867` terverifikasi ADA via halaman run GitHub: workflow `finetune-nazeio` run #6, dipicu manual oleh Ndolzz 2026-10-13 09:31 UTC pada commit `2eca536` (main), status Success, durasi 4j9m24s, 1 artifact (`nazeio-finetune-38376902867`).
- Artifact tersebut TIDAK dapat diunduh dari lingkungan kerja (tidak ada tool unduh artifact) → baseline produksi = BLOCKED pada lingkungan kerja; tidak ada klaim skor baseline. Workflow idbasic.yml mengunduh artifact dan memverifikasi checksum `params.npz` sebelum evaluasi, sehingga baseline CI memenuhi syarat "checkpoint terverifikasi".

Ambang penerimaan (OD-127 masih OPEN — usulan berikut MENUNGGU PERSETUJUAN OWNER; tidak ditetapkan retrospektif):

1. Gate regresi wajib (checkpoint eksperimen vs produksi): akurasi NazeIO tidak boleh turun di bawah gate produksi DECISION-023 (≥ 90%; model final saat ini 100%) dan perplexity sampel korpus umum ≤ 35.
2. Gate peningkatan ID-BASIC (pascatraining vs baseline, benchmark identik): tidak ada kategori yang menurun dan minimal 3 dari 5 kategori meningkat.
3. Ambang absolut per kategori baru ditetapkan owner SETELAH baseline terukur (menghindari penetapan pasca-hoc).

Deviasi desain terdokumentasi:

- `block_size` rehearsal = 96, bukan 64 (angka pra-kurasi pada TD §9): contoh train terpanjang 86 byte sehingga jendela 64 tidak memuat satu contoh utuh; 96 ≤ T_max 128 (D-018). Arsitektur, vocab 256, dan format checkpoint v2 tidak diubah.
- Rehearsal umum 256 KB/epoch (finetune_nazeio M-010 memakai 512 KB/epoch) agar 8 epoch muat dalam timeout CI 350 menit; pola campuran (idbasic + perintah + umum) tetap sama.

Status eksekusi jujur (tanpa klaim tanpa run):

- Lingkungan kerja tidak memiliki Python: validator, pytest (validator/evaluator/smoke), evaluator, dan rehearsal BELUM dijalankan lokal.
- Seluruh eksekusi menunggu pemtriggeran `workflow_dispatch` idbasic.yml oleh owner: (1) baseline — `run_id=38376902867`, `artifact_name=nazeio-finetune-38376902867`, `with_finetune=false`; (2) setelah baseline tercatat dan OD-127 disetujui — ulangi dengan `with_finetune=true`.
- T009 (perbandingan pra/pasca) baru dapat dikerjakan setelah kedua laporan CI tersedia; T010 menyusul setelahnya.
