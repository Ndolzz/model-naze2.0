# Technical Design — ID-BASIC-001: Kemampuan Dasar Bahasa Indonesia — Naze 2.0

**Format:** DECISION-017 (15 section) | **Status:** APPROVED (owner, 2026-10-13) | **Branch:** feature/id-basic-001
**Requirement:** REQ-401 | **Keputusan pendukung:** DECISION-018 (arsitektur beku), DECISION-023 (gate produksi terjaga), DECISION-025 (strategi dataset)

## 1. Purpose
Meningkatkan kemampuan dasar bahasa Indonesia model Naze (checkpoint naze_v1, seluruh gate DECISION-023 terpenuhi) pada lima kategori terukur: kosakata, tata bahasa, struktur kalimat, tanda baca, dan makna kontekstual — tanpa merusak kapabilitas eksisting (NazeIO 100%, holdout perplexity 1.747). Eksperimen terpisah dari checkpoint produksi; baseline diukur sebelum training; evaluasi pra/pasca memakai benchmark identik.

## 2. Scope
- Dataset berlabel lima kategori: sumber eksternal berlisensi terverifikasi (DECISION-025) dengan fallback contoh buatan sendiri untuk kategori tanpa sumber berlisensi jelas.
- Evaluator + benchmark tetap per kategori, terpisah dari data training, dengan pemeriksaan output kosong/tidak valid.
- Fine-tuning rehearsal (pola terbukti finetune_nazeio.py) pada checkpoint eksperimen baru.
- Baseline dan evaluasi pascatraining via workflow CI.

## 3. Non-goals
- Tidak mengubah arsitektur/tokenizer (DECISION-018: D=64 H=4 L=2 d_ff=128 T_max=128, vocab 256 fixed).
- Tidak menimpa checkpoint produksi naze_v1 (eksperimen di models/naze_idbasic_exp).
- Tidak mengklaim pemahaman bahasa dari penurunan loss semata (DECISION-016: bukti = benchmark).
- Tidak mencakup kapabilitas di luar lima kategori; tidak multimodal.

## 4. Requirements
- REQ-401 (utama): lima kategori terukur; baseline pra-training; ambang pasca-baseline (OD-127).
- REQ-101: seluruh eksperimen deterministik per-seed.
- REQ-102: checkpoint ≤ 1 MB; RAM < 2 GB (DECISION-021).
- REQ-105: seluruh artefak terdokumentasi (manifest, provenance, lisensi).
- REQ-301/302: spec-driven; no overengineering.

## 5. Architecture
Tidak ada perubahan core. Artefak baru:
- `data/idbasic/*.txt` — data latih berlabel per kategori (format pasangan terbukti: baris input + baris label).
- `data/idbasic/benchmark/*.json` — benchmark tetap per kategori (schema: id, category, prompt, expected, metric; metric: exact_match | rule_based).
- `data/idbasic/manifest.json` — provenance per sumber: asal, lisensi + bukti URL, metode derivasi/pembersihan, jumlah contoh aktual (dihitung oleh validator, bukan diklaim), versi, sha256.
- `scripts/validate_idbasic.py` — validasi UTF-8, struktur, label, duplikat, leakage, distribusi kategori; menghitung sha256 + jumlah aktual ke manifest.
- `scripts/benchmark_idbasic.py` — evaluator (pola benchmark_nazeio.py): greedy generate, exact match / rule-based, laporan JSON.
- `scripts/finetune_idbasic.py` — fine-tuning rehearsal (pola finetune_nazeio.py): korpus campuran data ID-BASIC ×N + rehearsal teks umum (train.txt) + rehearsal perintah (perintah_tingkat_1.txt), block 64, checkpoint tujuan models/naze_idbasic_exp.
- `.github/workflows/idbasic.yml` — workflow_dispatch: unduh artifact checkpoint sumber, baseline/evaluasi, fine-tuning, benchmark, upload artifact laporan.

## 6. Module Boundaries
scripts/* → naze.train / naze.lm / naze.data / naze.token / naze.inference (arah dependensi proyek tidak berubah). Tidak ada impor antar-script; data adalah satu-satunya antarmuka antar tahap.

## 7. Interfaces
- Data latih: `{jenis}: {input}\n{label_key}: {label}\n\n` per contoh (konkret per kategori ditetapkan di task breakdown sebelum berkas dibuat).
- Benchmark: JSON `{"version", "category", "items": [{"id", "prompt", "expected", "metric"}]}`.
- Laporan: `runs/idbasic_baseline.json` / `runs/idbasic_eval.json` — per kategori: n_items, n_correct, metric_values, contoh kesalahan (maks 10), konfigurasi, checkpoint hash, seed, versi dataset.

## 8. Data Flow
Sumber eksternal (unduh + derivasi sesuai lisensi, dicatat di manifest) → berkas data/idbasic → validator (fail-fast) → split eksplisit train/eval tanpa tumpang tindih → rehearsal mixing → fine-tuning → benchmark identik untuk baseline dan pasca.

## 9. Configuration
Block 64 (pasangan label terpanjang menentukan; dihitung aktual oleh validator), batch 8, lr 0.08, seed 0, epochs eksperimen default 8 (dinaikkan bertahap bila perlu, dengan budget CI ≤ 350 menit/run), idbasic_repeats dan rehearsal_bytes configurable. Perintah rehearsal ×25 (pola M-010).

## 10. Error Handling
Validator fail-fast: byte non-UTF-8, label kosong/tidak valid, duplikat antar contoh, contoh eval yang muncul di train. Evaluator menandai output kosong/panjang nol sebagai invalid (bukan correct). Training berhenti bila loss non-finite atau checkpoint rusak. Semua kegagalan dilaporkan apa adanya (BLOCKED/NOT RUN), tidak dipalsukan.

## 11. Testing Strategy
- Unit: validator (contoh valid/invalid sintetis), evaluator (exact match, rule-based, output kosong).
- Integrasi: smoke training pada korpus kecil (±20 KB, 2 epoch) di CI — verifikasi loss turun dan checkpoint eksperimen tersimpan, tanpa menyentuh models/naze_v1.
- Regresi: benchmark NazeIO 20 command (target: tidak turun dari 100%) dan holdout perplexity (dibandingkan baseline 1.747; ambang regresi ditetapkan pasca-baseline, sebelum hasil pasca dipakai — OD-127).

## 12. Resource Constraints
Checkpoint ≤ 1 MB (D-018); RAM < 2 GB (D-021); durasi workflow CI ≤ 350 menit; dataset per kategori direncanakan ribuan contoh (angka final = hasil penghitungan validator, bukan klaim).

## 13. Acceptance Criteria
1. Audit Tahap 0 terdokumentasi dengan bukti (DONE, 2026-10-13).
2. Dataset tervalidasi; manifest akurat (provenance + lisensi + sha256 + jumlah aktual).
3. Benchmark tetap terpisah dari data training; nol leakage (diverifikasi validator).
4. Baseline benar-benar dijalankan (atau BLOCKED dilaporkan eksplisit).
5. Training eksperimen dijalankan bila lingkungan memungkinkan; smoke test lolos sebelumnya.
6. Evaluasi pra/pasca memakai benchmark identik; tabel perbandingan + analisis kesalahan.
7. Ambang ditetapkan setelah baseline dan sebelum keputusan (OD-127); tidak retrospektif.
8. Regresi NazeIO/holdout dilaporkan apa adanya; checkpoint produksi tidak tertimpa.
9. Tes wajib hijau di CI; status Git dan CI dilaporkan jujur.

## 14. Risks
- Lisensi sumber eksternal berubah/tidak terverifikasi → fallback buatan sendiri (dicatat di manifest).
- Kapasitas model ±108 ribu parameter terbatas → ambang realistis per kategori (OD-127), laporan apa adanya.
- Catastrophic forgetting → rehearsal (pola M-010 terbukti) + regresi wajib.
- Sumber eksternal kualitas bervariasi → validasi + analisis kesalahan.

## 15. Open Decisions
- OD-127 — ambang penerimaan per kategori (setelah baseline).
- OD-128 — adopsi checkpoint eksperimen ke produksi (keputusan owner pasca-evaluasi).
