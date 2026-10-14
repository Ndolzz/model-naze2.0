# M011_TASKS — Naze 1.1: Korpus Dwibahasa & Suara Merek Naze

**Status:** IN PROGRESS — TECHNICAL DESIGN APPROVED (owner, 2026-10-14, tanpa revisi)
**Referensi:** docs/architecture/M011_TECHNICAL_DESIGN.md (DECISION-025)
**Dependencies:** M-001..M-010 DONE; DECISION-025 ACCEPTED (owner, 2026-10-14); REQ-012 APPROVED
**Requirements:** REQ-012 (utama); REQ-005/008/009/101..105/201..203/301/302/304

**Atribut:**
- Korpus target ~8 MB dengan komposisi DECISION-025 (40/20/15/15/10), toleransi ±10% per komponen.
- Seluruh data sintetis asisten WAJIB ditulis sesuai docs/spec/BRAND_VOICE.md.
- Setiap commit implementasi mereferensikan Task ID + Req ID (PROJECT_SPEC §7).
- Holdout TIDAK BOLEH digunakan untuk training atau hyperparameter tuning (DECISION-024/025).
- Tidak ada perubahan kode pada src/naze (REQ-302; TD §6).

---

## Kategori 1: Pekerjaan Rumah (prasyarat)

### M011-T001 — Rekonstruksi status splitting corpus_manifest.json
**Status:** DONE (2026-10-14)
**Deskripsi:** Rekonsiliasi inkonsistensi pra-existing: manifest v1 menyatakan dataset splitting NOT_YET_SPLIT sementara splits 80/10/10 sudah terintegrasi di data/corpus/splits/ (temuan pemeriksaan 2026-10-14). Perbarui manifest agar status konsisten dengan realita sebelum diganti oleh T008.
**AC:**
- [x] Status splitting pada manifest sesuai kondisi aktual file splits/ (manifest v1.0.1: SPLIT_INTEGRATED, referensi splits/split_manifest.json).
**Modul:** data/corpus/corpus_manifest.json
**Req:** REQ-105, REQ-304

### M011-T002 — Bersihkan direktori warisan data/korpus/
**Status:** DONE (2026-10-14)
**Deskripsi:** Direktori data/korpus/ (berisi perintah_tingkat_1.txt, 2,7 KB) adalah sisa pra-penataan data/corpus/. Periksa cakupan kontennya terhadap data/corpus/perintah/, dokumentasikan duplikasi, lalu hapus direktori warisan. **Keputusan owner (2026-10-14): konten diarsipkan verbatim ke docs/archive/korpus_legacy/ terlebih dahulu (dengan README analisis duplikasi), kemudian direktori warisan dihapus.**
**AC:**
- [x] Konten diperiksa dan duplikasi didokumentasikan (docs/archive/korpus_legacy/README.md).
- [x] Owner mengonfirmasi (2026-10-14: arsipkan terlebih dahulu); konten terarsip verbatim di docs/archive/korpus_legacy/; direktori warisan dihapus; referensi commands.json source_format diperbarui.
**Modul:** data/
**Blocker:** RESOLVED — keputusan owner 2026-10-14 (arsip verbatim ke docs/archive/korpus_legacy/, lalu hapus).
**Req:** REQ-105, REQ-304

---

## Kategori 2: Korpus Dwibahasa (DECISION-025)

### M011-T003 — Komponen Bahasa Inggris umum (~1,6 MB)
**Deskripsi:** Siapkan wikipedia_en.txt dari sumber berlisensi terbuka (Simple English Wikipedia, CC-BY-SA) ~1,6 MB. Preprocessing: pembersihan markup, normalisasi whitespace, deduplikasi.
**AC:**
- [ ] Lisensi terverifikasi dan terdokumentasi (WAJIB sebelum integrasi).
- [ ] Ukuran aktual ~1,6 MB (±10%); fingerprint SHA-256 tercatat.
**Modul:** data/corpus/umumnya/wikipedia_en.txt
**Req:** REQ-012, REQ-005, REQ-201/202

### M011-T004 — Pangkas komponen Bahasa Indonesia umum ke ~3,2 MB
**Deskripsi:** Ambil subhimpunan deterministik (seed=42) dari wikipedia_id.txt (5,04 MB) menjadi ~3,2 MB sesuai komposisi D-025.
**AC:**
- [ ] Subhimpunan deterministik seed=42; ukuran ~3,2 MB (±10%); fingerprint tercatat.
**Modul:** data/corpus/umumnya/wikipedia_id.txt
**Req:** REQ-012, REQ-005, REQ-101

### M011-T005 — Percakapan asisten bergaya Naze dwibahasa (~1,2 MB)
**Deskripsi:** Tulis percakapan sintetis dwibahasa (format User:/Naze:, satu pertukaran per blok, <= 128 token) sesuai BRAND_VOICE.md. Cakupan: perintah harian, tanya-jawab singkat, dorongan halus; seimbang antara ID dan EN.
**AC:**
- [ ] Seluruh pasangan lolos validasi format (User:/Naze:, <= 128 token).
- [ ] Peninjauan manual: respons sesuai BRAND_VOICE.md.
- [ ] Ukuran ~1,2 MB (±10%); fingerprint tercatat.
**Modul:** data/corpus/asisten/percakapan_naze.txt
**Req:** REQ-012

### M011-T006 — Ekspansi perintah asisten dwibahasa (~1,2 MB)
**Deskripsi:** Perluas perintah_asisten.txt dengan padanan dwibahasa: setiap perintah Indonesia memiliki padanan bahasa Inggris bermakna identik (BRAND_VOICE.md §9).
**AC:**
- [ ] Cakupan perintah nazeio lengkap dwibahasa; ukuran ~1,2 MB (±10%).
**Modul:** data/corpus/perintah/perintah_asisten.txt
**Req:** REQ-012

### M011-T007 — Ekspansi instruksi editing (~0,8 MB)
**Deskripsi:** Perluas instruksi_editing.txt menuju target ~0,8 MB (saat ini 4,4 KB) untuk mendukung Naze Motion Agent sesuai porsi D-025.
**AC:**
- [ ] Ukuran ~0,8 MB (±10%); cakupan instruksi editing konsisten dengan kontrak NMA.
**Modul:** data/corpus/editing/instruksi_editing.txt
**Req:** REQ-012

### M011-T008 — Manifest korpus v2
**Deskripsi:** Tulis corpus_manifest.json v2.0.0: komposisi D-025 (40/20/15/15/10), ukuran aktual, token count, fingerprint SHA-256 setiap file, status lisensi, status splitting.
**AC:**
- [ ] Manifest v2 lengkap dan valid (dapat di-load; fail-fast bila field hilang).
- [ ] Toleransi ±10% per komponen diverifikasi.
**Modul:** data/corpus/corpus_manifest.json
**Dependensi:** T003..T007
**Req:** REQ-012, REQ-105

### M011-T009 — Integrasi pipeline & splits ulang
**Deskripsi:** Regenerasi corpus_all_ids.npy dan splits 80/10/10 deterministik seed=42; verifikasi tanpa leakage lintas split; pipeline training memproses tanpa error.
**AC:**
- [ ] Splits deterministik (dua pembuatan menghasilkan split identik); tanpa leakage.
- [ ] Pipeline: manifest -> encode -> splits -> TextWindows berjalan tanpa error.
**Modul:** data/corpus/, data/corpus/splits/
**Dependensi:** T008
**Req:** REQ-005, REQ-101

---

## Kategori 3: Benchmark Dwibahasa

### M011-T010 — Ekspansi benchmark commands.json dwibahasa
**Deskripsi:** Perluas data/benchmark/commands.json dengan field language ("id"/"en") dan padanan perintah EN; pertahankan entri ID lama agar perbandingan baseline Naze 1.0 tetap mungkin.
**AC:**
- [ ] Benchmark valid JSON; setiap entri memiliki input, expected, language.
**Modul:** data/benchmark/commands.json
**Req:** REQ-012, REQ-008

### M011-T011 — Skrip benchmark akurasi per bahasa
**Deskripsi:** Perluas scripts/benchmark_nazeio.py agar melaporkan akurasi terpisah per bahasa (ID, EN) dan total.
**AC:**
- [ ] Laporan benchmark memuat akurasi per bahasa; metode identik dengan M-010 (exact match pada expected).
**Modul:** scripts/benchmark_nazeio.py
**Dependensi:** T010
**Req:** REQ-008, REQ-103

---

## Kategori 4: Training & Evaluasi

### M011-T012 — Eksperimen pembanding jalur training (resolusi OD-123)
**Deskripsi:** Jalankan eksperimen pembanding dua jalur pada korpus dwibahasa: (A) fine-tune checkpoint Naze 1.0 + rehearsal; (B) training ulang penuh. Evaluasi kerangka 5-basis (D-016); pilih jalur default; resolusi OD-123 (entri DECISION_LOG).
**AC:**
- [ ] Kedua jalur dievaluasi dengan konfigurasi reproducible (seed, epoch, lr tercatat).
- [ ] Laporan 5-basis; keputusan jalur default terdokumentasi.
**Modul:** scripts/ (finetune_naze11.py / train_naze11.py)
**Dependensi:** T009, T010, T011
**Blocker:** OD-123 — diresolusi oleh task ini.
**Req:** REQ-006, REQ-009, REQ-012

### M011-T013 — Training final jalur terpilih
**Deskripsi:** Training final Naze 1.1 dengan jalur terpilih; checkpoint models/naze_v1_1/ + run log + ringkasan (pola M-010).
**AC:**
- [ ] Checkpoint <= 1 MB (D-018); training loss menurun (konvergensi terdokumentasi).
- [ ] Run log + ringkasan tersimpan; resume berfungsi.
**Modul:** scripts/, models/
**Dependensi:** T012
**Req:** REQ-006, REQ-007, REQ-101, REQ-102

### M011-T014 — Evaluasi penuh & verifikasi non-degradasi
**Deskripsi:** Evaluasi holdout per komponen (ID umum, EN umum, asisten, perintah, editing): loss + perplexity; benchmark dwibahasa; verifikasi AC 6-7 TD §13.
**AC:**
- [ ] Laporan per komponen lengkap; seluruh target provisional TD §9 diverifikasi.
- [ ] Gate DECISION-023 pada kemampuan bahasa Indonesia tetap terpenuhi.
**Modul:** scripts/, docs/
**Dependensi:** T013
**Req:** REQ-009, REQ-012

### M011-T015 — Verifikasi konsistensi suara merek
**Deskripsi:** Sampel keluaran asisten model final (ID dan EN) dinilai terhadap BRAND_VOICE.md; dokumentasikan skor dan temuan.
**AC:**
- [ ] >= 95% sampel sesuai BRAND_VOICE.md; laporan peninjauan terdokumentasi.
**Modul:** docs/
**Dependensi:** T013
**Req:** REQ-012, REQ-105

---

## Kategori 5: Rilis & Dokumentasi

### M011-T016 — Update README.md dan CHANGELOG.md
**Deskripsi:** Perbarui README (status Naze 1.1, korpus dwibahasa, contoh generate dwibahasa) dan CHANGELOG.md.
**AC:**
- [ ] README memuat status M-011 dan contoh dwibahasa; CHANGELOG terbaru.
**Modul:** README.md, CHANGELOG.md
**Dependensi:** T014
**Req:** REQ-105, REQ-304

### M011-T017 — Update dokumen tata kelola
**Deskripsi:** Perbarui TRACEABILITY, ROADMAP, SPEC_REVIEW, REQUIREMENTS (status REQ-012) sesuai maintenance policy PROJECT_SPEC §7; entri resolusi OD-123 di DECISION_LOG.
**AC:**
- [ ] Seluruh dokumen tata kelola konsisten dengan hasil M-011.
**Modul:** docs/
**Dependensi:** T014, T015
**Req:** REQ-105, REQ-304

### M011-T018 — Validasi penuh & CI
**Deskripsi:** Eksekusi pytest penuh + coverage; pastikan test lama tidak dimodifikasi; laporan validasi final (AC §13 TD).
**AC:**
- [ ] pytest hijau; coverage >= 80% src/naze; test lama tidak berubah.
**Modul:** tests/, .github/workflows/
**Dependensi:** T009..T017
**Req:** REQ-103

### M011-T019 — Ekspor bobot & integrasi APK nazeio (opsional)
**Deskripsi:** Ekspor bobot float32 model final; integrasi ke APK nazeio sebagai target optimasi (bukan gate).
**AC:**
- [ ] Bobot float32 diekspor; ukuran APK <= 50 MB (target optimasi D-023).
**Modul:** kolaborasi nazeio
**Dependensi:** T014
**Catatan:** Target optimasi, bukan release gate; dapat ditunda owner.
**Req:** REQ-008

---

## Ringkasan Dependensi Task

```
T001, T002 (pekerjaan rumah — paralel)
T003..T007 (komponen korpus — paralel)
T008 <- T003..T007
T009 <- T008
T010, T011 (benchmark — paralel dengan T003..T008)
T012 <- T009, T010, T011 (BLOCKER utama: resolusi OD-123)
T013 <- T012
T014, T015 <- T013 (paralel)
T016, T017 <- T014, T015
T018 <- T009..T017
T019 <- T014 (opsional)
```

## Blocker Utama

1. **OD-123 (BARU): strategi training M-011** — jalur A fine-tune + rehearsal vs jalur B training ulang; OPEN; diresolusi via eksperimen pembanding M011-T012 (bukti empiris, bukan asumsi).
2. **Verifikasi lisensi sumber EN (T003)** — WAJIB sebelum integrasi.
3. **Konfirmasi owner atas T002 — RESOLVED (2026-10-14):** arsip verbatim ke docs/archive/korpus_legacy/, lalu hapus.

## Task yang Bisa Dikerjakan Setelah Persetujuan Owner atas TD

- [x] T001, T002: pekerjaan rumah — selesai 2026-10-14 (T002 via arsip docs/archive/korpus_legacy/)
- [ ] T003..T007: seluruh komponen korpus
- [ ] T010, T011: benchmark dwibahasa

## Definisi DONE untuk M-011

M-011 **DONE** jika:
1. [ ] Korpus dwibahasa ~8 MB terintegrasi (komposisi D-025, toleransi ±10%) — AC 1-4 TD §13.
2. [ ] Benchmark dwibahasa: akurasi ID >= 90% DAN EN >= 90% pada model final.
3. [ ] Non-degradasi: ppl komponen ID umum <= 2 × baseline Naze 1.0; gate D-023 kemampuan ID terpenuhi.
4. [ ] Konsistensi suara merek >= 95% sampel (BRAND_VOICE.md).
5. [ ] Seluruh task T001..T018 selesai (T019 opsional bila ditunda owner).
6. [ ] CI hijau; test lama tidak dimodifikasi; coverage >= 80%.
7. [ ] Dokumentasi lengkap dan konsisten (README, CHANGELOG, manifest, TRACEABILITY, ROADMAP, SPEC_REVIEW).

## Catatan Penting

- **OD-123 adalah blocker eksperimen, bukan blocker persiapan korpus:** T003..T011 dapat berjalan sebelum resolusi.
- **Holdout tidak boleh** untuk training/hyperparameter tuning (D-024/025).
- Target provisional (akurasi EN >= 90%, ppl <= 2×) ditetapkan per eksperimen di TD §9 sesuai DECISION-016 — dapat direvisi owner dengan bukti pengukuran.
- **Tidak boleh rilis** bila AC wajib TD §13 tidak terpenuhi pada model final.
