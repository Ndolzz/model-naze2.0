# M011 TECHNICAL DESIGN — Naze 1.1: Korpus Dwibahasa & Suara Merek Naze (Stage 9, lanjutan)

**Milestone:** M-011 — Naze 1.1
**Stage:** Stage 9 (lanjutan — tidak membuka Stage 10)
**Status:** APPROVED — owner, 2026-10-14, tanpa revisi; implementasi M-011 dimulai
**Version:** 0.1.0
**Source of truth:** REQ-012 (REQUIREMENTS v0.6.0), DECISION-025, docs/spec/BRAND_VOICE.md
**Dependencies:** M-001..M-010 DONE (Naze 1.0 v1.0.0); DECISION-025 ACCEPTED (owner, 2026-10-14)

---

## 1. Purpose

Mengimplementasikan REQ-012: model Naze menguasai bahasa Indonesia dan bahasa Inggris dengan gaya bicara merek Naze. Pencapaian dilakukan melalui data, bukan perubahan arsitektur: korpus dwibahasa ~8 MB (komposisi DECISION-025), data sintetis asisten yang ditulis sesuai BRAND_VOICE.md, benchmark NazeIO dwibahasa, dan training dengan strategi yang mencegah catastrophic forgetting (temuan M-010: fine-tuning murni memicu ppl umum 7,614 pada 500 epoch). Arsitektur model tidak berubah (DECISION-018: D=64, H=4, L=2, d_ff=128, T_max=128; checkpoint <= 1 MB).

## 2. Scope

- Korpus dwibahasa sesuai komposisi DECISION-025: 40% ID umum (~3,2 MB), 20% EN umum (~1,6 MB), 15% percakapan asisten bergaya Naze dwibahasa (~1,2 MB), 15% perintah asisten dwibahasa (~1,2 MB), 10% instruksi editing (~0,8 MB); total ~8 MB.
- Manifest korpus v2: komposisi, ukuran aktual, token count, fingerprint SHA-256, verifikasi lisensi.
- Splits 80/10/10 dibuat ulang, deterministik per-seed (seed=42), tanpa data leakage.
- Benchmark NazeIO dwibahasa: akurasi per bahasa dilaporkan terpisah.
- Eksperimen dua jalur training: (A) fine-tuning atas checkpoint Naze 1.0 dengan rehearsal; (B) training ulang penuh — resolusi OD-123 via eksperimen pembanding (kerangka 5-basis DECISION-016, basis 1).
- Evaluasi 5-basis + laporan loss/perplexity per komponen korpus.
- Pembaruan dokumentasi: README, CHANGELOG, manifest, TRACEABILITY, ROADMAP, SPEC_REVIEW.
- Pekerjaan rumah (temuan pemeriksaan 2026-10-14): rekonsiliasi status splitting pada corpus_manifest.json; pembersihan direktori warisan data/korpus/ (keputusan owner 2026-10-14: arsip verbatim ke docs/archive/korpus_legacy/, lalu hapus).

## 3. Non-goals

- Tidak mengubah arsitektur, dimensi, atau ukuran model (DECISION-018 tetap; REQ-302).
- Tidak menambah bahasa ketiga.
- Tidak membuka Stage 10 / multimodal (OD-108 tetap DEFERRED; M-012+).
- Tidak menggunakan pretrained model, API LLM eksternal, atau korpus berlisensi tidak jelas (REQ-201/202; verifikasi lisensi WAJIB).
- Tidak melakukan fine-tuning tanpa rehearsal.

## 4. Requirements

| Req | Relevan untuk M-011 | Status |
|---|---|---|
| REQ-012 | Kemampuan dwibahasa & suara merek | APPROVED — target utama M-011 |
| REQ-005 | Dataset pipeline (splits ulang) | DONE (M-005); dipakai ulang |
| REQ-008 | Inference (benchmark dwibahasa) | DONE (M-009); dipakai ulang |
| REQ-009 | Evaluation (5-basis, per komponen) | DONE (M-008); dipakai ulang |
| REQ-101 | Reproducibility (seed=42) | Berlaku |
| REQ-102 | Resource efficiency (D-018/D-021) | Berlaku |
| REQ-103 | Testability (CI, coverage >= 80%) | Berlaku (D-019) |
| REQ-104 | Modularity (tanpa perubahan src/naze) | Berlaku |
| REQ-105 | Documentation | Berlaku |
| REQ-201/202/203 | No pretrained core; no API LLM core; Python+NumPy | Permanen |
| REQ-301/302/304 | SDD; no overengineering; change log | Berlaku |

**Catatan:** REQ-012 adalah requirement baru; seluruh requirement lain sudah DONE atau berlaku lintas-stage. OD-123 (strategi training) dibuka di §15 dan diresolusi via eksperimen (M011-T012).

## 5. Architecture

Pipeline Naze 1.1:

```
=== KORPUS DWIBAHASA (DECISION-025) ===
Sumber (lisensi terbuka / sintetis BRAND_VOICE.md)
  -> preprocessing (pembersihan, normalisasi, deduplikasi)
  -> corpus_manifest v2 (komposisi 40/20/15/15/10, fingerprint, lisensi)
  -> ByteTokenizer.encode (vocab 256 fixed, D-010/D-015)
  -> corpus_all_ids v2 -> splits 80/10/10 (seed=42, tanpa leakage)

=== TRAINING (OD-123: dua jalur) ===
Jalur A: checkpoint Naze 1.0 + fine-tune + rehearsal (pola finetune_nazeio.py)
Jalur B: training ulang penuh (pola train_final.py)
  -> eksperimen pembanding 5-basis (D-016) -> pilih jalur default
  -> TrainingRun (M-008) -> checkpoint models/naze_v1_1/

=== EVALUASI & BENCHMARK ===
Holdout per komponen (ID umum / EN umum / asisten / perintah / editing)
  -> loss + perplexity per komponen
  -> benchmark NazeIO dwibahasa (akurasi per bahasa)
  -> verifikasi: tanpa degradasi gate DECISION-023 pada kemampuan bahasa Indonesia
```

**Prinsip desain:** tidak ada komponen neural baru; seluruh kemampuan baru ditanamkan melalui data. Deterministik penuh per-seed (REQ-101). Fail-fast pada semua tahap persiapan korpus.

## 6. Module Boundaries

| Modul | Peran | Baru/Diubah | Dependensi |
|---|---|---|---|
| data/corpus/ | Korpus dwibahasa + manifest v2 | Diubah (data, bukan kode) | — |
| data/benchmark/ | Benchmark dwibahasa | Diubah (data) | — |
| scripts/ | Eksperimen & training M-011 | Baru; adopsi pola script M-010 | src/naze |
| src/naze/ | Paket inti | TIDAK DIUBAH | — |
| docs/ | Dokumentasi SDD | Diubah | — |

**Boundary rule:** M-011 tidak menambah atau mengubah kode di src/naze (REQ-302; REQ-104). Seluruh perubahan terjadi pada data, scripts, dan docs.

## 7. Interfaces

### Struktur korpus M-011
```
data/corpus/
├── README.md                      # update: komposisi D-025
├── corpus_manifest.json           # v2.0.0
├── corpus_all_ids.npy             # regenerasi
├── umumnya/
│   ├── wikipedia_id.txt           # dipangkas ke ~3,2 MB (subhimpunan seed=42)
│   └── wikipedia_en.txt           # BARU ~1,6 MB (Simple English Wikipedia, CC-BY-SA)
├── asisten/
│   └── percakapan_naze.txt        # BARU ~1,2 MB (format User:/Naze:, BRAND_VOICE.md)
├── perintah/
│   └── perintah_asisten.txt       # ekspansi dwibahasa ~1,2 MB
├── editing/
│   └── instruksi_editing.txt      # ekspansi ~0,8 MB
└── splits/                        # regenerasi 80/10/10 seed=42
```

### Benchmark dwibahasa
```
data/benchmark/commands.json       # v2: field "language" ("id" | "en") pada setiap entri
```

### Format percakapan sintetis (BRAND_VOICE.md §9)
```text
User: buka galeri
Naze: Galeri sudah terbuka. Ada foto yang ingin Anda lihat?
```
Satu pertukaran maksimal ~128 token.

## 8. Data Flow

### Persiapan korpus
1. Verifikasi lisensi setiap sumber (WAJIB sebelum integrasi).
2. Siapkan wikipedia_en.txt (~1,6 MB); pangkas wikipedia_id.txt ke ~3,2 MB (subhimpunan deterministik seed=42).
3. Tulis data sintetis (percakapan, perintah dwibahasa, editing) sesuai BRAND_VOICE.md.
4. Manifest v2: ukuran aktual per komponen + fingerprint SHA-256 + toleransi ±10% per komponen.
5. Encode ByteTokenizer -> corpus_all_ids v2 -> splits 80/10/10 seed=42; verifikasi tanpa leakage (tidak ada baris identik lintas split).

### Training (M011-T012..T013)
1. Jalur A: muat checkpoint Naze 1.0 -> fine-tune lr rendah + rehearsal teks umum per epoch.
2. Jalur B: training ulang penuh pada korpus dwibahasa.
3. Eksperimen pembanding 5-basis -> pilih jalur default -> resolusi OD-123 (entri DECISION_LOG).
4. Training final jalur terpilih -> checkpoint v2 + run log (M-008).

### Evaluasi (M011-T014..T015)
1. Holdout: loss + perplexity per komponen (ID umum, EN umum, asisten, perintah, editing).
2. Benchmark dwibahasa: akurasi per bahasa via InferenceEngine (M-009).
3. Verifikasi non-degradasi: gate DECISION-023 pada kemampuan bahasa Indonesia tetap terpenuhi.
4. Verifikasi konsistensi suara merek: sampel respons dinilai terhadap BRAND_VOICE.md.

## 9. Configuration

### Default config M-011
- **Model:** Transformer config D-018 (tidak berubah).
- **Seed:** 42 (korpus, splits, training; REQ-101).
- **Splits:** 80/10/10.
- **Jalur A (fine-tune + rehearsal):** lr=0,02, block=64, rehearsal teks umum tiap epoch, epochs=8 (tunable per eksperimen).
- **Jalur B (training ulang):** lr=0,08, epochs=10 (pola M-010).

### Target provisional per eksperimen (D-016: ditetapkan di sini per eksperimen — bukan ambang universal)

| Metrik | Target Provisional | Keterangan |
|---|---|---|
| Akurasi NazeIO ID | >= 90% | tanpa degradasi gate D-023 |
| Akurasi NazeIO EN | >= 90% | kemampuan baru REQ-012 |
| Holdout ppl komponen ID umum | <= 2 × ppl baseline Naze 1.0 (komponen sama) | non-degradasi |
| Holdout ppl komponen EN umum | dilaporkan; menurun lintas epoch | basis 2/3 D-016 |
| Konsistensi suara merek | >= 95% sampel sesuai BRAND_VOICE.md (peninjauan manual) | REQ-012 poin (2) |

Angka di atas adalah target provisional (bukan hasil pengukuran, bukan jaminan), mengikuti pola DECISION-023; dapat direvisi owner dengan bukti pengukuran.

## 10. Error Handling

| Error | Handling | Tipe |
|---|---|---|
| File korpus hilang | FileNotFoundError (fail-fast) | Persiapan |
| Ukuran komponen di luar toleransi ±10% | ValueError (fail-fast, manifest) | Persiapan |
| Fingerprint SHA-256 mismatch | ValueError (fail-fast) | Persiapan |
| Lisensi sumber tidak terverifikasi | task BLOCKED — integrasi dilarang | Persiapan |
| Leakage lintas split terdeteksi | ValueError (fail-fast); splits dibuat ulang | Persiapan |
| Percakapan sintetis > 128 token / format salah | ValueError (validasi per baris) | Persiapan |
| Loss NaN/Inf saat training | ValueError (as_array) + abort run | Training |
| Checkpoint > 1 MB | ValueError (fail-fast, D-018) | Training |

**Prinsip:** fail-fast; tanpa silent failure (REQ-302).

## 11. Testing Strategy

### Unit tests
- Test manifest v2: komposisi, toleransi ukuran, fingerprint, lisensi terverifikasi.
- Test validasi percakapan sintetis: format User:/Naze:, panjang <= 128 token.
- Test determinisme splits seed=42 (dua pembuatan menghasilkan split identik).
- Test deteksi leakage (fixture berisi duplikat lintas split harus gagal).

### Integration tests
- Pipeline: manifest -> encode -> splits -> TextWindows -> training singkat -> generate ID dan EN.
- Benchmark dwibahasa berjalan pada model dummy (verifikasi metode, bukan kualitas).

### Regression tests
- Seluruh test lama tetap hijau tanpa modifikasi (DECISION-020 branch protection).
- Coverage >= 80% baris src/naze (DECISION-019) — tidak berubah karena src/naze tidak disentuh.

### Acceptance tests
- Verifikasi AC §13 via eksperimen pembanding 5-basis dan laporan validasi final.

## 12. Resource Constraints

| Sumber daya | Batas | Status |
|---|---|---|
| Checkpoint | <= 1 MB (D-018) | Berlaku — arsitektur tidak berubah |
| RAM training | < 2 GB (D-021) | Berlaku; korpus ~8 MB jauh di bawah batas |
| RAM inference | < 2 GB (D-021) | Berlaku |
| Sequence length | <= 128 (D-018) | Berlaku; percakapan dibatasi per desain |
| Ukuran korpus | ~8 MB teks mentah | D-025 |

**Estimasi:** training pada korpus ~8 MB mengikuti pola M-010 (CPU consumer-grade); RAM puncak didominasi params+grads+aktivasi (config D-018) + data terindeks.

## 13. Acceptance Criteria (Objective, Verifiable)

### Wajib (MUST)
1. Korpus dwibahasa terintegrasi sesuai komposisi DECISION-025, toleransi ±10% per komponen, total ~8 MB.
2. Lisensi semua sumber terverifikasi dan terdokumentasi di manifest v2.
3. Manifest v2 memuat ukuran aktual, token count, dan fingerprint SHA-256 setiap file.
4. Splits 80/10/10 deterministik seed=42; teruji tanpa leakage.
5. Seluruh percakapan sintetis lolos validasi format dan <= 128 token.
6. Benchmark NazeIO dwibahasa menghasilkan akurasi per bahasa; target provisional §9 tercapai pada model final (ID >= 90% DAN EN >= 90%).
7. Non-degradasi: holdout ppl komponen ID umum <= 2 × baseline Naze 1.0; gate DECISION-023 pada kemampuan bahasa Indonesia tetap terpenuhi.
8. Konsistensi suara merek: >= 95% sampel respons asisten dinilai sesuai BRAND_VOICE.md (peninjauan manual terdokumentasi).
9. Seluruh test lama tetap hijau tanpa modifikasi; CI hijau; coverage >= 80%.
10. Dokumentasi diperbarui: README, CHANGELOG, manifest, TRACEABILITY, ROADMAP, SPEC_REVIEW (maintenance policy PROJECT_SPEC §7).

### Target optimasi (SHOULD)
11. Evaluasi 5-basis lengkap dilaporkan (baseline, perbaikan, konvergensi, validasi, reproducible).
12. Eksperimen pembanding jalur A/B terdokumentasi; OD-123 RESOLVED.

**Definisi DONE M-011:** seluruh AC wajib terverifikasi pada satu model final + seluruh task M011-T001..T019 selesai (T019 opsional bila integrasi APK ditunda owner).

## 14. Risks

| Risiko | Dampak | Mitigasi | Status |
|---|---|---|---|
| Catastrophic forgetting saat fine-tune | Kemampuan ID terdegradasi | Rehearsal wajif; jalur B disiapkan; verifikasi non-degradasi AC 7 | Aktif |
| Kapasitas model kecil (~108k param) | Kelancaran EN terbatas pada fragmen pendek | Ekspektasi dikelola; evaluasi per komponen jujur | Aktif |
| Kualitas data sintetis EN | Distribusi tidak alami | Komponen EN umum dari sumber alami; sintetis hanya untuk asisten | Aktif |
| Lisensi sumber EN | Legal | Verifikasi WAJIB sebelum integrasi (AC 2) | Aktif |
| Ketimpangan kemampuan ID vs EN | EN lemah relatif | Evaluasi akurasi per bahasa terpisah; ekspansi EN dapat ditinjau owner | Dipantau |

## 15. Open Decisions

| ID | Deskripsi | Status | Dampak |
|---|---|---|---|
| OD-123 (BARU) | Strategi training M-011: jalur A fine-tune + rehearsal vs jalur B training ulang penuh | OPEN — diresolusi via eksperimen pembanding M011-T012 (basis 1 D-016) | M011-T013 menunggu resolusi |
| OD-108 | Multimodal / Physical AI (Stage 10) | OPEN — DEFERRED (M-012+) | Tidak blocking M-011 |
| OD-119 | Versioning & release policy | OPEN | Penamaan versi Naze 1.1 saat rilis |

**Catatan:** OD-123 dibuka oleh dokumen ini; resolusi melalui bukti empiris (eksperimen pembanding), bukan asumsi. Implementasi M-011 dimulai hanya setelah persetujuan owner atas dokumen ini (PROJECT_SPEC §7 tahap 5). **Persetujuan diberikan owner pada 2026-10-14, tanpa revisi; implementasi dimulai.**
