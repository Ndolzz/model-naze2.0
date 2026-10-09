# Corpus Dataset - Naze 1.0 (M-010)

**Status:** INTEGRATED (DECISION-024, APPROVED 2026-10-09)
**Milestone:** M-010 (Naze 1.0, Stage 9)
**Source of Truth:** DECISION-024 (OD-112 Resolution)

---

## Overview

Corpus final untuk training Naze 1.0 menggunakan **strategi hybrid** dengan komposisi:
- 70% Bahasa Indonesia Umum (~5 MB)
- 20% Perintah Asisten (~1.5 MB)  
- 10% Instruksi Editing (~0.5 MB)
- **Total Target:** ~7 MB (ukuran file teks UTF-8 mentah, sebelum preprocessing)

## Directory Structure

```
data/corpus/
├── README.md                    # Dokumentasi ini
├── umumnya/                    # 70% - Bahasa Indonesia Umum
│   ├── wikipedia_id.txt        # Wikipedia Indonesia (CC-BY-SA)
│   └── berita_id.txt           # Berita dengan lisensi eksplisit
├── perintah/                   # 20% - Perintah Asisten (untuk nazeio)
│   └── perintah_asisten.txt    # Command sintetic terkontrol
└── editing/                    # 10% - Instruksi Editing (untuk NMA)
    └── instruksi_editing.txt    # Instruksi editing sintetic
```

## File Descriptions

### 1. Bahasa Indonesia Umum (70%, ~5 MB)
- **Sumber:** Wikipedia Indonesia (CC-BY-SA 3.0), berita dengan lisensi eksplisit
- **Format:** Teks mentah (UTF-8)
- **Preprocessing:**
  - Pembersihan: markup, metadata, duplikasi dihapus
  - Normalisasi: lowercase, whitespace berlebih dihapus
  - Deduplikasi: teks yang muncul di multiple file dihapus
- **Lisensi:** CC-BY-SA 3.0 (Wikipedia), verifikasi lisensi WAJIB untuk berita
- **Status:** **NOT YET CREATED** - Menunggu owner untuk menyediakan/menyetujui sumber

### 2. Perintah Asisten (20%, ~1.5 MB)
- **Sumber:** Data sintetis terkontrol
- **Format:** Teks mentah (UTF-8), satu perintah per baris
- **Contoh:** "Buka aplikasi WhatsApp", "Setel alarm untuk jam 7 pagi"
- **Kegunaan:** Benchmark command accuracy untuk nazeio
- **Status:** **NOT YET CREATED** - Menunggu owner untuk menyetujui format dan konten

### 3. Instruksi Editing (10%, ~0.5 MB)
- **Sumber:** Data sintetis untuk workflow editing
- **Format:** Teks mentah (UTF-8), instruksi editing
- **Contoh:** "Potong video dari detik 10 sampai 20", "Tambahkan transisi fade"
- **Kegunaan:** Mendukung kemampuan Naze Motion Agent
- **Status:** **NOT YET CREATED** - Menunggu owner untuk menyetujui format dan konten

## Dataset Splitting

Setelah corpus final terintegrasi, dataset akan dibagi:
- **Training:** 80% dari total corpus
- **Validation:** 10% dari total corpus
- **Holdout:** 10% dari total corpus (TIDAK BOLEH digunakan untuk training atau hyperparameter tuning)

Pembagian dilakukan secara **deterministik per-seed** (REQ-101) untuk memastikan reproduktibilitas.

## Tokenization

- **Tokenizer:** ByteTokenizer (DECISION-010, DECISION-015)
- **Vocabulary Size:** 256 (fixed, lossless by construction)
- **Token Count:** Akan dihitung setelah corpus final terintegrasi

## Fingerprints

SHA-256 fingerprint setiap file corpus akan terdokumentasi di sini setelah file dibuat:

```
File: wikipedia_id.txt
SHA-256: [TBD]
Size: [TBD] bytes
Tokens: [TBD]

File: berita_id.txt
SHA-256: [TBD]
Size: [TBD] bytes
Tokens: [TBD]

File: perintah_asisten.txt
SHA-256: [TBD]
Size: [TBD] bytes
Tokens: [TBD]

File: instruksi_editing.txt
SHA-256: [TBD]
Size: [TBD] bytes
Tokens: [TBD]
```

## Integration Pipeline

Corpus akan diintegrasikan ke pipeline training menggunakan:
1. `ByteTokenizer.encode()` untuk konversi teks ke token IDs
2. `TextWindows` untuk sliding window sampling
3. `batches_pos()` untuk per-posisi training

## Acceptance Criteria (DECISION-024)

- [ ] Sumber data memiliki lisensi yang memungkinkan redistribusi
- [ ] Ukuran aktual corpus terdokumentasi (byte mentah, teks UTF-8, token count)
- [ ] Fingerprint (SHA-256) setiap file corpus terdokumentasi
- [ ] Tidak ada data leakage antara split
- [ ] Pipeline training dapat memproses corpus tanpa error

## Risks dan Keterbatasan

1. **Lisensi:** Verifikasi lisensi Wikipedia dan berita **WAJIB** dilakukan sebelum training
2. **Data Sintetis:** Tidak mencerminkan distribusi bahasa nyata sepenuhnya
3. **Ukuran:** 7 MB adalah estimasi; ukuran aktual tergantung pada sumber dan preprocessing
4. **Alternatif:** Jika lisensi sumber tidak memungkinkan, alternatif: corpus sintetis penuh (~7 MB)

## Dependencies

- DECISION-010: Byte-level tokenizer
- DECISION-015: Vocab size 256 fixed
- DECISION-011: Dataset configurable
- DECISION-018: Config D-018 (D=64, H=4, L=2, d_ff=128, T_max=128)
- DECISION-024: OD-112 Resolution (strategi hybrid ~7 MB)

## Next Steps

1. Owner menyediakan/menyetujui sumber corpus
2. Buat file corpus di direktori yang sesuai
3. Verifikasi lisensi setiap sumber
4. Preprocess corpus (pembersihan, normalisasi, deduplikasi)
5. Dokumentasikan fingerprint dan ukuran
6. Integrasikan ke pipeline training (T004)

## See Also

- [DECISION-024](docs/decisions/DECISION_LOG.md#decision-024) - OD-112 Resolution
- [M010_TECHNICAL_DESIGN.md](docs/architecture/M010_TECHNICAL_DESIGN.md) - Technical Design M-010
- [M010_TASKS.md](docs/tasks/M010_TASKS.md) - Task Breakdown M-010
