# OD-112 CORPUS RECOMMENDATION

> **Status:** DRAFT  Rekomendasi untuk Open Decision-112
> **Open Decision:** OD-112  Korpus final training Naze
> **Milestone:** M-010 (Stage 9)
> **Tanggal:** 2026-10-09
> **Dokumen Terkait:** M010_TECHNICAL_DESIGN.md, DECISION-023

---

## 1. Ringkasan

Dokumen ini menyajikan **rekomendasi** untuk penyelesaian **OD-112** (Korpus final training Naze). Rekomendasi ini **BUKAN keputusan final** dan memerlukan **persetujuan owner** sebelum diimplementasikan.

**Tujuan:** Menyediakan korpus training yang memungkinkan model Naze (config D-018) mencapai target **DECISION-023**:
- Training loss <= 2.5
- Validation perplexity <= 35
- NazeIO command accuracy >= 90%

---

## 2. Kebutuhan Korpus

### 2.1 Persyaratan Teknis

| Persyaratan | Nilai | Sumber |
|-------------|-------|--------|
| **Vocab Size** | 256 (fixed) | DECISION-010, DECISION-015 |
| **Tokenizer** | Byte-level | DECISION-010 |
| **Max Sequence Length** | 128 | DECISION-018 |
| **Model Capacity** | ~108k params | DECISION-018 |
| **Checkpoint Size** | <= 1 MB (float64), ~0.43 MB (float32) | DECISION-018 |

### 2.2 Kebutuhan Domain

**Model Naze 1.0 berfokus pada:**
1. **Perintah Tingkat 1** (untuk nazeio)
2. **Instruksi Editing Video** (untuk naze-motion-agent)
3. **Bahasa Indonesia** (primary)
4. **Teks Pendek** (<= 128 token)

---

## 3. Sumber Korpus yang Direkomendasikan

### 3.1 Korpus yang Sudah Ada

| Sumber | Status | Ukuran | Keterangan |
|--------|--------|--------|------------|
| `data/korpus/perintah_tingkat_1.txt` | ✅ **TERSEDIA** | ~30+ perintah | Sudah ada di model-naze2.0 |
| Toy corpus (M-006, M-007) | ✅ **TERSEDIA** | ~100-200 token | Untuk testing |

**Catatan:** Korpus yang sudah ada **TIDAK CUKUP** untuk training skala produksi.

### 3.2 Sumber Eksternal yang Direkomendasikan

#### 3.2.1 Korpus Bahasa Indonesia

| Sumber | Jenis | Ukuran | Lisensi | Keterangan |
|--------|------|--------|---------|------------|
| **Indonesian Wikipedia** | Teks | ~50-100 MB | CC BY-SA | Perlu cleaning |
| **Indonesian News (Kompas, Detik)** | Teks | ~100-500 MB | Beragam | Perlu scraping + cleaning |
| **Indonesian Books (Project Gutenberg)** | Teks | ~10-50 MB | Public Domain | Terbatas |
| **Indonesian Subtitles** | Teks | ~50-200 MB | Beragam | Format srt/ass |
| **Indonesian Social Media** | Teks | ~100-1000 MB | Beragam | Perlu anonymization |

#### 3.2.2 Korpus Domain-Spesifik

| Sumber | Domain | Ukuran | Keterangan |
|--------|--------|--------|------------|
| **nazeio Command List** | Perintah suara | ~100-500 perintah | Perlu diperluas |
| **Video Editing Tutorials** | Instruksi | ~1-5 MB | Perlu scraping |
| **Mobile App Usage** | Instruksi | ~1-5 MB | Perlu scraping |

### 3.3 Sumber Internal (Jika Tersedia)

| Sumber | Status | Keterangan |
|--------|--------|------------|
| Log perintah user nazeio | ❓ UNCONFIRMED | Memerlukan izin owner |
| Log instruksi user naze-motion-agent | ❓ UNCONFIRMED | Memerlukan izin owner |
| Dataset pribadi owner | ❓ UNCONFIRMED | Memerlukan konfirmasi |

---

## 4. Rekomendasi Korpus Final

### 4.1 Opsi 1: Korpus Publik + Domain-Spesifik (Recommended)

**Komposisi:**

| Komponen | Sumber | Ukuran Target | Proporsi |
|----------|--------|---------------|-----------|
| **Bahasa Indonesia Umum** | Wikipedia + News | ~5 MB | 70% |
| **Perintah Tingkat 1** | nazeio + custom | ~1 MB | 20% |
| **Instruksi Editing** | Tutorials + custom | ~1 MB | 10% |
| **Total** | | **~7 MB** | 100% |

**Keuntungan:**
- Cakupan bahasa Indonesia yang baik
- Domain-spesifik untuk nazeio dan naze-motion-agent
- Ukuran cukup untuk config D-018
- Tersedia secara publik

**Keterbatasan:**
- Memerlukan cleaning dan preprocessing
- Memerlukan izin untuk sumber tertentu
- Memerlukan scraping untuk news/tutorials

### 4.2 Opsi 2: Korpus Sintetis (Fallback)

**Komposisi:**

| Komponen | Sumber | Ukuran Target | Proporsi |
|----------|--------|---------------|-----------|
| **Teks Acak (Random)** | Generator | ~5 MB | 50% |
| **Perintah Tingkat 1** | nazeio | ~1 MB | 20% |
| **Instruksi Editing** | Custom | ~1 MB | 20% |
| **Pattern Teks** | Template | ~1 MB | 10% |
| **Total** | | **~8 MB** | 100% |

**Keuntungan:**
- Tidak memerlukan izin
- Dapat dikontrol kualitasnya
- Dapat disesuaikan dengan kebutuhan

**Keterbatasan:**
- Kurang representatif bahasa alami
- Risiko overfitting ke pattern
- Kurang variasi

### 4.3 Opsi 3: Korpus Minimal (Untuk Testing)

**Komposisi:**

| Komponen | Sumber | Ukuran Target | Proporsi |
|----------|--------|---------------|-----------|
| **Perintah Tingkat 1** | nazeio | ~0.5 MB | 50% |
| **Instruksi Editing** | Custom | ~0.5 MB | 50% |
| **Total** | | **~1 MB** | 100% |

**Keuntungan:**
- Cepat untuk training
- Ukuran sangat kecil
- Cukup untuk testing

**Keterbatasan:**
- **TIDAK CUKUP** untuk produksi
- Akurasi terbatas
- Generalisasi buruk

---

## 5. Preprocessing dan Cleaning

### 5.1 Pipeline Preprocessing

```
RAW TEXT
  │
  ▼
┌─────────────────────────┐
│  1. Normalisasi Teks      │
│  - Lowercase              │
│  - Hapus tanda baca       │
│  - Rapikan spasi          │
│  - Hapus whitespace berlebih
└─────────────┬─────────────┘
              │
              ▼
┌─────────────────────────┐
│  2. Tokenisasi            │
│  - ByteTokenizer.encode   │
│  - Vocab 256 (fixed)      │
└─────────────┬─────────────┘
              │
              ▼
┌─────────────────────────┐
│  3. Filtering             │
│  - Hapus sequence kosong  │
│  - Filter terlalu panjang │
│  - Hapus duplikasi        │
└─────────────┬─────────────┘
              │
              ▼
┌─────────────────────────┐
│  4. Sliding Window        │
│  - block_size = 128       │
│  - stride = 1            │
│  - Generate (x, y) pairs  │
└─────────────┬─────────────┘
              │
              ▼
         TRAINING DATA
```

### 5.2 Aturan Cleaning

| Aturan | Deskripsi | Contoh |
|--------|-----------|--------|
| **Lowercase** | Semua huruf kecil | "Hello" -> "hello" |
| **Hapus Tanda Baca** | Hapus ,.!?:;"'()[]{} | "hello!" -> "hello" |
| **Rapikan Spasi** | Hapus spasi berlebih | "hello   world" -> "hello world" |
| **Hapus Whitespace** | Hapus \n\t\r | "hello\nworld" -> "hello world" |
| **Hapus URL** | Hapus http/https links | "lihat https://..." -> "lihat" |
| **Hapus Email** | Hapus alamat email | "email a@b.com" -> "email" |
| **Hapus HTML Tags** | Hapus <...> | "<b>hello</b>" -> "hello" |
| **Hapus Unicode Non-BMP** | Hapus emoji, dll. | "hello 😊" -> "hello" |
| **Normalisasi Angka** | Konversi ke digit | "satu" -> "1" (opsional) |

### 5.3 Deduplikasi

**Metode:**
1. **Exact Match:** Hapus teks yang identik
2. **Fuzzy Match:** Hapus teks yang mirip (>95% similarity)
3. **N-gram Deduplikasi:** Hapus teks dengan n-gram overlap tinggi

**Tools:**
- Python: `datasets` library (jika tersedia)
- Custom: Hash-based deduplication

### 5.4 Pembagian Dataset

**Proporsi:**
- **Training:** 80%
- **Validation:** 10%
- **Holdout/Test:** 10%

**Metode:**
- **Deterministik:** Seed yang sama -> pembagian yang sama
- **Stratified:** Mempertahankan distribusi kelas (jika ada label)
- **Random:** Shuffle sebelum pembagian

**Contoh:**
```python
from naze.core import seeded_rng
import numpy as np

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=0):
    rng = seeded_rng(seed)
    n = len(data)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        "train": [data[i] for i in indices[:train_end]],
        "val": [data[i] for i in indices[train_end:val_end]],
        "test": [data[i] for i in indices[val_end:]]
    }
```

---

## 6. Pencegahan Data Leakage

### 6.1 Aturan Pencegahan

| Aturan | Deskripsi |
|--------|-----------|
| **Pisah Waktu** | Training dan validation data tidak boleh overlap dalam waktu |
| **Pisah Sumber** | Training dan validation data tidak boleh dari sumber yang sama |
| **Pisah Konten** | Training data tidak boleh muncul di validation/test |
| **No Lookahead** | Validation data tidak boleh digunakan untuk hyperparameter tuning |

### 6.2 Validasi Pencegahan

**Cek sebelum training:**
1. **Interseksi Kosong:** Training ∩ Validation = ∅
2. **Interseksi Kosong:** Training ∩ Test = ∅
3. **Interseksi Kosong:** Validation ∩ Test = ∅

**Contoh:**
```python
def check_leakage(train, val, test):
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    
    assert train_set.isdisjoint(val_set), "Leakage: train ∩ val != ∅"
    assert train_set.isdisjoint(test_set), "Leakage: train ∩ test != ∅"
    assert val_set.isdisjoint(test_set), "Leakage: val ∩ test != ∅"
```

---

## 7. Benchmark Terpisah

### 7.1 Benchmark untuk NazeIO

**Dataset:** Daftar perintah Tingkat 1

**Format:**
```json
{
  "name": "nazeio_command_benchmark",
  "version": "1.0",
  "tests": [
    {
      "id": "cmd_001",
      "input": "buka aplikasi whatsapp",
      "expected_intent": "BUKA_APLIKASI",
      "expected_target": "whatsapp",
      "min_confidence": 0.8
    },
    {
      "id": "cmd_002",
      "input": "pasang pengingat jam 10 pagi",
      "expected_intent": "PASANG_PENGINGAT",
      "expected_target": null,
      "min_confidence": 0.7
    }
  ]
}
```

**Metrik:**
- **Accuracy:** `(benar) / (total) * 100%`
- **Target:** >= 90% (DECISION-023)

### 7.2 Benchmark untuk Naze Motion Agent

**Dataset:** Daftar instruksi editing

**Format:**
```json
{
  "name": "nma_editing_benchmark",
  "version": "1.0",
  "tests": [
    {
      "id": "edit_001",
      "input": "buat video dengan gambar foto1.jpg dan foto2.jpg",
      "expected_task": "create_video",
      "expected_actions": [
        {"type": "CREATE_PROJECT"},
        {"type": "ADD_MEDIA", "parameters": {"path": "foto1.jpg"}},
        {"type": "ADD_MEDIA", "parameters": {"path": "foto2.jpg"}}
      ],
      "min_confidence": 0.7
    }
  ]
}
```

**Metrik:**
- **ActionPlan Accuracy:** `(ActionPlan valid) / (total) * 100%`
- **Target:** >= 85% (untuk instruksi sederhana)

---

## 8. Estimasi Resource

### 8.1 Ukuran Dataset

| Ukuran | Token (byte-level) | Sequence (T=128) | Estimasi Waktu Training |
|--------|-------------------|------------------|--------------------------|
| 1 MB | ~1M token | ~8k sequences | ~1-5 menit |
| 5 MB | ~5M token | ~40k sequences | ~10-30 menit |
| 10 MB | ~10M token | ~80k sequences | ~30-60 menit |
| 50 MB | ~50M token | ~400k sequences | ~5-10 jam |

**Catatan:**
- Config D-018: ~108k parameters
- Batch size: 8 (default)
- Hardware: CPU (laptop/PC umum)
- Precision: float64

### 8.2 Memory Usage

| Komponen | Ukuran (float64) | Ukuran (float32) |
|---------|------------------|------------------|
| Parameters | ~0.87 MB | ~0.43 MB |
| Gradients | ~0.87 MB | - |
| Activations (per layer) | ~4 MB | ~2 MB |
| Batch Data | ~0.01 MB | ~0.01 MB |
| **Total Training** | **~5-10 MB** | - |
| **Total Inference** | **~1-2 MB** | **~0.5-1 MB** |

**Target:** RAM training < 2 GB (DECISION-021) ✅ **TERPENUHI**

### 8.3 Storage Usage

| Komponen | Ukuran |
|---------|--------|
| Model (float32) | ~0.43 MB |
| Checkpoint (float64) | ~0.87 MB |
| Metadata | ~1 KB |
| **Total per Model** | **< 1 MB** |

**Target:** Checkpoint <= 1 MB (DECISION-018) ✅ **TERPENUHI**

---

## 9. Rekomendasi Final

### 9.1 Rekomendasi Utama (Opsi 1)

**Korpus:** **Bahasa Indonesia Umum + Domain-Spesifik**

**Komposisi:**
- 70%: Bahasa Indonesia umum (Wikipedia + News)
- 20%: Perintah Tingkat 1 (nazeio)
- 10%: Instruksi Editing (naze-motion-agent)

**Ukuran:** **~7 MB** (setelah preprocessing)

**Sumber:**
1. **Indonesian Wikipedia** (CC BY-SA)
   - Scrape halaman Wikipedia Bahasa Indonesia
   - Filter: Hanya teks, hapus markup
   - Ukuran target: ~3.5 MB

2. **Indonesian News** (Public Domain/Creative Commons)
   - Scrape berita dari situs dengan lisensi terbuka
   - Filter: Hanya teks, hapus HTML
   - Ukuran target: ~1.5 MB

3. **nazeio Command List**
   - Perluas `perintah_tingkat_1.txt` ke 100+ perintah
   - Tambah variasi ejaan
   - Ukuran target: ~1 MB

4. **Video Editing Instructions**
   - Scrape tutorial editing dari sumber terbuka
   - Buat instruksi sintetis
   - Ukuran target: ~1 MB

**Keuntungan:**
- ✅ Cakupan bahasa Indonesia yang luas
- ✅ Domain-spesifik untuk kedua proyek
- ✅ Ukuran cukup untuk config D-018
- ✅ Dapat mencapai target DECISION-023

**Risiko:**
- ⚠️ Memerlukan scraping dan cleaning
- ⚠️ Memerlukan izin untuk beberapa sumber
- ⚠️ Memerlukan waktu untuk persiapan

### 9.2 Rekomendasi Cadangan (Opsi 2)

**Korpus:** **Sintetis + Domain-Spesifik**

**Komposisi:**
- 50%: Teks acak (random text generation)
- 20%: Perintah Tingkat 1
- 20%: Instruksi Editing
- 10%: Pattern teks

**Ukuran:** **~8 MB**

**Keuntungan:**
- ✅ Tidak memerlukan izin
- ✅ Dapat dikontrol kualitasnya
- ✅ Cepat untuk disiapkan

**Risiko:**
- ❌ Kurang representatif bahasa alami
- ❌ Risiko overfitting
- ❌ Akurasi mungkin tidak mencapai target

### 9.3 Rekomendasi untuk Testing (Opsi 3)

**Korpus:** **Minimal**

**Ukuran:** **~1 MB**

**Keuntungan:**
- ✅ Cepat untuk training
- ✅ Cukup untuk testing

**Keterbatasan:**
- ❌ **TIDAK CUKUP** untuk produksi
- ❌ Akurasi terbatas

---

## 10. Trade-offs

### 10.1 Ukuran vs Kualitas

| Ukuran | Keuntungan | Keterbatasan |
|--------|-----------|---------------|
| **Kecil (1-5 MB)** | Cepat training, ukuran APK kecil | Akurasi terbatas, generalisasi buruk |
| **Sedang (5-10 MB)** | Balance, cukup untuk target | Memerlukan waktu persiapan |
| **Besar (10-50 MB)** | Akurasi tinggi, generalisasi baik | Lambat training, ukuran APK besar |

**Rekomendasi:** **5-10 MB** (balance optimal)

### 10.2 Publik vs Sintetis

| Sumber | Keuntungan | Keterbatasan |
|--------|-----------|---------------|
| **Publik** | Representatif, variasi tinggi | Memerlukan izin, cleaning kompleks |
| **Sintetis** | Kontrol penuh, tidak memerlukan izin | Kurang representatif, risiko overfitting |

**Rekomendasi:** **Kombinasi** (70% publik + 30% sintetis)

### 10.3 Bahasa Indonesia vs Multilingual

| Opsi | Keuntungan | Keterbatasan |
|------|-----------|---------------|
| **Indonesia Only** | Fokus, ukuran kecil, akurasi tinggi | Terbatas pada bahasa Indonesia |
| **Multilingual** | Cakupan luas, fleksibel | Ukuran besar, akurasi per-bahasa rendah |

**Rekomendasi:** **Indonesia Only** (focus pada target utama)

---

## 11. Keputusan yang Memerlukan Owner

### 11.1 Keputusan Wajib

| Pertanyaan | Opsi | Rekomendasi | Dampak |
|-----------|-------|-------------|--------|
| **Sumber korpus utama** | Wikipedia/News/Sintetis | **Wikipedia + News** | Kualitas dan legalitas |
| **Ukuran korpus** | 1-5 MB / 5-10 MB / 10-50 MB | **5-10 MB** | Balance kualitas/ukuran |
| **Komposisi korpus** | Umum only / Domain-only / Kombinasi | **Kombinasi** | Cakupan dan akurasi |
| **Bahasa** | Indonesia only / Multilingual | **Indonesia only** | Fokus dan ukuran |

### 11.2 Keputusan Tambahan

| Pertanyaan | Opsi | Rekomendasi | Dampak |
|-----------|-------|-------------|--------|
| **Pembersihan teks** | Minimal / Standard / Agresif | **Standard** | Kualitas data |
| **Deduplikasi** | Exact only / Fuzzy / N-gram | **Fuzzy + N-gram** | Kualitas data |
| **Pembagian dataset** | 80/10/10 / 70/15/15 / Lainnya | **80/10/10** | Evaluasi |
| **Format korpus** | Teks mentah / JSON / Lainnya | **Teks mentah** | Kesederhanaan |

---

## 12. Checklist Persiapan Korpus

- [ ] **Sumber diputuskan** (owner decision)
- [ ] **Ukuran ditentukan** (owner decision)
- [ ] **Komposisi ditentukan** (owner decision)
- [ ] **Lisensi diverifikasi** (untuk sumber publik)
- [ ] **Izin diperoleh** (jika diperlukan)
- [ ] **Data diunduh/scrape**
- [ ] **Preprocessing dilakukan**
- [ ] **Cleaning dilakukan**
- [ ] **Deduplikasi dilakukan**
- [ ] **Pembagian dataset dilakukan**
- [ ] **Leakage dicek**
- [ ] **Benchmark dataset disiapkan** (nazeio + naze-motion-agent)
- [ ] **Korpus final divalidasi**
- [ ] **Training dijalankan**
- [ ] **Metrik DECISION-023 divalidasi**

---

## 13. Ringkasan

| Aspek | Rekomendasi | Status |
|-------|-------------|--------|
| **Sumber** | Wikipedia + News + Domain-Spesifik | ⏳ Menunggu owner |
| **Ukuran** | ~7 MB | ⏳ Menunggu owner |
| **Komposisi** | 70% umum + 20% perintah + 10% editing | ⏳ Menunggu owner |
| **Bahasa** | Indonesia only | ⏳ Menunggu owner |
| **Preprocessing** | Standard cleaning | ✅ Siap |
| **Deduplikasi** | Fuzzy + N-gram | ✅ Siap |
| **Pembagian** | 80/10/10 | ✅ Siap |
| **Benchmark** | Terpisah untuk nazeio dan NMA | ✅ Siap |

**Status OD-112:** **BLOCKING M-010** 
**Tindakan:** Menunggu keputusan owner untuk sumber, ukuran, dan komposisi korpus final.

---

## 14. Lampiran

### 14.1 Contoh Korpus (Format)

```
# === BAHASA INDONESIA UMUM ===

ini adalah contoh teks dalam bahasa indonesia
bahasa indonesia adalah bahasa resmi negara indonesia
indonesia memiliki banyak pulau dan budaya

# === PERINTAH TINGKAT 1 ===

perintah: buka aplikasi whatsapp
intent: BUKA_APLIKASI
target: whatsapp

perintah: buka pengaturan
intent: BUKA_PENGATURAN
target: null

perintah: pasang pengingat jam 10 pagi
intent: PASANG_PENGINGAT
target: null
parameters: {"time": "10:00"}

# === INSTRUKSI EDITING ===

buat video dengan gambar foto1.jpg dan foto2.jpg
tambahkan teks hello world di tengah
potong video dari detik 5 sampai 15
```

### 14.2 Referensi

- [M010_TECHNICAL_DESIGN.md](M010_TECHNICAL_DESIGN.md)
- [DECISION_LOG.md](../decisions/DECISION_LOG.md)
- [DECISION-018](../decisions/DECISION_LOG.md#decision-018)
- [DECISION-023](../decisions/DECISION_LOG.md#decision-023)
- [NAZEIO_INTEGRATION_CONTRACT.md](NAZEIO_INTEGRATION_CONTRACT.md)
- [NAZE_MOTION_AGENT_INTEGRATION_CONTRACT.md](NAZE_MOTION_AGENT_INTEGRATION_CONTRACT.md)
