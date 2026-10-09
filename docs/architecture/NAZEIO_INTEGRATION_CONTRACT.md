# NAZEIO INTEGRATION CONTRACT

> **Status:** DRAFT  berdasarkan spesifikasi nazeio yang tersedia
> **Repository:** `Ndolzz/nazeio` (eksternal)
> **Referensi:** `specs/16_naze_engine.md`, `docs/rencana_integrasi_naze_engine.md`
> **Model Repository:** `Ndolzz/model-naze2.0` (M-001..M-009 DONE)
> **Tanggal:** 2026-10-09
> **Milestone:** M-010 (Stage 9)

---

## 1. Ringkasan

Dokumen ini mendefinisikan **kontrak integrasi** antara proyek **model-naze2.0** (sebagai provider model dan inference engine) dengan proyek **nazeio** (sebagai aplikasi asisten berbasis suara). Kontrak ini berfokus pada antarmuka teknis, format data, tanggung jawab masing-masing pihak, dan batasan yang harus dipatuhi.

**Tujuan:** Memungkinkan nazeio menggunakan model Naze untuk mengenali dan mengeksekusi **perintah Tingkat 1** secara lokal (offline) tanpa bergantung pada API eksternal.

---

## 2. Peran dan Tanggung Jawab

### 2.1 Peran model-naze2.0

| Komponen | Tanggung Jawab | Output |
|---------|---------------|--------|
| **Model Terlatih** | Menyediakan model Transformer (config D-018) yang terlatih | Bobot model (params.npz) |
| **Inference Engine** | Menyediakan forward-only inference path | `InferenceEngine` class |
| **Export Format** | Ekspor bobot ke format npz (float32) | `params.npz` + metadata |
| **Benchmark** | Menyediakan metrik performa inference | `BenchmarkResult` |
| **Korpus Perintah** | Menyediakan korpus perintah Tingkat 1 | `data/korpus/perintah_tingkat_1.txt` |

### 2.2 Peran nazeio

| Komponen | Tanggung Jawab | Output |
|---------|---------------|--------|
| **Pengenalan Suara** | Menerima audio, mengonversi ke teks (spesifikasi 05) | Teks input |
| **Normalisasi Teks** | Membersihkan teks: lowercase, spasi, tanda baca | Teks dinormalisasi |
| **Naze Engine** | Memuat model, menjalankan inference | Intent + confidence score |
| **Matching Intent** | Mencocokkan output model ke daftar intent | Intent teridentifikasi |
| **Eksekusi Aksi** | Menjalankan aksi lokal (buka aplikasi, dll.) | Aksi ter eksekusi |
| **Fallback** | Jika Naze Engine gagal, diteruskan ke tanya jawab (spesifikasi 03) | Response dari LLM eksternal |

---

## 3. Antarmuka Kontrak

### 3.1 Format Input untuk Naze Engine

```json
{
  "input_type": "text",
  "text": "buka aplikasi whatsapp",
  "timestamp": "2026-10-09T10:00:00Z",
  "session_id": "uuid-v4",
  "context": {
    "previous_intent": null,
    "device_info": {
      "os": "Android",
      "version": "14",
      "architecture": "ARMv7"
    }
  }
}
```

**Keterangan:**
- `input_type`: Selalu "text" (Naze Engine **TIDAK** memproses audio)
- `text`: Teks yang sudah dinormalisasi (lowercase, spasi dirapikan)
- `timestamp`: Waktu request (opsional, untuk logging)
- `session_id`: ID sesi untuk tracking (opsional)
- `context`: Konteks tambahan (opsional)

### 3.2 Format Output dari Naze Engine

```json
{
  "status": "success" | "error" | "fallback",
  "intent": {
    "name": "BUKA_APLIKASI",
    "confidence": 0.95,
    "action": "buka",
    "target": "whatsapp",
    "parameters": {}
  } | null,
  "alternatives": [
    {
      "name": "BUKA_PENGATURAN",
      "confidence": 0.03
    }
  ],
  "processing_time_ms": 150,
  "model_version": "1.0.0",
  "error": null | {"code": "...", "message": "..."}
}
```

**Keterangan:**
- `status`: "success" (intent ditemukan), "error" (kesalahan internal), "fallback" (tidak yakin)
- `intent`: Intent utama yang terdeteksi (null jika status != "success")
- `confidence`: Skor keyakinan (0.0 - 1.0), **WAJIB >= threshold** untuk dianggap valid
- `alternatives`: Daftar intent alternatif dengan confidence rendah
- `processing_time_ms`: Waktu pemrosesan
- `model_version`: Versi model yang digunakan
- `error`: Detail error (jika status == "error")

---

## 4. Format Data Internal (model-naze2.0 -> nazeio)

### 4.1 Format Bobot Model (npz)

**Spesifikasi:**
- Format: **NumPy .npz** (zip archive of .npy files)
- Precision: **float32** (untuk efisiensi memori di ARMv7)
- Ukuran: **<= 0.43 MB** (config D-018: ~108k params x 4 bytes)
- Checksum: **SHA-256** (termasuk di metadata)

**Kunci Parameter:**
```
# Embedding
emb.E          # (vocab_size=256, d_model=64)

# Positional
pos.P          # (max_sequence_length=128, d_model=64)

# Transformer Blocks (num_layers=2)
blk0.attn.q.W   # (d_model=64, d_model=64)
blk0.attn.q.b   # (d_model=64,)
blk0.attn.k.W   # (d_model=64, d_model=64)
blk0.attn.k.b   # (d_model=64,)
blk0.attn.v.W   # (d_model=64, d_model=64)
blk0.attn.v.b   # (d_model=64,)
blk0.attn.o.W   # (d_model=64, d_model=64)
blk0.attn.o.b   # (d_model=64,)
blk0.attn.norm.gamma  # (d_model=64,)
blk0.attn.norm.beta   # (d_model=64,)
blk0.ffn.fc1.W  # (d_model=64, d_ff=128)
blk0.ffn.fc1.b  # (d_ff=128,)
blk0.ffn.fc2.W  # (d_ff=128, d_model=64)
blk0.ffn.fc2.b  # (d_model=64,)
blk0.ffn.norm.gamma  # (d_model=64,)
blk0.ffn.norm.beta   # (d_model=64,)
# ... blk1 (sama)

# Final LayerNorm
final.gamma     # (d_model=64,)
final.beta      # (d_model=64,)

# LM Head
head.W          # (vocab_size=256, d_model=64)
head.b          # (vocab_size=256,)

# Metadata
__metadata__    # JSON string: {"version": "1.0.0", "config": {...}, "checksum": "sha256", "naze_version": "1.0.0"}
```

### 4.2 Format Korpus Perintah

**File:** `data/korpus/perintah_tingkat_1.txt` (sudah ada di model-naze2.0)

**Format:**
```
# Komentar: kategori perintah
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
```

**Catatan:**
- Setiap perintah dipisahkan oleh baris kosong
- Format: `perintah: <teks>` + `intent: <nama>` + opsional `target:`/`parameters:`
- Semua teks **lowercase**, tanpa tanda baca

---

## 5. Alur Data End-to-End

```
┌─────────────────────────────────────────────────────────────────┐
│                      NAZEIO APPLICATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐  │
│  │   AUDIO     │────▶│  SPEECH TO  │────▶│  TEXT           │  │
│  │   INPUT     │     │   TEXT       │     │  NORMALIZATION  │  │
│  └─────────────┘     └─────────────┘     └─────────────────┘  │
│           (spesifikasi 05)          │                │              │
│                                      ▼                ▼              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    NAZE ENGINE                                ││
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  ││
│  │  │  LOAD MODEL  │────▶│  INFERENCE   │────▶│  PARSE       │  ││
│  │  │  (params.npz)│     │  (Kotlin)    │     │  OUTPUT      │  ││
│  │  └─────────────┘     └─────────────┘     └─────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                        │                        │          │
│  ┌────────┴────────┐    ┌────────┴────────┐    ┌────────┴─────┐ │
│  │   INTENT        │    │  ALTERNATIVE   │    │   FALLBACK   │ │
│  │   MATCHING      │    │  INTENTS       │    │   TO Q&A     │ │
│  └────────┬────────┘    └─────────────────┘    └───────────────┘ │
│           │                                                        │
│  ┌────────▼────────────────────────────────────────────────────┐│
│  │                    ACTION EXECUTION                            ││
│  │  - Buka Aplikasi                                               ││
│  │  - Buka Pengaturan                                             ││
│  │  - Pasang Pengingat                                            ││
│  │  - ... (daftar intent Tingkat 1)                              ││
│  └───────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Kontrak Teknis

### 6.1 Kontrak Model

**model-naze2.0 menyediakan:**

1. **Config Model (D-018):**
   - `d_model = 64`
   - `num_heads = 4`
   - `num_layers = 2`
   - `d_ff = 128`
   - `max_sequence_length = 128`
   - `vocab_size = 256` (fixed, DECISION-015)
   - `seed = 0` (default, deterministik)

2. **Precision:**
   - Training: **float64** (DECISION-007)
   - Inference (nazeio): **float32** (untuk efisiensi)

3. **Ukuran:**
   - Checkpoint (float64): **~0.87 MB**
   - Checkpoint (float32): **~0.43 MB**
   - APK tambahan: **< 20 MB** (target nazeio)

4. **Determinisme:**
   - Semua operasi **deterministik per-seed** (DECISION-007)
   - Seed yang sama -> output yang sama

### 6.2 Kontrak Inference

**nazeio mengimplementasikan di Kotlin:**

```kotlin
// Interface yang harus diimplementasi
interface NazeEngine {
    // Inisialisasi: muat model dari assets
    fun init(assets: AssetManager, modelPath: String): Boolean
    
    // Proses teks: input -> intent
    fun process(text: String): NazeResult
    
    // Benchmark: ukur performa
    fun benchmark(texts: List<String>): BenchmarkResult
    
    // Status: apakah engine siap
    fun isReady(): Boolean
}

// Result structure
data class NazeResult(
    val status: String, // "success", "error", "fallback"
    val intent: Intent?, // null if not success
    val alternatives: List<IntentWithConfidence>,
    val processingTimeMs: Long,
    val modelVersion: String,
    val error: NazeError?
)

data class Intent(
    val name: String,
    val confidence: Float, // 0.0 - 1.0
    val action: String,
    val target: String?,
    val parameters: Map<String, String>
)

data class IntentWithConfidence(
    val name: String,
    val confidence: Float
)

data class NazeError(
    val code: String,
    val message: String
)
```

### 6.3 Kontrak Korpus Perintah

**model-naze2.0 menyediakan:**

1. **File:** `data/korpus/perintah_tingkat_1.txt`
2. **Format:** Lihat Section 4.2
3. **Isi:** 30+ perintah Tingkat 1 (sudah ada)
4. **Kategori:**
   - Buka aplikasi
   - Buka pengaturan
   - Pasang pengingat
   - Bluetooth
   - Tangkap layar
   - Pencarian Google
   - Widget dan animasi
   - Tanggal dan waktu
   - Telepon
   - Maps

**nazeio bertanggung jawab:**
- Menambahkan variasi perintah sesuai kebutuhan
- Memelihara daftar intent dan mapping ke aksi
- Memvalidasi korpus sebelum training

---

## 7. Error Handling

### 7.1 Error Codes

| Code | Deskripsi | Penanganan |
|------|-----------|------------|
| `MODEL_NOT_LOADED` | Model belum dimuat | Return status="error", fallback ke Q&A |
| `MODEL_CORRUPT` | Bobot model rusak | Return status="error", fallback ke Q&A |
| `INVALID_INPUT` | Input kosong/terlalu panjang | Return status="error" |
| `INFERENCE_TIMEOUT` | Inference terlalu lama | Return status="error", fallback |
| `NO_INTENT_MATCH` | Tidak ada intent yang cocok | Return status="fallback" |
| `LOW_CONFIDENCE` | Confidence < threshold | Return status="fallback" |

### 7.2 Timeout dan Cancellation

- **Timeout default:** 2 detik per request
- **Max sequence length:** 128 token (D-018)
- **Cancellation:** Dapat dibatalkan (thread-safe)

### 7.3 Logging

- **Tidak boleh** mencatat input user (privasi)
- **Boleh** mencatat:
  - Waktu pemrosesan
  - Status (success/error/fallback)
  - Model version
  - Error code (tanpa detail sensitif)

---

## 8. Pemisahan Tanggung Jawab

### 8.1 Speech-to-Text
- **Tanggung jawab:** nazeio (spesifikasi 05)
- **Model:** TIDAK menggunakan Naze
- **Input ke Naze Engine:** Teks (bukan audio)

### 8.2 Inference
- **Tanggung jawab:** nazeio (implementasi Kotlin)
- **Model:** Dari model-naze2.0 (bobot npz)
- **Engine:** Naze Engine (Kotlin)

### 8.3 Intent Matching
- **Tanggung jawab:** nazeio
- **Input:** Output model (token IDs -> teks)
- **Output:** Intent + confidence
- **Daftar Intent:** Dikelola nazeio

### 8.4 Action Execution
- **Tanggung jawab:** nazeio
- **Input:** Intent teridentifikasi
- **Output:** Aksi ter eksekusi

### 8.5 Text-to-Speech
- **Tanggung jawab:** nazeio (jika diperlukan)
- **Model:** TIDAK menggunakan Naze

### 8.6 Fallback ke Q&A
- **Tanggung jawab:** nazeio (spesifikasi 03)
- **Trigger:** status="fallback" atau "error"

---

## 9. Benchmark dan Metrik

### 9.1 Metrik Wajib (DECISION-023)

| Metrik | Target | Metode | Status |
|--------|--------|--------|--------|
| **Command Accuracy** | >= 90% | Daftar perintah uji | **WAJIB** (release gate) |
| **Inference Latency** | <= 2 detik | Raspberry Pi 3, config D-018 | Target optimasi |
| **APK Size** | <= 20 MB tambahan | Ukuran APK dengan model | Target optimasi |

### 9.2 Daftar Perintah Uji

**Sumber:** `data/korpus/perintah_tingkat_1.txt` + variasi

**Format uji:**
```json
{
  "tests": [
    {
      "input": "buka aplikasi whatsapp",
      "expected_intent": "BUKA_APLIKASI",
      "expected_target": "whatsapp",
      "min_confidence": 0.8
    },
    {
      "input": "pasang pengingat jam 10 pagi",
      "expected_intent": "PASANG_PENGINGAT",
      "expected_target": null,
      "min_confidence": 0.7
    }
  ]
}
```

### 9.3 Metode Evaluasi

1. **Akurasi:** `(jumlah benar) / (jumlah total) * 100%`
2. **Latency:** Rata-rata waktu per request (ms)
3. **Determinisme:** Output yang sama untuk input yang sama

---

## 10. Deployment

### 10.1 Local Inference (On-Device)

**Keuntungan:**
- Offline (tidak perlu internet)
- Cepat (tidak ada latency jaringan)
- Privat (data tidak keluar perangkat)

**Keterbatasan:**
- Model terbatas ukuran (<= 0.43 MB)
- Kapasitas terbatas (config D-018)
- Hanya perintah Tingkat 1

### 10.2 Remote Inference (Opsional)

**Tidak direkomendasikan** untuk nazeio karena:
- Menambah latency jaringan
- Memerlukan server
- Bertentangan dengan tujuan offline

**Jika diperlukan:**
- Gunakan API yang sama
- Model bisa lebih besar
- Memerlukan autentikasi

### 10.3 Model Versioning

- **Format:** SemVer (1.0.0, 1.0.1, dll.)
- **Lokasi:** `assets/models/naze_v{version}.npz`
- **Fallback:** Jika model baru gagal, gunakan model lama

---

## 11. Kompatibilitas Versi

### 11.1 Model Version

| Versi | Config | Ukuran | Status |
|-------|--------|--------|--------|
| 1.0.0 | D-018 | ~0.43 MB | Target M-010 |

### 11.2 Backward Compatibility

- **Bobot:** Format npz stabil, kompatibel dengan versi lama
- **API:** Kontrak JSON stabil
- **Intent:** Daftar intent bisa diperluas

### 11.3 Migration Path

- Model baru diunduh via update aplikasi
- Model lama tetap berfungsi sampai update
- Tidak ada breaking changes tanpa peringatan

---

## 12. Keamanan

### 12.1 Command Execution Safety

**ATURAN PENTING:**
- **TIDAK BOLEH** mengeksekusi perintah yang tidak terdaftar
- **TIDAK BOLEH** mengeksekusi perintah dengan confidence < threshold
- **TIDAK BOLEH** mengeksekusi aksi berbahaya (hapus file, dll.)
- **WAJIB** validasi input sebelum eksekusi
- **WAJIB** fallback ke Q&A jika tidak yakin

### 12.2 Sandboxing

- Naze Engine berjalan di thread terpisah
- Tidak ada akses ke:
  - File system (kecuali assets model)
  - Network
  - System commands
  - Personal data

### 12.3 Data Privacy

- **TIDAK BOLEH** mencatat input user
- **TIDAK BOLEH** mengirim input user ke server
- **TIDAK BOLEH** menyimpan input user
- Input diproses di memori, dibuang setelah selesai

---

## 13. Testing

### 13.1 Unit Tests (Kotlin)

- Test pemuatan model
- Test inference deterministik
- Test intent matching
- Test error handling
- Test timeout

### 13.2 Integration Tests

- Test end-to-end: audio -> teks -> intent -> aksi
- Test fallback ke Q&A
- Test performance (latency, memori)

### 13.3 Golden Tests

- Output model dibandingkan dengan referensi Python
- Toleransi: 1e-4 (float64 vs float32)
- Test determinisme (bit-exact untuk float32)

---

## 14. Capability Discovery

**model-naze2.0 menyediakan:**

```json
{
  "capabilities": {
    "model": {
      "type": "transformer",
      "config": {"d_model": 64, "num_heads": 4, "num_layers": 2, ...},
      "vocab_size": 256,
      "max_sequence_length": 128
    },
    "features": [
      "text_inference",
      "greedy_generation",
      "temperature_sampling"
    ],
    "limitations": [
      "no_speech_recognition",
      "no_image_understanding",
      "no_tool_calling"
    ]
  }
}
```

**nazeio menggunakan:**
- Mengecek capabilities sebelum penggunaan
- Menyesuaikan behavior berdasarkan fitur yang tersedia

---

## 15. Open Issues dan Assumsi

### 15.1 Assumsi yang Belum Dikonfirmasi

| ID | Assumsi | Status | Dampak |
|----|---------|--------|--------|
| A-001 | nazeio dapat mengimplementasi inference Kotlin | **UNCONFIRMED** | Blocker integrasi |
| A-002 | Format npz dapat dibaca di Kotlin | **UNCONFIRMED** | Blocker inference |
| A-003 | Model float32 cukup akurat | **UNCONFIRMED** | Risiko akurasi |
| A-004 | Raspberry Pi 3 adalah target ARMv7 | **UNCONFIRMED** | Risiko performa |

### 15.2 Keputusan yang Memerlukan Owner

| Keputusan | Pertanyaan | Status |
|-----------|------------|--------|
| OD-112 | Korpus final training Naze | **BLOCKING M-010** | Memengaruhi akurasi |
| OD-119 | Versioning policy | OPEN | Memengaruhi deployment |
| - | Threshold confidence untuk intent matching | OPEN | Memengaruhi akurasi |
| - | Daftar intent Tingkat 1 final | OPEN | Memengaruhi cakupan |

---

## 16. Ringkasan Kontrak

| Aspek | model-naze2.0 | nazeio |
|-------|---------------|--------|
| **Model** | ✅ Menyediakan (config D-018) | ⏳ Implementasi Kotlin |
| **Inference** | ✅ InferenceEngine (Python) | ⏳ Implementasi Kotlin |
| **Export** | ✅ export_model_weights | ⏳ Pembaca npz |
| **Korpus** | ✅ perintah_tingkat_1.txt | ⏳ Validasi + perluasan |
| **Benchmark** | ✅ benchmark_model | ⏳ Integrasi |
| **Testing** | ✅ 423 tests, 93.40% coverage | ⏳ Unit tests Kotlin |

**Status Integrasi:** **BLOCKED** oleh:
1. Implementasi Kotlin di nazeio (A-001, A-002)
2. Korpus final (OD-112)
3. Threshold confidence (belum ditentukan)

---

## 17. Next Steps

### 17.1 model-naze2.0

- [x] M-001..M-009 DONE
- [ ] Selesaikan OD-112 (korpus final)
- [ ] Train model final (T005)
- [ ] Validasi metrik DECISION-023 (T006, T008)
- [ ] Ekspor model float32 (T005 output)

### 17.2 nazeio

- [ ] Implementasi Naze Engine di Kotlin
- [ ] Implementasi pembaca npz
- [ ] Implementasi inference float32
- [ ] Integrasi dengan alur aplikasi
- [ ] Test end-to-end
- [ ] Benchmark ARMv7

### 17.3 Kolaborasi

- [ ] Verifikasi kontrak ini dengan owner nazeio
- [ ] Sinkronisasi daftar perintah
- [ ] Tentukan threshold confidence
- [ ] Tentukan format output final

---

## 18. Lampiran

### 18.1 Contoh Implementasi Python (model-naze2.0)

```python
# Ekspor model untuk nazeio
from naze.lm import TransformerLM, TransformerConfig
from naze.export import export_model_weights

config = TransformerConfig(
    d_model=64, num_heads=4, num_layers=2,
    d_ff=128, max_sequence_length=128, vocab_size=256, seed=0
)
model = TransformerLM(config)
# ... training ...
export_model_weights(model, "naze_v1.npz", version="1.0.0", float32=True)
```

### 18.2 Contoh Inferensi (model-naze2.0)

```python
from naze.inference import InferenceEngine, InferenceBatch
from naze.token import ByteTokenizer

engine = InferenceEngine(model)
tok = ByteTokenizer()

# Single prompt
prompt = tok.encode("buka aplikasi whatsapp")
batch = InferenceBatch([prompt], max_length=128)
output = engine.forward(batch)
generated_ids = engine.generate(prompt, max_new=10, temperature=0.8, seed=0)
generated_text = tok.decode(generated_ids)
```

### 18.3 Referensi

- [model-naze2.0 Repository](https://github.com/Ndolzz/model-naze2.0)
- [nazeio Repository](https://github.com/Ndolzz/nazeio)
- [nazeio specs/16_naze_engine.md](https://github.com/Ndolzz/nazeio/blob/main/specs/16_naze_engine.md)
- [nazeio docs/rencana_integrasi_naze_engine.md](https://github.com/Ndolzz/nazeio/blob/main/docs/rencana_integrasi_naze_engine.md)
