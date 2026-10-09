# NAZE MOTION AGENT INTEGRATION CONTRACT

> **Status:** DRAFT  berdasarkan spesifikasi naze-motion-agent yang tersedia
> **Repository:** `Ndolzz/naze-motion-agent` (eksternal)
> **Model Repository:** `Ndolzz/model-naze2.0` (M-001..M-009 DONE)
> **Tanggal:** 2026-10-09
> **Milestone:** M-010 (Stage 9)

---

## 1. Ringkasan

Dokumen ini mendefinisikan **kontrak integrasi** antara proyek **model-naze2.0** (sebagai provider model bahasa) dengan proyek **naze-motion-agent** (sebagai aplikasi motion/video editing). Kontrak ini berfokus pada bagaimana Naze dapat digunakan untuk memahami instruksi natural language dan menghasilkan **ActionPlan** yang valid untuk dieksekusi oleh motion agent.

**Tujuan:** Memungkinkan naze-motion-agent menggunakan model Naze untuk:
1. Memahami instruksi user dalam bahasa alami
2. Mengonversi instruksi ke ActionPlan terstruktur
3. Validasi ActionPlan sebelum eksekusi

---

## 2. Peran dan Tanggung Jawab

### 2.1 Peran model-naze2.0

| Komponen | Tanggung Jawab | Output |
|---------|---------------|--------|
| **Model Terlatih** | Menyediakan model untuk pemahaman bahasa | Bobot model (params.npz) |
| **Inference Engine** | Menyediakan forward-only inference | `InferenceEngine` class |
| **Text Generation** | Menghasilkan teks respons | `generate()` function |
| **Export Format** | Ekspor bobot ke format terstandar | `params.npz` + metadata |

### 2.2 Peran naze-motion-agent

| Komponen | Tanggung Jawab | Output |
|---------|---------------|--------|
| **AI Planner** | Memahami instruksi, menghasilkan ActionPlan | `ActionPlan` object |
| **AI Provider** | Abstraksi provider AI (NMA-AI-003) | `AIProvider` interface |
| **Action Dispatcher** | Mengeksekusi ActionPlan (NMA-ACTION-003) | Executed actions |
| **Schema Validator** | Validasi ActionPlan (NMA-AI-004) | Validated ActionPlan |
| **Safety Validator** | Validasi keamanan (NMA-AI-005) | Safe/Rejected |

---

## 3. Arsitektur naze-motion-agent (Berdasarkan Dokumen)

### 3.1 AI Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   NAZE MOTION AGENT                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    AI PLANNER                                ││
│  │  ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐││
│  │  │ USER INPUT  │────▶│  AI PROVIDER     │────▶│  JSON       │││
│  │  │ (instruction)│     │  (abstraction)   │     │  PARSER     │││
│  │  └─────────────┘     └─────────────────┘     └─────────────┘││
│  │                          │                        │             ││
│  │                          ▼                        ▼             ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │                    ACTION PLAN                               │││
│  │  │  - task: "create_video", "edit_video", etc.                 │││
│  │  │  - actions: [Action]                                        │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                        │
│                          ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              VALIDATION PIPELINE                              ││
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  ││
│  │  │ SCHEMA      │────▶│ SAFETY       │────▶│ ACTION      │  ││
│  │  │ VALIDATOR   │     │ VALIDATOR    │     │ VALIDATOR   │  ││
│  │  └─────────────┘     └─────────────┘     └─────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                        │
│                          ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              ACTION DISPATCHER                               ││
│  │  - OPEN_APP, WAIT, TAP, LONG_PRESS, SWIPE, SCROLL, TYPE_TEXT  ││
│  │  - PRESS_BACK, SCREENSHOT, FIND_ELEMENT, CREATE_PROJECT        ││
│  │  - ADD_MEDIA, ADD_TEXT, EXPORT (NMA-ACTION-002)                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 AI Provider Interface (NMA-AI-003)

```kotlin
interface AIProvider {
    fun generate(
        prompt: String,
        context: Map<String, String> = emptyMap(),
        timeoutMs: Long = 30000,
        onPartial: (String) -> Unit = {},
        onError: (Throwable) -> Unit = {}
    ): Flow<String>
    
    fun close()
}

// Implementasi untuk Naze
class NazeAIProvider(
    private val modelPath: String,
    private val config: NazeConfig
) : AIProvider {
    override fun generate(prompt: String, ...): Flow<String> {
        // Load model, run inference, yield tokens
    }
}
```

---

## 4. Integrasi Naze sebagai AI Provider

### 4.1 Peran Naze di naze-motion-agent

Naze **BUKAN** pengganti penuh untuk provider LLM eksternal (Gemini, Claude, dll.), tetapi dapat digunakan untuk:

1. **Pemahaman Instruksi Sederhana:**
   - "Buat video dengan gambar A, B, C"
   - "Tambahkan teks 'Hello' di tengah"
   - "Potong video dari detik 10 sampai 20"

2. **Generasi ActionPlan Terstruktur:**
   - Output dalam format JSON yang sesuai dengan ACTION_SPEC
   - Validasi schema sebelum pengiriman

3. **Offline Capability:**
   - Berfungsi tanpa internet
   - Cepat (target: < 2 detik)

### 4.2 Batasan Naze

Naze **TIDAK** dapat:
- Memahami gambar/video (tidak ada computer vision)
- Melakukan rendering video (tidak ada rendering engine)
- Mengakses file system secara langsung
- Mengeksekusi perintah sistem
- Menggunakan API eksternal

**Semua kemampuan ini tetap ditangani oleh naze-motion-agent.**

---

## 5. Kontrak Input/Output

### 5.1 Input ke Naze AI Provider

```json
{
  "prompt": "Buat video dengan gambar foto1.jpg, foto2.jpg, foto3.jpg dengan durasi 5 detik masing-masing",
  "context": {
    "current_project": "project_123",
    "available_media": ["foto1.jpg", "foto2.jpg", "foto3.jpg", "musik1.mp3"],
    "current_state": "idle",
    "device_info": {
      "os": "Android",
      "version": "14"
    }
  },
  "options": {
    "temperature": 0.8,
    "max_tokens": 256,
    "stream": false
  }
}
```

**Keterangan:**
- `prompt`: Instruksi user dalam bahasa alami
- `context`: Konteks aplikasi (opsional, untuk grounding)
- `options`: Opsi inference (opsional)

### 5.2 Output dari Naze AI Provider

**Format 1: Streaming (jika stream=true)**
```
"{\n  \"task\": \"create_video\",\n  \"actions\": [\n    {\n      \"type\": \"CREATE_PROJECT\",\n      \"pa"
"{\n  \"task\": \"create_video\",\n  \"actions\": [\n    {\n      \"type\": \"CREATE_PROJECT\",\n      \"pa"
```

**Format 2: Non-streaming (default)**
```json
{
  "task": "create_video",
  "actions": [
    {
      "type": "CREATE_PROJECT",
      "parameters": {
        "name": "video_baru",
        "resolution": "1920x1080",
        "fps": 30
      }
    },
    {
      "type": "ADD_MEDIA",
      "parameters": {
        "path": "foto1.jpg",
        "duration": 5.0,
        "start_time": 0.0
      }
    },
    {
      "type": "ADD_MEDIA",
      "parameters": {
        "path": "foto2.jpg",
        "duration": 5.0,
        "start_time": 5.0
      }
    },
    {
      "type": "ADD_MEDIA",
      "parameters": {
        "path": "foto3.jpg",
        "duration": 5.0,
        "start_time": 10.0
      }
    },
    {
      "type": "EXPORT",
      "parameters": {
        "format": "mp4",
        "path": "output/video_baru.mp4"
      }
    }
  ],
  "metadata": {
    "model": "naze-1.0.0",
    "tokens_used": 42,
    "processing_time_ms": 1500
  }
}
```

---

## 6. Schema ActionPlan

### 6.1 Action Types (NMA-ACTION-002)

| Type | Deskripsi | Parameter Wajib |
|------|-----------|-----------------|
| `OPEN_APP` | Buka aplikasi | `package_name` |
| `WAIT` | Tunggu | `duration_ms` |
| `TAP` | Ketuk layar | `x`, `y` |
| `LONG_PRESS` | Tekan lama | `x`, `y`, `duration_ms` |
| `SWIPE` | Geser | `x1`, `y1`, `x2`, `y2`, `duration_ms` |
| `SCROLL` | Scroll | `direction`, `amount` |
| `TYPE_TEXT` | Ketik teks | `text` |
| `PRESS_BACK` | Tekan tombol back | - |
| `SCREENSHOT` | Tangkap layar | `path` (opsional) |
| `FIND_ELEMENT` | Cari elemen | `text` |
| `CREATE_PROJECT` | Buat proyek baru | `name`, `resolution`, `fps` |
| `ADD_MEDIA` | Tambah media | `path`, `duration` |
| `ADD_TEXT` | Tambah teks | `text`, `style`, `position` |
| `EXPORT` | Ekspor proyek | `format`, `path` |

### 6.2 Action Schema

```json
{
  "type": "ADD_MEDIA",
  "parameters": {
    "path": "string",
    "duration": "number",
    "start_time": "number",
    "volume": "number (0.0-1.0)",
    "fade_in": "number (ms)",
    "fade_out": "number (ms)"
  },
  "timeout_ms": 5000,
  "retry_policy": {
    "max_attempts": 2
  },
  "verification": {
    "type": "media_loaded",
    "expected": true
  }
}
```

### 6.3 Full ActionPlan Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "task": {
      "type": "string",
      "enum": ["create_video", "edit_video", "export_video", "add_effect", "remove_effect"]
    },
    "actions": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/action"
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "model": {"type": "string"},
        "tokens_used": {"type": "integer"},
        "processing_time_ms": {"type": "integer"}
      }
    }
  },
  "definitions": {
    "action": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": ["OPEN_APP", "WAIT", "TAP", "LONG_PRESS", "SWIPE", "SCROLL", "TYPE_TEXT", "PRESS_BACK", "SCREENSHOT", "FIND_ELEMENT", "CREATE_PROJECT", "ADD_MEDIA", "ADD_TEXT", "EXPORT"]
        },
        "parameters": {
          "type": "object"
        },
        "timeout_ms": {
          "type": "integer",
          "default": 5000
        },
        "retry_policy": {
          "type": "object",
          "properties": {
            "max_attempts": {"type": "integer", "default": 2}
          }
        },
        "verification": {
          "type": "object"
        }
      },
      "required": ["type"]
    }
  },
  "required": ["task", "actions"]
}
```

---

## 7. Validasi dan Safety

### 7.1 Schema Validation (NMA-AI-004)

**WAJIB:**
- Semua ActionPlan **WAJIB** valid sesuai schema
- Malformed JSON **WAJIB** ditolak
- Unknown action type **WAJIB** ditolak (NMA-ACTION-009)

**Contoh Error:**
```json
{
  "error": "schema_validation_failed",
  "message": "Unknown action type: PLAY_VIDEO",
  "path": "/actions/2/type",
  "valid_types": ["OPEN_APP", "WAIT", ...]
}
```

### 7.2 Safety Validation (NMA-AI-005)

**WAJIB ditolak:**
- Perintah yang berbahaya (hapus file, format device, dll.)
- Perintah yang tidak didukung (akses sistem, dll.)
- Perintah yang ambigu (tidak jelas action-nya)

**Contoh Safety Rules:**
```kotlin
val SAFETY_RULES = listOf<
    SafetyRule(
        pattern = "hapus.*file|delete.*file|format.*device",
        action = REJECT,
        reason = "Dangerous: file deletion"
    ),
    SafetyRule(
        pattern = "akses.*root|sudo|admin",
        action = REJECT,
        reason = "Dangerous: system access"
    ),
    SafetyRule(
        pattern = ".*",
        maxTokens = 512,
        action = REJECT_IF_EXCEED,
        reason = "Too long"
    )
>
```

### 7.3 Action Validation (NMA-AI-004)

**WAJIB:**
- Parameter action **WAJIB** valid
- Target action **WAJIB** dapat diselesaikan
- Action **WAJIB** didukung oleh aplikasi

**Contoh:**
```json
{
  "error": "action_validation_failed",
  "message": "File 'foto999.jpg' not found",
  "action": "ADD_MEDIA",
  "parameter": "path"
}
```

---

## 8. Kontrak Teknis

### 8.1 Kontrak Model (Sama dengan NazeIO)

- **Config:** D-018 (D=64, H=4, L=2, d_ff=128, T_max=128, vocab=256)
- **Precision:** float64 (training), float32 (inference)
- **Ukuran:** ~0.43 MB (float32)
- **Determinisme:** Per-seed

### 8.2 Kontrak Inference

**Naze AI Provider mengimplementasikan:**

```kotlin
class NazeAIProvider : AIProvider {
    override fun generate(
        prompt: String,
        context: Map<String, String>,
        timeoutMs: Long,
        onPartial: (String) -> Unit,
        onError: (Throwable) -> Unit
    ): Flow<String> {
        // 1. Tokenize input
        val tokens = tokenizer.encode(prompt)
        
        // 2. Run inference
        val outputTokens = inferenceEngine.generate(
            promptIds = tokens,
            maxNew = 256,
            temperature = 0.8f,
            seed = 42
        )
        
        // 3. Decode to text
        val response = tokenizer.decode(outputTokens)
        
        // 4. Emit as flow
        emit(response)
    }
}
```

### 8.3 Kontrak Output Format

**Naze menghasilkan teks JSON yang valid:**
- **WAJIB** valid JSON
- **WAJIB** sesuai dengan ACTION_SPEC schema
- **WAJIB** dapat di-parse oleh JSON parser

**Jika Naze gagal:**
- **TIDAK BOLEH** mengembalikan teks random
- **WAJIB** mengembalikan error yang jelas
- **WAJIB** fallback ke provider lain (jika tersedia)

---

## 9. Pemisahan Tanggung Jawab

### 9.1 Naze (model-naze2.0)

**Bertanggung jawab untuk:**
- Memahami instruksi natural language
- Menghasilkan teks respons (JSON ActionPlan)
- Menyediakan model yang akurat
- Determinisme dan reproducibility

**TIDAK bertanggung jawab untuk:**
- Validasi ActionPlan (tanggung jawab naze-motion-agent)
- Eksekusi ActionPlan (tanggung jawab naze-motion-agent)
- Safety validation (tanggung jawab naze-motion-agent)
- Rendering video (tanggung jawab naze-motion-agent)
- File system access (tanggung jawab naze-motion-agent)

### 9.2 naze-motion-agent

**Bertanggung jawab untuk:**
- Validasi ActionPlan (schema, safety, action)
- Eksekusi ActionPlan (ActionDispatcher)
- Rendering video
- File system access
- User interface

**TIDAK bertanggung jawab untuk:**
- Pemahaman bahasa (dapat menggunakan Naze)
- Generasi teks (dapat menggunakan Naze)

---

## 10. Benchmark dan Metrik

### 10.1 Metrik Wajib

| Metrik | Target | Metode | Status |
|--------|--------|--------|--------|
| **Instruksi Valid** | >= 85% | Instruksi editing yang valid | **WAJIB** |
| **ActionPlan Valid** | >= 90% | ActionPlan sesuai schema | **WAJIB** |
| **Inference Latency** | <= 2 detik | Raspberry Pi 3, config D-018 | Target |

### 10.2 Dataset Benchmark

**Contoh instruksi untuk benchmark:**

```
# Instruksi Sederhana
"Buat video kosong dengan durasi 10 detik"
"Tambahkan gambar foto1.jpg ke proyek"
"Ekspor proyek ke format MP4"

# Instruksi Kompleks
"Buat video dengan gambar foto1.jpg, foto2.jpg, foto3.jpg masing-masing 3 detik, lalu tambahkan teks 'Hello World' di tengah, dan ekspor ke output.mp4"

# Instruksi dengan Parameter
"Potong video dari detik 5 sampai 15 dan simpan sebagai video_baru.mp4"
"Tambahkan efek fade-in selama 1 detik pada gambar pertama"

# Instruksi Invalid (harus ditolak)
"Hapus semua file di perangkat"
"Format ulang hard drive"
"Jalankan sebagai root"
```

### 10.3 Metode Evaluasi

1. **Akurasi Instruksi:**
   - `(jumlah instruksi valid) / (jumlah total) * 100%`
   - Instruksi valid = ActionPlan sesuai schema + safety valid

2. **Akurasi ActionPlan:**
   - `(jumlah ActionPlan benar) / (jumlah total) * 100%`
   - ActionPlan benar = action yang tepat untuk instruksi

3. **Latency:**
   - Rata-rata waktu dari input sampai ActionPlan siap

---

## 11. Deployment

### 11.1 Local Inference (On-Device)

**Keuntungan:**
- Offline (tidak perlu internet)
- Cepat (tidak ada latency jaringan)
- Privat (data tidak keluar perangkat)

**Keterbatasan:**
- Model terbatas ukuran (<= 0.43 MB)
- Kapasitas terbatas (config D-018)
- Hanya instruksi sederhana

### 11.2 Remote Inference (Opsional)

**Untuk instruksi kompleks:**
- Gunakan provider LLM eksternal (Gemini, Claude)
- Naze dapat digunakan sebagai fallback untuk instruksi sederhana
- Memerlukan API key (NMA-AI-007)

### 11.3 Model Versioning

- **Format:** SemVer (1.0.0, 1.0.1, dll.)
- **Lokasi:** `assets/models/naze_v{version}.npz`
- **Fallback:** Jika model baru gagal, gunakan model lama

---

## 12. Keamanan

### 12.1 Safety First

**ATURAN PENTING:**
- **TIDAK BOLEH** mengeksekusi ActionPlan yang tidak divalidasi
- **TIDAK BOLEH** mengeksekusi action yang tidak didukung
- **TIDAK BOLEH** mengeksekusi perintah berbahaya
- **WAJIB** validasi schema sebelum parsing
- **WAJIB** validasi safety sebelum eksekusi
- **WAJIB** validasi action sebelum eksekusi

### 12.2 Sandboxing

- Naze berjalan di thread terpisah
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

### 13.1 Unit Tests

- Test tokenize/decode
- Test inference deterministik
- Test JSON generation
- Test schema validation
- Test safety validation

### 13.2 Integration Tests

- Test end-to-end: instruksi -> ActionPlan -> eksekusi
- Test fallback ke provider lain
- Test performance (latency, memori)

### 13.3 Golden Tests

- Output model dibandingkan dengan referensi
- Toleransi: 1e-4 (float64 vs float32)
- Test determinisme (bit-exact untuk float32)

---

## 14. Capability Discovery

**Naze mendeklarasikan kemampuannya:**

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
      "text_generation",
      "json_output",
      "action_plan_generation",
      "deterministic_output"
    ],
    "limitations": [
      "no_image_understanding",
      "no_video_understanding",
      "no_tool_calling",
      "no_system_access",
      "limited_context_window"
    ],
    "domain_knowledge": [
      "video_editing_instructions",
      "basic_commands",
      "indonesian_language"
    ]
  }
}
```

**naze-motion-agent menggunakan:**
- Mengecek capabilities sebelum penggunaan
- Menyesuaikan behavior berdasarkan fitur yang tersedia
- Fallback ke provider lain jika Naze tidak mendukung

---

## 15. Open Issues dan Assumsi

### 15.1 Assumsi yang Belum Dikonfirmasi

| ID | Assumsi | Status | Dampak |
|----|---------|--------|--------|
| B-001 | Naze dapat menghasilkan JSON yang valid | **UNCONFIRMED** | Blocker integrasi |
| B-002 | Naze dapat memahami instruksi editing | **UNCONFIRMED** | Risiko akurasi |
| B-003 | ActionPlan dari Naze valid | **UNCONFIRMED** | Risiko safety |
| B-004 | Latency Naze <= 2 detik di ARMv7 | **UNCONFIRMED** | Risiko UX |

### 15.2 Keputusan yang Memerlukan Owner

| Keputusan | Pertanyaan | Status |
|-----------|------------|--------|
| OD-112 | Korpus final training Naze | **BLOCKING M-010** | Memengaruhi akurasi |
| - | Apakah Naze akan digunakan untuk editing? | OPEN | Memengaruhi prioritas |
| - | Threshold confidence untuk ActionPlan | OPEN | Memengaruhi safety |
| - | Daftar action types yang didukung | OPEN | Memengaruhi cakupan |

---

## 16. Perbedaan dengan NazeIO

| Aspek | NazeIO | Naze Motion Agent |
|-------|--------|-------------------|
| **Tujuan** | Asisten suara | Aplikasi editing |
| **Input** | Perintah Tingkat 1 | Instruksi editing |
| **Output** | Intent + target | ActionPlan terstruktur |
| **Action Types** | Buka app, pengaturan, dll. | CREATE_PROJECT, ADD_MEDIA, dll. |
| **Fallback** | Q&A (spesifikasi 03) | Provider LLM eksternal |
| **Offline** | ✅ Wajib | ✅ Diharapkan |

---

## 17. Ringkasan Kontrak

| Aspek | model-naze2.0 | naze-motion-agent |
|-------|---------------|-------------------|
| **Model** | ✅ Menyediakan (config D-018) | ⏳ Integrasi |
| **Inference** | ✅ InferenceEngine | ⏳ AI Provider |
| **Output Format** | ✅ Teks/JSON | ⏳ Parser ActionPlan |
| **Validation** | ❌ Tidak bertanggung jawab | ✅ Schema + Safety + Action |
| **Execution** | ❌ Tidak bertanggung jawab | ✅ ActionDispatcher |
| **Benchmark** | ✅ benchmark_model | ⏳ Dataset instruksi |

**Status Integrasi:** **BLOCKED** oleh:
1. Kemampuan Naze untuk memahami instruksi editing (B-001, B-002)
2. Korpus final dengan instruksi editing (OD-112)
3. Validasi ActionPlan (B-003)

---

## 18. Rekomendasi

### 18.1 Untuk model-naze2.0

1. **Training dengan Korpus Editing:**
   - Tambahkan korpus instruksi editing ke training data
   - Fine-tune model untuk domain editing
   - Validasi output JSON

2. **Output JSON:**
   - Train model untuk menghasilkan JSON yang valid
   - Gunakan prompt engineering: "Generate valid JSON ActionPlan for video editing: ..."
   - Validasi output sebelum training

3. **Benchmark:**
   - Buat dataset instruksi editing
   - Ukur akurasi ActionPlan
   - Optimasi threshold confidence

### 18.2 Untuk naze-motion-agent

1. **Integrasi Naze:**
   - Implementasi NazeAIProvider
   - Tambah Naze sebagai opsi provider
   - Fallback ke provider lain jika gagal

2. **Validasi:**
   - Schema validation (NMA-AI-004)
   - Safety validation (NMA-AI-005)
   - Action validation (NMA-ACTION-009)

3. **Testing:**
   - Test integrasi Naze
   - Test fallback
   - Test performance

---

## 19. Lampiran

### 19.1 Contoh Instruksi dan ActionPlan

**Instruksi:**
```
"Buat video dengan gambar foto1.jpg dan foto2.jpg masing-masing 3 detik, lalu tambahkan teks 'Hello' di tengah, dan ekspor ke output.mp4"
```

**ActionPlan (target):**
```json
{
  "task": "create_video",
  "actions": [
    {
      "type": "CREATE_PROJECT",
      "parameters": {
        "name": "video_baru",
        "resolution": "1920x1080",
        "fps": 30
      }
    },
    {
      "type": "ADD_MEDIA",
      "parameters": {
        "path": "foto1.jpg",
        "duration": 3.0,
        "start_time": 0.0
      }
    },
    {
      "type": "ADD_MEDIA",
      "parameters": {
        "path": "foto2.jpg",
        "duration": 3.0,
        "start_time": 3.0
      }
    },
    {
      "type": "ADD_TEXT",
      "parameters": {
        "text": "Hello",
        "style": "default",
        "position": "center",
        "start_time": 1.5,
        "duration": 3.0
      }
    },
    {
      "type": "EXPORT",
      "parameters": {
        "format": "mp4",
        "path": "output.mp4"
      }
    }
  ],
  "metadata": {
    "model": "naze-1.0.0",
    "tokens_used": 89,
    "processing_time_ms": 1800
  }
}
```

### 19.2 Referensi

- [model-naze2.0 Repository](https://github.com/Ndolzz/model-naze2.0)
- [naze-motion-agent Repository](https://github.com/Ndolzz/naze-motion-agent)
- [naze-motion-agent docs/AI_AGENT_SPEC.md](https://github.com/Ndolzz/naze-motion-agent/blob/main/docs/AI_AGENT_SPEC.md)
- [naze-motion-agent docs/ACTION_SPEC.md](https://github.com/Ndolzz/naze-motion-agent/blob/main/docs/ACTION_SPEC.md)
