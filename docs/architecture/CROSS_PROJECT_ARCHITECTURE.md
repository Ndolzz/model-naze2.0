# CROSS-PROJECT ARCHITECTURE

> **Status:** DRAFT  Arsitektur integrasi lintas proyek
> **Repository Utama:** `Ndolzz/model-naze2.0`
> **Proyek Terintegrasi:** `Ndolzz/nazeio`, `Ndolzz/naze-motion-agent`
> **Tanggal:** 2026-10-09
> **Milestone:** M-010 (Stage 9)

---

## 1. Ringkasan

Dokumen ini mendefinisikan **arsitektur integrasi lintas proyek** untuk ekosistem Naze. Tujuannya adalah:
1. Menghindari ketergantungan langsung yang tidak perlu
2. Memastikan kontrak API/interface yang jelas
3. Memungkinkan evolusi independen masing-masing proyek
4. Menjaga kompatibilitas backward

---

## 2. Prinsip Arsitektur

### 2.1 Decoupling
- **TIDAK ADA** ketergantungan langsung antara proyek
- Setiap proyek memiliki **interface yang stabil**
- Komunikasi via **kontrak yang terdefinisi**

### 2.2 Single Responsibility
- **model-naze2.0:** Model AI (training, inference, export)
- **nazeio:** Aplikasi asisten suara (speech-to-text, intent matching, action execution)
- **naze-motion-agent:** Aplikasi editing (AI planning, action dispatching, rendering)

### 2.3 Versioning
- Semua kontrak **berversi**
- Backward compatibility **wajib**
- Breaking changes **dilarang** tanpa peringatan

### 2.4 Offline-First
- Prioritas: **Local inference** (on-device)
- Remote inference: **Opsional** (fallback)
- Semua proyek **bisa berfungsi offline**

---

## 3. Arsitektur Lintas Proyek

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ECOSYSTEM NAZE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐     ┌─────────────────────────┐         │
│  │    model-naze2.0         │     │    nazeio               │         │
│  │  (Core Model Provider)   │     │  (Voice Assistant)      │         │
│  │                         │     │                         │         │
│  │  ┌───────────────────┐  │◄────┤  ┌───────────────────┐  │         │
│  │  │  Model Training     │  │     │  │  Naze Engine       │  │         │
│  │  │  (Python + NumPy)   │  │     │  │  (Kotlin)          │  │         │
│  │  └───────────────────┘  │     │  └───────────────────┘  │         │
│  │                         │     │                         │         │
│  │  ┌───────────────────┐  │     │  ┌───────────────────┐  │         │
│  │  │  Inference Engine  │  │     │  │  Speech-to-Text    │  │         │
│  │  │  (Python)          │  │     │  │  (spesifikasi 05)  │  │         │
│  │  └───────────────────┘  │     │  └───────────────────┘  │         │
│  │                         │     │                         │         │
│  │  ┌───────────────────┐  │     │  ┌───────────────────┐  │         │
│  │  │  Model Export      │  │────▶│  │  Intent Matching   │  │         │
│  │  │  (params.npz)      │  │     │  │  (Tingkat 1)       │  │         │
│  │  └───────────────────┘  │     │  └───────────────────┘  │         │
│  │                         │     │                         │         │
│  │  ┌───────────────────┐  │     │  ┌───────────────────┐  │         │
│  │  │  Benchmark         │  │     │  │  Action Execution  │  │         │
│  │  │  (Python)          │  │     │  │  (Local)          │  │         │
│  │  └───────────────────┘  │     │  └───────────────────┘  │         │
│  └─────────────────────────┘     └─────────────────────────┘         │
│           │                              │                              │         │
│           │                              │                              │         │
│           ▼                              ▼                              ▼         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        SHARED CONTRACTS                              ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  ││
│  │  │  Model      │  │  Inference  │  │  ActionPlan / Intent Schema      │  ││
│  │  │  Format    │  │  Interface  │  │  (JSON Schema)                   │  ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │    naze-motion-agent                                                  │
│  │  (Video Editing Application)                                          │
│  │                                                                     │
│  │  ┌───────────────────┐     ┌───────────────────┐     ┌──────────┐│
│  │  │  AI Planner        │     │  Action Dispatcher  │     │  Render  ││
│  │  │  (Naze/External)   │────▶│  (Executor)         │────▶│  Engine  ││
│  │  └───────────────────┘     └───────────────────┘     └──────────┘│
│  │                         │                             │            │
│  │  ┌───────────────────┐     ┌───────────────────┐              │
│  │  │  Schema Validator  │◄────┤  Safety Validator   │◄─────────────┘
│  │  └───────────────────┘     └───────────────────┘              │
│  └─────────────────────────────────────────────────────────────────────┘
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Kontrak Bersama

### 4.1 Kontrak Model Format

**Tujuan:** Format bobot model yang konsisten lintas proyek.

**Spesifikasi:**
- **Format:** NumPy .npz (zip archive)
- **Precision:** float32 (untuk inference di perangkat)
- **Metadata:** JSON string di kunci `__metadata__`
- **Checksum:** SHA-256 (untuk integritas)

**Schema Metadata:**
```json
{
  "schema_version": "1.0",
  "model_version": "1.0.0",
  "naze_version": "1.0.0",
  "config": {
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "d_ff": 128,
    "max_sequence_length": 128,
    "vocab_size": 256
  },
  "checksum_sha256": "abc123...",
  "export_timestamp": "2026-10-09T10:00:00Z",
  "training_metrics": {
    "final_loss": 2.3,
    "final_perplexity": 10.0
  }
}
```

**Kunci Parameter:** Lihat [NAZEIO_INTEGRATION_CONTRACT.md](NAZEIO_INTEGRATION_CONTRACT.md#41-format-bobot-model-npz)

---

### 4.2 Kontrak Inference Interface

**Tujuan:** Antarmuka inference yang konsisten.

**Abstraksi:**
```
interface InferenceProvider {
    // Inisialisasi
    init(modelPath: String, config: InferenceConfig) -> Result<Unit, InferenceError>
    
    // Inference
    infer(input: InferenceInput) -> Result<InferenceOutput, InferenceError>
    
    // Generate
    generate(prompt: String, options: GenerateOptions) -> Result<String, InferenceError>
    
    // Benchmark
    benchmark(input: BenchmarkInput) -> BenchmarkResult
    
    // Status
    isReady() -> Boolean
    
    // Cleanup
    close()
}
```

**Implementasi per Proyek:**
- **model-naze2.0:** `InferenceEngine` (Python)
- **nazeio:** `NazeEngine` (Kotlin, implementasi Kotlin)
- **naze-motion-agent:** `NazeAIProvider` (Kotlin, wrapper)

---

### 4.3 Kontrak Schema Output

**Tujuan:** Format output yang konsisten untuk intent/action.

#### 4.3.1 Intent Schema (untuk nazeio)

```json
{
  "$schema": "https://naze.io/schemas/intent/v1.json",
  "type": "object",
  "properties": {
    "intent": {
      "type": "string",
      "enum": ["BUKA_APLIKASI", "BUKA_PENGATURAN", "PASANG_PENGINGAT", ...]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "action": {"type": "string"},
    "target": {"type": ["string", "null"]},
    "parameters": {
      "type": "object",
      "additionalProperties": true
    }
  },
  "required": ["intent", "confidence"]
}
```

#### 4.3.2 ActionPlan Schema (untuk naze-motion-agent)

```json
{
  "$schema": "https://naze.io/schemas/actionplan/v1.json",
  "type": "object",
  "properties": {
    "task": {
      "type": "string",
      "enum": ["create_video", "edit_video", "export_video", ...]
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
        }
      },
      "required": ["type"]
    }
  },
  "required": ["task", "actions"]
}
```

---

## 5. Adapter Per Aplikasi

### 5.1 Adapter untuk nazeio

**Tujuan:** Mengadaptasi `InferenceEngine` (Python) ke `NazeEngine` (Kotlin).

**Arsitektur:**
```
┌─────────────────┐     ┌─────────────────┐
│  model-naze2.0   │     │    nazeio       │
│  (Python)        │     │  (Kotlin)       │
│                 │     │                 │
│  ┌─────────────┐│     │  ┌─────────────┐│
│  │ Inference  │◄┼────▶│  │ Naze Engine  ││
│  │ Engine     ││     │  │ (Kotlin)     ││
│  └─────────────┘│     │  └─────────────┘│
│                 │     │                 │
│  ┌─────────────┐│     │  ┌─────────────┐│
│  │ export.py   │◄─────┤  │ npz Reader   ││
│  │ (export)    │      │  │ (Kotlin)     ││
│  └─────────────┘      │  └─────────────┘│
└─────────────────┘     └─────────────────┘
```

**Implementasi:**
1. **Ekspor di Python:** `export_model_weights()` -> `params.npz`
2. **Impor di Kotlin:** Baca `params.npz`, load ke tensor
3. **Inference di Kotlin:** Implementasi forward pass di Kotlin
4. **Mapping Output:** Token IDs -> Intent

---

### 5.2 Adapter untuk naze-motion-agent

**Tujuan:** Mengadaptasi Naze sebagai AI Provider.

**Arsitektur:**
```
┌─────────────────┐     ┌─────────────────────────────┐
│  model-naze2.0   │     │  naze-motion-agent            │
│  (Python)        │     │  (Kotlin)                     │
│                 │     │                             │
│  ┌─────────────┐│     │  ┌─────────────────────────┐│
│  │ Inference  │◄┼────▶│  │  NazeAIProvider           ││
│  │ Engine     ││     │  │  (implements AIProvider)  ││
│  └─────────────┘│     │  └─────────────────────────┘│
│                 │     │                             │
│  ┌─────────────┐│     │  ┌─────────────────────────┐│
│  │ export.py   │◄─────┤  │  JSON Parser              ││
│  │ (export)    │      │  │  (ActionPlan validation)   ││
│  └─────────────┘      │  └─────────────────────────┘│
└─────────────────┘     └─────────────────────────────┘
```

**Implementasi:**
1. **Ekspor di Python:** `export_model_weights()` -> `params.npz`
2. **Impor di Kotlin:** Baca `params.npz`, load ke tensor
3. **Inference di Kotlin:** Generate teks JSON
4. **Parsing di Kotlin:** Parse JSON -> ActionPlan
5. **Validation:** Schema + Safety + Action validation

---

## 6. Error Codes dan Logging

### 6.1 Error Codes Bersama

| Code | Deskripsi | Severity |
|------|-----------|----------|
| `MODEL_NOT_FOUND` | Model tidak ditemukan | HIGH |
| `MODEL_CORRUPT` | Bobot model rusak | HIGH |
| `MODEL_VERSION_UNSUPPORTED` | Versi model tidak didukung | MEDIUM |
| `INVALID_INPUT` | Input tidak valid | MEDIUM |
| `INFERENCE_TIMEOUT` | Inference timeout | MEDIUM |
| `INVALID_OUTPUT` | Output tidak valid | HIGH |
| `SCHEMA_VALIDATION_FAILED` | Output tidak sesuai schema | HIGH |
| `SAFETY_VALIDATION_FAILED` | Output berbahaya | HIGH |
| `ACTION_VALIDATION_FAILED` | Action tidak valid | MEDIUM |

### 6.2 Logging Policy

**TIDAK BOLEH dilog:**
- Input user (privasi)
- Output yang mengandung data sensitif
- API keys atau credentials
- Path file lokal (kecuali untuk debugging dengan izin)

**BOLEH dilog:**
- Waktu pemrosesan
- Status (success/error)
- Model version
- Error code (tanpa detail sensitif)
- Metrik performa (latency, tokens, dll.)

**Format Log:**
```json
{
  "timestamp": "2026-10-09T10:00:00Z",
  "level": "INFO" | "WARN" | "ERROR",
  "component": "NazeEngine" | "InferenceProvider",
  "event": "inference_start" | "inference_complete" | "error",
  "model_version": "1.0.0",
  "processing_time_ms": 150,
  "status": "success" | "error",
  "error_code": null | "MODEL_NOT_FOUND",
  "session_id": "uuid"
}
```

---

## 7. Capability Discovery

### 7.1 Discovery Mechanism

**Setiap proyek menyediakan manifest capabilities:**

```json
{
  "capabilities": {
    "version": "1.0",
    "features": [
      {
        "name": "text_inference",
        "description": "Generate text from text input",
        "supported": true,
        "limitations": ["max_length=128", "vocab=256"]
      },
      {
        "name": "json_generation",
        "description": "Generate structured JSON output",
        "supported": true,
        "limitations": ["schema must be provided"]
      },
      {
        "name": "intent_recognition",
        "description": "Recognize intent from text",
        "supported": true,
        "domains": ["general", "mobile", "editing"]
      },
      {
        "name": "action_plan_generation",
        "description": "Generate ActionPlan for video editing",
        "supported": false,  // Memerlukan training tambahan
        "required_training": ["editing_instructions"]
      }
    ],
    "constraints": {
      "max_tokens": 256,
      "max_sequence_length": 128,
      "deterministic": true,
      "offline": true
    }
  }
}
```

### 7.2 Discovery Flow

```
┌─────────────────┐     ┌─────────────────┐
│   Aplikasi      │     │   Naze Model     │
│                 │     │                 │
│  1. Get        │────▶│  capabilities   │
│     capabilities│     │                 │
│  2. Check      │◄────┤  manifest.json  │
│     feature     │     │                 │
│  3. Use/       │     │                 │
│     Fallback    │     │                 │
└─────────────────┘     └─────────────────┘
```

---

## 8. Version Compatibility

### 8.1 Semantic Versioning

- **MAJOR:** Breaking changes (tidak kompatibel backward)
- **MINOR:** Fitur baru (kompatibel backward)
- **PATCH:** Bug fixes (kompatibel backward)

### 8.2 Compatibility Matrix

| Model Version | nazeio Support | naze-motion-agent Support |
|---------------|----------------|----------------------------|
| 1.0.0 | ✅ | ⚠️ (tergantung JSON generation) |
| 1.0.x | ✅ | ✅ |
| 2.0.0 | ❌ (breaking) | ❌ (breaking) |

### 8.3 Migration Strategy

1. **Deprecation Warning:**
   - Versi lama diberi warning deprecation
   - Dokumentasi diperbarui
   - User diberi peringatan

2. **Dual Support:**
   - Support versi lama dan baru selama periode transisi
   - Default: versi baru
   - Fallback: versi lama

3. **Full Migration:**
   - Versi lama dihapus
   - Hanya versi baru yang didukung

---

## 9. Testing Strategy

### 9.1 Unit Testing

- Test masing-masing komponen secara terpisah
- Test kontrak format (npz, JSON)
- Test determinisme

### 9.2 Integration Testing

- Test integrasi model-naze2.0 -> nazeio
- Test integrasi model-naze2.0 -> naze-motion-agent
- Test end-to-end workflow

### 9.3 Contract Testing

- Test kontrak model format
- Test kontrak inference interface
- Test kontrak schema output
- Test backward compatibility

### 9.4 Golden Testing

- Output model dibandingkan dengan referensi
- Toleransi: 1e-4 (float64 vs float32)
- Test determinisme (bit-exact untuk float32)

---

## 10. Deployment Strategy

### 10.1 Model Distribution

**Opsi 1: Bundled (Recommended)**
- Model disertakan dalam APK
- Ukuran APK: +~0.43 MB
- Keuntungan: Offline, cepat, tidak perlu download
- Keterbatasan: Ukuran APK bertambah

**Opsi 2: Download on Demand**
- Model diunduh saat pertama kali dibutuhkan
- Keuntungan: Ukuran APK kecil
- Keterbatasan: Memerlukan internet, latency download

**Opsi 3: Hybrid**
- Model kecil (D-018) bundled
- Model besar (jika ada) download on demand
- Keuntungan: Balance antara ukuran dan kapasitas

### 10.2 Version Management

- Model versi disimpan di server
- Aplikasi mengecek update model secara berkala
- User dapat memilih versi model
- Fallback ke model bundled jika download gagal

### 10.3 Rollback Strategy

- Jika model baru gagal, rollback ke model lama
- Model lama tetap tersedia di perangkat
- Tidak ada downtime

---

## 11. Open Issues dan Keputusan

### 11.1 Open Issues

| ID | Issue | Status | Dampak |
|----|-------|--------|--------|
| C-001 | Format npz di Kotlin | UNCONFIRMED | Blocker inference |
| C-002 | JSON generation di Naze | UNCONFIRMED | Blocker ActionPlan |
| C-003 | Performance di ARMv7 | UNCONFIRMED | Risiko UX |
| C-004 | Ukuran APK dengan model | UNCONFIRMED | Risiko deployment |

### 11.2 Keputusan yang Memerlukan Owner

| Keputusan | Pertanyaan | Status |
|-----------|------------|--------|
| OD-112 | Korpus final training Naze | **BLOCKING** | Memengaruhi semua integrasi |
| OD-119 | Versioning policy | OPEN | Memengaruhi deployment |
| - | Format kontrak final | OPEN | Memengaruhi implementasi |
| - | Threshold confidence | OPEN | Memengaruhi safety |

---

## 12. Ringkasan dan Next Steps

### 12.1 Status Saat Ini

| Komponen | Status | Catatan |
|---------|--------|---------|
| Model Format Contract | ✅ **DONE** | npz + metadata |
| Inference Interface Contract | ✅ **DONE** | Abstraksi InferenceProvider |
| Intent Schema | ✅ **DONE** | Untuk nazeio |
| ActionPlan Schema | ✅ **DONE** | Untuk naze-motion-agent |
| Adapter Design | ✅ **DONE** | Per aplikasi |
| Error Codes | ✅ **DONE** | Bersama |
| Capability Discovery | ✅ **DONE** | Manifest |

### 12.2 Blocker

1. **OD-112:** Korpus final training Naze
   - **Impact:** Semua integrasi memerlukan model terlatih
   - **Status:** OPEN, menunggu owner

2. **Implementasi Kotlin:**
   - **Impact:** nazeio dan naze-motion-agent memerlukan implementasi Kotlin
   - **Status:** Belum dimulai, memerlukan keputusan owner

3. **JSON Generation:**
   - **Impact:** naze-motion-agent memerlukan Naze yang dapat menghasilkan JSON
   - **Status:** Memerlukan training dengan korpus editing

### 12.3 Next Steps

**model-naze2.0:**
- [ ] Selesaikan OD-112 (korpus final)
- [ ] Train model final (T005)
- [ ] Validasi metrik DECISION-023
- [ ] Ekspor model float32
- [ ] Buat dataset instruksi editing (untuk naze-motion-agent)

**nazeio:**
- [ ] Implementasi Naze Engine di Kotlin
- [ ] Implementasi pembaca npz
- [ ] Implementasi inference float32
- [ ] Integrasi dengan alur aplikasi
- [ ] Test end-to-end

**naze-motion-agent:**
- [ ] Implementasi NazeAIProvider
- [ ] Integrasi dengan AI Planner
- [ ] Validasi ActionPlan
- [ ] Test integrasi

**Kolaborasi:**
- [ ] Verifikasi kontrak dengan owner masing-masing proyek
- [ ] Sinkronisasi format dan schema
- [ ] Tentukan threshold dan parameter

---

## 13. Lampiran

### 13.1 Referensi Dokumen

- [NAZEIO_INTEGRATION_CONTRACT.md](NAZEIO_INTEGRATION_CONTRACT.md)
- [NAZE_MOTION_AGENT_INTEGRATION_CONTRACT.md](NAZE_MOTION_AGENT_INTEGRATION_CONTRACT.md)
- [M010_TECHNICAL_DESIGN.md](M010_TECHNICAL_DESIGN.md)
- [DECISION_LOG.md](../decisions/DECISION_LOG.md)

### 13.2 Referensi Eksternal

- [nazeio Repository](https://github.com/Ndolzz/nazeio)
- [naze-motion-agent Repository](https://github.com/Ndolzz/naze-motion-agent)
- [nazeio specs/16_naze_engine.md](https://github.com/Ndolzz/nazeio/blob/main/specs/16_naze_engine.md)
- [nazeio docs/rencana_integrasi_naze_engine.md](https://github.com/Ndolzz/nazeio/blob/main/docs/rencana_integrasi_naze_engine.md)
- [naze-motion-agent docs/AI_AGENT_SPEC.md](https://github.com/Ndolzz/naze-motion-agent/blob/main/docs/AI_AGENT_SPEC.md)
- [naze-motion-agent docs/ACTION_SPEC.md](https://github.com/Ndolzz/naze-motion-agent/blob/main/docs/ACTION_SPEC.md)
