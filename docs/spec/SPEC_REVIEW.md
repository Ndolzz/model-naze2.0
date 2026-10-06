# SPEC REVIEW REPORT — Naze 2.0

**Reviewer:** Senior AI Architect / SDD Reviewer
**Date:** 2026-10-06
**Review version:** 1.1.0 (review tahap kedua — verifikasi menyeluruh ulang atas v1.0.0)
**Scope:** PROJECT_SPEC, REQUIREMENTS, ARCHITECTURE, DECISION_LOG, ROADMAP, TASKS, TRACEABILITY (v1.0.0)

---

## 1. Review Summary

Review tahap kedua dilakukan terhadap seluruh dokumen SDD yang sama, dengan checklist diperdalam — khususnya **SDD Integrity** (kelengkapan setiap tahap workflow) yang di v1.0.0 belum tersorot penuh. Hasil: spesifikasi tetap **konsisten, from-scratch, dan realistis untuk hardware terbatas**. Temuan v1.0.0 **masih berlaku dan belum diselesaikan owner** (ISSUE-004 dan ISSUE-006 tetap MAJOR & blocking-implementasi). Ditemukan **3 temuan baru** (ISSUE-012..014, semua minor/proses) dan **1 gap workflow** yang kini teridentifikasi formal: **tahap Technical Design belum memiliki template/dokumen yang didefinisikan**. Tidak ada perubahan besar yang dilakukan otomatis.

**Kesimpulan status: tetap READY FOR TECHNICAL DESIGN — dengan kondisi wajib yang sama + syarat tambahan (lihat §9).**

---

## 2. Valid Requirements (terverifikasi ulang)

Semua 26 requirement: **ID unik** ✅, tujuan jelas ✅, memiliki AC ✅, priority ✅, status ✅, konsisten dengan scope ✅ — kecuali deviasi yang tercatat sebagai ISSUE di §3.

| ID | Verdict |
|---|---|
| REQ-001, REQ-002 | VALID (catatan API docs, ISSUE-002) |
| REQ-003 | VALID (catatan CI, ISSUE-003) |
| REQ-004 | **KONTRADIKTIF** dengan DECISION-010 (ISSUE-004) |
| REQ-005 | VALID dengan gap "cleaning" (ISSUE-005) |
| REQ-006 | VALID namun **AC ambigu** — ambang loss tidak numerik (ISSUE-006) |
| REQ-007 | VALID — 2 AC belum terimplementasi (ISSUE-007/008) |
| REQ-008, REQ-011 | VALID (PLANNED, mapping benar) |
| REQ-009 | VALID namun **mapping stage ambigu** (ISSUE-012, baru) |
| REQ-010 | VALID (catatan ISSUE-009) |
| REQ-101..105 | VALID sebagai NFR (parsial terpenuhi, wajar) |
| REQ-201..203, 301..304 | VALID — konsisten lintas seluruh dokumen |

---

## 3. Issues Found

### Temuan v1.0.0 — status tidak berubah (belum ditindaklanjuti owner)

ISSUE-001 — Consistency (minor): penomoran DECISION stabil sejak b53259b. *KEEP.*
ISSUE-002 — REQ-001 API docs belum ada. *MODIFY: jadwalkan task; dokumen: TASKS.md.*
ISSUE-003 — REQ-003 "diuji di CI" tidak actionable. *OPEN DECISION-114; MODIFY AC sementara.*
ISSUE-004 ⚠️ MAJOR — REQ-004 "vocab configurable" kontradiksi DECISION-010 (byte-level fixed 256). *CLARIFY — menunggu owner memilih (a) MODIFY REQ-004 atau (b) MODIFY DECISION-010 pra-Stage 6. Belum diputuskan → tetap blocking.*
ISSUE-005 — REQ-005 "cleaning" tidak terdefinisi. *OPEN DECISION-115 / CLARIFY.*
ISSUE-006 ⚠️ MAJOR — AC "loss menurun sesuai ambang" tanpa ambang numerik. *OPEN DECISION-116 — menunggu owner. Tetap blocking.*
ISSUE-007 — Resume training & RNG state di checkpoint belum dispesifikasikan. *MODIFY REQ-007; task M-008.*
ISSUE-008 — Checksum integritas belum ada. *KEEP; task M-008.*
ISSUE-009 — Load-checkpoint→generate belum teruji eksplisit. *MODIFY test plan M-008.*
ISSUE-010 — Metadata per-run belum dispesifikasikan. *KEEP; M-008.*
ISSUE-011 — Metrik memori REQ-102 tidak terdefinisi. *CLARIFY; OPEN DECISION-117.*

### Temuan baru v1.1.0

ISSUE-012
- **Type:** Architecture mapping ambiguity
- **Related:** REQ-009 ↔ ARCHITECTURE.md
- **Problem:** REQUIREMENTS memetakan REQ-009 (Evaluation) ke Stage 7; ARCHITECTURE menyebut evaluasi berkala di Stage 7 (training) namun harness evaluasi penuh dekat Stage 8 (inference benchmark). Mapping REQ-009 ke stage tidak tunggal.
- **Impact:** Rendah — tidak melanggar urutan dependency, hanya menimbulkan risiko salah milestone saat M-008/M-009 dibuka.
- **Recommendation:** CLARIFY — tetapkan satu rumusan: "evaluation harness = Stage 7 (dipakai oleh training), benchmark inference = Stage 8". Dokumen terpengaruh: REQUIREMENTS (REQ-009 priority line), ARCHITECTURE.

ISSUE-013
- **Type:** SDD workflow gap (proses)
- **Related:** PROJECT_SPEC §7 (SDD workflow) vs dokumen yang ada
- **Problem:** Tahap **Technical Design** dalam workflow resmi tidak memiliki template/dokumen/lokasi yang didefinisikan (docs/ tidak memiliki home untuk technical design per-milestone). v1.0.0 tidak mencatat gap ini secara formal.
- **Impact:** Sedang — reviewer dan implementer tidak punya artefak baku untuk tahap ini; risiko technical design dilewati diam-diam.
- **Recommendation:** MODIFY — definisikan docs/design/ (satu dokumen per milestone, mis. DESIGN-M007.md) di PROJECT_SPEC §7. OPEN DECISION-120: format/templat technical design (menunggu owner). Dokumen: PROJECT_SPEC, (baru) docs/design/.

ISSUE-014
- **Type:** Documentation consistency (minor)
- **Related:** TASKS.md, ROADMAP.md vs TRACEABILITY/SPEC_REVIEW
- **Problem:** TRACEABILITY.md dan SPEC_REVIEW.md (dibuat v1.0.0) belum terdaftar di struktur dokumen ROADMAP/TASKS/PROJECT_SPEC; tidak ada aturan kapan matriks traceability harus diperbarui (mis. per milestone selesai).
- **Impact:** Rendah — risiko matriks usang saat requirement berubah.
- **Recommendation:** MODIFY — tambahkan kebijakan: TRACEABILITY diperbarui setiap perubahan REQUIREMENTS (bagian dari langkah 10 workflow). Dokumen: PROJECT_SPEC §7, REQUIREMENTS changelog.

---

## 4. Architecture Issues

**ARCH-REVIEW-1 (dikonfirmasi ulang): Urutan stage VALID.** Project Foundation → Neural Engine → Autodiff → Tokenizer → Dataset → Language Model → Transformer → Training → Inference → Naze 1.0 — tidak ada lompatan dependency. **Tidak ada perubahan dependency yang direkomendasikan.**

**ARCH-REVIEW-2:** M-006 mengambil bagian Stage 7 lebih awal (SGD, checkpoint) — terdokumentasi ("Stage 5.5"), bukan pelanggaran; dipantau.

**ARCH-REVIEW-3:** Stage 10 sengaja tidak didesain — sesuai no-overengineering. Benar.

**ARCH-REVIEW-4:** Versi ringkas ARCHITECTURE mengompresi detail per-stage versi 0.1.0 — tertelusur via git history. KEEP.

**ARCH-REVIEW-5 (baru):** Ambiguitas mapping REQ-009 (Stage 7 vs 8) — lihat ISSUE-012. Tidak mengubah dependency; hanya perlu klarifikasi rumusan.

---

## 5. From-Scratch Validation

**ALLOWED (sesuai spesifikasi):**
- NumPy (fondasi numerik — DECISION-003) dan Python stdlib
- pytest, ruff (tooling development — bukan core)
- Format file stdlib-level (npz)
- Native extension buatan sendiri untuk hot-path (bila OPEN DECISION-101 diputuskan demikian)

**NOT ALLOWED (di core — REQ-201/202/203, permanen):**
- Pretrained LLM / pretrained Transformer / bobot pretrained apa pun
- External LLM API sebagai core
- Hugging Face model sebagai core
- PyTorch / TensorFlow / JAX — sebagai engine maupun sumber bobot

**Hasil: spesifikasi dan implementasi saat ini TIDAK melanggar.** Tidak ditemukan celah yang secara tidak sengaja mengizinkan item NOT ALLOWED. (Penguat opsional dari v1.0.0 tetap berlaku: checklist deklarasi bebas-pretrained per PR yang menyentuh src/naze.)

---

## 6. Hardware Constraints Evaluation

- **RAM:** hemat memori (dataset index-based; float64 = 2x float32 — trade-off terdokumentasi DECISION-007, sudah ada catatan tinjauan untuk inference). OK.
- **CPU:** NumPy-only, model kecil. OK.
- **Storage & checkpoint size:** npz params-only — kecil. OK.
- **Training time:** tidak ditetapkan — benar, jangan mengarang angka. OK.
- **Model size:** LM tahap awal "jutaan parameter" (konseptual); **Transformer Stage 6 belum punya batas ukuran** → OPEN DECISION-118 (butuh spesifikasi hardware owner — tidak boleh dikarang).
- **Dataset size:** OPEN DECISION-112 (korpus final). Benar terbuka.

**Tidak ada angka hardware yang dikarang oleh spesifikasi.** ✅

---

## 7. Missing Requirements (dikonfirmasi ulang + tambahan)

- **MISS-001** — Regression/performance smoke suite (usul REQ-012, DEFERRED).
- **MISS-002** — Kebijakan error handling (kontrak exception) — CLARIFY di technical design.
- **MISS-003** — Versioning & release policy Naze 1.0 — OPEN DECISION-119.
- **MISS-004** — Konfigurasi eksperimen terstruktur (file config run) — usul M-008.
- **MISS-005 (baru)** — **Aturan pemeliharaan TRACEABILITY/SPEC_REVIEW** (kapan diperbarui, siapa pemilik) — lihat ISSUE-014.

---

## 8. Open Decisions

**Existing (menunggu owner):** OD-101 (accelerasi), OD-107 (sukses Naze 1.0), OD-108 (multimodal), OD-112 (korpus final), OD-113 (coverage target), OD-114 (CI), OD-115 (cleaning), OD-116 (ambang loss — **blocking**), OD-117 (metrik memori), OD-118 (batas ukuran Transformer), OD-119 (release policy).

**Baru dari review ini:**
- **OD-120** — Format & lokasi technical design per milestone (template docs/design/DESIGN-MXXX.md?) — dibutuhkan sebelum M-007 masuk technical design.

---

## 9. Specification Readiness

**Verdict: ✅ READY FOR TECHNICAL DESIGN — dengan kondisi wajib:**

1. **Owner memutuskan ISSUE-004** (REQ-004 ↔ DECISION-010) — masih terbuka sejak v1.0.0.
2. **Owner memutuskan OD-116** (ambang loss numerik) — masih terbuka sejak v1.0.0.
3. **(Baru) Owner memutuskan OD-120** (format technical design) — syarat proses agar tahap berikutnya punya artefak baku.

Issue lain tidak memblokir dan dapat ditangani bertahap di technical design M-007/M-008.

**Milestone berikutnya yang siap masuk technical design: M-007 — Transformer (Stage 6)** — dengan rekomendasi OD-113 & OD-118 juga diputuskan sebelum implementasinya dimulai.

**Perubahan yang dilakukan review ini:** tidak ada perubahan requirement/arsitektur otomatis. Dokumen diperbarui: TRACEABILITY.md (v1.1.0) dan SPEC_REVIEW.md (laporan ini). Semua rekomendasi menunggu persetujuan owner sesuai change policy (KEEP/MODIFY/CLARIFY/REMOVE/OPEN DECISION).
