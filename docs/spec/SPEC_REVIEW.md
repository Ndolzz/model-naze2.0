# SPEC REVIEW REPORT — Naze 2.0

**Reviewer:** Senior AI Architect / SDD Reviewer
**Date:** 2026-10-06
**Scope:** PROJECT_SPEC, REQUIREMENTS, ARCHITECTURE, DECISION_LOG, ROADMAP, TASKS (+ bukti pemenuhan AC dari hasil M-001..M-006)
**Review version:** 1.0.0

---

## 1. Review Summary

Spesifikasi Naze 2.0 secara keseluruhan **konsisten, from-scratch, dan realistis untuk hardware terbatas**. Struktur SDD utuh (spec → architecture → decisions → roadmap → tasks), changelog terjaga, tidak ada requirement yang dihapus tanpa dokumentasi, dan tidak ada angka hardware yang dikarang. Ditemukan **9 issue** (2 major, 7 minor), **0 kontradiksi fatal**, **4 requirement/kemampuan yang hilang**, dan **6 open decision baru** yang diusulkan. Tidak ada perubahan besar yang dilakukan otomatis — semua rekomendasi berkategori KEEP/MODIFY/CLARIFY/OPEN DECISION dan menunggu persetujuan project owner.

**Kesimpulan: READY FOR TECHNICAL DESIGN — dengan 2 kondisi wajib (lihat §9).**

---

## 2. Valid Requirements (terverifikasi konsisten)

| ID | Verdict |
|---|---|
| REQ-001, REQ-002 | VALID — jelas, teruji, AC terpenuhi (kecuali API docs, ISSUE-002) |
| REQ-003 | VALID — pendekatan diputuskan (DECISION-009), AC terpenuhi; catatan CI (ISSUE-003) |
| REQ-010, REQ-011 | VALID — arsitektur diputuskan (DECISION-012); Transformer belum dibuka (benar) |
| REQ-101..REQ-105 | VALID sebagai NFR; parsial terpenuhi (wajar tahap ini) |
| REQ-201, REQ-202, REQ-203, REQ-301..REQ-304 | VALID — constraint konsisten lintas semua dokumen |

---

## 3. Issues Found

ISSUE-001
- **Type:** Consistency (dokumen, minor)
- **Related:** DECISION_LOG histori
- **Problem:** Penomoran DECISION sempat tidak stabil antar versi dokumen sebelum commit b53259b; versi terkini konsisten.
- **Impact:** Rendah.
- **Recommendation:** KEEP versi terkini + catatan bahwa penomoran stabil sejak b53259b.

ISSUE-002
- **Type:** Missing deliverable
- **Related:** REQ-001 (AC: "Dokumentasi API tersedia")
- **Problem:** API docs belum ada; AC REQ-001 belum bisa ditutup.
- **Impact:** Rendah.
- **Recommendation:** MODIFY — jadwalkan API docs sebagai task M-007; jangan tutup AC diam-diam. Dokumen terpengaruh: TASKS.md.

ISSUE-003
- **Type:** Ambiguous AC (not enforceable)
- **Related:** REQ-003 ("diuji di CI")
- **Problem:** CI tidak didefinisikan di spec/roadmap mana pun; AC tidak actionable.
- **Impact:** Sedang.
- **Recommendation:** OPEN DECISION-114 (CI: GitHub Actions / lokal / none). MODIFY AC sementara → "diuji otomatis via suite". Dokumen: REQUIREMENTS, DECISION_LOG.

ISSUE-004 ⚠️ MAJOR
- **Type:** Contradictory requirement
- **Related:** REQ-004 vs DECISION-010
- **Problem:** AC REQ-004: "vocab size dapat dikonfigurasi". DECISION-010 (byte-level): vocab fixed 256. Implementasi M-004 memenuhi decision tapi **melanggar AC REQ-004** → validasi-vs-spec untuk REQ-004 tidak bisa lulus.
- **Impact:** Tinggi (proses SDD).
- **Recommendation:** CLARIFY — owner memilih: (a) MODIFY REQ-004 (vocab configurable berlaku untuk tokenizer subword masa depan; byte-level dikecualikan dengan catatan), atau (b) MODIFY DECISION-010 saat revisi tokenizer pra-Stage 6. **Tidak diubah otomatis.** Dokumen: REQUIREMENTS atau DECISION_LOG.

ISSUE-005
- **Type:** Scope gap (requirement promise, undefined)
- **Related:** REQ-005 ("loading, cleaning, batching")
- **Problem:** "Cleaning" tidak didefinisikan dan tidak diimplementasikan.
- **Impact:** Sedang.
- **Recommendation:** CLARIFY — definisikan minimal (mis. normalisasi newline) atau tandai DEFERRED beralasan. OPEN DECISION-115. Dokumen: REQUIREMENTS.

ISSUE-006 ⚠️ MAJOR
- **Type:** Ambiguous acceptance criteria
- **Related:** REQ-006, REQ-010, ROADMAP M-006 ("loss menurun sesuai ambang spec")
- **Problem:** Tidak ada ambang numerik loss di spec mana pun; test memakai heuristik ad-hoc (turun >50%) yang tidak tercatat di REQUIREMENTS → "validate against specification" tidak objektif.
- **Impact:** Tinggi.
- **Recommendation:** MODIFY — tambahkan ambang eksplisit di AC REQ-006/REQ-010; angka final = OPEN DECISION-116 (owner; menetapkan sendiri = melanggar no-invention). Dokumen: REQUIREMENTS, ROADMAP.

ISSUE-007
- **Type:** Missing capability
- **Related:** REQ-006 ("stoppable & resumable"), REQ-007
- **Problem:** Resume training belum dispesifikasikan; RNG state tidak disimpan di checkpoint → resume penuh belum deterministik.
- **Impact:** Sedang.
- **Recommendation:** MODIFY — perluas REQ-007 (checkpoint = params + step + rng state); task M-008. Dokumen: REQUIREMENTS, TASKS.

ISSUE-008
- **Type:** Unimplemented AC
- **Related:** REQ-007 ("integritas terverifikasi")
- **Problem:** Tidak ada checksum/hash pada checkpoint.
- **Impact:** Rendah.
- **Recommendation:** KEEP; jadwalkan task (SHA-256) di M-008. Dokumen: TASKS.

ISSUE-009
- **Type:** AC not fully validated
- **Related:** REQ-010 ("generate dari checkpoint")
- **Problem:** Kombinasi load-checkpoint → generate belum dites eksplisit (komponennya masing-masing teruji).
- **Impact:** Rendah.
- **Recommendation:** MODIFY test plan (bukan spec) di M-008.

ISSUE-010
- **Type:** Partial NFR fulfillment (expected at this stage)
- **Related:** REQ-101 (config+seed+versi per run)
- **Problem:** Metadata run terstruktur belum dispesifikasikan.
- **Impact:** Sedang (kritikal saat training serius).
- **Recommendation:** KEEP; jadwalkan M-008.

ISSUE-011
- **Type:** NFR unenforceable as written
- **Related:** REQ-102 ("batas memori yang dapat diuji")
- **Problem:** Metrik memori tidak terdefinisi (peak RSS? ukuran array aktif?).
- **Impact:** Sedang.
- **Recommendation:** CLARIFY di technical design M-008. OPEN DECISION-117.

---

## 4. Architecture Issues

**ARCH-REVIEW-1 — Urutan stage VALID.** 0→1→2→3→4→5→6→7→8→9→10 tidak melompati dependency: engine→autodiff→(tokenizer→dataset)→LM→Transformer→training→inference→1.0. **Tidak ada perubahan dependency yang direkomendasikan.**

**ARCH-REVIEW-2 — Catatan minor (bukan pelanggaran):** M-006 mengimplementasikan bagian Stage 7 (SGD, checkpoint) lebih awal. Terdokumentasi (ROADMAP M-006, ARCHITECTURE "Stage 5.5"), tidak melanggar dependency — acceptable, dipantau agar tidak jadi kebiasaan.

**ARCH-REVIEW-3 — Stage 10 sengaja tidak didesain** — sesuai no-overengineering. Benar.

**ARCH-REVIEW-4 — Detail per-stage versi 0.1.0** (Purpose/Inputs/Outputs lengkap) terkompresi di versi ringkas. KEEP struktur terkini; versi lengkap tetap tertelusur via git history.

---

## 5. From-Scratch Validation

**ALLOWED:** NumPy (fondasi numerik, DECISION-003); Python stdlib; pytest & ruff (tooling dev, bukan core); format file stdlib (npz); native extension sendiri bila OD-101 diputuskan.

**NOT ALLOWED di core (REQ-201/202/203, permanen):** pretrained LLM/Transformer/bobot pretrained apa pun; API LLM eksternal; model Hugging Face; PyTorch/TensorFlow/JAX (engine maupun sumber bobot).

**Hasil: spesifikasi dan implementasi saat ini TIDAK melanggar; tidak ada celah yang mengizinkan hal di atas secara tidak sengaja.**
**Recommendation penguat (opsional, MODIFY):** template checklist PR untuk perubahan pada src/naze: deklarasi bebas-pretrained & bebas-framework-DL.

---

## 6. Hardware Constraints Evaluation

- **RAM:** desain hemat memori benar (index-based dataset; float64 = 2x float32, trade-off terdokumentasi di DECISION-007 dan sudah ada catatan tinjauan ulang untuk inference). OK.
- **CPU/latency:** NumPy-only + model kecil — realistis. OK.
- **Storage/checkpoint:** params-only npz — kecil. OK.
- **Training time:** tidak ada target waktu — wajar, jangan dikarang. OK.
- **Model size:** LM tahap awal dibatasi konsep "jutaan parameter"; **Transformer (Stage 6) belum punya batas ukuran** → OPEN DECISION-118 (butuh spesifikasi hardware owner yang belum diketahui — tidak boleh dikarang).
- **Dataset size:** OD-112 (korpus final) masih open — benar.

**Tidak ada angka hardware yang dikarang.**

---

## 7. Missing Requirements

- **MISS-001 — Regression/performance smoke suite:** integration test ada, tapi tidak ada requirement regression. *Rekomendasi: pertimbangkan REQ-012 (DEFERRED).*
- **MISS-002 — Kebijakan error handling:** kontrak exception (raise vs silent) tidak dispesifikasikan; praktik kode konsisten raise ValueError/RuntimeError tapi tidak termutakan di spec. *Rekomendasi: CLARIFY di technical design.*
- **MISS-003 — Versioning & release policy:** tidak ada requirement semver/artefact rilis untuk Naze 1.0. *Rekomendasi: OPEN DECISION-119, sebelum Stage 9.*
- **MISS-004 — Konfigurasi eksperimen terstruktur (file config run):** REQ-101 menyiratkan tapi tidak menuntut. *Rekomendasi: pertimbangkan di M-008.*

---

## 8. Open Decisions (terkini + baru)

**Existing (belum diputuskan owner):** OD-101 (accelerasi), OD-107 (sukses Naze 1.0), OD-108 (multimodal), OD-112 (korpus final), OD-113 (coverage target).

**Baru (diusulkan review ini):**
- OD-114 — CI platform (GitHub Actions / lokal / none)
- OD-115 — definisi "cleaning" REQ-005 (atau DEFERRED)
- OD-116 — ambang numerik loss untuk AC REQ-006/REQ-010
- OD-117 — metrik pengukuran memori untuk REQ-102
- OD-118 — batas ukuran model Transformer Stage 6 (butuh spesifikasi hardware owner)
- OD-119 — versioning & release policy Naze 1.0

---

## 9. Specification Readiness

**Verdict: ✅ READY FOR TECHNICAL DESIGN — dengan kondisi wajib sebelum implementasi lanjut:**

1. **Owner memutuskan ISSUE-004** (kontradiksi REQ-004 ↔ DECISION-010) — satu-satunya kontradiksi nyata; tidak boleh didiamkan.
2. **Owner memutuskan OD-116** (ambang loss) agar validasi-vs-spec objektif.

Issue lain dapat ditangani bertahap di technical design M-007/M-008 dan tidak memblokir.

**Milestone berikutnya yang siap masuk technical design: M-007 — Transformer (Stage 6).** Sebelum implementasinya, OD-113 dan OD-118 sebaiknya juga diputuskan.

**Perubahan yang dilakukan review ini:** tidak ada perubahan requirement/arsitektur otomatis. Dokumen dibuat: TRACEABILITY.md (matriks baru) dan SPEC_REVIEW.md (laporan ini). Semua rekomendasi menunggu persetujuan owner sesuai change policy.
