# SPEC REVIEW REPORT — Naze 2.0

> **[RECONSTRUCTED GOVERNANCE — 2026-10-07]**
> Dokumen ini adalah RECONSTRUCTED GOVERNANCE, bukan pemulihan file asli. Versi lokal SPEC_REVIEW v1.2.0 hilang dan tidak pernah di-commit.
> **Basis rekonstruksi:** SPEC_REVIEW v1.1.0 dari GitHub commit `bab6eb9` (Section 1–9 — verbatim dari git history, akurat) + Section 10 baru yang mendokumentasikan resolusi governance dan status M-007 sesuai DECISION-015/016/017, TRACEABILITY v1.3.0, dan M007_TECHNICAL_DESIGN/M007_TASKS.
> Wording Section 10 BUKAN teks asli yang hilang.

**Reviewer:** Senior AI Architect / SDD Reviewer
**Date:** 2026-10-06 (v1.1.0) · 2026-10-07 (v1.2.0 — rekonstruksi) · 2026-10-09 (v1.3.0)
**Review version:** 1.3.0 (M-007 IMPLEMENTED: T001..T018 selesai; update §11; dasar v1.2.0 = governance update)
**Scope:** PROJECT_SPEC, REQUIREMENTS, ARCHITECTURE, DECISION_LOG, ROADMAP, TASKS, TRACEABILITY (v1.0.0 → v1.4.0)

---

## 1. Review Summary

Review tahap kedua dilakukan terhadap seluruh dokumen SDD yang sama, dengan checklist diperdalam — khususnya **SDD Integrity** (kelengkapan setiap tahap workflow) yang di v1.0.0 belum tersorot penuh. Hasil: spesifikasi tetap **konsisten, from-scratch, dan realistis untuk hardware terbatas**. Temuan v1.0.0 **masih berlaku dan belum diselesaikan owner** (ISSUE-004 dan ISSUE-006 tetap MAJOR & blocking-implementasi). Ditemukan **3 temuan baru** (ISSUE-012..014, semua minor/proses) dan **1 gap workflow** yang kini teridentifikasi formal: **tahap Technical Design belum memiliki template/dokumen yang didefinisikan**. Tidak ada perubahan besar yang dilakukan otomatis.

**Kesimpulan status: tetap READY FOR TECHNICAL DESIGN — dengan kondisi wajib yang sama + syarat tambahan (lihat §9).** *(Status historis v1.1.0; lihat §10 untuk update v1.2.0.)*

---

## 2. Valid Requirements (terverifikasi ulang)

Semua **23 requirement** (jumlah dikoreksi dari klaim "26" pra-existing per keputusan owner 2026-10-07 — lihat §10.4): **ID unik** ✅, tujuan jelas ✅, memiliki AC ✅, priority ✅, status ✅, konsisten dengan scope ✅ — kecuali deviasi yang tercatat sebagai ISSUE di §3. *(Teks asli v1.1.0 menyebut "Semua 26 requirement" — kesalahan hitung pra-existing; tidak ada 3 requirement tambahan yang valid ditemukan di git history.)*

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

### Temuan v1.0.0 — status tidak berubah (belum ditindaklanjuti owner) *(status historis v1.1.0; lihat §10)*

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

*(Status historis v1.1.0; resolusi di §10.)*

---

## 9. Specification Readiness

**Verdict (v1.1.0): ✅ READY FOR TECHNICAL DESIGN — dengan kondisi wajib:**

1. **Owner memutuskan ISSUE-004** (REQ-004 ↔ DECISION-010) — masih terbuka sejak v1.0.0.
2. **Owner memutuskan OD-116** (ambang loss numerik) — masih terbuka sejak v1.0.0.
3. **(Baru) Owner memutuskan OD-120** (format technical design) — syarat proses agar tahap berikutnya punya artefak baku.

Issue lain tidak memblokir dan dapat ditangani bertahap di technical design M-007/M-008.

**Milestone berikutnya yang siap masuk technical design: M-007 — Transformer (Stage 6)** — dengan rekomendasi OD-113 & OD-118 juga diputuskan sebelum implementasinya dimulai.

**Perubahan yang dilakukan review ini:** tidak ada perubahan requirement/arsitektur otomatis. Dokumen diperbarui: TRACEABILITY.md (v1.1.0) dan SPEC_REVIEW.md (laporan ini). Semua rekomendasi menunggu persetujuan owner sesuai change policy (KEEP/MODIFY/CLARIFY/REMOVE/OPEN DECISION).

---

## 10. Governance Update (v1.2.0) — [RECONSTRUCTED]

> Bagian ini direkonstruksi (teks asli v1.2.0 hilang). Substansi sesuai keputusan yang terdokumentasi di DECISION_LOG (D-015/016/017), TRACEABILITY v1.3.0, M007_TECHNICAL_DESIGN.md, dan M007_TASKS.md.

### 10.1 Resolusi kondisi wajib §9 (v1.1.0)

| Kondisi wajib (v1.1.0 §9) | Status | Resolusi |
|---|---|---|
| ISSUE-004 (REQ-004 ↔ DECISION-010, MAJOR) | ✅ **RESOLVED** | **DECISION-015**: vocab byte-level FIXED 256; "configurable" ≠ vocab size. AC REQ-004 direvisi (REQUIREMENTS v0.4.0). |
| OD-116 / ISSUE-006 (ambang loss, MAJOR) | ✅ **RESOLVED** | **DECISION-016**: kerangka evaluasi 5-basis (baseline, improvement terukur, konvergensi/stabilitas, validasi, config reproducible); tanpa ambang loss numerik universal. AC REQ-006/009/010 direvisi (REQUIREMENTS v0.4.0). |
| OD-120 / ISSUE-013 (format technical design) | ✅ **RESOLVED** | **DECISION-017**: `docs/architecture/M###_TECHNICAL_DESIGN.md`, 15 section wajib. PROJECT_SPEC §7 → workflow 9 tahap eksplisit + governance maintenance policy (sekaligus menutup ISSUE-014). |

### 10.2 Kemajuan setelah resolusi

- **Technical Design M-007 selesai:** `docs/architecture/M007_TECHNICAL_DESIGN.md` — 15 section sesuai DECISION-017 (terverifikasi); 21 komponen arsitektur dengan invariant D = H × Dh; 17 AC objective; estimasi resource dev/test (~89k param).
- **Task Breakdown M-007 selesai:** `docs/tasks/M007_TASKS.md` — M007-T001..T018, urutan kritis terdefinisi, setiap task memuat AC + Req ID + referensi TD.
- **TRACEABILITY diperbarui ke v1.3.0** sesuai maintenance policy: seluruh requirement MAPPED (0 UNMAPPED); kolom Technical Design terisi untuk M-007; chain of traceability per DECISION-017.

### 10.3 Status isu & open decision lain

- **ISSUE-012** (mapping REQ-009): klarifikasi diterapkan di REQUIREMENTS v0.4.0 — evaluation harness = Stage 7; benchmark inference = Stage 8. *[RECONSTRUCTED — tidak dapat dipastikan apakah klarifikasi ini sudah tercatat di v1.2.0 asli atau hanya di REQUIREMENTS.]* Rumusan ini konsisten dengan baris REQ-009 TRACEABILITY v1.3.0.
- **ISSUE-013/014**: tertutup via DECISION-017 + maintenance policy PROJECT_SPEC §7 (v0.4.0).
- **ISSUE-001/002/003/005/007/008/009/010/011**: tidak berubah — jadwal M-008 sesuai v1.1.0.
- **OD-113** (coverage): OPEN — DEFERRED ke fase implementasi M-007 (diputuskan sebelum implementasi dinyatakan selesai penuh).
- **OD-114** (CI): OPEN — memengaruhi enforcement AC test-plan di masa depan.
- **OD-118** (batas ukuran Transformer): OPEN — **BLOCKER untuk training skala serius; BUKAN blocker untuk konfigurasi dev/test M-007** (TD §15).
- **OD-121** (skema positional; baru dari TD M-007): OPEN — default learned; owner dapat mengubah sebelum/awal M007-T003.
- **MISS-005**: tertutup via governance maintenance policy.

### 10.4 Koreksi jumlah requirement (owner decision, 2026-10-07) [RECONSTRUCTED]

Verifikasi final terhadap git history (REQUIREMENTS v0.1.0 commit `6722376` dan v0.3.0 commit `b53259b`) membuktikan jumlah REQ ID unik = **23** (REQ-001..011, REQ-101..105, REQ-201..203, REQ-301..304). Klaim "26 requirement" pada SPEC_REVIEW v1.0.0/v1.1.0 §2 dan ringkasan TRACEABILITY v1.1.0/v1.3.0 adalah **kesalahan hitung pra-existing** — tidak ditemukan 3 requirement tambahan yang valid.

**Keputusan owner (2026-10-07):** 23 adalah jumlah yang benar; seluruh klaim "26" dikoreksi menjadi "23"; **tidak ada REQ baru yang dibuat** hanya untuk mencapai angka 26 (melanggar REQ-302/no-invention). Konsistensi dijaga lintas dokumen: REQUIREMENTS (23 REQ ID terdefinisi), TRACEABILITY v1.3.1 (**23/23 MAPPED, 0 UNMAPPED**), SPEC_REVIEW §2 (23 requirement valid).

### 10.5 Verdict (v1.2.0)

**✅ READY FOR IMPLEMENTATION — M-007 (Transformer, Stage 6).**

Seluruh kondisi wajib v1.1.0 §9 terpenuhi (DECISION-015/016/017). Technical Design + Task Breakdown M-007 selesai dan terlacak penuh di TRACEABILITY v1.3.0. Implementasi M007-T001..T018 dimulai hanya setelah persetujuan owner atas M007_TECHNICAL_DESIGN.md; commit implementasi wajib mereferensikan Task ID + Req ID. OD-113/OD-114/OD-118/OD-121 tercatat sebagai kondisi non-blocking untuk dev/test config (lihat TD §15).

---

## 11. Implementation Update (v1.3.0, 2026-10-09)

> Update pasca-implementasi M-007 sesuai maintenance policy PROJECT_SPEC §7 (implementasi selesai → review/status diperbarui).

### 11.1 Status implementasi M-007

**M-007 IMPLEMENTED — M007-T001..T018 selesai (2026-10-09).**

- Komponen (T001..T012): `src/naze/nn/transformer.py` — config (vocab 256 fixed, invariant D = H × Dh), embedding scatter-add, positional learned, QKV, causal attention (mask stabil), MHA, LayerNorm, FFN tanh, residual, block, stacking, LM head (D-015).
- Integrasi (T013): `src/naze/lm/transformer_lm.py` — TransformerLM (loss/backward 22 kunci params datar) + transformer_generate (greedy/temperature, konteks dipotong ke max_sequence_length); MLPLM tidak berubah.
- Test (T014..T016): 3 suite baru — unit (per komponen, ≥1 nilai acuan), numeric (grad-check 1e-5, mask correctness level logits & attention, determinisme per-seed, LN stats), integration (tokenizer→dataset→model→loss, generate adapter, sanity 5-basis D-016 lulus: loss 5.58 → 2.48, holdout turun > 1.0).
- Regresi (T017): audit commit membuktikan tidak ada test lama yang
  dimodifikasi; workflow CI `.github/workflows/tests.yml` ditambahkan
  (pytest penuh, Python 3.11, PYTHONPATH=src) — sebelumnya pytest tidak
  pernah dieksekusi otomatis. Run CI pertama menemukan 23 failure (bug
  dtype engine + test lama pra-existing yang belum pernah tereksekusi);
  diperbaiki commit `91982fe`/`7a53871` (fix dtype `Linear`/`LayerNorm`,
  penyesuaian test). Run CI `7a53871`: pytest penuh HIJAU — AC T017
  terverifikasi.
- Dokumentasi (T018): TRACEABILITY v1.4.0, REQUIREMENTS v0.5.0, ROADMAP (M-007 DONE), M007_TASKS status — update ini.

### 11.2 Catatan teknis signifikan

- Grad bias K analitik ≈ 0 (softmax invarian pergeseran konstanta per baris) — dites absolut di suite numeric, didokumentasikan di docstring test (bukan defect; konsekuensi struktural arsitektur).
- Evaluasi sanity 5-basis D-016 dieksekusi tanpa ambang numerik universal sesuai DECISION-016.

### 11.3 Status isu & OD pasca-implementasi

- OD-121 (positional): default learned DIPERTAHANKAN (implementasi T003); owner masih dapat mengubah dengan perubahan berikutnya.
- OD-113 (coverage): tetap OPEN — coverage form belum diukur; suite penuh kini via CI.
- OD-114 (CI): tereduksi sebagian — CI pytest ada (T017); kebijakan enforcement formal tetap menunggu owner.
- OD-118 (batas ukuran): ✅ RESOLVED — DECISION-018 (owner, 2026-10-09): checkpoint final ≤ 1 MB (float64); config produksi D=64, H=4, L=2, d_ff=128, T_max=128 (~108k param ≈ 0.87 MB f64). Constraint formal M-008; nazeio spec 16 mengacu keputusan ini.
- ISSUE-002/003/005/007..011: jadwal tetap M-008.

### 11.4 Verdict (v1.3.0)

**✅ M-007 SELESAI (IMPLEMENTED) — pytest penuh hijau di CI (commit
`7a53871`, 2026-10-09). Milestone berikutnya: M-008 — Training System
lengkap (PLANNED, menunggu persetujuan owner).**
