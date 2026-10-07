# DECISION LOG — Naze 2.0

> **[RECONSTRUCTED GOVERNANCE — 2026-10-07]**
> Dokumen ini adalah RECONSTRUCTED GOVERNANCE, bukan pemulihan file asli. Versi governance lokal yang memuat DECISION-015/016/017 hilang dan tidak pernah di-commit.
> **Basis rekonstruksi:** (1) versi GitHub commit `b53259b` untuk DECISION-001..014 — bagian ini akurat (verbatim dari git history); (2) keputusan governance yang terdokumentasi lintas TRACEABILITY v1.3.0, M007_TECHNICAL_DESIGN.md, M007_TASKS.md, dan SPEC_REVIEW v1.1.0.
> **Entri DECISION-015/016/017:** substansi sesuai keputusan yang ditetapkan; WORDING BUKAN TEKS ASLI yang hilang.

**Status legend:** ACCEPTED / PROPOSED / OPEN DECISION / SUPERSEDED

---

## Keputusan Diterima (ACCEPTED)

## DECISION-001 — SDD sebagai workflow resmi — ACCEPTED
Spesifikasi = source of truth. Define Spec → Review → Architecture → Tasks → Implement → Test → Validate → Review → Commit → Update Spec.
Alternatif: ad-hoc (tidak traceable), waterfall (kaku). Konsekuensi: perubahan lewat proses spec.

## DECISION-002 — Python sebagai bahasa utama — ACCEPTED
Alternatif C++/Rust ditolak tahap ini. Performa bergantung NumPy.

## DECISION-003 — NumPy fondasi numerik; framework DL dilarang untuk core — ACCEPTED
Kontrol penuh; stabil untuk model kecil di hardware terbatas.

## DECISION-004 — Larangan pretrained model & API LLM eksternal sebagai core (permanen) — ACCEPTED
Prinsip project owner. Development dari nol lebih lama; model awal kecil.

## DECISION-005 — Arsitektur bertahap Stage 0-10, dibuka berurutan — ACCEPTED
Modular, tervalidasi bertahap. Stage 10 tidak didesain sekarang.

## DECISION-006 — Tooling: pytest + ruff — ACCEPTED (owner, 2026-10-06)

## DECISION-007 — Dtype float64 + seluruh randomness via seeded_rng (deterministik); validasi NaN/Inf — ACCEPTED (M-002)
Presisi gradien + reproducibility. float32 dapat ditinjau untuk inference.

## DECISION-008 — Abstraksi Layer + Sequential, He-normal init, forward-only (Stage 1) — ACCEPTED (M-002)

## DECISION-009 — [RESOLUSI OPEN DECISION-105] Backprop terstruktur per-layer — ACCEPTED (owner delegation, 2026-10-06)
- **Context:** Trade-off graph-based reverse-mode autodiff vs manual backprop terstruktur.
- **Decision:** Backprop terstruktur per-layer: forward meng-cache input; backward(grad_out) menghitung grad_input + grad parameter; Sequential.backward komposisi terbalik. Validasi via numerical gradient check (tol 1e-5, float64).
- **Reason:** Transparan penuh (visi project), sederhana, memori efisien (tanpa graph), cukup untuk arsitektur saat ini. Graph-based autodiff ditambah hanya jika kompleksitas model menuntut.
- **Alternatives:** graph-based reverse-mode (ditunda — overengineering sekarang); finite-difference only (terlalu lambat untuk training).
- **Consequences:** Setiap layer baru wajib mengimplementasikan backward + lulus gradient check.

## DECISION-010 — [RESOLUSI OPEN DECISION-102] Tokenizer: byte-level — ACCEPTED (owner delegation, 2026-10-06)
- **Decision:** Byte-level tokenizer (vocab 256 tetap, lossless by construction, tanpa training, deterministik penuh).
- **Reason:** Paling sederhana yang memenuhi REQ-004; tidak ada keputusan korpus yang menunggu. Lossless dijamin konstruksi.
- **Alternatives:** BPE (lebih pendek sequence tapi butuh training + keputusan vocab size), word-level (OOV problem), hybrid.
- **Consequences:** Sequence lebih panjang (trade-off diketahui); dapat direvisi ke BPE sebelum Stage 6 bila owner memutuskan.

## DECISION-011 — [RESOLUSI OPEN DECISION-103] Dataset: text configurable — ACCEPTED (owner delegation, 2026-10-06)
- **Decision:** Dataset berupa token ID dari teks apa pun (string); TextWindows sliding-window, batch deterministik per-seed, memori terkendali. Korpus final training Naze tetap OPEN (belum diputuskan owner).
- **Consequences:** Pipeline siap dipakai dengan korpus apa pun saat owner memutuskan.

## DECISION-012 — [RESOLUSI OPEN DECISION-104] First LM: MLP over context window (Bengio-style) — ACCEPTED (owner delegation, 2026-10-06)
- **Decision:** ids (B,T) -> embedding lookup -> flatten -> Linear -> tanh -> Linear -> logits. Loss softmax-CE. Sampling greedy + temperature.
- **Reason:** Paling sederhana yang memvalidasi seluruh pipeline end-to-end (REQ-010), sebelum Transformer (Stage 6).
- **Alternatives:** bigram (terlalu lemah untuk validasi pipeline), RNN (backprop through time — kompleksitas belum dibutuhkan).
- **Consequences:** Terbukti trainable end-to-end (loss turun, generation jalan). Kapasitas terbatas — digantikan Transformer di Stage 6.

## DECISION-013 — [RESOLUSI OPEN DECISION-106] Optimizer: SGD saja untuk saat ini — ACCEPTED (owner delegation, 2026-10-06)
- **Reason:** No overengineering. Adam/AdamW ditambah bila SGD terbukti tidak cukup (diukum, bukan dikarang).
- **Consequences:** Konvergensi lebih lambat; acceptable untuk model kecil.

## DECISION-014 — Checkpoint: npz (params + step), save/load deterministik — ACCEPTED (M-006)
- **Reason:** Format stdlib-level, sederhana, cukup untuk REQ-007 saat ini. Format lain (safetensors dsb.) tidak dibutuhkan.

## DECISION-015 — [RESOLUSI ISSUE-004] Vocabulary tokenizer byte-level FIXED 256; "configurable" ≠ vocab size — ACCEPTED [RECONSTRUCTED]
- **Context:** ISSUE-004 (MAJOR, SPEC_REVIEW v1.0.0 & v1.1.0): AC REQ-004 "vocab configurable" kontradiktif dengan DECISION-010 (byte-level, 256). Blocking M-007 sejak review pertama.
- **Decision:** Vocab tokenizer byte-level tetap FIXED 256 (seluruh ruang byte). "Configurable" TIDAK berlaku untuk vocab size — parameter configurable mengacu pada dimensi model lain (d_model, num_heads, num_layers, dst.). AC REQ-004 direvisi di REQUIREMENTS v0.4.0.
- **Reason:** Byte-level 256 = lossless by construction (D-010); fixed 256 konsisten dengan LM head logits (B, T, 256) pada arsitektur Transformer M-007 (TD §9); menghilangkan kontradiksi spesifikasi tanpa menambah kompleksitas.
- **Alternatives:** vocab size configurable (menunda implementasi, tidak ada kebutuhan terukur — REQ-302); migrasi BPE (butuh training, bertentangan dengan kesederhanaan D-010).
- **Consequences:** Tokenizer M-004 tidak berubah (DONE); TransformerConfig M-007 memvalidasi vocab_size == 256 (M007-T001); ISSUE-004 RESOLVED.

## DECISION-016 — [RESOLUSI OPEN DECISION-116 / ISSUE-006] Evaluasi tanpa ambang loss numerik universal; kerangka 5-basis — ACCEPTED [RECONSTRUCTED]
- **Context:** ISSUE-006 (MAJOR, SPEC_REVIEW v1.0.0 & v1.1.0): AC "loss menurun sesuai ambang" tanpa ambang numerik terdefinisi; OD-116 blocking sejak review pertama.
- **Decision:** Tidak ada ambang loss numerik universal. Evaluasi kualitas training divalidasi via kerangka 5-basis: (1) perbandingan terhadap baseline, (2) perbaikan loss yang terukur, (3) bukti konvergensi/stabilitas, (4) hasil validasi, (5) konfigurasi evaluasi yang reproducible. Angka target hanya ditetapkan per eksperimen (korpus/konfigurasi spesifik).
- **Reason:** Ambang universal tidak dapat dipertanggungjawabkan lintas korpus/hardware tanpa dasar empiris (melanggar prinsip no-invention); kerangka 5-basis objektif, auditable, dan berlaku untuk setiap eksperimen; konsisten dengan 17 AC objective TD M-007 §13 (tanpa ambang loss universal).
- **Alternatives:** ambang numerik tetap (mis. loss < X — dikarang tanpa dasar); tanpa kriteria apa pun (tidak verifiable).
- **Consequences:** AC REQ-006/009/010 direvisi di REQUIREMENTS v0.4.0; ISSUE-006 & OD-116 RESOLVED; angka target spesifik (bila kelak ditetapkan) menjadi bagian eksperimen/OD-107 (definisi sukses Naze 1.0).

## DECISION-017 — [RESOLUSI OPEN DECISION-120 / ISSUE-013] Format Technical Design per milestone — ACCEPTED [RECONSTRUCTED]
- **Context:** ISSUE-013 (SPEC_REVIEW v1.1.0) & OD-120: tahap Technical Design dalam workflow SDD tidak memiliki template/dokumen/lokasi baku.
- **Decision:** Technical Design per milestone disimpan sebagai `docs/architecture/M###_TECHNICAL_DESIGN.md` (satu dokumen per milestone), dengan 15 section wajib: Purpose, Scope, Non-goals, Requirements, Architecture, Module Boundaries, Interfaces, Data Flow, Configuration, Error Handling, Testing Strategy, Resource Constraints, Acceptance Criteria, Risks, Open Decisions.
- **Reason:** Memberi artefak baku untuk tahap Technical Design (menutup gap workflow ISSUE-013); lokasi `docs/architecture/` konsisten dengan struktur dokumentasi yang ada; 15 section memaksa desain eksplisit sebelum task breakdown dan implementasi.
- **Alternatives:** `docs/design/DESIGN-MXXX.md` (usulan SPEC_REVIEW v1.1.0 — ditolak demi konsistensi struktur folder yang sudah ada); tanpa template (status quo — gap workflow tetap terbuka).
- **Consequences:** PROJECT_SPEC §7 diperbarui: workflow 9 tahap eksplisit + governance maintenance policy (v0.4.0); `docs/architecture/M007_TECHNICAL_DESIGN.md` ditulis sesuai format ini (15 section, terverifikasi); ISSUE-013 & OD-120 RESOLVED; TRACEABILITY memuat kolom Technical Design (v1.3.0).

---

## OPEN DECISIONS (menunggu project owner)

> **[RECONSTRUCTED]** Konsolidasi daftar ini dari SPEC_REVIEW v1.1.0 §8 (OD-114..120) + M007_TECHNICAL_DESIGN §15 (OD-121). Daftar pada versi GitHub sebelumnya hanya memuat OD-101/107/108/112/113; isi entri OD-101/107/108/112/113 di bawah verbatim dari versi tersebut.

## OPEN DECISION-101 — Strategi accelerasi (GPU / native extension)
Dibutuhkan sebelum: Stage 6+ skala besar. Alternatif diketahui: NumPy+CPU; CUDA custom; Rust/C++ hot-path.

## OPEN DECISION-107 — Definisi sukses Naze 1.0 (metrik, target, dataset evaluasi)
Dibutuhkan sebelum: Stage 9.

## OPEN DECISION-108 — Roadmap multimodal / physical AI (Stage 10)
Dibutuhkan sebelum: Stage 10. Jauh di masa depan.

## OPEN DECISION-112 — Korpus final training Naze (bahasa, ukuran, sumber, lisensi)
Dibutuhkan sebelum: training skala serius (pasca-M-006). Pipeline sudah siap menerima korpus apa pun.

## OPEN DECISION-113 — Coverage target test suite
Dibutuhkan sebelum: implementasi M-007 dinyatakan selesai penuh (TD M-007 §15: DEFERRED ke fase implementasi) — menetapkan ambang coverage formal.

## OPEN DECISION-114 — CI platform
Memengaruhi enforcement AC test-plan (TD M-007 §15; SPEC_REVIEW ISSUE-003: "diuji di CI" belum actionable).

## OPEN DECISION-115 — Definisi "cleaning" dataset (ISSUE-005)

## OPEN DECISION-117 — Metrik memori formal REQ-102 (ISSUE-011)

## OPEN DECISION-118 — Batas ukuran Transformer final
BLOCKER untuk training skala serius; BUKAN blocker untuk konfigurasi dev/test M-007 (TD §15: d_model, num_layers, T_max produksi, budget RAM/waktu perangkat owner). Konfigurasi TD §9 hanya dev/test.

## OPEN DECISION-119 — Versioning & release policy Naze 1.0 (MISS-003)

## OPEN DECISION-121 — Skema positional final Transformer
Default: learned positional embedding (TD M-007 §9/§15); owner dapat mengubah sebelum/awal implementasi M007-T003.

**CLOSED oleh governance 2026-10-06/07:** OD-116 → DECISION-016; OD-120 → DECISION-017; ISSUE-004 → DECISION-015. ([HISTORICAL DETAIL UNAVAILABLE] — tanggal persis keputusan owner tidak tersedia pada sumber yang dapat dipulihkan; rekonstruksi dicatat 2026-10-07.)

## Riwayat Resolusi

| Open Decision / Issue | Resolusi | Tanggal | Cara |
|---|---|---|---|
| OD-105 (autodiff) | DECISION-009: backprop terstruktur per-layer | 2026-10-06 | Delegasi owner ("saya izinkan") |
| OD-102 (tokenizer) | DECISION-010: byte-level | 2026-10-06 | Delegasi owner |
| OD-103 (dataset) | DECISION-011: text configurable (korpus final tetap OPEN-112) | 2026-10-06 | Delegasi owner |
| OD-104 (first LM) | DECISION-012: MLP Bengio-style | 2026-10-06 | Delegasi owner |
| OD-106 (optimizer) | DECISION-013: SGD (Adam ditunda) | 2026-10-06 | Delegasi owner |
| ISSUE-004 (REQ-004 ↔ D-010) | DECISION-015: vocab byte-level FIXED 256; "configurable" ≠ vocab size | [HISTORICAL DETAIL UNAVAILABLE] | Keputusan owner (governance M-007) |
| OD-116 / ISSUE-006 (ambang loss) | DECISION-016: kerangka evaluasi 5-basis; tanpa ambang numerik universal | [HISTORICAL DETAIL UNAVAILABLE] | Keputusan owner (governance M-007) |
| OD-120 / ISSUE-013 (format TD) | DECISION-017: `docs/architecture/M###_TECHNICAL_DESIGN.md`, 15 section wajib | [HISTORICAL DETAIL UNAVAILABLE] | Keputusan owner (governance M-007) |
