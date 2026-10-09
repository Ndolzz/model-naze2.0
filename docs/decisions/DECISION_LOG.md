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

## DECISION-018 — [RESOLUSI OPEN DECISION-118] Batas ukuran Transformer final: checkpoint ≤ 1 MB — ACCEPTED (owner, 2026-10-09)
- **Context:** OD-118 terbuka sejak SPEC_REVIEW v1.0.0 — blocker training skala serius; menunggu owner menetapkan d_model, num_layers, T_max produksi, budget RAM. TD M-007 §12/§15 membatasi estimasi ke config dev/test (~89k param, 0.7 MB) tanpa batas final.
- **Decision:** Ukuran checkpoint final ≤ 1 MB (float64 npz params-only). Konfigurasi produksi: d_model=64, num_heads=4, num_layers=2, d_ff=128, T_max=128 — ~108k param ≈ 0.87 MB float64 (≈ 0.43 MB float32 saat inference).
- **Reason:** Paling ringan dan tercepat untuk training CPU (REQ-102 hardware terbatas); konsisten target on-device nazeio (ARMv7 total app < 20 MB — model ≤ 1 MB menyisakan headroom besar); no-overengineering (REQ-302) — identik config dev/test TD §9 kecuali T_max 64→128.
- **Alternatives:** ≤ 5 MB (D=128, L=4 — ditolak owner: belum ada kebutuhan terukur); ≤ 10-20 MB (menunda training serius di CPU); defer (membiarkan blocker untuk M-008).
- **Consequences:** OD-118 RESOLVED; M-008 training system memakai batas ini sebagai constraint formal; batas dapat direvisi owner bila kapasitas terbukti kurang (diukur, bukan diasumsikan); nazeio spesifikasi 16 dapat ditetapkan mengacu keputusan ini.

---

## DECISION-019 — [RESOLUSI OPEN DECISION-113] Coverage target: ≥ 80% baris di src/naze — ACCEPTED (owner, 2026-10-09)
- **Context:** OD-113 terbuka sejak SPEC_REVIEW v1.0.0; ISSUE-003/AC "diuji di CI" menunggu ambang coverage formal. M-007 selesai tanpa coverage terukur (pytest penuh hijau, 335 test).
- **Decision:** Target coverage formal: **≥ 80% baris** pada `src/naze` (pytest-cov), diverifikasi via CI.
- **Reason:** Praktis dan standar industri; realistis untuk repositori ukuran ini; mencegah regresi modul yang tak tersentuh test (23 failure pra-existing yang baru terlihat setelah CI ada membuktikan pentingnya pengukuran).
- **Alternatives:** ≥ 90% (ditolak: biaya marginal tinggi pada kode validasi fail-fast); tanpa angka (ditolak: tidak verifiable).
- **Consequences:** CI menambah pengukuran coverage (pytest-cov) — implementasi menyusul di M-008; kegagalan ambang = pipeline merah.

## DECISION-020 — [RESOLUSI OPEN DECISION-114] CI enforcement: pytest wajib hijau untuk main — ACCEPTED (owner, 2026-10-09)
- **Context:** OD-114 terbuka sejak SPEC_REVIEW v1.0.0; ISSUE-003: "diuji di CI" belum actionable. Workflow CI ada sejak M007-T017 tetapi pasif.
- **Decision:** CI pytest **wajib hijau** untuk push/merge ke `main` — diaktifkan via branch protection rule (required status check `pytest`); PR/push yang merah ditolak.
- **Reason:** Menutup gap enforcement AC test-plan secara formal; mencegah main berada dalam keadaan merah seperti run CI pertama (23 failure).
- **Alternatives:** CI pasif (ditolak: tanpa enforcement, red main terjadi lagi); CI + ruff gate (ditunda: lint belum ada kebutuhan terukur — REQ-302).
- **Consequences:** Branch protection diaktifkan di GitHub (aturan repo, bukan file); ISSUE-003 tertutup untuk M-007 ke atas.

## DECISION-021 — [RESOLUSI OPEN DECISION-117 / ISSUE-011] Metrik memori formal REQ-102: checkpoint ≤ 1 MB + RAM training < 2 GB — ACCEPTED (owner, 2026-10-09)
- **Context:** REQ-102 (resource constraints) tanpa metrik terdefinisi sejak ISSUE-011 (SPEC_REVIEW v1.0.0); TD M-007 §12 hanya estimasi dev/test.
- **Decision:** Dua metrik formal: (1) **ukuran checkpoint params-only ≤ 1 MB** (mengacu DECISION-018); (2) **RAM puncak proses training < 2 GB** (params + grads + aktivasi cache + batch data), diukur pada config produksi D-018.
- **Reason:** Angka konkret dan terukur (menutup "tidak actionable"); < 2 GB aman untuk laptop/PC umum; konsisten target on-device nazeio (RAM perangkat ARMv7 jauh lebih kecil — inference float32).
- **Alternatives:** tanpa angka RAM (tidak verifiable); batas RAM lebih ketat 1 GB (ditolak: float64 + aktivasi scores bisa mendekati batas pada T_max=128).
- **Consequences:** REQ-102 AC diperbarui di REQUIREMENTS (menyusul per change policy); M-008 training loop wajib memverifikasi kedua metrik.

## DECISION-022 — [RESOLUSI OPEN DECISION-121] Skema positional final: learned positional embedding — ACCEPTED (owner, 2026-10-09)
- **Context:** OD-121 dibuka di TD M-007 §15: default learned; owner dapat mengubah sebelum/awal M007-T003. T003 terimplementasi dengan learned dan lulus seluruh test M-007.
- **Decision:** Skema positional final = **learned positional embedding** (tabel (T_max, D), init normal(0, 0.1) per-seed). Tidak bermigrasi ke sinusoidal.
- **Reason:** Sudah terimplementasi dan teruji penuh (CI hijau); paling sederhana dengan engine yang ada; sinusoidal tidak menawarkan keuntungan terukur pada T_max ≤ 128 (REQ-302).
- **Alternatives:** sinusoidal (ditolak: revisi tanpa kebutuhan terukur; learned berkinerja baik pada sekuens pendek).
- **Consequences:** OD-121 RESOLVED; komponen PositionalRepr tidak berubah; revisi masa depan tetap mungkin lewat change policy.

## DECISION-024 — [RESOLUSI OPEN DECISION-112] Korpus final training: strategi hybrid ~7 MB — ACCEPTED (owner, 2026-10-09)
- **Context:** OD-112 blocking training skala serius sejak SPEC_REVIEW v1.0.0. Pipeline sudah siap menerima korpus apa pun (DECISION-011), tetapi korpus final untuk Naze 1.0 belum ditetapkan owner. DECISION-018 (batas checkpoint ≤ 1 MB) dan DECISION-023 (target sukses) menunggu keputusan ini.
- **Decision:** Tetapkan **strategi corpus hybrid** untuk training Naze 1.0:
  - **70% Bahasa Indonesia Umum:** ~5 MB, dari sumber dengan lisensi terbuka (Wikipedia, berita dengan lisensi CC/BY/mitra).
  - **20% Perintah Asisten:** ~1.5 MB, data sintetis terkontrol untuk mendukung kemampuan command (nazeio).
  - **10% Instruksi Editing:** ~0.5 MB, data sintetis untuk mendukung kemampuan editing (Naze Motion Agent).
  - **Total target:** ~7 MB (ukuran file teks UTF-8 mentah, sebelum preprocessing).
- **Sumber dan Lisensi:** Wikipedia (CC-BY-SA), berita dengan lisensi eksplisit. Verifikasi lisensi WAJIB sebelum penggunaan. Data sintetis dibuat untuk proyek.
- **Preprocessing:** Pembersihan, normalisasi, deduplikasi. Tokenisasi: ByteTokenizer (D-010, D-015) vocab 256 fixed.
- **Pembagian Dataset:** Training 80%, Validation 10%, Holdout 10% (TIDAK BOLEH untuk training). Deterministik per-seed (REQ-101).
- **Acceptance Criteria:** Sumber data memiliki lisensi redistribusi. Ukuran aktual terdokumentasi. Fingerprint SHA-256. Tidak ada data leakage.
- **Risiko:** Verifikasi lisensi WAJIB. Data sintetis tidak mencerminkan distribusi nyata. Ukuran estimasi.
- **Alternatives:** Wikipedia+News Only (~10 MB), Synthetic Only (~5-7 MB), Minimal (~2 MB).
- **Consequences:** OD-112 RESOLVED. Training M-010 dapat dimulai setelah corpus final disiapkan.


---


- **Context:** OD-112 blocking training skala serius sejak SPEC_REVIEW v1.0.0. Pipeline sudah siap menerima korpus apa pun (DECISION-011), tetapi korpus final untuk Naze 1.0 belum ditetapkan owner. DECISION-018 (batas checkpoint ≤ 1 MB) dan DECISION-023 (target sukses) menunggu keputusan ini.
- **Decision:** Tetapkan **strategi corpus hybrid** untuk training Naze 1.0:
  - **70% Bahasa Indonesia Umum:** ~5 MB, dari sumber dengan lisensi terbuka (Wikipedia, berita dengan lisensi CC/BY/mitra).
  - **20% Perintah Asisten:** ~1.5 MB, data sintetis terkontrol untuk mendukung kemampuan command (nazeio).
  - **10% Instruksi Editing:** ~0.5 MB, data sintetis untuk mendukung kemampuan editing (Naze Motion Agent).
  - **Total target:** ~7 MB (ukuran file teks UTF-8 mentah, sebelum preprocessing).
- **Sumber dan Lisensi:** Wikipedia (CC-BY-SA), berita dengan lisensi eksplisit. Verifikasi lisensi WAJIB sebelum penggunaan. Data sintetis dibuat untuk proyek.
- **Preprocessing:** Pembersihan, normalisasi, deduplikasi. Tokenisasi: ByteTokenizer (D-010, D-015) vocab 256 fixed.
- **Pembagian Dataset:** Training 80%, Validation 10%, Holdout 10% (TIDAK BOLEH untuk training). Deterministik per-seed (REQ-101).
- **Acceptance Criteria:** Sumber data memiliki lisensi redistribusi. Ukuran aktual terdokumentasi. Fingerprint SHA-256. Tidak ada data leakage.
- **Risiko:** Verifikasi lisensi WAJIB. Data sintetis tidak mencerminkan distribusi nyata. Ukuran estimasi.
- **Alternatives:** Wikipedia+News Only (~10 MB), Synthetic Only (~5-7 MB), Minimal (~2 MB).
- **Consequences:** OD-112 RESOLVED. Training M-010 dapat dimulai setelah corpus final disiapkan.
## OPEN DECISIONS (menunggu project owner)

> **[RECONSTRUCTED]** Konsolidasi daftar ini dari SPEC_REVIEW v1.1.0 §8 (OD-114..120) + M007_TECHNICAL_DESIGN §15 (OD-121). Daftar pada versi GitHub sebelumnya hanya memuat OD-101/107/108/112/113; isi entri OD-101/107/108/112/113 di bawah verbatim dari versi tersebut.

## OPEN DECISION-101 — Strategi accelerasi (GPU / native extension)
Dibutuhkan sebelum: Stage 6+ skala besar. Alternatif diketahui: NumPy+CPU; CUDA custom; Rust/C++ hot-path.
## DECISION-023  [RESOLUSI OPEN DECISION-107] Definisi sukses Naze 1.0: metrik, target provisional, dataset, release gate  ACCEPTED (2026-10-09, provisional)
- **Context:** OD-107 blocking M-010 (Stage 9) sejak SPEC_REVIEW v1.0.0. M-001..M-009 DONE, pipeline end-to-end siap, tapi tanpa definisi sukses formal untuk rilis produksi. DECISION-016 menegaskan tidak ada ambang loss numerik universal; target harus spesifik per eksperimen.
- **Decision:** Tetapkan **target provisional** (bukan hasil pengukuran, bukan jaminan) untuk Naze 1.0:

  | Metrik | Target Provisional | Satuan | Release Gate |
  |---|---|---|---|
  | Training loss | <= 2.5 | average cross-entropy per token | **WAJIB** |
  | Validation perplexity | <= 35 | exp(validation loss) | **WAJIB** |
  | NazeIO command accuracy | >= 90% | persentase command benar | **WAJIB** |
  | ARMv7 inference latency | <= 2 detik/token | rata-rata per token (prefill+generate) | Target optimasi |
  | Model artifact dalam APK | <= 50 MB | ukuran total dengan model | Target optimasi |

- **Definisi dan Metode Evaluasi:**
  - **Training loss:** Average cross-entropy per token pada dataset training, dihitung via `naze.lm.cross_entropy` (softmax + negative log likelihood). Metode: rata-rata loss per batch selama training, dilaporkan di akhir epoch.
  - **Validation perplexity:** exp(validation loss) pada holdout dataset (OD-112). Validation loss = average cross-entropy per token. Metode: evaluasi penuh pada holdout set, dilaporkan sebagai `perplexity` di `naze.train.EvalResult`.
  - **NazeIO command accuracy:** Persentase command yang dieksekusi dengan benar. Metode: benchmark dengan command list eksplisit (input, expected output, aturan penilaian). Dataset command = bagian dari OD-112.
  - **ARMv7 inference latency:** Rata-rata waktu per token (prefill + generation). Konfigurasi referensi: Raspberry Pi 3 (ARMv7, 4-core @1.2GHz), config D-018 (D=64, H=4, L=2, d_ff=128, T_max=128), float32 inference. Metode: `naze.inference.benchmark_model` dengan warmup=3, runs=5, rata-rata.
  - **Model artifact dalam APK:** Ukuran total APK Android dengan model terintegrasi. Target berlaku untuk integrasi Android (nazeio), bukan ukuran paket PyPI. Metode: pengukuran ukuran APK final.

- **Konsistensi Matematis:**
  - Loss function: cross-entropy = -mean(log(p[target])) per token (lihat `src/naze/lm/mlp_lm.py::cross_entropy`).
  - Perplexity = exp(loss) - valid untuk cross-entropy per-token.
  - Training loss <= 2.5 => Perplexity training <= exp(2.5) ~ 12.18 (konsisten, target training lebih ketat dari validation).
  - Validation perplexity <= 35 => Validation loss <= ln(35) ~ 3.55 (target validation lebih longgar, sesuai dengan generalisasi).
  - **Catatan:** Target loss/perplexity adalah **provisional** dan harus divalidasi empiris. Jika training pada corpus final (OD-112) tidak mencapai target, target direvisi (bukan model di-overfit).

- **Kondisi Pengujian:** Config model: DECISION-018 (D=64, H=4, L=2, d_ff=128, T_max=128, vocab=256). Training: SGD optimizer (DECISION-013), float64 precision (DECISION-007). Inference: float32 precision untuk nazeio.

- **Acceptance Criteria:**
  - **WAJIB (Release Gate):** Training loss <= 2.5 AND Validation perplexity <= 35 AND NazeIO command accuracy >= 90%.
  - **Target Optimasi:** ARMv7 latency <= 2 detik/token dan APK size <= 50 MB. Jika tidak terpenuhi, rilis tetap dapat dilakukan dengan catatan keterbatasan.

- **Risiko dan Keterbatasan:** Target provisional (belum divalidasi OD-112), ARMv7 latency tergantung hardware, APK size tergantung integrasi nazeio. **Bloker:** OD-112 **WAJIB** diselesaikan sebelum training.

- **Dependensi:** OD-112 (korpus final), OD-119 (versioning policy), nazeio (command benchmark, APK integration).

- **Alternatives:** Tidak menetapkan target (tidak actionable), target lebih ketat (risiko tidak konvergen), target lebih longgar (risiko kualitas rendah).

- **Consequences:** OD-107 **RESOLVED** dengan status **PROVISIONAL**. Target dapat direvisi owner berdasarkan hasil empiris.


## OPEN DECISION-107  Definisi sukses Naze 1.0 (metrik, target, dataset evaluasi)
Dibutuhkan sebelum: Stage 9.

**RESOLVED:** Lihat DECISION-023 di atas. Status: **ACCEPTED (2026-10-09, owner approval)**.


## OPEN DECISION-108 — Roadmap multimodal / physical AI (Stage 10)
Dibutuhkan sebelum: Stage 10. Jauh di masa depan.


Dibutuhkan sebelum: training skala serius (pasca-M-006). Pipeline sudah siap menerima korpus apa pun.



## OPEN DECISION-115 — Definisi "cleaning" dataset (ISSUE-005)

## OPEN DECISION-119 — Versioning & release policy Naze 1.0 (MISS-003)

**CLOSED oleh governance 2026-10-06/07:** OD-116 → DECISION-016; OD-120 → DECISION-017; ISSUE-004 → DECISION-015. ([HISTORICAL DETAIL UNAVAILABLE] — tanggal persis keputusan owner tidak tersedia pada sumber yang dapat dipulihkan; rekonstruksi dicatat 2026-10-07.) **2026-10-09:** OD-118 → DECISION-018. **2026-10-09 (batch):** OD-113 → DECISION-019; OD-114 → DECISION-020; OD-117/ISSUE-011 → DECISION-021; OD-121 → DECISION-022.

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
| OD-118 (batas ukuran Transformer) | DECISION-018: checkpoint ≤ 1 MB (f64); D=64 H=4 L=2 d_ff=128 T_max=128 | 2026-10-09 | Keputusan owner |
| OD-113 (coverage target) | DECISION-019: ≥ 80% baris src/naze via pytest-cov di CI | 2026-10-09 | Keputusan owner (batch) |
| OD-114 (CI enforcement) | DECISION-020: pytest wajib hijau utk main (branch protection) | 2026-10-09 | Keputusan owner (batch) |
| OD-117 / ISSUE-011 (metrik memori) | DECISION-021: checkpoint ≤ 1 MB + RAM training < 2 GB | 2026-10-09 | Keputusan owner (batch) |
| OD-121 (skema positional) | DECISION-022: learned positional embedding (final) | 2026-10-09 | Keputusan owner (batch) |
