# M008 TECHNICAL DESIGN — Training System lengkap (Stage 7 penuh)

**Milestone:** M-008 — Training System lengkap
**Stage:** Stage 7
**Status:** DRAFT — awaiting project owner approval
**Version:** 0.1.0
**Source of truth:** PROJECT_SPEC v0.4.0, REQUIREMENTS v0.5.0
(REQ-006/007/009/101/102/103), DECISION-013/014/016/018/019/020/021/022,
M007_TECHNICAL_DESIGN (§8 catatan per-posisi)
---

## 1. Purpose

Mendefinisikan desain teknis training system penuh Naze: loop epoch terstruktur,
logging per-run, evaluasi berkala (loss/perplexity), checkpoint dengan integritas
dan resume penuh, konfigurasi eksperimen reproducible, serta verifikasi metrik
resource formal (DECISION-018/021). Dokumen ini adalah acuan tunggal
implementasi M-008.

## 2. Scope

- TrainConfig: konfigurasi eksperimen terstruktur + serialisasi (MISS-004).
- Loss per-posisi penuh untuk Transformer (M007 TD §8: tugas M-008).
- Training loop epoch-based dengan resume deterministik penuh (ISSUE-007).
- Checkpoint v2: checksum SHA-256 (ISSUE-008), metadata per-run (ISSUE-010).
- Evaluation harness: mean CE loss + perplexity pada holdout (REQ-009,
  kerangka 5-basis DECISION-016); eval berkala selama training.
- Logging terstruktur JSONL + stdout ringkas.
- Test load checkpoint -> generate (ISSUE-009).
- Metrik resource formal: RAM puncak training < 2 GB, checkpoint <= 1 MB
  (DECISION-021) pada config produksi DECISION-018.
- Coverage CI: pytest-cov >= 80% baris src/naze (DECISION-019).

## 3. Non-goals

- Optimizer baru (Adam/AdamW) — DECISION-013: SGD sampai terbukti kurang
  (diukur pada eksperimen konkret, bukan diasumsikan).
- Learning-rate schedule / early-stopping — tanpa kebutuhan terukur (REQ-302).
- Benchmark kecepatan inference (REQ-009 mapping: Stage 8 = M-009).
- Data cleaning korpus — OD-115/OD-112 tetap OPEN; pipeline menerima teks
  apa pun.
- API docs lengkap (ISSUE-002) — dijadwalkan M-009.
- Distributed/multi-process training — jauh di luar hardware target.

## 4. Requirements

| Req | Relevan untuk M-008 |
|---|---|
| REQ-006 Training | Loop epoch penuh, logging, eval berkala (bagian M-008) |
| REQ-007 Checkpointing | Checksum SHA-256 + epoch state resume (ISSUE-007/008) |
| REQ-009 Evaluation | Harness loss/perplexity dari checkpoint; 5-basis D-016 |
| REQ-101 Reproducibility | Metadata per-run lengkap; resume deterministik identik |
| REQ-102 Resource | Metrik formal D-021: RAM < 2 GB, checkpoint <= 1 MB |
| REQ-103 Testability | Coverage >= 80% via pytest-cov di CI (D-019) |
| REQ-201/202/203 | Permanen: tanpa pretrained/API/framework DL |
| REQ-301/302 | Spec-driven; fitur hanya yang dibutuhkan milestone ini |

## 5. Architecture

Pipeline training run tunggal (deterministik end-to-end per TrainConfig):

```
TrainConfig (JSON) -> build model (MLPLM | TransformerLM per config.model)
  -> TrainingRun:
       epoch e = start_epoch..epochs-1:
         batches = TextWindows.batches_pos(seed = cfg.seed + e)  # per-posisi
         per batch: loss = model.loss_pos(logits, y_pos)
                    grads = model.backward_pos()
                    SGD update (p -= lr * g)
         per eval_interval epoch: evaluate(holdout) -> log
       checkpoint akhir + metadata.json + checksum
  -> resume: load params + epoch -> lanjut epoch berikutnya IDENTIK
```

Kunci desain resume deterministik: urutan batch epoch-e ditentukan penuh oleh
`seeded_rng(cfg.seed + e)` — run tanpa interrupt dan run resume menghasilkan
urutan batch identik per epoch, sehingga params hasil akhir identik eksak.
Tidak ada state RNG global yang tersimpan; state cukup (epoch, step, params).

## 6. Module Boundaries

| Module | Responsibility | Baru/ubah |
|---|---|---|
| naze/train/config.py | TrainConfig dataclass + save/load JSON | baru |
| naze/train/loop.py | TrainingRun: epoch loop, eval/log, resume | baru |
| naze/train/checkpoint.py | Checkpoint v2: npz + meta.json + SHA-256 | baru |
| naze/train/evaluate.py | mean CE loss + perplexity pada dataset | baru |
| naze/train/runlog.py | JSONL writer + RunMetadata | baru |
| naze/data/dataset.py | tambah batches_pos() (shift-1 per-posisi) | ubah |
| naze/lm/transformer_lm.py | tambah loss_pos/backward_pos | ubah |
| naze/train/trainer.py | SGDTrainer/save/load lama TIDAK diubah | tetap |
| .github/workflows/tests.yml | tambah pytest-cov >= 80% | ubah |

Boundary rule tetap (M007 TD §6): train -> lm -> nn -> core; train tidak
diimpor oleh lm/nn. Modul lama (trainer.py) tidak diubah — API lama tetap
dipakai test M-006 (regressi); modul baru memakai SGDTrainer sebagai engine
step agar tidak ada duplikasi logika update.

## 7. Interfaces

```
TrainConfig: dataclass
  model: "mlplm" | "transformer"     # transformer = default M-008
  corpus_path: str; holdout_path: str | None
  block_size: int; batch_size: int
  lr: float; epochs: int; seed: int
  eval_every: int (=1); checkpoint_path: str | None
  to_json() / from_json() — serialisasi lossless

TextWindows.batches_pos(): Iterator[(x (B,T), y (B,T))]
  # y = ids[i+1 : i+T+1] per window (shift-1); batches() lama tetap

TransformerLM.loss_pos(logits (B,T,256), targets (B,T)) -> float
TransformerLM.backward_pos() -> dict[str, Array]  # pola backward lama

save_checkpoint_v2(dir, model, run_state, config)
  # params.npz + meta.json {step, epoch, config-hash, naze-version,
  # created} + params.npz.sha256
load_checkpoint_v2(dir, model, *, verify=True) -> run_state
  # verify: recompute SHA-256; mismatch -> ValueError (fail-fast)

evaluate(model, data: TextWindows, *, per_pos=False) -> EvalResult
  # EvalResult: mean_loss, perplexity=exp(loss), n_batches, n_tokens

TrainingRun(config, model).run(start_epoch=0) -> RunSummary
  # log JSONL per eval; resume = load_checkpoint_v2 -> run(start_epoch)

measure_peak_rss() -> int | None   # resource.getrusage ru_maxrss (KB)
checkpoint_size(path) -> int       # byte; AC D-021/D-018
```

## 8. Data Flow

1. TrainConfig dimuat (JSON) -> validasi fail-fast (dimensi > 0, lr > 0,
   epochs > 0, path korpus ada).
2. Korpus -> ByteTokenizer.encode -> ids -> TextWindows(block, batch).
3. Epoch e: batches_pos(seed=seed+e) -> tiap batch: forward -> loss_pos ->
   backward_pos -> SGD update; loss train dicatat per epoch (mean).
4. Tiap eval_every epoch: evaluate(holdout) -> perplexity -> JSONL+stdout.
5. Akhir run (atau tiap checkpoint interval): save_checkpoint_v2 + meta.
6. Resume: load_checkpoint_v2 (verifikasi checksum) -> run(start_epoch=e+1).
7. Pasca-run: generate dari checkpoint final (ISSUE-009) — deterministik.

## 9. Configuration

```
model: "transformer" (default M-008; config produksi DECISION-018:
  d_model=64, num_heads=4, num_layers=2, d_ff=128, T_max=128, seed=0)
  | "mlplm" (block_size=6, d_embed=8, d_hidden=16 — legacy sanity)
lr: 0.08 (Transformer toy; per eksperimen — tanpa ambang D-016)
epochs, batch_size, block_size: per eksperimen
eval_every: 1 (default)
holdout_path: opsional; tanpa holdout -> eval train-only (dicatat di meta)
dtype: float64 (D-007); checkpoint v2 params npz float64
checksum: SHA-256 (hashlib stdlib — bukan dependensi baru)
```

## 10. Error Handling

- Validasi config invalid -> ValueError (fail-fast, pola repo).
- Checksum mismatch saat load -> ValueError (corrupt, tidak lanjut
  diam-diam).
- Param hilang/shape mismatch saat load -> KeyError/ValueError (pola lama).
- Resume epoch >= epochs -> no-op dengan warning eksplisit di log.
- Loss/grad non-finite di epoch -> ValueError (as_array pola engine);
  run berhenti fail-fast, metadata mencatat epoch gagal.
- measure_peak_rss() -> None pada platform tanpa getrusage (dicatat,
  bukan error — CI = Ubuntu Linux selalu tersedia).

## 11. Testing Strategy

- **Unit:** TrainConfig serialisasi roundtrip; batches_pos shape/shift
  benar & deterministik per-seed; loss_pos vs loss lama pada kasus y
  seragam; checksum mismatch ditolak; evaluate mean/perplexity pada kasus
  acuan (logits identik -> perplexity 1.0); measure_peak_rss/size valid.
- **Integration:** run 3 epoch toy corpus -> loss turun (basis 2 D-016),
  JSONL + metadata lengkap; run 2 epoch -> save -> resume 1 epoch IDENTIK
  dengan run 3 epoch langsung (array_equal params); load -> generate
  deterministik (ISSUE-009); RAM puncak < 2 GB & checkpoint <= 1 MB pada
  config D-018 (D-021).
- **Regression:** pytest penuh via CI hijau; test lama tidak dimodifikasi;
  coverage >= 80% (D-019) di-enforce CI (--cov-fail-under=80).
- Evaluasi kualitas mengikuti kerangka 5-basis D-016 (baseline, perbaikan
  terukur, konvergensi/stabilitas, validasi holdout, config reproducible) —
  tanpa ambang loss numerik universal.

## 12. Resource Constraints

- Metrik formal DECISION-021: RAM puncak proses training < 2 GB
  (resource.getrusage ru_maxrss, Linux KB) dan checkpoint <= 1 MB (D-018)
  pada config produksi; keduanya diukur otomatis tiap integration run.
- Estimasi RAM pada config D-018: params+grads ~108k x 8B x 2 ≈ 1.7 MB;
  aktivasi dominan scores (B,H,T,T) per layer ≈ 8x4x128^2x8B ≈ 4 MB/layer
  x2 + konteks — total proses jauh < 200 MB (interpreter ~50 MB overhead).
  Headroom besar terhadap batas 2 GB.
- Perplexity hanya dievaluasi pada holdout kecil — biaya O(n_windows x T).
- Tidak ada dependensi baru: hashlib/json/resource = stdlib; pytest-cov =
  tooling development (REQ-203 tidak dilanggar).

## 13. Acceptance Criteria (objective, verifiable)

1. TrainConfig serialisasi roundtrip lossless; config invalid ditolak.
2. batches_pos menghasilkan y = shift-1 benar; deterministik per-seed.
3. loss_pos = loss lama pada kasus target seragam semua posisi (padanan
   eksak diverifikasi numerik).
4. Checkpoint v2 menyimpan params + step + epoch + config + checksum;
   load memverifikasi checksum; mismatch ditolak ValueError.
5. Resume deterministik: params akhir run-resume == run-langsung (eksak).
6. Evaluasi berkala tercatat di JSONL; perplexity = exp(mean loss) benar.
7. Metadata per-run lengkap (config, versi, checksum, waktu, hasil akhir).
8. Load checkpoint -> generate deterministik per-seed (ISSUE-009).
9. RAM puncak < 2 GB dan checkpoint <= 1 MB pada config D-018 (D-021).
10. CI menjalankan pytest penuh + coverage >= 80% baris src/naze (D-019).
11. Test lama tidak dimodifikasi; seluruh suite hijau via CI.
12. Tanpa pretrained/API LLM/framework DL (REQ-201/202/203 permanen).
13. SGD tetap satu-satunya optimizer (D-013 — tidak ada optimizer baru).
14. Evaluasi kualitas mengikuti kerangka 5-basis D-016 pada toy corpus
    (loss menurun terukur; holdout dievaluasi bila tersedia).

## 14. Risks

1. Resume identik-eksak sensitif terhadap urutan operasi float — mitigasi:
   test array_equal params (bukan allclose) pada kasus toy; bila di CI
   ditemukan perbedaan bit akibat BLAS non-determinisme, AC diberi catatan
   dan diuji ulang (keputusan owner, tidak diasumsikan).
2. loss_pos Transformer mengubah basis loss vs M-007 (y (B,) vs (B,T)) —
   mitigasi: AC 3 memastikan padanan eksak pada kasus seragam; test lama
   M-007 (y (B,)) tidak berubah dan tetap hijau.
3. ru_maxrss mencakup seluruh proses Python (overhead interpreter) —
   angka tetap jauh di bawah 2 GB; dicatat transparan di metadata.
4. Coverage 80% mungkin gagal awal pada modul util kecil — mitigasi:
   jalankan pengukuran dulu, tambah test hanya untuk celah nyata
   (bukan test kosong demi angka — melanggar semangat REQ-302).

## 15. Open Decisions

- Tidak ada OD baru dari desain ini. OD lama yang tersisa tidak memblokir
  M-008: OD-112 (korpus final — pipeline menerima teks apa pun), OD-115
  (cleaning — non-goal), OD-101 (akselerasi — dev CPU memadai untuk
  eksperimen toy), OD-107/108/119 (jauh pasca-M-008).
- Keputusan yang dipakai desain ini: D-013 (SGD), D-014 (npz), D-016
  (evaluasi 5-basis), D-018 (batas ukuran), D-019 (coverage 80%), D-020
  (CI enforcement), D-021 (metrik memori), D-022 (positional learned).
