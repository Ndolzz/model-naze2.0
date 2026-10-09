# M008_TASKS — Training System Lengkap (Stage 7 Penuh)

**Status:** ACTIVE (dimulai 2026-10-09)
**Referensi:** docs/architecture/M008_TECHNICAL_DESIGN.md (disetujui owner)
**Requirements:** REQ-007 (training system), REQ-101 (reproducibility),
REQ-102 (checkpoint & resume), REQ-103 (evaluasi terukur); keputusan
D-018 (checkpoint <=1 MB, config D=64/H=4/L=2/d_ff=128/T_max=128),
D-019 (coverage >=80% baris src/naze), D-021 (metrik memori: checkpoint
<=1 MB + peak RSS training <2 GB), D-022 (positional learned).

**Aturan:** test lama TIDAK dimodifikasi; perubahan engine bersifat
aditif-backward-compatible; kode <=100 kolom; docstring Bahasa Indonesia.

---

## T001 — TrainConfig (src/naze/train/config.py)
Frozen dataclass konfigurasi training lengkap: model ("transformer" |
"mlplm"), corpus_path, holdout_path, block_size=8, batch_size=8,
lr=0.08, epochs=3, seed, eval_every=1, checkpoint_path, dimensi
transformer (default D-018) dan mlplm. Validasi fail-fast di
__post_init__ (ValueError). to_dict/to_json/from_json (sort_keys),
config_hash = SHA-256 JSON kanonik, save/load.
**AC:** [ ] validasi menolak konfigurasi tidak valid; [ ] roundtrip
JSON identik; [ ] config_hash stabil.

## T002 — Checkpoint v2 (src/naze/train/checkpoint.py)
save_checkpoint_v2(directory, model, *, step, epoch, config=None):
params.npz (+__step, __epoch), params.npz.sha256, meta.json (step,
epoch, created UTC, naze_version, size_bytes, checksum_sha256, config,
config_hash). load_checkpoint_v2(directory, model, *, verify=True)
memverifikasi checksum (mismatch -> ValueError), cek param/shape.
**AC:** [ ] roundtrip identik byte-level per param; [ ] checksum
tampered ditolak; [ ] meta lengkap.

## T003 — Evaluasi (src/naze/train/evaluate.py)
EvalResult frozen dataclass {mean_loss, perplexity, n_batches,
n_tokens}. evaluate(model, data, *, per_pos=False, seed=None):
iterasi batches_pos (transformer) atau batches (mlplm); perplexity =
exp(mean_loss) dengan guard overflow (-> inf).

**AC:** [ ] perplexity == exp(loss); [ ] dataset kosong -> ValueError.

## T004 — Run log & metrik (src/naze/train/runlog.py)
RunLog: appender JSONL (sort_keys, flush per record, context manager,
reader records()). measure_peak_rss() via resource.getrusage (None
bila platform tidak mendukung). checkpoint_size(path) via os.path.getsize.

**AC:** [ ] record tertulis utuh per baris; [ ] metrik tersedia.

## T005 — batches_pos (src/naze/data/dataset.py, ADITIF)
TextWindows.batches(*, seed=None) — default perilaku lama tidak berubah
(seed instance). batches_pos(*, seed=None): window per-posisi
x = ids[i:i+T], y = ids[i+1:i+T+1]; butuh >= block_size+2 token
(ValueError). Urutan deterministik per-seed.

**AC:** [ ] test lama batches tetap lulus tanpa modifikasi; [ ] y
adalah x digeser 1; [ ] deterministik per-seed.

## T006 — loss_pos/backward_pos (src/naze/lm/transformer_lm.py, ADITIF)
TransformerLM.loss_pos(logits, targets (B,T)): cross-entropy
per-posisi, cache _p_pos/_rows_pos/_pos_shape. backward_pos():
dlogits = (p - onehot)/n -> head -> model -> dict datar. Loss lama
(loss/backward) tidak diubah.

**AC:** [ ] loss_pos == loss lama ketika target sama di semua posisi;
[ ] grads backward_pos array_equal backward lama pada kasus itu.

## T007 — TrainingRun (src/naze/train/loop.py)
TrainingRun(config, model, data, *, holdout=None). run(*,
start_epoch=0, log=None): seed per epoch = cfg.seed + e; SGD update
per batch; eval holdout tiap eval_every epoch (record "eval");
checkpoint v2 di akhir bila checkpoint_path diisi; summary dict
{model, start_epoch, epochs_run, steps, no_op, final_train_loss,
eval_loss, perplexity, peak_rss_kb, checkpoint_dir,
checkpoint_size_bytes}. start_epoch == epochs -> no-op (record +
warning). start_epoch di luar [0, epochs] -> ValueError.

**AC:** [ ] loss train menurun (basis 2, D-016); [ ] resume eksak:
3 epoch langsung == 2 epoch + load + run(start_epoch=2); [ ] no-op
aman; [ ] mlplm mode jalan.

## T008 — CI coverage (D-019)
.github/workflows/tests.yml: install pytest pytest-cov numpy; jalankan
pytest --cov=naze --cov-fail-under=80 (PYTHONPATH=src).

**AC:** [ ] CI memaksa coverage >=80% baris src/naze.

## T009 — Test M-008 (tests/test_m008_units.py, tests/test_m008_integration.py)
Unit: TrainConfig roundtrip/validasi/save-load; batches_pos shift &
determinisme; loss_pos ekuivalensi loss lama; checkpoint v2 roundtrip
+ tampered; evaluate (mock + real); RunLog; metrik D-021 (checkpoint
<=1 MB, peak RSS <2 GB bila terukur). Integration: run 3 epoch (loss
turun, 3 record eval), resume eksak, load -> generate deterministik
(temp=0, ISSUE-009), metadata lengkap (naze_version "0.0.2"), no-op
resume, mlplm smoke.

**AC:** [ ] semua test baru lulus; [ ] test lama tidak tersentuh.

## T010 — Dokumentasi pasca-CI-hijau
ROADMAP M-008 DONE + changelog; TRACEABILITY v1.5.0; REQUIREMENTS
v0.6.0; SPEC_REVIEW section baru; status task di file ini.

**AC:** [ ] dokumen konsisten dengan implementasi; [ ] commit CI hijau.
