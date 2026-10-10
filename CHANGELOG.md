# Changelog

Semua perubahan penting pada proyek ini didokumentasikan dalam berkas ini. Format mengikuti [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-10

Persiapan rilis Naze 1.0 (milestone M-010, Stage 9). Status: training final dalam proses — release gate DECISION-023 (training loss ≤ 2.5; validation perplexity ≤ 35; NazeIO command accuracy ≥ 90%) belum diverifikasi. Versi 1.0.0 efektif dirilis setelah seluruh gate terpenuhi.

### Ditambahkan
- Workflow CI training manual (`.github/workflows/training.yml`): workflow_dispatch dengan input `epochs`/`resume`, timeout 350 menit, artifact `naze-training-<run_id>` (retensi 30 hari) — M010-T004.
- Korpus hybrid ±7 MB (DECISION-024) dengan pembagian train/val/holdout 80/10/10 di `data/corpus/splits/`.
- `docs/api.md` — API reference FINAL v1.0.0, sinkron dengan `pyproject.toml` (T001/T011).
- `docs/tutorial_train.md` — tutorial training final (M010-T012).
- Changelog ini (M010-T013).

### Diubah
- Training system: checkpoint disimpan setiap epoch, dan `scripts/train_final.py` memuat checkpoint secara eksplisit saat `--resume` — run CI yang terputus kini dapat dilanjutkan tanpa kehilangan bobot.
- `README.md` diperbarui: status M-001..M-010, struktur modul, instruksi training final, tautan dokumentasi (M010-T014).
- `tests/test_smoke.py`: asersi versi disinkronkan ke 1.0.0 (perbaikan failure pra-existing pada CI).

### Ringkasan Milestone
- **M-001..M-006** (DONE 2026-10-06): fondasi proyek, neural engine, backprop terstruktur + gradient check, byte tokenizer, dataset pipeline, MLP LM pertama + training minimal end-to-end.
- **M-007** (DONE 2026-10-09): Transformer (config DECISION-018: D=64, H=4, L=2, d_ff=128, T_max=128) serta CI enforcement dan coverage ≥ 80%.
- **M-008** (DONE 2026-10-09): training system — TrainConfig, TrainingRun, checkpoint v2, RunLog JSONL, evaluasi.
- **M-009** (DONE 2026-10-09): inference engine (forward-only, batching, benchmark) + export bobot float32.
- **M-010** (ACTIVE 2026-10-10): training final + evaluasi gate DECISION-023.

### Catatan Rilis
- Nama paket PyPI: `naze-ai` (dipilih; `naze` juga tersedia — diverifikasi 2026-10-10). Kebijakan versioning dan rilis lengkap masih OPEN DECISION-119.
- Tidak ada pretrained model atau API LLM eksternal sebagai core (DECISION-004, permanen).
