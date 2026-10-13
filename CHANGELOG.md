# Changelog

Semua perubahan penting pada proyek ini didokumentasikan dalam berkas ini. Format mengikuti [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-13

Rilis Naze 1.0 (milestone M-010, Stage 9). Seluruh release gate DECISION-023 terverifikasi pada satu model final (2026-10-13): training loss 0.5186 (≤ 2.5); holdout perplexity 1.747 (≤ 35); NazeIO command accuracy 100% / 20 dari 20 (≥ 90%). Model final: checkpoint 3 epoch korpus hybrid (DECISION-024) + fine-tuning command dengan rehearsal via workflow `finetune-nazeio`.

### Ditambahkan
- `scripts/finetune_nazeio.py` + workflow `finetune-nazeio` (`.github/workflows/finetune-nazeio.yml`): fine-tuning command NazeIO dari checkpoint final pada `data/korpus/perintah_tingkat_1.txt` (format pemetaan intent), disusul benchmark otomatis — menutup gap antara format korpus perintah hybrid (kalimat) dan benchmark (pemetaan intent).
- Workflow CI benchmark NazeIO (`.github/workflows/benchmark.yml`): memuat checkpoint dari artifact run training, menjalankan `scripts/benchmark_nazeio.py`, dan mengunggah laporan akurasi — M010-T008.
- Workflow CI training manual (`.github/workflows/training.yml`): workflow_dispatch dengan input `epochs`/`resume`, timeout 350 menit, artifact `naze-training-<run_id>` (retensi 30 hari) — M010-T004.
- Korpus hybrid ±7 MB (DECISION-024) dengan pembagian train/val/holdout 80/10/10 di `data/corpus/splits/`.
- `docs/api.md` — API reference FINAL v1.0.0, sinkron dengan `pyproject.toml` (T001/T011).
- `docs/tutorial_train.md` — tutorial training final (M010-T012).
- Changelog ini (M010-T013).

### Diubah
- Fine-tuning command memakai `block_size=64`: jendela 32 hanya melatih posisi 0..31, sementara pasangan perintah terpanjang ±61 byte — generasi benchmark melebihi posisi terlatih.
- Fine-tuning NazeIO kini memakai korpus campuran (blok perintah diulang + rehearsal teks umum tersebar merata dari `train.txt`) — fine-tuning murni pada korpus perintah terbukti memicu catastrophic forgetting (ppl umum 2.679 pada 100 epoch).
- Training system: checkpoint disimpan setiap epoch, dan `scripts/train_final.py` memuat checkpoint secara eksplisit saat `--resume` — run CI yang terputus kini dapat dilanjutkan tanpa kehilangan bobot.
- `README.md` diperbarui: status M-001..M-010, struktur modul, instruksi training final, tautan dokumentasi (M010-T014).
- `tests/test_smoke.py`: asersi versi disinkronkan ke 1.0.0 (perbaikan failure pra-existing pada CI).

### Ringkasan Milestone
- **M-001..M-006** (DONE 2026-10-06): fondasi proyek, neural engine, backprop terstruktur + gradient check, byte tokenizer, dataset pipeline, MLP LM pertama + training minimal end-to-end.
- **M-007** (DONE 2026-10-09): Transformer (config DECISION-018: D=64, H=4, L=2, d_ff=128, T_max=128) serta CI enforcement dan coverage ≥ 80%.
- **M-008** (DONE 2026-10-09): training system — TrainConfig, TrainingRun, checkpoint v2, RunLog JSONL, evaluasi.
- **M-009** (DONE 2026-10-09): inference engine (forward-only, batching, benchmark) + export bobot float32.
- **M-010** (DONE 2026-10-13): training final 3 epoch + fine-tuning command dengan rehearsal; seluruh gate DECISION-023 terpenuhi (train loss 0.5186; holdout ppl 1.747; akurasi NazeIO 100%).

### Catatan Rilis
- Nama paket PyPI: `naze-ai` (dipilih; `naze` juga tersedia — diverifikasi 2026-10-10). Kebijakan versioning dan rilis lengkap masih OPEN DECISION-119.
- Tidak ada pretrained model atau API LLM eksternal sebagai core (DECISION-004, permanen).
