# Tutorial: Training Final Naze 1.0 (M-010)

Panduan langkah demi langkah menjalankan training final Naze 1.0: persiapan, training (lokal atau CI), resume, evaluasi release gate (DECISION-023), generate teks, dan export bobot model.

## 1. Prasyarat

Python 3.11 dengan dependensi terpasang:

```bash
pip install -e ".[dev]"
```

Korpus hybrid (DECISION-024) tersedia di `data/corpus/splits/`: `train.txt` (80%), `val.txt` (10%), `holdout.txt` (10%). Holdout tidak dipakai untuk training atau tuning — hanya dievaluasi sekali di akhir.

## 2. Menjalankan training

Dari root repository:

```bash
python scripts/train_final.py --epochs 10
```

Argumen (nilai default):

| Argumen | Default | Keterangan |
|---|---|---|
| `--epochs` | 10 | jumlah epoch training |
| `--block-size` | 32 | ukuran context window |
| `--batch-size` | 8 | ukuran batch |
| `--lr` | 0.08 | learning rate SGD (DECISION-013) |
| `--seed` | 0 | seed deterministik (DECISION-007) |
| `--checkpoint-dir` | models/naze_v1 | direktori checkpoint |
| `--run-log` | runs/m010_train.jsonl | log per epoch (JSONL) |
| `--summary` | runs/m010_summary.json | ringkasan akhir |
| `--resume` | 0 | start_epoch (melanjutkan dari checkpoint) |

Output:
- `models/naze_v1/` — checkpoint v2 (params + metadata + checksum).
- `runs/m010_train.jsonl` — run log per epoch (loss train/val, perplexity, metrik memori).
- `runs/m010_summary.json` — ringkasan akhir: train loss, val perplexity, holdout loss/perplexity, ukuran checkpoint, peak RSS, config hash.

## 3. Training via GitHub Actions

1. Buka tab Actions, pilih workflow "training", lalu klik Run workflow.
2. Isi input:
   - `epochs` (default 10): jumlah epoch.
   - `resume` (default 0): start_epoch untuk resume penuh.
   - `resume_run_id` (default kosong): ID run sebelumnya; jika diisi, artifact run tersebut diunduh lebih dahulu sehingga training melanjutkan dari checkpoint-nya.
3. Setelah selesai, unduh artifact `naze-training-<run_id>` yang berisi `models/naze_v1/` dan `runs/` (retensi 30 hari).

Catatan desain:
- Langkah training memiliki timeout step 330 menit, di bawah timeout job 350 menit. Jika langkah training melewati batas step, step tersebut gagal, tetapi langkah unggah artifact tetap berjalan (`if: always()`) sehingga checkpoint dan run log tetap terselamatkan.
- Untuk training yang tidak muat dalam satu run, gunakan strategi multi-run: jalankan run pertama dengan epoch per batch (misal 5), lalu jalankan run berikutnya dengan `resume_run_id` = ID run sebelumnya dan `resume` = epoch terakhir yang tercapai ditambah satu (cek `runs/m010_train.jsonl` pada artifact run sebelumnya). Hyperparameter (block size, batch size, lr, seed) wajib identik antar run.

## 4. Resume

```bash
python scripts/train_final.py --epochs 10 --resume 5
```

`--resume N` menetapkan start_epoch dan melanjutkan dari checkpoint di `models/naze_v1/`. Jika `start_epoch == epochs`, training menjadi no-op dan peringatan ditampilkan.

## 5. Evaluasi release gate (DECISION-023)

Baca `runs/m010_summary.json`. Rilis Naze 1.0 memenuhi tiga gate wajib:

| Metrik | Gate | Sumber |
|---|---|---|
| Training loss | ≤ 2.5 (rata-rata cross-entropy per token) | `runs/m010_summary.json` |
| Validation perplexity | ≤ 35 (exp(val loss)) | `runs/m010_summary.json` |
| NazeIO command accuracy | ≥ 90% | `python scripts/benchmark_nazeio.py` dengan `data/benchmark/commands.json` |

Angka holdout hanya dievaluasi sekali di akhir script dan tidak boleh dipakai untuk tuning (DECISION-024).

## 6. Generate teks

```python
import sys
sys.path.insert(0, "src")

from naze.inference import InferenceEngine
from naze.lm import TransformerLM, TransformerConfig
from naze.token import ByteTokenizer

tok = ByteTokenizer()
config = TransformerConfig(
    d_model=64, num_heads=4, num_layers=2,
    d_ff=128, max_sequence_length=128, vocab_size=256, seed=0,
)
model = TransformerLM(config)  # latih model atau muat bobot terlatih
engine = InferenceEngine(model)
prompt = tok.encode("Halo, saya Naze.")
generated = engine.generate(prompt, max_new=50, temperature=0.8, seed=0)
print(tok.decode(prompt + generated))
```

Detail antarmuka: `docs/api.md`.

## 7. Export bobot (untuk integrasi)

```python
from naze.export import export_model_weights

export_model_weights(model, "naze_v1.npz", version="1.0.0", float32=True)
```

Menghasilkan `params.npz` float32 (±0.43 MB untuk config D-018) beserta metadata JSON dengan checksum. Inference on-device (nazeio) memakai presisi float32 (DECISION-023).

## 8. Troubleshooting

- **RAM tinggi atau OOM.** Config produksi D-018 dirancang di bawah 2 GB RAM puncak (DECISION-021); turunkan `--batch-size` bila perlu.
- **Peringatan checkpoint > 1 MB.** Batas formal DECISION-018; laporkan bila muncul.
- **Resume tidak berpengaruh.** Pastikan checkpoint v2 ada di `models/naze_v1/` dan `--resume` bernilai lebih kecil dari `--epochs`; `start_epoch == epochs` adalah no-op.
- **Hasil tidak reproduksi.** Seluruh randomness mengikuti seed (DECISION-007); gunakan `--seed` dan hyperparameter yang sama.
