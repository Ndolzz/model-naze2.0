# Naze 2.0

Model AI yang dibangun **dari nol** — tanpa pretrained model, tanpa API LLM eksternal sebagai core. Seluruh komponen (neural engine, tokenizer, dataset pipeline, Transformer, training system, inference engine, export) dibuat sendiri dengan Python + NumPy.

## Status: Naze 1.0 — Training Final (M-010 ACTIVE)

M-001..M-009 DONE. Pipeline end-to-end lengkap: byte-level tokenizer → dataset → Transformer (config DECISION-018: D=64, H=4, L=2, d_ff=128, T_max=128) → training system (checkpoint v2 + run log) → inference engine → export float32. Korpus hybrid ±7 MB (DECISION-024) dengan pembagian train/val/holdout 80/10/10 terintegrasi di `data/corpus/splits/`.

M-010 (training final) berjalan. Rilis Naze 1.0 ditentukan oleh release gate DECISION-023: training loss ≤ 2.5, validation perplexity ≤ 35, NazeIO command accuracy ≥ 90%.

## Prinsip
- **Spec-Driven Development (SDD):** spesifikasi = source of truth (`docs/spec/`).
- Keputusan belum jelas = OPEN DECISION (`docs/decisions/DECISION_LOG.md`).
- No overengineering.

## Struktur

```
docs/                 spesifikasi SDD (spec, architecture, decisions, tasks)
data/corpus/splits/   korpus hybrid D-024 (train/val/holdout 80/10/10)
scripts/              train_final.py, benchmark_nazeio.py
src/naze/
├── core/             numerical core + gradient check
├── nn/               layers, activations, komponen Transformer
├── token/            byte-level tokenizer (vocab 256)
├── data/             dataset pipeline (sliding window)
├── lm/               MLPLM, TransformerLM, cross_entropy, generate
├── train/            TrainConfig, TrainingRun, checkpoint v2, evaluasi, run log
├── inference/        inference engine (forward-only, batching, benchmark)
└── export.py         export bobot float32
tests/                test suite (pytest)
```

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check .
```

CI: pytest wajib hijau untuk `main` (DECISION-020); coverage baris ≥ 80% pada `src/naze` (DECISION-019).

## Training Final (M-010)

Lokal, dari root repository:

```bash
python scripts/train_final.py --epochs 10
```

Output: checkpoint `models/naze_v1/`, run log `runs/m010_train.jsonl`, ringkasan `runs/m010_summary.json`. Resume: tambahkan `--resume <start_epoch>`.

Alternatif: workflow GitHub Actions "training" (tab Actions → Run workflow, input `epochs` dan `resume`). Panduan lengkap: `docs/tutorial_train.md`.

## Contoh: Train & Generate (MLP, M-006)

```python
from naze.data import TextWindows
from naze.lm import MLPLM, generate
from naze.token import ByteTokenizer
from naze.train import SGDTrainer

tok = ByteTokenizer()
text = open("corpus.txt", encoding="utf-8").read()
ids = tok.encode(text)

model = MLPLM(vocab_size=tok.vocab_size, block_size=16, d_embed=32, d_hidden=128, seed=0)
trainer = SGDTrainer(model, lr=0.3)
for epoch in range(10):
    for x, y in TextWindows(ids, block_size=16, batch_size=64, seed=epoch).batches():
        trainer.train_step(x, y)

new_ids = generate(model, ids[:16], 50, temperature=0.8, seed=0)
print(tok.decode(ids[:16] + new_ids))
```

## Contoh: Inference Transformer (M-009)

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
prompt = tok.encode("Halo")
generated = engine.generate(prompt, max_new=20, temperature=0.8, seed=0)
print(tok.decode(prompt + generated))
```

## Milestones
- M-001 Project Foundation ✅ | M-002 Neural Engine ✅ | M-003 Autodiff ✅
- M-004 Tokenizer ✅ | M-005 Dataset ✅ | M-006 First LM + training minimal ✅ (2026-10-06)
- M-007 Transformer ✅ | M-008 Training System ✅ | M-009 Inference & Export ✅ (2026-10-09)
- M-010 Training Final — ACTIVE (2026-10-10)
- Berikutnya: evaluasi gate DECISION-023 → rilis Naze 1.0.

## Dokumentasi
- `docs/api.md` — API reference (FINAL, v1.0.0)
- `docs/tutorial_train.md` — tutorial training final (M-010)
- `CHANGELOG.md` — riwayat perubahan
- `docs/decisions/DECISION_LOG.md` — log keputusan (governance)
- `docs/tasks/ROADMAP.md` — roadmap milestone
