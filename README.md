# Naze 2.0

Model AI yang dibangun **dari nol** — tanpa pretrained model, tanpa API LLM eksternal sebagai core. Seluruh komponen (neural engine, autodiff, tokenizer, dataset pipeline, training, inference) dibuat sendiri dengan Python + NumPy.

## Status: End-to-End Pipeline Berfungsi (M-001..M-006)

Encode teks -> batch -> train MLP LM (SGD) -> checkpoint -> generate teks. Deterministik per-seed, teruji (gradient check + integration test).

## Prinsip
- **Spec-Driven Development (SDD):** spesifikasi = source of truth (`docs/spec/`).
- Keputusan belum jelas = OPEN DECISION (`docs/decisions/DECISION_LOG.md`).
- No overengineering.

## Struktur

```
docs/             spesifikasi SDD (spec, architecture, decisions, tasks)
src/naze/
├── core/         numerical core + gradient check
├── nn/           layers & activations (forward + backward)
├── token/        byte-level tokenizer
├── data/         dataset pipeline (sliding window)
├── lm/           MLP language model + generate
└── train/        SGD trainer + checkpoint
tests/            test suite (pytest)
```

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check .
```

## Contoh: Train & Generate

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

## Milestones
- M-001 Project Foundation ✅ | M-002 Neural Engine ✅ | M-003 Autodiff ✅
- M-004 Tokenizer ✅ | M-005 Dataset ✅ | M-006 First LM + training minimal ✅
- Berikutnya: M-007 Transformer (Stage 6) — menunggu persetujuan owner.
