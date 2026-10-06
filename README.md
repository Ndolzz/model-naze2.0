# Naze 2.0

Model AI yang dibangun **dari nol** — tanpa pretrained model, tanpa API LLM eksternal sebagai core. Seluruh komponen inti (neural engine, autodiff, tokenizer, dataset pipeline, training, inference) dibuat sendiri dengan Python + NumPy.

## Prinsip

- **Spec-Driven Development (SDD):** spesifikasi adalah source of truth. Lihat `docs/spec/`.
- Keputusan arsitektur yang belum jelas dicatat sebagai OPEN DECISION di `docs/decisions/DECISION_LOG.md`.
- No overengineering: hanya fitur yang dibutuhkan milestone aktif.

## Struktur Repo

```
docs/
├── spec/          PROJECT_SPEC.md, REQUIREMENTS.md
├── architecture/  ARCHITECTURE.md (Stage 0–10)
├── decisions/     DECISION_LOG.md
└── tasks/         ROADMAP.md, TASKS.md
src/naze/
├── core/          numerical core (seeded RNG, dtype konvensi, validasi array)
└── nn/            layers (Linear/Sequential), activations — forward-only (Stage 1)
tests/             test suite (pytest)
```

## Development

```bash
pip install -e ".[dev]"
pytest          # menjalankan test suite
ruff check .     # lint
ruff format .    # format
```

## Contoh Penggunaan (Stage 1 — forward pass)

```python
import numpy as np
from naze.nn import Linear, Sequential, relu

model = Sequential([
    Linear(4, 16, seed=1),
    relu,
    Linear(16, 3, seed=2),
])
y = model(np.zeros((8, 4)))  # -> (8, 3)
```

> Catatan: `relu` berfungsi langsung sebagai callable; untuk container layer gunakan `Activation(relu, "relu")`.

## Status

- **MILESTONE-001 — Project Foundation:** DONE
- **MILESTONE-002 — Neural Network Engine (Stage 1):** IN REVIEW — forward pass + numerical core teruji; backprop menyusul di Stage 2 (menunggu OPEN DECISION-105).
