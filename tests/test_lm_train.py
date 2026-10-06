"""Integration test: MLP LM end-to-end — train, generate, checkpoint (M-006).

Acceptance criteria Stage 5: training end-to-end berhasil, loss menurun,
generation berjalan; REQ-007: checkpoint save/load deterministik.
"""

import tempfile
from pathlib import Path

import numpy as np

from naze.data import TextWindows
from naze.lm import MLPLM, generate
from naze.token import ByteTokenizer
from naze.train import SGDTrainer, load_checkpoint, save_checkpoint

V = 4  # toy vocab: {a,b,c,d}


def _toy_model() -> MLPLM:
    return MLPLM(vocab_size=V, block_size=4, d_embed=8, d_hidden=32, seed=0)


def _toy_batches(seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    ids = (list(range(V)) * 50)  # pola periodik abcdbcd...
    return list(TextWindows(ids, block_size=4, batch_size=16, seed=seed).batches())


def test_lm_gradient_check() -> None:
    from naze.core.gradcheck import assert_close, numeric_grad

    model = MLPLM(vocab_size=V, block_size=4, d_embed=4, d_hidden=6, seed=0)
    x = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    logits = model.forward(x)
    model.loss(logits, y)
    grads = model.backward()

    def f() -> float:
        return model.loss(model.forward(x), y)

    for name in ["E", "fc1.W", "fc2.W"]:
        assert_close(grads[name], numeric_grad(f, model.params()[name]), tol=1e-4)


def test_training_loss_decreases() -> None:
    model = _toy_model()
    trainer = SGDTrainer(model, lr=0.5)
    batches = _toy_batches()
    first = last = 0.0
    for epoch in range(20):
        for x, y in batches:
            last = trainer.train_step(x, y)
        if epoch == 0:
            first = last
    assert last < first * 0.5, f"loss tidak turun cukup: {first:.3f} -> {last:.3f}"


def test_generation_valid_and_deterministic() -> None:
    model = _toy_model()
    out1 = generate(model, [0, 1, 2, 3], 8, temperature=0.5, seed=42)
    out2 = generate(model, [0, 1, 2, 3], 8, temperature=0.5, seed=42)
    assert len(out1) == 8
    assert all(0 <= t < V for t in out1)
    assert out1 == out2  # deterministik per-seed


def test_greedy_generation() -> None:
    model = _toy_model()
    out = generate(model, [0, 1, 2, 3], 5, temperature=0.0)
    assert len(out) == 5


def test_checkpoint_roundtrip() -> None:
    model = _toy_model()
    trainer = SGDTrainer(model, lr=0.1)
    for x, y in _toy_batches():
        trainer.train_step(x, y)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.npz"
        save_checkpoint(path, model, step=trainer.step_count)

        model2 = _toy_model()
        step = load_checkpoint(path, model2)
        assert step == trainer.step_count

        ctx = np.array([[0, 1, 2, 3]], dtype=np.int64)
        assert np.allclose(model.forward(ctx), model2.forward(ctx))


def test_byte_lm_end_to_end() -> None:
    """Sanity end-to-end: encode -> dataset -> train beberapa step -> generate -> decode."""
    tok = ByteTokenizer()
    text = "naze belajar dari nol " * 20
    ids = tok.encode(text)
    model = MLPLM(vocab_size=tok.vocab_size, block_size=8, d_embed=16, d_hidden=32, seed=0)
    ds = TextWindows(ids, block_size=8, batch_size=32, seed=0)
    trainer = SGDTrainer(model, lr=0.3)
    losses = [trainer.train_step(x, y) for x, y in ds.batches()]
    assert losses[-1] < losses[0]
    out = generate(model, ids[:8], 4, temperature=0.5, seed=0)
    assert isinstance(tok.decode(list(ids[:8]) + out), str)
