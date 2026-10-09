"""Test unit M-008 (T009): config, batches_pos, loss_pos, ckpt v2, evaluate, runlog."""

from __future__ import annotations

import math

import naze
import numpy as np
import pytest

from naze.data.dataset import TextWindows
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.train.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from naze.train.config import TrainConfig
from naze.train.evaluate import evaluate
from naze.train.runlog import RunLog, checkpoint_size, measure_peak_rss


def _tcfg(**over) -> TrainConfig:
    base = dict(model="transformer", d_model=4, num_heads=2, num_layers=1,
                d_ff=8, max_sequence_length=8, block_size=6, batch_size=8,
                epochs=1, corpus_path="korpus")
    base.update(over)
    return TrainConfig(**base)


def _small_model(seed: int = 0) -> TransformerLM:
    cfgt = TransformerConfig(d_model=4, num_heads=2, num_layers=1, d_ff=8,
                             max_sequence_length=8)
    return TransformerLM(cfgt, seed=seed)


# --- T001: TrainConfig ---

def test_trainconfig_default_valid() -> None:
    cfg = TrainConfig()  # default D-018
    assert cfg.model == "transformer"
    assert (cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.d_ff) == (64, 4, 2, 128)
    assert cfg.max_sequence_length == 128
    assert cfg.lr == 0.08 and cfg.epochs == 3 and cfg.eval_every == 1


@pytest.mark.parametrize("over", [
    dict(model="adam"),
    dict(lr=0.0),
    dict(epochs=0),
    dict(eval_every=0),
    dict(seed=-1),
    dict(block_size=200),  # > max_sequence_length default 128
    dict(num_heads=5),     # 64 % 5 != 0
    dict(model="mlplm", d_embed=0),
])
def test_trainconfig_invalid(over) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**over)


def test_trainconfig_json_roundtrip_stable_hash() -> None:
    cfg = _tcfg()
    again = TrainConfig.from_json(cfg.to_json())
    assert again == cfg
    assert again.to_json() == cfg.to_json()
    assert cfg.config_hash() == again.config_hash()
    assert _tcfg(epochs=2).config_hash() != cfg.config_hash()


def test_trainconfig_save_load(tmp_path) -> None:
    cfg = _tcfg()
    p = tmp_path / "cfg.json"
    cfg.save(p)
    assert TrainConfig.load(p) == cfg


# --- T005: batches_pos ---

def test_batches_pos_shift_and_deterministic() -> None:
    data = TextWindows(list(range(50)), block_size=6, batch_size=100, seed=0)
    x, y = next(iter(data.batches_pos()))
    assert x.shape == (43, 6) and y.shape == (43, 6)
    for row in range(x.shape[0]):  # y = window x digeser 1 pada korpus asli
        i = int(x[row, 0])
        assert np.array_equal(y[row], np.arange(i + 1, i + 7))
    x2, y2 = next(iter(data.batches_pos(seed=0)))
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    x3, y3 = next(iter(data.batches()))  # batches lama tetap kompatibel
    assert y3.ndim == 1 and x3.shape[1] == 6


def test_batches_pos_too_short() -> None:
    data = TextWindows(list(range(7)), block_size=6, batch_size=4)  # 7 = T + 1
    with pytest.raises(ValueError):
        next(iter(data.batches_pos()))


# --- T006: loss_pos / backward_pos ---

def test_loss_pos_matches_legacy_when_repeated_targets() -> None:
    m1 = _small_model(0)
    m2 = _small_model(0)
    x = np.zeros((3, 5), dtype=np.int64)
    x[:, 2] = 7
    t = np.array([10, 20, 30], dtype=np.int64)
    l1 = m1.loss(m1.forward(x), t)
    tpos = np.repeat(t[:, None], 5, axis=1)
    l2 = m2.loss_pos(m2.forward(x), tpos)
    assert l1 == pytest.approx(l2)
    g1 = m1.backward()
    g2 = m2.backward_pos()
    assert set(g1) == set(g2)
    for name in g1:
        assert np.array_equal(g1[name], g2[name])


def test_loss_pos_invalid_shapes() -> None:
    m = _small_model(0)
    logits = np.zeros((2, 3, 256))
    with pytest.raises(ValueError):
        m.loss_pos(logits, np.zeros((2, 4), dtype=np.int64))  # shape beda
    with pytest.raises(ValueError):
        m.loss_pos(logits, np.zeros((2, 3), dtype=np.float64))  # bukan int
    with pytest.raises(ValueError):
        m.loss_pos(logits, np.full((2, 3), 999, dtype=np.int64))  # out of range
    with pytest.raises(RuntimeError):
        m.backward_pos()  # belum loss_pos


# --- T002: checkpoint v2 ---

def test_checkpoint_v2_roundtrip_and_meta(tmp_path) -> None:
    m1 = _small_model(0)
    cfg = _tcfg()
    meta = save_checkpoint_v2(tmp_path, m1, step=12, epoch=2, config=cfg)
    assert meta["step"] == 12 and meta["epoch"] == 2
    assert meta["naze_version"] == naze.__version__  # rilis 1.0.0 (M-010)
    assert meta["config_hash"] == cfg.config_hash()
    assert meta["size_bytes"] == checkpoint_size(tmp_path / "params.npz")
    m2 = _small_model(99)  # init beda — load harus menimpa seluruhnya
    loaded = load_checkpoint_v2(tmp_path, m2)
    assert loaded["step"] == 12 and loaded["epoch"] == 2
    assert loaded["meta"]["checksum_sha256"] == meta["checksum_sha256"]
    for name, p in m1.params().items():
        assert np.array_equal(p, m2.params()[name])


def test_checkpoint_v2_tampered_checksum(tmp_path) -> None:
    m = _small_model(0)
    save_checkpoint_v2(tmp_path, m, step=1, epoch=1)
    (tmp_path / "params.npz.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_checkpoint_v2(tmp_path, _small_model(1))
    assert load_checkpoint_v2(tmp_path, _small_model(1), verify=False)["step"] == 1


# --- T003: evaluate ---

class _PerfectPos:
    def forward(self, ids: np.ndarray) -> np.ndarray:
        return np.zeros((ids.shape[0], ids.shape[1], 256))

    def loss_pos(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return 0.0

    def loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return 0.0


class _EmptyData:
    def batches(self, *, seed=None):
        return iter([])

    def batches_pos(self, *, seed=None):
        return iter([])


def test_evaluate_perfect_model() -> None:
    data = TextWindows(list(range(50)), block_size=6, batch_size=100, seed=0)
    res = evaluate(_PerfectPos(), data, per_pos=True)
    assert res.mean_loss == 0.0
    assert res.perplexity == 1.0
    assert res.n_batches == 1
    assert res.n_tokens == 43 * 6


def test_evaluate_real_transformer_ppl() -> None:
    data = TextWindows(list(range(50)), block_size=6, batch_size=10, seed=0)
    res = evaluate(_small_model(0), data, per_pos=True, seed=0)
    assert math.isclose(res.perplexity, math.exp(res.mean_loss))
    assert res.n_tokens > 0


def test_evaluate_empty_dataset() -> None:
    with pytest.raises(ValueError):
        evaluate(_PerfectPos(), _EmptyData())


# --- T004: runlog & metrik ---

def test_runlog_and_metrics(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    with RunLog(p) as log:
        log.write({"b": 1, "a": "x"})
        log.write({"type": "eval", "epoch": 0})
    assert RunLog(p).records() == [{"a": "x", "b": 1}, {"epoch": 0, "type": "eval"}]
    rss = measure_peak_rss()
    assert rss is None or rss > 0
    with pytest.raises(RuntimeError):
        RunLog(p).write({"tanpa": "context manager"})
