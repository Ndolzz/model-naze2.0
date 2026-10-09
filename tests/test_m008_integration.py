"""Test integrasi M-008 (T009): loop penuh, resume eksak, resource D-021."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from naze.data.dataset import TextWindows
from naze.lm.mlp_lm import MLPLM
from naze.lm.transformer_lm import TransformerLM, transformer_generate
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer
from naze.train.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from naze.train.config import TrainConfig
from naze.train.loop import TrainingRun
from naze.train.runlog import RunLog, checkpoint_size, measure_peak_rss

KORPUS = "naze kecil belajar bahasa. " * 20
HOLDOUT = "naze kecil belajar bahasa. " * 5


def _tcfg(**over) -> TrainConfig:
    base = dict(model="transformer", d_model=8, num_heads=2, num_layers=1,
                d_ff=16, max_sequence_length=16, block_size=8, batch_size=8,
                lr=0.08, epochs=3, eval_every=1, seed=0,
                corpus_path="korpus", holdout_path="holdout")
    base.update(over)
    return TrainConfig(**base)


def _model(cfg: TrainConfig) -> TransformerLM:
    cfgt = TransformerConfig(d_model=cfg.d_model, num_heads=cfg.num_heads,
                             num_layers=cfg.num_layers, d_ff=cfg.d_ff,
                             max_sequence_length=cfg.max_sequence_length)
    return TransformerLM(cfgt, seed=cfg.seed)


def _data(text: str, cfg: TrainConfig) -> TextWindows:
    tok = ByteTokenizer()
    return TextWindows(tok.encode(text), cfg.block_size, cfg.batch_size, seed=cfg.seed)


def test_run_loss_decreases_and_logs_eval(tmp_path) -> None:
    cfg = _tcfg()
    model = _model(cfg)
    log_path = tmp_path / "run.jsonl"
    with RunLog(log_path) as log:
        summary = TrainingRun(cfg, model, _data(KORPUS, cfg),
                              holdout=_data(HOLDOUT, cfg)).run(log=log)
    evals = [r for r in RunLog(log_path).records() if r["type"] == "eval"]
    assert len(evals) == 3  # eval_every=1 selama 3 epoch
    assert evals[-1]["train_loss"] < evals[0]["train_loss"]  # basis 2 (D-016)
    assert summary["model"] == "transformer"
    assert summary["epochs_run"] == 3 and summary["no_op"] is False
    assert summary["steps"] > 0
    assert summary["final_train_loss"] < evals[0]["train_loss"]
    assert summary["eval_loss"] is not None
    assert summary["perplexity"] > 1.0
    assert summary["checkpoint_dir"] is None  # checkpoint_path tidak diisi


def test_resume_exact(tmp_path) -> None:
    cfg = _tcfg()
    direct = _model(cfg)
    TrainingRun(cfg, direct, _data(KORPUS, cfg),
               holdout=_data(HOLDOUT, cfg)).run()
    ckpt = tmp_path / "ckpt"
    cfg2 = replace(cfg, epochs=2, checkpoint_path=str(ckpt))
    m2 = _model(cfg2)
    TrainingRun(cfg2, m2, _data(KORPUS, cfg2), holdout=_data(HOLDOUT, cfg2)).run()
    m3 = _model(cfg)
    assert load_checkpoint_v2(ckpt, m3)["epoch"] == 2
    cfg3 = replace(cfg, epochs=3)
    TrainingRun(cfg3, m3, _data(KORPUS, cfg3),
                holdout=_data(HOLDOUT, cfg3)).run(start_epoch=2)
    for name, p in direct.params().items():
        assert np.array_equal(p, m3.params()[name])  # resume eksak


def test_generate_deterministic_after_load(tmp_path) -> None:
    cfg = replace(_tcfg(epochs=1), checkpoint_path=str(tmp_path / "ckpt"))
    model = _model(cfg)
    TrainingRun(cfg, model, _data(KORPUS, cfg)).run()
    m2 = _model(cfg)
    load_checkpoint_v2(tmp_path / "ckpt", m2)
    tok = ByteTokenizer()
    out1 = transformer_generate(m2, tok.encode("naze kecil"), 4, temperature=0.0)
    out2 = transformer_generate(m2, tok.encode("naze kecil"), 4, temperature=0.0)
    assert out1 == out2 and len(out1) == 4  # ISSUE-009: greedy deterministik


def test_resource_d021_checkpoint_and_rss(tmp_path) -> None:
    full = TrainConfig()  # default D-018: D=64, H=4, L=2, d_ff=128, T_max=128
    assert (full.d_model, full.num_heads, full.num_layers, full.d_ff) == (64, 4, 2, 128)
    cfgt = TransformerConfig(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                             max_sequence_length=128)
    model = TransformerLM(cfgt, seed=0)
    ckpt = tmp_path / "ckpt"
    save_checkpoint_v2(ckpt, model, step=0, epoch=0, config=full)
    assert checkpoint_size(ckpt / "params.npz") <= 1_000_000  # D-018/D-021
    rss = measure_peak_rss()
    assert rss is None or rss < 2 * 1024 * 1024  # peak < 2 GB (dalam KB)


def test_meta_lengkap(tmp_path) -> None:
    cfg = _tcfg(epochs=1)
    model = _model(cfg)
    meta = save_checkpoint_v2(tmp_path, model, step=5, epoch=1, config=cfg)
    for key in ("step", "epoch", "created", "naze_version", "size_bytes",
                "checksum_sha256", "config", "config_hash"):
        assert key in meta
    assert meta["naze_version"] == "0.0.2"
    loaded = load_checkpoint_v2(tmp_path, _model(cfg))
    assert loaded["meta"]["config_hash"] == cfg.config_hash()


def test_noop_resume() -> None:
    cfg = _tcfg(epochs=2)
    model = _model(cfg)
    TrainingRun(cfg, model, _data(KORPUS, cfg)).run()
    before = {k: p.copy() for k, p in model.params().items()}
    s2 = TrainingRun(cfg, model, _data(KORPUS, cfg)).run(start_epoch=2)
    assert s2["no_op"] is True and s2["epochs_run"] == 0
    for k, p in model.params().items():
        assert np.array_equal(p, before[k])  # no-op tidak menyentuh params


def test_invalid_start_epoch() -> None:
    cfg = _tcfg(epochs=2)
    with pytest.raises(ValueError):
        TrainingRun(cfg, _model(cfg), _data(KORPUS, cfg)).run(start_epoch=3)


def test_mlplm_smoke() -> None:
    cfg = _tcfg(model="mlplm", block_size=6, epochs=2)
    tok = ByteTokenizer()
    model = MLPLM(256, block_size=6, d_embed=8, d_hidden=16, seed=0)
    data = TextWindows(tok.encode(KORPUS), 6, cfg.batch_size, seed=cfg.seed)
    summary = TrainingRun(cfg, model, data).run()
    assert summary["model"] == "mlplm" and summary["epochs_run"] == 2
    assert summary["final_train_loss"] is not None
    assert summary["final_train_loss"] > 0.0
