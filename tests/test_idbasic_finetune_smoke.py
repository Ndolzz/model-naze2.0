"""Smoke test rehearsal fine-tuning ID-BASIC-001 (IDB-T008, pytest).

Menguji: kategori manifest, pemuatan teks training, komposisi korpus campuran,
guard anti-penimpaan checkpoint, penolakan checkpoint sumber hilang, dan
pipeline training kecil 2 epoch pada slice dataset aktual (bukan benchmark)
dengan roundtrip checkpoint v2. Konfigurasi kecil (d_model=32, L=1) agar tes
ringan; arsitektur tetap TransformerLM sesuai produksi.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import finetune_idbasic as fid  # noqa: E402

from naze.data.dataset import TextWindows  # noqa: E402
from naze.lm.transformer_lm import TransformerLM  # noqa: E402
from naze.nn.transformer import TransformerConfig  # noqa: E402
from naze.token import ByteTokenizer  # noqa: E402
from naze.train.checkpoint import load_checkpoint_v2  # noqa: E402
from naze.train.config import TrainConfig  # noqa: E402
from naze.train.evaluate import evaluate  # noqa: E402
from naze.train.loop import TrainingRun  # noqa: E402
from naze.train.runlog import RunLog  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "idbasic"
SMOKE_MODEL_CFG = {"d_model": 32, "num_heads": 4, "num_layers": 1,
                   "d_ff": 64, "max_sequence_length": 128}


def test_manifest_categories_sorted():
    assert fid.manifest_categories() == [
        "contextual_meaning", "grammar", "punctuation",
        "sentence_structure", "vocabulary",
    ]


def test_load_idbasic_train_text_covers_all_categories():
    if not DATA_DIR.exists():
        pytest.skip("data/idbasic tidak ditemukan")
    text = fid.load_idbasic_train_text()
    for category in fid.manifest_categories():
        source = (DATA_DIR / f"{category}.txt").read_text(encoding="utf-8").strip()
        assert source in text


def test_build_mixed_text_repeats():
    text = fid.build_mixed_text("A", "B", "C", idbasic_repeats=3,
                                command_repeats=2)
    assert text.count("A") == 3
    assert text.count("B") == 2
    assert text.count("C") == 1


def test_output_checkpoint_guard_rejects_same_dir(tmp_path):
    with pytest.raises(SystemExit):
        fid.check_output_checkpoint(tmp_path, tmp_path)


def test_output_checkpoint_guard_rejects_production_dir(tmp_path):
    with pytest.raises(SystemExit):
        fid.check_output_checkpoint(tmp_path / "src", REPO_ROOT / "models" / "naze_v1")


def test_main_rejects_missing_source_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        fid.main(["--checkpoint", str(tmp_path / "tidak_ada"),
                  "--out-checkpoint", str(tmp_path / "exp")])


def test_smoke_training_two_epochs_checkpoint_roundtrip(tmp_path):
    if not DATA_DIR.exists():
        pytest.skip("data/idbasic tidak ditemukan")

    out_dir = tmp_path / "exp"
    text = fid.load_idbasic_train_text()
    small = text[:2048]  # slice deterministik; hanya data training, tanpa benchmark
    ids = ByteTokenizer().encode(small)
    assert len(ids) > 256

    cfg = TrainConfig(model="transformer", corpus_path="smoke", block_size=96,
                      batch_size=4, lr=0.08, epochs=2, seed=0, eval_every=1,
                      checkpoint_path=str(out_dir), **SMOKE_MODEL_CFG)
    model_cfg = TransformerConfig(**SMOKE_MODEL_CFG, seed=0)
    model = TransformerLM(model_cfg, seed=0)

    data = TextWindows(ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
    holdout = TextWindows(ids[:512], 32, cfg.batch_size, seed=cfg.seed)

    before = evaluate(model, data, per_pos=True, seed=cfg.seed).mean_loss
    with RunLog(tmp_path / "smoke_runlog.jsonl") as log:
        summary = TrainingRun(cfg, model, data, holdout=holdout).run(log=log)
    after = evaluate(model, data, per_pos=True, seed=cfg.seed).mean_loss

    assert summary["epochs_run"] == 2
    assert summary["steps"] > 0
    assert summary["final_train_loss"] is not None
    assert after < before  # loss menurun pada korpus mainan
    assert (out_dir / "params.npz").exists()
    assert (out_dir / "params.npz.sha256").exists()
    assert (out_dir / "meta.json").exists()

    model2 = TransformerLM(model_cfg, seed=0)
    loaded = load_checkpoint_v2(out_dir, model2)  # verifikasi checksum terpasang
    assert loaded["epoch"] == 2
    for name, param in model2.params().items():
        assert np.array_equal(param, model.params()[name])
