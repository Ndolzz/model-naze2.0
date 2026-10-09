"""Training final Naze 1.0 (M010-T005/T006, DECISION-023).

Menjalankan training Transformer (config D-018) pada korpus hybrid
(DECISION-024) dengan pipeline M-008: TrainingRun + checkpoint v2 +
RunLog JSONL. Holdout TIDAK dipakai untuk training/tuning — hanya
dievaluasi sekali di akhir (metrik DECISION-023).

Jalankan dari root repo:
    python scripts/train_final.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naze.data.dataset import TextWindows
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer
from naze.train.config import TrainConfig
from naze.train.evaluate import evaluate
from naze.train.loop import TrainingRun
from naze.train.runlog import RunLog, checkpoint_size, measure_peak_rss

SPLITS = ROOT / "data" / "corpus" / "splits"


def load_ids(path: Path) -> list[int]:
    return ByteTokenizer().encode(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Training final Naze 1.0 (M-010)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="models/naze_v1")
    parser.add_argument("--run-log", type=str, default="runs/m010_train.jsonl")
    parser.add_argument("--summary", type=str, default="runs/m010_summary.json")
    parser.add_argument("--resume", type=int, default=0, help="start_epoch (resume penuh)")
    args = parser.parse_args()

    train_ids = load_ids(SPLITS / "train.txt")
    val_ids = load_ids(SPLITS / "val.txt")
    holdout_ids = load_ids(SPLITS / "holdout.txt")
    print(f"[data] train={len(train_ids)} val={len(val_ids)} holdout={len(holdout_ids)} token")

    cfg = TrainConfig(
        model="transformer",
        corpus_path=str(SPLITS / "train.txt"),
        holdout_path=str(SPLITS / "val.txt"),
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        eval_every=1,
        checkpoint_path=str(ROOT / args.checkpoint_dir),
    )
    model_cfg = TransformerConfig(
        d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_sequence_length=cfg.max_sequence_length, seed=cfg.seed,
    )
    model = TransformerLM(model_cfg, seed=cfg.seed)

    train_data = TextWindows(train_ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
    val_data = TextWindows(val_ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)

    log_path = ROOT / args.run_log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with RunLog(log_path) as log:
        summary = TrainingRun(cfg, model, train_data, holdout=val_data).run(
            start_epoch=args.resume, log=log
        )

    holdout_data = TextWindows(holdout_ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
    res = evaluate(model, holdout_data, per_pos=True, seed=cfg.seed)
    summary["holdout_loss"] = res.mean_loss
    summary["holdout_perplexity"] = res.perplexity
    summary["holdout_n_tokens"] = res.n_tokens
    summary["train_config"] = cfg.to_dict()
    summary["config_hash"] = cfg.config_hash()

    ckpt_npz = ROOT / args.checkpoint_dir / "params.npz"
    if ckpt_npz.exists():
        size = checkpoint_size(ckpt_npz)
        summary["checkpoint_size_bytes"] = size
        if size > 1_000_000:
            print(f"[WARN] checkpoint {size} byte > 1 MB (DECISION-018)")
    if measure_peak_rss() is not None:
        summary["peak_rss_kb_final"] = measure_peak_rss()

    out = ROOT / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
