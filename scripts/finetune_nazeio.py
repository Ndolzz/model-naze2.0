"""Fine-tuning command NazeIO (M-010, melengkapi DECISION-024).

Korpus hybrid menyertakan perintah asisten dalam format kalimat, tetapi
benchmark NazeIO (DECISION-023) memakai format pemetaan
"perintah: {teks}\nintent: {intent}" dari data/korpus/perintah_tingkat_1.txt
(berkas tersebut memang ditujukan untuk fine-tuning Naze Engine).
Script ini melanjutkan training dari checkpoint final pada korpus tersebut,
lalu checkpoint fine-tuned disimpan kembali ke models/naze_v1.

Jalankan dari root repo:
    python scripts/finetune_nazeio.py --checkpoint models/naze_v1 --epochs 100
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
from naze.train.checkpoint import load_checkpoint_v2
from naze.train.config import TrainConfig
from naze.train.loop import TrainingRun
from naze.train.runlog import RunLog

PERINTAH = ROOT / "data" / "korpus" / "perintah_tingkat_1.txt"


def load_perintah_text(path: Path) -> str:
    # Pertahankan struktur baris asli: "perintah: X\nintent: Y\n\nperintah: ...".
    # (Jangan gabungkan semua baris dengan \n\n - konteks kondisional
    # "perintah -> intent" harus identik dengan format benchmark.)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning command NazeIO (M-010)")
    parser.add_argument("--checkpoint", type=str, default="models/naze_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--report", type=str, default="runs/finetune_nazeio_summary.json")
    args = parser.parse_args()

    ckpt_dir = ROOT / args.checkpoint
    if not (ckpt_dir / "params.npz").exists():
        raise SystemExit(f"checkpoint tidak ditemukan: {ckpt_dir / 'params.npz'}")
    text = load_perintah_text(PERINTAH)
    ids = ByteTokenizer().encode(text)
    print(f"[finetune] korpus perintah: {len(ids)} token, epochs={args.epochs}")

    cfg = TrainConfig(
        model="transformer",
        corpus_path=str(PERINTAH),
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_every=args.eval_every,
        seed=args.seed,
        checkpoint_path=str(ckpt_dir),
    )
    model_cfg = TransformerConfig(
        d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_sequence_length=cfg.max_sequence_length, seed=cfg.seed,
    )
    model = TransformerLM(model_cfg, seed=cfg.seed)
    loaded = load_checkpoint_v2(ckpt_dir, model)
    print(f"[finetune] checkpoint dimuat: epoch={loaded['epoch']} step={loaded['step']}")

    holdout_data = TextWindows(ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)  # self-eval: target tahap ini adalah hafalan (loss mendekati 0)
    log_path = ROOT / "runs" / "finetune_nazeio.jsonl"
    with RunLog(log_path) as log:
        data = TextWindows(ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
        summary = TrainingRun(cfg, model, data, holdout=holdout_data).run(log=log)
    summary["finetune_corpus_tokens"] = len(ids)
    summary["loaded_epoch"] = loaded["epoch"]
    summary["finetune_stage"] = "nazeio_command"

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
