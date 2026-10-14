"""Fine-tuning command NazeIO dengan rehearsal (M-010, DECISION-023).

Benchmark NazeIO memerlukan format pemetaan "perintah: {teks}\nintent: {intent}"
dari docs/archive/korpus_legacy/perintah_tingkat_1.txt (arsip M011-T002; asal:
data/korpus/, konten identik). Fine-tuning murni pada korpus perintah
terbukti memicu catastrophic forgetting (ppl umum 2679 pada 100 epoch; 7614 pada
500 epoch), maka script ini melatih pada korpus CAMPURAN: perintah (diulang)
+ potongan teks umum (rehearsal) dari train.txt yang tersebar merata,
sehingga akurasi perintah dan kemampuan bahasa umum terjaga dalam satu model.

Jalankan dari root repo:
    python scripts/finetune_nazeio.py --checkpoint models/naze_v1 --epochs 8
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
from naze.train.evaluate import evaluate
from naze.train.loop import TrainingRun
from naze.train.runlog import RunLog

PERINTAH = ROOT / "docs" / "archive" / "korpus_legacy" / "perintah_tingkat_1.txt"  # M011-T002: arsip (asal: data/korpus/)
VAL_SPLIT = ROOT / "data" / "corpus" / "splits" / "val.txt"
DEFAULT_GENERAL_CORPUS = ROOT / "data" / "corpus" / "splits" / "train.txt"
VAL_SAMPLE_BYTES = 65_536  # sampel deterministik; evaluasi penuh tetap lewat train_final (no-op)


def load_perintah_text(path: Path) -> str:
    # Pertahankan struktur baris asli: "perintah: X\nintent: Y\n\nperintah: ...".
    # (Jangan gabungkan semua baris dengan \n\n - konteks kondisional
    # "perintah -> intent" harus identik dengan format benchmark.)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    return "\n".join(lines).strip() + "\n"


def load_general_rehearsal(path: Path, total_bytes: int, n_chunks: int = 8) -> str:
    # Rehearsal deterministik: n_chunks potongan tersebar merata di seluruh file
    # agar keragaman distribusi umum tetap terwakili.
    data = path.read_text(encoding="utf-8")
    if total_bytes <= 0 or not data:
        return ""
    chunk_size = max(1, total_bytes // n_chunks)
    parts = []
    for i in range(n_chunks):
        start = i * len(data) // n_chunks
        parts.append(data[start:start + chunk_size])
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning command NazeIO (M-010)")
    parser.add_argument("--checkpoint", type=str, default="models/naze_v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--command-repeats", type=int, default=25,
                        help="pengulangan blok perintah per epoch")
    parser.add_argument("--general-corpus", type=str, default=str(DEFAULT_GENERAL_CORPUS))
    parser.add_argument("--general-bytes", type=int, default=524288,
                        help="jumlah byte rehearsal teks umum per epoch (0 = nonaktif)")
    parser.add_argument("--report", type=str, default="runs/finetune_nazeio_summary.json")
    args = parser.parse_args()

    ckpt_dir = ROOT / args.checkpoint
    if not (ckpt_dir / "params.npz").exists():
        raise SystemExit(f"checkpoint tidak ditemukan: {ckpt_dir / 'params.npz'}")

    cmd_text = load_perintah_text(PERINTAH)
    general_text = load_general_rehearsal(Path(args.general_corpus), args.general_bytes)
    text = (cmd_text + "\n") * args.command_repeats + "\n" + general_text + "\n"
    ids = ByteTokenizer().encode(text)
    print(f"[finetune] korpus campuran: {len(ids)} token "
          f"(perintah x{args.command_repeats}, rehearsal umum {len(general_text)} char), "
          f"epochs={args.epochs}")

    cfg = TrainConfig(
        model="transformer",
        corpus_path=str(PERINTAH),
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        eval_every=args.eval_every,
        checkpoint_path=str(ckpt_dir),
    )
    model_cfg = TransformerConfig(
        d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_sequence_length=cfg.max_sequence_length, seed=cfg.seed,
    )
    model = TransformerLM(model_cfg, seed=cfg.seed)
    loaded = load_checkpoint_v2(ckpt_dir, model)
    print(f"[finetune] checkpoint dimuat: epoch={loaded['epoch']} step={loaded['step']}")

    cmd_ids = ByteTokenizer().encode(cmd_text)
    holdout_data = TextWindows(cmd_ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
    log_path = ROOT / "runs" / "finetune_nazeio.jsonl"
    with RunLog(log_path) as log:
        data = TextWindows(ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
        summary = TrainingRun(cfg, model, data, holdout=holdout_data).run(log=log)
    summary["finetune_stage"] = "nazeio_command_mixed"
    summary["command_repeats"] = args.command_repeats
    summary["general_rehearsal_bytes"] = len(general_text)
    summary["loaded_epoch"] = loaded["epoch"]

    # Evaluasi kemampuan bahasa umum pasca fine-tuning (gate DECISION-023: ppl <= 35).
    val_text = VAL_SPLIT.read_text(encoding="utf-8")[:VAL_SAMPLE_BYTES]
    val_ids = ByteTokenizer().encode(val_text)
    val_data = TextWindows(val_ids, 32, cfg.batch_size, seed=cfg.seed)
    general = evaluate(model, val_data, per_pos=True, seed=cfg.seed)
    summary["general_val_sample_loss"] = general.mean_loss
    summary["general_val_sample_perplexity"] = general.perplexity
    summary["general_val_sample_tokens"] = general.n_tokens
    print(f"[finetune] sampel korpus umum: loss={general.mean_loss:.4f} "
          f"ppl={general.perplexity:.4f} (gate: ppl <= 35)")

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
