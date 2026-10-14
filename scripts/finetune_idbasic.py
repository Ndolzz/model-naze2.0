#!/usr/bin/env python3
"""Fine-tuning rehearsal kemampuan dasar Bahasa Indonesia (IDB-T008, REQ-401).

Rehearsal terisolasi: melatih salinan bobot dari checkpoint SUMBER ke
checkpoint EKSPERIMEN baru (default models/naze_idbasic_exp). Checkpoint
sumber (model produksi) tidak pernah ditimpa: guard menolak direktori
tujuan yang sama dengan sumber maupun models/naze_v1.

Korpus campuran mengikuti pola finetune_nazeio.py (M-010) untuk mencegah
catastrophic forgetting:
- blok idbasic (data/idbasic/*.txt, HANYA berkas training) diulang
  --idbasic-repeats kali;
- rehearsal perintah NazeIO (data/korpus/perintah_tingkat_1.txt) diulang
  --command-repeats kali;
- rehearsal teks umum dari data/corpus/splits/train.txt sebesar
  --general-bytes per epoch (potongan tersebar merata, deterministik).

Benchmark data/idbasic/benchmark TIDAK PERNAH masuk korpus (anti-kebocoran;
juga tidak dipakai untuk validasi gradien/pemilihan hyperparameter).
Dataset wajib lolos scripts/validate_idbasic.py sebelum training; proses
berhenti (exit 2) bila validasi gagal.

block_size default 96 (deviasi terdokumentasi dari angka pra-kurasi 64 pada
TD): contoh train terpanjang 86 byte sehingga jendela 64 tidak cukup;
96 <= T_max 128 tetap terpenuhi (D-018). Arsitektur, vocab (256, D-015),
dan format checkpoint v2 tidak diubah; kompatibilitas diverifikasi oleh
load_checkpoint_v2 (nama & shape seluruh parameter).

Jalankan dari root repo:
    python scripts/finetune_idbasic.py --checkpoint models/naze_v1 \
        --out-checkpoint models/naze_idbasic_exp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from naze.data.dataset import TextWindows
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer
from naze.train.checkpoint import load_checkpoint_v2
from naze.train.config import TrainConfig
from naze.train.evaluate import evaluate
from naze.train.loop import TrainingRun
from naze.train.runlog import RunLog

import validate_idbasic

IDBASIC_DIR = ROOT / "data" / "idbasic"
PERINTAH = ROOT / "data" / "korpus" / "perintah_tingkat_1.txt"
VAL_SPLIT = ROOT / "data" / "corpus" / "splits" / "val.txt"
DEFAULT_GENERAL_CORPUS = ROOT / "data" / "corpus" / "splits" / "train.txt"
VAL_SAMPLE_BYTES = 65_536
PRODUCTION_DIRS = ("models/naze_v1",)


def manifest_categories(data_dir: Path = IDBASIC_DIR) -> list[str]:
    """Kategori training = kunci splits.train pada manifest selain 'total'."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["splits"]["train"]
    categories = sorted(k for k in counts if k != "total")
    if not categories:
        raise SystemExit("manifest idbasic: splits.train kosong")
    return categories


def load_idbasic_train_text(data_dir: Path = IDBASIC_DIR,
                            categories: list[str] | None = None) -> str:
    """Gabungkan berkas training idbasic; pertahankan pemisah blok antar kategori."""
    if categories is None:
        categories = manifest_categories(data_dir)
    parts = []
    for category in categories:
        path = data_dir / f"{category}.txt"
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"berkas training kosong: {path}")
        parts.append(text)
    return "\n\n".join(parts) + "\n"


def load_perintah_text(path: Path) -> str:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    return "\n".join(lines).strip() + "\n"


def load_general_rehearsal(path: Path, total_bytes: int, n_chunks: int = 8) -> str:
    """Rehearsal deterministik: potongan tersebar merata di seluruh file."""
    data = path.read_text(encoding="utf-8")
    if total_bytes <= 0 or not data:
        return ""
    chunk_size = max(1, total_bytes // n_chunks)
    parts = []
    for i in range(n_chunks):
        start = i * len(data) // n_chunks
        parts.append(data[start:start + chunk_size])
    return "\n".join(parts)


def build_mixed_text(idbasic_text: str, perintah_text: str, general_text: str,
                     idbasic_repeats: int, command_repeats: int) -> str:
    return ((idbasic_text + "\n") * idbasic_repeats + "\n"
            + (perintah_text + "\n") * command_repeats + "\n"
            + general_text + "\n")


def check_output_checkpoint(source_dir: Path, out_dir: Path) -> None:
    """Tolak penimpaan checkpoint sumber/produksi (isolasi eksperimen)."""
    src = source_dir.resolve()
    out = out_dir.resolve()
    if src == out:
        raise SystemExit(f"checkpoint tujuan tidak boleh sama dengan sumber: {out}")
    for prod in PRODUCTION_DIRS:
        if out == (ROOT / prod).resolve():
            raise SystemExit(f"checkpoint tujuan dilarang menimpa produksi: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning rehearsal ID-BASIC-001")
    parser.add_argument("--checkpoint", type=str, default="models/naze_v1",
                        help="direktori checkpoint SUMBER (hanya dibaca)")
    parser.add_argument("--out-checkpoint", type=str, default="models/naze_idbasic_exp",
                        help="direktori checkpoint EKSPERIMEN keluaran (ditimpa)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=96,
                        help="jendela konteks (contoh terpanjang 86 byte; TD pra-kurasi 64)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--idbasic-repeats", type=int, default=12,
                        help="pengulangan blok idbasic per epoch")
    parser.add_argument("--command-repeats", type=int, default=25,
                        help="pengulangan rehearsal perintah NazeIO per epoch")
    parser.add_argument("--general-corpus", type=str, default=str(DEFAULT_GENERAL_CORPUS))
    parser.add_argument("--general-bytes", type=int, default=262_144,
                        help="byte rehearsal teks umum per epoch (0 = nonaktif)")
    parser.add_argument("--report", type=str, default="runs/finetune_idbasic_summary.json")
    args = parser.parse_args(argv)

    src_dir = ROOT / args.checkpoint
    out_dir = ROOT / args.out_checkpoint
    check_output_checkpoint(src_dir, out_dir)
    if not (src_dir / "params.npz").exists():
        raise SystemExit(f"checkpoint sumber tidak ditemukan: {src_dir / 'params.npz'}")

    errors = validate_idbasic.validate_dataset_dir(IDBASIC_DIR)
    if errors:
        print(f"[idbasic-finetune] dataset GAGAL validasi ({len(errors)} masalah); "
              "training dibatalkan")
        for err in errors:
            print(f"  - {err}")
        return 2

    idbasic_text = load_idbasic_train_text()
    perintah_text = load_perintah_text(PERINTAH)
    general_text = load_general_rehearsal(Path(args.general_corpus), args.general_bytes)
    text = build_mixed_text(idbasic_text, perintah_text, general_text,
                            args.idbasic_repeats, args.command_repeats)
    ids = ByteTokenizer().encode(text)
    print(f"[idbasic-finetune] korpus campuran: {len(ids)} token "
          f"(idbasic x{args.idbasic_repeats} ({len(idbasic_text)} char/blok), "
          f"perintah x{args.command_repeats}, rehearsal umum {len(general_text)} char), "
          f"epochs={args.epochs}, block={args.block_size}")

    cfg = TrainConfig(
        model="transformer",
        corpus_path=str(IDBASIC_DIR),
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        eval_every=args.eval_every,
        checkpoint_path=str(out_dir),
    )
    model_cfg = TransformerConfig(
        d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_sequence_length=cfg.max_sequence_length, seed=cfg.seed,
    )
    model = TransformerLM(model_cfg, seed=cfg.seed)
    # load_checkpoint_v2 memverifikasi checksum dan kecocokan nama+shape
    # seluruh parameter (kompatibilitas dimensi & format checkpoint v2).
    loaded = load_checkpoint_v2(src_dir, model)
    src_sha = loaded["meta"].get("checksum_sha256")
    print(f"[idbasic-finetune] checkpoint sumber dimuat: epoch={loaded['epoch']} "
          f"step={loaded['step']} checksum={src_sha or 'n/a'}")

    idbasic_ids = ByteTokenizer().encode(idbasic_text)
    holdout_data = TextWindows(idbasic_ids, cfg.block_size, cfg.batch_size,
                               seed=cfg.seed)
    log_path = ROOT / "runs" / "finetune_idbasic.jsonl"
    started = time.perf_counter()
    with RunLog(log_path) as log:
        data = TextWindows(ids, cfg.block_size, cfg.batch_size, seed=cfg.seed)
        summary = TrainingRun(cfg, model, data, holdout=holdout_data).run(log=log)
    duration = time.perf_counter() - started

    summary["finetune_stage"] = "idbasic_mixed_rehearsal"
    summary["source_checkpoint"] = str(src_dir)
    summary["source_checkpoint_epoch"] = loaded["epoch"]
    summary["source_checkpoint_step"] = loaded["step"]
    summary["source_checkpoint_sha256"] = src_sha
    summary["out_checkpoint"] = str(out_dir)
    summary["idbasic_repeats"] = args.idbasic_repeats
    summary["command_repeats"] = args.command_repeats
    summary["general_rehearsal_bytes"] = len(general_text)
    summary["corpus_tokens"] = len(ids)
    summary["duration_seconds"] = round(duration, 3)
    summary["config_hash"] = cfg.config_hash()

    # Evaluasi kemampuan bahasa umum pasca fine-tuning (rujukan gate M-010:
    # ppl sampel <= 35, DECISION-023). Nilai ini pemantauan, bukan ambang OD-127.
    val_text = VAL_SPLIT.read_text(encoding="utf-8")[:VAL_SAMPLE_BYTES]
    val_ids = ByteTokenizer().encode(val_text)
    val_data = TextWindows(val_ids, 32, cfg.batch_size, seed=cfg.seed)
    general = evaluate(model, val_data, per_pos=True, seed=cfg.seed)
    summary["general_val_sample_loss"] = general.mean_loss
    summary["general_val_sample_perplexity"] = general.perplexity
    summary["general_val_sample_tokens"] = general.n_tokens
    print(f"[idbasic-finetune] sampel korpus umum: loss={general.mean_loss:.4f} "
          f"ppl={general.perplexity:.4f} (rujukan gate M-010: ppl <= 35)")

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
