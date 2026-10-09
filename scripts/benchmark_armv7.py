"""Benchmark latensi inference proxy ARmv7 (M010-T009, DECISION-023).

Mengukur latensi per token generation (batch 1) pada host lokal sebagai
PROXY untuk Raspberry Pi 3 (ARmv7). Target DECISION-023: <= 2 ms/token
(optimis — diukur owner pada perangkat sungguhan). Mendukung presisi
float32 (export nazeio) untuk membandingkan latensi float64 vs float32.

Jalankan dari root repo:
    python scripts/benchmark_armv7.py --checkpoint models/naze_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from naze.inference.engine import InferenceEngine
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer
from naze.train.checkpoint import load_checkpoint_v2
from naze.train.runlog import measure_peak_rss

TARGET_MS_PER_TOKEN = 2.0


def load_model(checkpoint_dir: Path, *, float32: bool) -> TransformerLM:
    cfg = TransformerConfig(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                             max_sequence_length=128, seed=0)
    model = TransformerLM(cfg, seed=0)
    load_checkpoint_v2(checkpoint_dir, model)
    if float32:
        for p in model.params().values():
            p[...] = p.astype(np.float32)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark latensi proxy ARmv7 (M-010)")
    parser.add_argument("--checkpoint", type=str, default="models/naze_v1")
    parser.add_argument("--float32", dest="float32", action="store_true", default=True)
    parser.add_argument("--no-float32", dest="float32", action="store_false")
    parser.add_argument("--n-new", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report", type=str, default="runs/benchmark_armv7_report.json")
    args = parser.parse_args()

    tok = ByteTokenizer()
    model = load_model(ROOT / args.checkpoint, float32=args.float32)
    engine = InferenceEngine(model)
    prompt = tok.encode("perintah: buka whatsapp\nintent:")

    for _ in range(args.warmup):
        engine.generate(prompt, 16, temperature=0.0, seed=0)

    per_token_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        engine.generate(prompt, args.n_new, temperature=0.0, seed=0)
        per_token_ms.append((time.perf_counter() - t0) * 1000.0 / args.n_new)

    mean_ms = sum(per_token_ms) / len(per_token_ms)
    report = {
        "proxy_note": "host lokal sebagai proxy ARmv7 (Raspberry Pi 3), bukan device",
        "dtype": "float32" if args.float32 else "float64",
        "n_new": args.n_new, "warmup": args.warmup, "runs": args.runs,
        "per_token_ms": per_token_ms,
        "mean_ms_per_token": mean_ms,
        "min_ms_per_token": min(per_token_ms),
        "max_ms_per_token": max(per_token_ms),
        "tokens_per_second": 1000.0 / mean_ms,
        "target_ms_per_token": TARGET_MS_PER_TOKEN,
        "achieved_target": mean_ms <= TARGET_MS_PER_TOKEN,
        "peak_rss_kb": measure_peak_rss(),
    }
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"[armv7-proxy] {report['dtype']}: {mean_ms:.3f} ms/token "
          f"({report['tokens_per_second']:.1f} tok/s) "
          f"— target <= {TARGET_MS_PER_TOKEN} ms/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
