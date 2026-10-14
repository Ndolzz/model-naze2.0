"""Benchmark akurasi command NazeIO dwibahasa (M010-T008 + M011-T011, DECISION-023/025).

Memuat checkpoint final, menjalankan setiap command dari
data/benchmark/commands.json (v2: dwibahasa) lewat InferenceEngine
(greedy), lalu menilai exact-match terhadap expected_output.
Laporan memuat akurasi total dan akurasi terpisah per bahasa
("id", "en"). Target DECISION-025: akurasi >= 90% per bahasa
(diukur owner pada model final M-011); subset "id" menjaga
perbandingan baseline Naze 1.0 (DECISION-023).

Jalankan dari root repo:
    python scripts/benchmark_nazeio.py --checkpoint models/naze_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naze.inference.engine import InferenceEngine
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer
from naze.train.checkpoint import load_checkpoint_v2

DEFAULT_COMMANDS = ROOT / "data" / "benchmark" / "commands.json"


def load_model(checkpoint_dir: Path) -> TransformerLM:
    cfg = TransformerConfig(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                             max_sequence_length=128, seed=0)
    model = TransformerLM(cfg, seed=0)
    load_checkpoint_v2(checkpoint_dir, model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark command NazeIO dwibahasa (M-011)")
    parser.add_argument("--checkpoint", type=str, default="models/naze_v1")
    parser.add_argument("--commands", type=str, default=str(DEFAULT_COMMANDS))
    parser.add_argument("--report", type=str, default="runs/benchmark_nazeio_report.json")
    args = parser.parse_args()

    bench = json.loads(Path(args.commands).read_text(encoding="utf-8"))
    tok = ByteTokenizer()
    engine = InferenceEngine(load_model(ROOT / args.checkpoint))

    target_total = bench.get("target_accuracy_percent", 90)
    target_per_language = bench.get("target_accuracy_per_language_percent", target_total)

    results = []
    n_correct = 0
    per_language: dict[str, dict[str, int]] = {}
    for cmd in bench["commands"]:
        language = cmd.get("language", "id")
        prompt_ids = tok.encode(cmd["prompt"])
        expected = cmd["expected_output"]
        gen = engine.generate(prompt_ids, len(tok.encode(expected)),
                              temperature=0.0, seed=0)
        output = tok.decode(gen)
        ok = output == expected
        n_correct += int(ok)
        stats = per_language.setdefault(language, {"n": 0, "n_correct": 0})
        stats["n"] += 1
        stats["n_correct"] += int(ok)
        results.append({"id": cmd["id"], "category": cmd["category"],
                        "language": language,
                        "expected": expected, "output": output, "correct": ok})

    accuracy = 100.0 * n_correct / len(bench["commands"])
    per_language_report = {}
    for language, stats in sorted(per_language.items()):
        acc = 100.0 * stats["n_correct"] / stats["n"]
        per_language_report[language] = {
            "n_commands": stats["n"],
            "n_correct": stats["n_correct"],
            "accuracy_percent": acc,
            "achieved_target": acc >= target_per_language,
        }
    report = {
        "version": bench["version"],
        "metric": bench["metric"],
        "n_commands": len(bench["commands"]),
        "n_correct": n_correct,
        "accuracy_percent": accuracy,
        "target_accuracy_percent": target_total,
        "achieved_target": accuracy >= target_total,
        "target_accuracy_per_language_percent": target_per_language,
        "per_language": per_language_report,
        "results": results,
    }
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"[nazeio] akurasi total {accuracy:.2f}% ({n_correct}/{len(bench['commands'])}) "
          f"— target {target_total}% (DECISION-023)")
    for language, lang in per_language_report.items():
        print(f"[nazeio] akurasi [{language}] {lang['accuracy_percent']:.2f}% "
              f"({lang['n_correct']}/{lang['n_commands']}) "
              f"— target per bahasa {target_per_language}% (DECISION-025)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
