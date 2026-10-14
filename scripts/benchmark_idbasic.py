#!/usr/bin/env python3
"""Evaluator benchmark kemampuan dasar Bahasa Indonesia (IDB-T006, REQ-401).

Memuat checkpoint Naze 2.0, menjalankan setiap item benchmark lima kategori
(data/idbasic/benchmark/*.json) lewat InferenceEngine greedy
(temperature=0.0, seed=0, deterministik), lalu menilai exact-match KETAT
terhadap 'expected' TANPA normalisasi apa pun: prediksi benar hanya bila
identik karakter-per-karakter dengan label (termasuk spasi awal,
kapitalisasi, dan tanda baca). Normalisasi whitespace/lowercase sengaja
tidak dilakukan agar kesalahan tata bahasa, kapitalisasi, dan tanda baca
tetap terhitung sebagai kesalahan.

Kategori dievaluasi dibaca dari manifest (splits.benchmark), bukan dikode
ulang di sini. Evaluator tidak menetapkan sendiri ambang kelulusan:
ambang pasca-baseline adalah OD-127 (menunggu keputusan owner).

Kegagalan dipisahkan dan tidak pernah disamakan dengan skor nol:
1. kegagalan data benchmark  -> exit 2 tanpa skor;
2. kegagalan pemuatan model  -> exit 2 tanpa skor;
3. kegagalan prediksi per item (exception di tengah evaluasi) -> exit 2.
Prediksi yang salah adalah hasil model yang sah dan hanya menurunkan skor.

Jalankan dari root repo:
    python scripts/benchmark_idbasic.py --checkpoint models/naze_v1 \
        --report runs/idbasic_baseline.json --label baseline

Kode keluar: 0 = evaluasi selesai dengan skor sah (berapa pun nilainya);
2 = kegagalan data/model/infrastruktur (tidak ada skor yang dihasilkan).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naze.inference.engine import InferenceEngine
from naze.lm.transformer_lm import TransformerLM
from naze.nn.transformer import TransformerConfig
from naze.token import ByteTokenizer
from naze.train.checkpoint import load_checkpoint_v2

DEFAULT_DATA_DIR = ROOT / "data" / "idbasic"
PROMPT_SUFFIX = "\njawab:"
EXPECTED_PREFIX = " "


class BenchmarkDataError(Exception):
    """Kegagalan struktur/isi berkas benchmark (bukan skor model)."""


class ModelLoadError(Exception):
    """Kegagalan pemuatan checkpoint (bukan skor model)."""


class PredictionError(Exception):
    """Kegagalan infrastruktur prediksi per item (bukan skor nol)."""


def load_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    if not path.exists():
        raise BenchmarkDataError(f"manifest tidak ditemukan: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkDataError(f"manifest tidak dapat dibaca: {exc}") from exc
    if not isinstance(manifest.get("splits", {}).get("benchmark"), dict):
        raise BenchmarkDataError("manifest tanpa splits.benchmark yang valid")
    return manifest


def benchmark_categories(manifest: dict) -> list[str]:
    """Kategori dievaluasi = kunci splits.benchmark selain 'total'."""
    counts = manifest["splits"]["benchmark"]
    categories = sorted(k for k in counts if k != "total")
    if not categories:
        raise BenchmarkDataError("splits.benchmark kosong pada manifest")
    return categories


def load_category_benchmark(data_dir: Path, category: str) -> list[dict]:
    """Muat dan validasi struktur satu berkas benchmark kategori."""
    path = data_dir / "benchmark" / f"{category}.json"
    if not path.exists():
        raise BenchmarkDataError(f"berkas benchmark tidak ditemukan: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkDataError(f"{path}: tidak dapat dibaca: {exc}") from exc
    if data.get("category") != category:
        raise BenchmarkDataError(f"{path}: 'category' != {category!r}")
    if data.get("metric") != "exact_match":
        raise BenchmarkDataError(f"{path}: 'metric' bukan 'exact_match'")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise BenchmarkDataError(f"{path}: 'items' kosong atau bukan list")
    seen_ids: set[str] = set()
    for item in items:
        item_id = item.get("id", "")
        if not item_id:
            raise BenchmarkDataError(f"{path}: item tanpa 'id'")
        if item_id in seen_ids:
            raise BenchmarkDataError(f"{path}: id duplikat {item_id!r}")
        seen_ids.add(item_id)
        prompt = item.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.endswith(PROMPT_SUFFIX):
            raise BenchmarkDataError(
                f"{path} [{item_id}]: prompt tanpa akhiran baris 'jawab:'"
            )
        expected = item.get("expected", "")
        if (not isinstance(expected, str) or not expected.startswith(EXPECTED_PREFIX)
                or not expected.strip()):
            raise BenchmarkDataError(
                f"{path} [{item_id}]: 'expected' harus diawali spasi dan tidak kosong"
            )
    return items


def is_correct(output: str, expected: str) -> bool:
    """Exact-match ketat: identik karakter-per-karakter, tanpa normalisasi."""
    return output == expected


def evaluate_items(engine: Any, tok: Any, items: list[dict]) -> tuple[int, list[dict]]:
    """Evaluasi satu daftar item; return (jumlah benar, hasil per item)."""
    n_correct = 0
    results: list[dict] = []
    for item in items:
        prompt_ids = tok.encode(item["prompt"])
        expected = item["expected"]
        try:
            gen = engine.generate(prompt_ids, len(tok.encode(expected)),
                                  temperature=0.0, seed=0)
        except Exception as exc:  # kegagalan infrastruktur, bukan skor nol
            raise PredictionError(f"{item['id']}: gagal generate: {exc}") from exc
        output = tok.decode(gen)
        ok = is_correct(output, expected)
        n_correct += int(ok)
        results.append({"id": item["id"], "expected": expected,
                        "output": output, "correct": ok})
    return n_correct, results


def load_model(checkpoint_dir: Path) -> tuple[TransformerLM, dict]:
    """Muat checkpoint (D-018) dan verifikasi checksum; return (model, meta)."""
    if not (checkpoint_dir / "params.npz").exists():
        raise ModelLoadError(f"checkpoint tidak ditemukan: {checkpoint_dir / 'params.npz'}")
    cfg = TransformerConfig(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                             max_sequence_length=128, seed=0)
    model = TransformerLM(cfg, seed=0)
    try:
        loaded = load_checkpoint_v2(checkpoint_dir, model)
    except (OSError, ValueError, KeyError) as exc:
        raise ModelLoadError(f"gagal memuat checkpoint {checkpoint_dir}: {exc}") from exc
    meta = {"epoch": loaded["epoch"], "step": loaded["step"],
            "checksum_sha256": loaded["meta"].get("checksum_sha256")}
    return model, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluator benchmark ID-BASIC-001")
    parser.add_argument("--checkpoint", type=str, default="models/naze_v1",
                        help="direktori checkpoint (relatif root repo)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--report", type=str, default="runs/idbasic_report.json")
    parser.add_argument("--label", type=str, default="",
                        help="label tahap evaluasi (mis. baseline / post_training)")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(Path(args.data_dir))
        categories = benchmark_categories(manifest)
        benchmarks = {c: load_category_benchmark(Path(args.data_dir), c)
                      for c in categories}
    except BenchmarkDataError as exc:
        print(f"[idbasic-bench] KEGAGALAN DATA: {exc}")
        return 2

    try:
        model, ckpt_meta = load_model(ROOT / args.checkpoint)
    except ModelLoadError as exc:
        print(f"[idbasic-bench] KEGAGALAN MODEL: {exc}")
        return 2

    tok = ByteTokenizer()
    engine = InferenceEngine(model)

    per_category: dict[str, dict] = {}
    total_items = 0
    total_correct = 0
    try:
        for category in categories:
            n_correct, results = evaluate_items(engine, tok, benchmarks[category])
            n_items = len(results)
            accuracy = 100.0 * n_correct / n_items
            per_category[category] = {
                "n_items": n_items,
                "n_correct": n_correct,
                "n_failed": n_items - n_correct,
                "exact_match_accuracy_percent": accuracy,
                "failed": [r for r in results if not r["correct"]],
                "results": results,
            }
            total_items += n_items
            total_correct += n_correct
            print(f"[idbasic-bench] {category}: {n_correct}/{n_items} "
                  f"exact-match ({accuracy:.2f}%)")
    except PredictionError as exc:
        print(f"[idbasic-bench] KEGAGALAN PREDIKSI: {exc}")
        return 2

    overall = 100.0 * total_correct / total_items
    report = {
        "label": args.label,
        "metric": "exact_match",
        "normalization": "none",
        "note_normalisasi": ("exact-match ketat; kesalahan kapitalisasi, spasi, "
                            "dan tanda baca tetap dihitung salah"),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_meta["epoch"],
        "checkpoint_step": ckpt_meta["step"],
        "checkpoint_checksum_sha256": ckpt_meta["checksum_sha256"],
        "n_items": total_items,
        "n_correct": total_correct,
        "exact_match_accuracy_percent": overall,
        "per_category": per_category,
    }
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[idbasic-bench] TOTAL: {total_correct}/{total_items} "
          f"exact-match ({overall:.2f}%) — laporan: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
