"""Tes unit evaluator benchmark ID-BASIC-001 (IDB-T006, pytest)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

import benchmark_idbasic as bie  # noqa: E402


class FakeTok:
    """Tokenizer ASCII sederhana untuk pengujian tanpa bobot model."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class FakeEngine:
    """Engine pengganti: generate selalu mengembalikan keluaran tetap."""

    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt_ids, max_new, *, temperature=None,
                 top_k=None, seed=None):
        return [ord(ch) for ch in self.output]


class FailingEngine:
    """Engine pengganti yang selalu gagal (kegagalan infrastruktur)."""

    def generate(self, prompt_ids, max_new, *, temperature=None,
                 top_k=None, seed=None):
        raise RuntimeError("infrastruktur rusak")


def _bench_file(tmp_path: Path, category: str, payload: dict) -> Path:
    bench_dir = tmp_path / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"{category}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


VALID_ITEM = {"id": "voc-b01", "prompt": "sinonim: pandai\njawab:",
              "expected": " pintar"}


def _valid_payload() -> dict:
    return {"version": "1.0.0", "category": "vocabulary",
            "metric": "exact_match", "items": [dict(VALID_ITEM)]}


def test_is_correct_strict_no_normalization():
    assert bie.is_correct(" pintar", " pintar") is True
    # tanpa spasi awal -> salah
    assert bie.is_correct("pintar", " pintar") is False
    # spasi tambahan -> salah
    assert bie.is_correct(" pintar ", " pintar") is False
    # kapitalisasi berbeda -> salah
    assert bie.is_correct(" Pintar", " pintar") is False
    # keluaran kosong -> salah, bukan error
    assert bie.is_correct("", " pintar") is False


def test_evaluate_items_correct_prediction():
    items = [dict(VALID_ITEM)]
    n_correct, results = bie.evaluate_items(FakeEngine(" pintar"), FakeTok(), items)
    assert n_correct == 1
    assert results[0]["correct"] is True
    assert results[0]["output"] == " pintar"


def test_evaluate_items_wrong_prediction_scores_zero():
    items = [dict(VALID_ITEM)]
    n_correct, results = bie.evaluate_items(FakeEngine(" bodoh"), FakeTok(), items)
    assert n_correct == 0
    assert results[0]["correct"] is False
    assert results[0]["output"] == " bodoh"


def test_evaluate_items_empty_output_is_wrong_not_error():
    items = [dict(VALID_ITEM)]
    n_correct, results = bie.evaluate_items(FakeEngine(""), FakeTok(), items)
    assert n_correct == 0
    assert results[0]["correct"] is False
    assert results[0]["output"] == ""


def test_evaluate_items_generate_failure_raises_prediction_error():
    items = [dict(VALID_ITEM)]
    with pytest.raises(bie.PredictionError):
        bie.evaluate_items(FailingEngine(), FakeTok(), items)


def test_load_category_benchmark_valid(tmp_path):
    _bench_file(tmp_path, "vocabulary", _valid_payload())
    items = bie.load_category_benchmark(tmp_path, "vocabulary")
    assert len(items) == 1
    assert items[0]["id"] == "voc-b01"


def test_load_category_benchmark_rejects_bad_metric(tmp_path):
    payload = _valid_payload()
    payload["metric"] = "fuzzy"
    _bench_file(tmp_path, "vocabulary", payload)
    with pytest.raises(bie.BenchmarkDataError):
        bie.load_category_benchmark(tmp_path, "vocabulary")


def test_load_category_benchmark_rejects_missing_file(tmp_path):
    with pytest.raises(bie.BenchmarkDataError):
        bie.load_category_benchmark(tmp_path, "vocabulary")


def test_load_category_benchmark_rejects_expected_without_space(tmp_path):
    payload = _valid_payload()
    payload["items"] = [{"id": "voc-b01",
                         "prompt": "sinonim: pandai\njawab:",
                         "expected": "pintar"}]
    _bench_file(tmp_path, "vocabulary", payload)
    with pytest.raises(bie.BenchmarkDataError):
        bie.load_category_benchmark(tmp_path, "vocabulary")


def test_load_category_benchmark_rejects_duplicate_id(tmp_path):
    payload = _valid_payload()
    payload["items"] = [dict(VALID_ITEM), dict(VALID_ITEM)]
    _bench_file(tmp_path, "vocabulary", payload)
    with pytest.raises(bie.BenchmarkDataError):
        bie.load_category_benchmark(tmp_path, "vocabulary")


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(bie.BenchmarkDataError):
        bie.load_manifest(tmp_path)


def test_load_manifest_without_benchmark_splits(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"splits": {}}),
                                            encoding="utf-8")
    with pytest.raises(bie.BenchmarkDataError):
        bie.load_manifest(tmp_path)


def test_benchmark_categories_sorted_and_excludes_total():
    manifest = {"splits": {"benchmark": {"vocabulary": 44, "grammar": 46,
                                          "total": 90}}}
    assert bie.benchmark_categories(manifest) == ["grammar", "vocabulary"]


def test_benchmark_categories_empty_raises():
    manifest = {"splits": {"benchmark": {"total": 90}}}
    with pytest.raises(bie.BenchmarkDataError):
        bie.benchmark_categories(manifest)


def test_load_model_missing_checkpoint_raises(tmp_path):
    with pytest.raises(bie.ModelLoadError):
        bie.load_model(tmp_path / "models" / "naze_v1")


def test_accuracy_reported_not_claimed_on_model_failure(tmp_path):
    # Kegagalan pemuatan model harus exit 2 tanpa laporan skor.
    _bench_file(tmp_path, "vocabulary", _valid_payload())
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"benchmark": {"vocabulary": 1, "total": 1}}}),
        encoding="utf-8")
    report_path = tmp_path / "runs" / "report.json"
    rc = bie.main(["--data-dir", str(tmp_path),
                   "--checkpoint", str(tmp_path / "tidak_ada"),
                   "--report", str(report_path),
                   "--label", "baseline"])
    assert rc == 2
    assert not report_path.exists()
