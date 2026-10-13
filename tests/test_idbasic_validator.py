"""Tes unit validator dataset ID-BASIC-001 (pytest)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

import validate_idbasic as vid  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "idbasic"


def test_parse_examples_valid():
    text = "sinonim: pintar\njawab: cerdas\n\nantonim: panas\njawab: dingin\n\n"
    examples = vid.parse_examples(text)
    assert examples == [
        ("sinonim: pintar", "jawab: cerdas"),
        ("antonim: panas", "jawab: dingin"),
    ]


def test_parse_examples_incomplete_block_raises():
    with pytest.raises(ValueError):
        vid.parse_examples("sinonim: pintar\njawab: cerdas\n\nantonim: panas\n")


def test_missing_answer_prefix_detected():
    errors: list[str] = []
    vid.validate_train_text("soal: a\njawaban: b\n\n", "vocabulary", errors)
    assert any("jawab" in e for e in errors)


def test_duplicate_prompt_detected():
    text = "soal: a\njawab: satu\n\nsoal: a\njawab: dua\n\n"
    errors: list[str] = []
    vid.validate_train_text(text, "vocabulary", errors)
    assert any("duplikat" in e for e in errors)


def test_empty_label_detected():
    errors: list[str] = []
    vid.validate_train_text("soal: a\njawab: \n\n", "vocabulary", errors)
    assert any("label kosong" in e for e in errors)


def test_too_long_example_detected():
    prompt = "soal: " + "x" * 100
    errors: list[str] = []
    vid.validate_train_text(prompt + "\njawab: jawaban\n\n", "vocabulary", errors)
    assert any("byte" in e for e in errors)


def test_valid_train_text_no_errors():
    errors: list[str] = []
    examples = vid.validate_train_text("soal: a\njawab: satu\n\n", "vocabulary", errors)
    assert errors == []
    assert len(examples) == 1


def test_benchmark_expected_must_start_with_space():
    data = {
        "items": [
            {"id": "b-01", "prompt": "soal: a\njawab:", "expected": "tanpa-spasi"},
        ]
    }
    errors: list[str] = []
    vid.validate_benchmark_data(data, "vocabulary", errors)
    assert any("expected" in e for e in errors)


def test_benchmark_prompt_must_end_with_marker():
    data = {"items": [{"id": "b-01", "prompt": "soal: a", "expected": " x"}]}
    errors: list[str] = []
    vid.validate_benchmark_data(data, "vocabulary", errors)
    assert any("jawab" in e for e in errors)


def test_leakage_detected():
    train = {"vocabulary": {"soal: a"}}
    bench = {"vocabulary": ["soal: a\njawab:"]}
    assert vid.check_leakage(train, bench)


def test_no_leakage_passes():
    train = {"vocabulary": {"soal: a"}}
    bench = {"vocabulary": ["soal: b\njawab:"]}
    assert vid.check_leakage(train, bench) == []


def test_real_dataset_validates():
    if not DATA_DIR.exists():
        pytest.skip("direktori data/idbasic tidak ditemukan")
    errors = vid.validate_dataset_dir(DATA_DIR)
    assert errors == []
