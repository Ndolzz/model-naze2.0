#!/usr/bin/env python3
"""Validasi dataset ID-BASIC-001 (hanya stdlib).

Pemeriksaan: encoding UTF-8, format blok dua baris, awalan jawaban, label
kosong, prompt duplikat, panjang byte contoh, struktur benchmark, kebocoran
prompt benchmark ke training, jumlah contoh, dan hash SHA-256 berkas
terhadap manifest.

Penggunaan:
    python scripts/validate_idbasic.py [--data-dir data/idbasic]

Kode keluar: 0 bila seluruh pemeriksaan lolos, 1 bila ada kegagalan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CATEGORIES = [
    "vocabulary",
    "grammar",
    "sentence_structure",
    "punctuation",
    "contextual_meaning",
]
ANSWER_PREFIX = "jawab: "
PROMPT_SUFFIX = "\njawab:"
MAX_EXAMPLE_BYTES = 96


def parse_examples(text: str) -> list[tuple[str, str]]:
    """Uraikan korpus latih menjadi pasangan (prompt, baris_jawaban)."""
    examples: list[tuple[str, str]] = []
    pending: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        pending.append(line)
        if len(pending) == 2:
            examples.append((pending[0], pending[1]))
            pending = []
    if pending:
        raise ValueError("blok tidak lengkap: ada baris tanpa pasangan jawaban")
    return examples


def validate_train_text(text: str, category: str, errors: list[str]) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    try:
        examples = parse_examples(text)
    except ValueError as exc:
        errors.append(f"{category}: {exc}")
        return examples
    seen: set[str] = set()
    for prompt, answer in examples:
        if not answer.startswith(ANSWER_PREFIX):
            errors.append(
                f"{category}: baris jawaban tanpa awalan 'jawab: ': {prompt[:48]!r}"
            )
            continue
        label = answer[len(ANSWER_PREFIX):]
        if not label.strip():
            errors.append(f"{category}: label kosong: {prompt[:48]!r}")
        if prompt in seen:
            errors.append(f"{category}: prompt duplikat: {prompt[:48]!r}")
        seen.add(prompt)
        block = f"{prompt}\n{answer}"
        if len(block.encode("utf-8")) > MAX_EXAMPLE_BYTES:
            errors.append(
                f"{category}: contoh melebihi {MAX_EXAMPLE_BYTES} byte: {prompt[:48]!r}"
            )
    return examples


def validate_benchmark_data(data: dict, category: str, errors: list[str]) -> list[str]:
    items = data.get("items")
    if not isinstance(items, list) or not items:
        errors.append(f"benchmark/{category}: 'items' kosong atau bukan list")
        return []
    prompts: list[str] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"benchmark/{category}: item bukan objek")
            continue
        item_id = item.get("id", "")
        prompt = item.get("prompt", "")
        expected = item.get("expected", "")
        if not item_id:
            errors.append(f"benchmark/{category}: 'id' kosong")
        elif item_id in seen_ids:
            errors.append(f"benchmark/{category}: id duplikat: {item_id}")
        seen_ids.add(item_id)
        if not prompt.endswith(PROMPT_SUFFIX):
            errors.append(
                f"benchmark/{category}: prompt tanpa akhiran baris 'jawab:': {item_id}"
            )
            continue
        if not expected.startswith(" ") or not expected.strip():
            errors.append(
                f"benchmark/{category}: 'expected' tanpa spasi awal atau kosong: {item_id}"
            )
        if prompt in prompts:
            errors.append(f"benchmark/{category}: prompt duplikat: {item_id}")
        prompts.append(prompt)
        block = f"{prompt}{expected}"
        if len(block.encode("utf-8")) > MAX_EXAMPLE_BYTES:
            errors.append(
                f"benchmark/{category}: contoh melebihi {MAX_EXAMPLE_BYTES} byte: {item_id}"
            )
    return prompts


def check_leakage(
    train_prompts: dict[str, set[str]], bench_prompts: dict[str, list[str]]
) -> list[str]:
    errors: list[str] = []
    all_train: set[str] = set()
    for prompts in train_prompts.values():
        all_train.update(prompts)
    for category, prompts in bench_prompts.items():
        for prompt in prompts:
            stem = prompt[: -len(PROMPT_SUFFIX)]
            if stem in all_train:
                errors.append(
                    f"leakage: prompt benchmark {category} muncul di training: {stem[:48]!r}"
                )
    return errors


def _read_text(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: gagal dibaca sebagai UTF-8: {exc}")
        return ""


def validate_dataset_dir(data_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_text = _read_text(data_dir / "manifest.json", errors)
    manifest: dict = {}
    if manifest_text:
        try:
            manifest = json.loads(manifest_text)
        except json.JSONDecodeError as exc:
            errors.append(f"manifest.json: JSON tidak valid: {exc}")

    train_prompts: dict[str, set[str]] = {}
    train_counts: dict[str, int] = {}
    for category in CATEGORIES:
        text = _read_text(data_dir / f"{category}.txt", errors)
        if not text:
            train_counts[category] = 0
            train_prompts[category] = set()
            continue
        examples = validate_train_text(text, category, errors)
        train_counts[category] = len(examples)
        train_prompts[category] = {prompt for prompt, _ in examples}

    bench_prompts: dict[str, list[str]] = {}
    bench_counts: dict[str, int] = {}
    for category in CATEGORIES:
        text = _read_text(data_dir / "benchmark" / f"{category}.json", errors)
        if not text:
            bench_counts[category] = 0
            bench_prompts[category] = []
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"benchmark/{category}.json: JSON tidak valid: {exc}")
            data = {}
        if data.get("category") != category:
            errors.append(f"benchmark/{category}.json: 'category' tidak sesuai")
        if data.get("metric") != "exact_match":
            errors.append(f"benchmark/{category}.json: 'metric' bukan 'exact_match'")
        prompts = validate_benchmark_data(data, category, errors)
        bench_counts[category] = len(prompts)
        bench_prompts[category] = prompts

    errors.extend(check_leakage(train_prompts, bench_prompts))

    if manifest:
        splits = manifest.get("splits", {})
        for section, counts in (("train", train_counts), ("benchmark", bench_counts)):
            expected = splits.get(section, {})
            for category in CATEGORIES:
                if expected.get(category) != counts[category]:
                    errors.append(
                        f"manifest: jumlah {section}/{category} pada manifest "
                        f"({expected.get(category)}) tidak sama dengan isi berkas "
                        f"({counts[category]})"
                    )
        train_names = {f"{c}.txt" for c in CATEGORIES}
        bench_names = {f"benchmark/{c}.json" for c in CATEGORIES}
        for entry in manifest.get("files", []):
            rel = entry.get("path", "")
            text = _read_text(data_dir / rel, errors)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if entry.get("sha256") != digest:
                errors.append(f"manifest: sha256 tidak cocok untuk {rel}")
            actual = None
            if rel in train_names:
                actual = train_counts.get(Path(rel).stem)
            elif rel in bench_names:
                actual = bench_counts.get(Path(rel).stem)
            if actual is not None and entry.get("examples") != actual:
                errors.append(
                    f"manifest: jumlah contoh {rel} pada manifest "
                    f"({entry.get('examples')}) tidak sama dengan isi berkas ({actual})"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data/idbasic", help="direktori dataset idbasic"
    )
    args = parser.parse_args(argv)
    errors = validate_dataset_dir(Path(args.data_dir))
    if errors:
        print(f"[idbasic] validasi GAGAL: {len(errors)} masalah")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("[idbasic] validasi dataset lolos seluruh pemeriksaan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
