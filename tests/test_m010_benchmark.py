"""Test M-010 (T007): dataset benchmark command + validasi skrip/workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS_PATH = ROOT / "data" / "benchmark" / "commands.json"
SCRIPTS = [
    ROOT / "scripts" / "train_final.py",
    ROOT / "scripts" / "benchmark_nazeio.py",
    ROOT / "scripts" / "benchmark_armv7.py",
]


def test_commands_json_schema() -> None:
    data = json.loads(COMMANDS_PATH.read_text(encoding="utf-8"))
    assert data["metric"] == "exact_match"
    assert data["target_accuracy_percent"] == 90  # DECISION-023
    commands = data["commands"]
    assert len(commands) >= 20
    for cmd in commands:
        for key in ("id", "prompt", "expected_output", "category"):
            assert isinstance(cmd[key], str) and cmd[key]
        assert cmd["prompt"].startswith("perintah: ")  # format korpus perintah
        assert cmd["prompt"].endswith("\nintent:")
        assert cmd["expected_output"].startswith(" ")
    assert len({c["id"] for c in commands}) == len(commands)  # ID unik


def test_scripts_parse() -> None:
    for script in SCRIPTS:
        ast.parse(script.read_text(encoding="utf-8"))  # valid sintaks Python


def test_workflows() -> None:
    tests_yml = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "--cov-fail-under=80" in tests_yml  # D-019
    release_yml = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "python -m build" in release_yml  # build sdist+wheel (T017)
    assert "tags" in release_yml
