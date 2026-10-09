"""Run log JSONL + metrik resource (M008-T004, REQ-103, D-021)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def measure_peak_rss() -> int | None:
    """Peak RSS proses dalam KB (ru_maxrss); None bila platform tak dukung."""
    try:
        import resource
    except ImportError:
        return None
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except AttributeError:
        return None


def checkpoint_size(path) -> int:
    """Ukuran file checkpoint dalam byte (metrik D-021)."""
    return os.path.getsize(path)


class RunLog:
    """Appender JSONL: satu record JSON per baris, sort_keys, flush/record."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "RunLog":
        self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: dict) -> None:
        """Tulis satu record (dict) sebagai baris JSON; flush segera."""
        if self._handle is None:
            raise RuntimeError("RunLog harus dipakai sebagai context manager")
        self._handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._handle.flush()

    def records(self) -> list[dict]:
        """Baca seluruh record JSONL (untuk verifikasi/analisis pasca-run)."""
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
