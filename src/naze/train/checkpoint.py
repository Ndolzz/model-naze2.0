"""Checkpoint v2 — params + checksum + metadata (M008-T002, REQ-102, D-018).

Struktur direktori: params.npz (params + __step/__epoch), params.npz.sha256
(checksum SHA-256 file), meta.json (step, epoch, created UTC, naze_version,
size_bytes, checksum_sha256, config JSON, config_hash).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _sha256_file(path: Path) -> str:
    """SHA-256 file, dibaca per chunk 1 MB (file besar tetap hemat memori)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _naze_version() -> str:
    import naze  # import lokal: hindari siklus saat naze/__init__ memuat train

    return naze.__version__


def save_checkpoint_v2(directory, model, *, step: int, epoch: int, config=None) -> dict:
    """Simpan params.npz + checksum + meta.json; return meta (M008-T002)."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "params.npz"
    np.savez(npz_path, **model.params(), __step=np.int64(step), __epoch=np.int64(epoch))
    checksum = _sha256_file(npz_path)
    (out / "params.npz.sha256").write_text(checksum + "\n", encoding="utf-8")
    config_json = config.to_json() if config is not None else None
    config_hash = config.config_hash() if config is not None else None
    meta = {
        "step": int(step),
        "epoch": int(epoch),
        "created": datetime.now(timezone.utc).isoformat(),
        "naze_version": _naze_version(),
        "size_bytes": npz_path.stat().st_size,
        "checksum_sha256": checksum,
        "config": config_json,
        "config_hash": config_hash,
    }
    (out / "meta.json").write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_checkpoint_v2(directory, model, *, verify: bool = True) -> dict:
    """Muat checkpoint v2; verifikasi checksum bila verify (M008-T002).

    Return {"step", "epoch", "meta"}. Checksum mismatch -> ValueError.
    """
    out = Path(directory)
    npz_path = out / "params.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"checkpoint tidak ditemukan: {npz_path}")
    if verify:
        expected = (out / "params.npz.sha256").read_text(encoding="utf-8").strip()
        if _sha256_file(npz_path) != expected:
            raise ValueError("checksum params.npz tidak cocok (korup/tampered)")
    data = np.load(npz_path)
    params = model.params()
    for name in params:
        if name not in data:
            raise KeyError(f"checkpoint tidak memuat param {name!r}")
        if data[name].shape != params[name].shape:
            raise ValueError(
                f"shape {name!r} tidak cocok: {data[name].shape} vs {params[name].shape}"
            )
    for name, p in params.items():
        p[...] = data[name]
    meta = {}
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"step": int(data["__step"]), "epoch": int(data["__epoch"]), "meta": meta}
