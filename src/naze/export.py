"""Ekspor bobot model ke npz float32 (M-009-T017, untuk integrasi nazeio).

Skrip ini mengekspor parameter model (TransformerLM/MLPLM) ke format npz
untuk digunakan di aplikasi Android (nazeio). Bobot dikonversi ke float32
untuk menghemat memori (0.43 MB vs 0.87 MB untuk config D-018).

Metadata disimpan sebagai kunci tambahan di npz:
- d_model, num_heads, num_layers, d_ff, max_sequence_length
- vocab_size (256 fixed)
- version: versi bobot
- checksum: SHA-256 parameter
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np


def export_model_weights(
    model,
    path: str,
    *,
    version: str = "0.0.1",
    float32: bool = True,
) -> dict:
    """Ekspor parameter model ke npz float32.

    Args:
        model: Model (TransformerLM atau MLPLM) dengan method params().
        path: Path file output (.npz).
        version: Versi bobot (default "0.0.1").
        float32: Konversi ke float32 (default True).

    Returns:
        Metadata yang disimpan.
    """
    params = model.params()

    # Konversi ke float32 jika diminta
    export_params = {}
    for name, array in params.items():
        if float32 and np.issubdtype(array.dtype, np.floating):
            export_params[name] = array.astype(np.float32)
        else:
            export_params[name] = array

    # Metadata
    metadata = {
        "version": version,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "float32": float32,
        "dtype": "float32" if float32 else "float64",
    }

    # Ambil konfigurasi model jika TransformerLM
    try:
        config = model.config
        metadata.update({
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "d_ff": config.d_ff,
            "max_sequence_length": config.max_sequence_length,
            "vocab_size": config.vocab_size,
        })
    except AttributeError:
        # MLPLM
        metadata["model_type"] = "MLPLM"

    # Checksum SHA-256 parameter
    checksum = hashlib.sha256()
    for name in sorted(export_params.keys()):
        array = export_params[name]
        checksum.update(array.tobytes())
    metadata["checksum_sha256"] = checksum.hexdigest()

    # Simpan ke npz
    np.savez(path, **export_params, **{"__metadata__": json.dumps(metadata, sort_keys=True)})

    return metadata


def load_model_weights(path: str) -> tuple[dict, dict]:
    """Muat parameter model dari npz.

    Args:
        path: Path file npz.

    Returns:
        (params, metadata): parameter dan metadata.
    """
    data = np.load(path)
    params = {k: v for k, v in data.items() if not k.startswith("__")}
    metadata = json.loads(data["__metadata__"].item())
    return params, metadata


def export_reference(
    model,
    prompt_ids: list[int],
    path: str,
    *,
    max_new: int = 10,
    temperature: float = 0.0,  # greedy
    seed: int = 0,
) -> dict:
    """Ekspor referensi logits untuk uji kesetaraan Kotlin.

    Args:
        model: Model (TransformerLM atau MLPLM).
        prompt_ids: Token IDs prompt.
        path: Path file output (.json).
        max_new: Jumlah token yang digenerate.
        temperature: Suhu (0.0 = greedy).
        seed: Seed untuk RNG.

    Returns:
        Dict dengan input, logits, dan output referensi.
    """
    from naze.core.numeric import as_array

    # Forward untuk prompt
    x = np.array([prompt_ids], dtype=np.int64)
    logits = model.forward(x)

    # Generate
    from naze.lm.mlp_lm import generate as mlp_generate
    from naze.lm.transformer_lm import transformer_generate, TransformerLM

    if isinstance(model, TransformerLM):
        new_ids = transformer_generate(model, prompt_ids, max_new, temperature=temperature, seed=seed)
    else:
        new_ids = mlp_generate(model, prompt_ids, max_new, temperature=temperature, seed=seed)

    reference = {
        "input_ids": prompt_ids,
        "logits_shape": logits.shape,
        "logits_dtype": str(logits.dtype),
        "generated_ids": new_ids,
        "model_type": type(model).__name__,
        "temperature": temperature,
        "seed": seed,
    }

    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, sort_keys=True, ensure_ascii=False)

    return reference


if __name__ == "__main__":
    # Contoh penggunaan (untuk pengujian)
    import sys
    sys.path.insert(0, "src")

    from naze.lm.mlp_lm import MLPLM
    from naze.token import ByteTokenizer

    # Buat model MLPLM kecil
    tok = ByteTokenizer()
    model = MLPLM(
        vocab_size=tok.vocab_size,
        block_size=16,
        d_embed=8,
        d_hidden=16,
        seed=0,
    )

    # Ekspor bobot
    metadata = export_model_weights(model, "test_weights.npz", version="0.0.1-test")
    print(f"Ekspor selesai: {metadata}")

    # Ekspor referensi
    export_reference(model, [1, 2, 3, 4], "test_reference.json")
    print("Referensi selesai")
