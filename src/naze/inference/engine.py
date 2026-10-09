"""InferenceEngine: forward-only wrapper untuk MLPLM/TransformerLM (M-009-T002..T004/T007..T008, REQ-008).

InferenceEngine menyediakan:
- forward: batch -> logits (B,T,256)
- generate: autoregressive (greedy/sampling)
- sliding window: untuk sequence panjang
- InferenceConfig: konfigurasi inference
- InferenceOutput: hasil forward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import numpy as np

from naze.core.numeric import Array, seeded_rng
from naze.nn.activations import softmax

if TYPE_CHECKING:
    from naze.lm.mlp_lm import MLPLM
    from naze.lm.transformer_lm import TransformerLM


@dataclass(frozen=True)
class InferenceOutput:
    """Hasil forward model (M-009-T002).

    Atribut:
        logits: (B, T, 256) float64, output model.
        probs: (B, T, 256) float64, optional (softmax logits).
        top_k: int, default 5 (jumlah token teratas yang disimpan).
    """

    logits: Array
    probs: Array | None = None
    top_k: int = 5


@dataclass(frozen=True)
class InferenceConfig:
    """Konfigurasi inference (M-009-T008).

    Atribut:
        max_sequence_length: Panjang sequence maksimum (default 128, DECISION-018).
        pad_id: ID token untuk padding (default 0).
        return_probs: Apakah menghitung probs (softmax) (default False).
        temperature: Suhu untuk sampling (default 1.0).
        top_k: Jumlah token teratas (default 5).
        seed: Seed untuk RNG (default 0).
        warmup_runs: Jumlah warmup runs untuk benchmark (default 3).
        benchmark_runs: Jumlah runs untuk benchmark (default 5).
    """

    max_sequence_length: int = 128
    pad_id: int = 0
    return_probs: bool = False
    temperature: float = 1.0
    top_k: int = 5
    seed: int = 0
    warmup_runs: int = 3
    benchmark_runs: int = 5

    def __post_init__(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length harus > 0")
        if self.pad_id < 0 or self.pad_id >= 256:
            raise ValueError("pad_id harus di [0, 256)")
        if self.temperature < 0:
            raise ValueError("temperature harus >= 0")
        if self.top_k <= 0:
            raise ValueError("top_k harus > 0")
        if self.seed < 0:
            raise ValueError("seed harus >= 0")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs harus >= 0")
        if self.benchmark_runs < 1:
            raise ValueError("benchmark_runs harus >= 1")


class InferenceEngine:
    """Forward-only wrapper untuk MLPLM/TransformerLM (M-009-T003).

    InferenceEngine menyediakan interface inference yang seragam untuk
    MLPLM dan TransformerLM, dengan dukungan batching dan sliding window.
    """

    def __init__(
        self,
        model: MLPLM | TransformerLM,
        *,
        config: InferenceConfig | None = None,
    ) -> None:
        """Inisialisasi InferenceEngine.

        Args:
            model: Model LM (MLPLM atau TransformerLM).
            config: Konfigurasi inference (default InferenceConfig()).
        """
        self.model = model
        self.config = config if config is not None else InferenceConfig()

    def forward(
        self,
        batch: "InferenceBatch",
        *,
        return_probs: bool | None = None,
    ) -> InferenceOutput:
        """Forward pass untuk batch (M-009-T003).

        Args:
            batch: InferenceBatch dengan token IDs.
            return_probs: Apakah menghitung probs (default dari config).

        Returns:
            InferenceOutput dengan logits dan (optional) probs.
        """
        from naze.inference.batch import InferenceBatch

        if not isinstance(batch, InferenceBatch):
            raise TypeError(f"batch harus InferenceBatch, dapat {type(batch).__name__}")

        use_probs = (
            return_probs if return_probs is not None else self.config.return_probs
        )

        # Get model's expected block_size
        try:
            model_block_size = self.model.block_size
        except AttributeError:
            # TransformerLM doesn't have block_size, use max_sequence_length
            model_block_size = self.config.max_sequence_length

        # Potong atau pad input ke model_block_size
        # Untuk MLPLM: input harus (B, block_size)
        # Untuk TransformerLM: input bisa (B, T) dengan T <= max_sequence_length
        x = batch.x
        if hasattr(self.model, 'block_size'):
            # MLPLM: harus tepat block_size
            if x.shape[1] != model_block_size:
                # Potong ke block_size
                x = x[:, :model_block_size]
                if x.shape[1] < model_block_size:
                    # Pad dengan pad_id
                    pad_width = model_block_size - x.shape[1]
                    x = np.pad(x, ((0, 0), (0, pad_width)), constant_values=self.config.pad_id)

        # Forward model
        logits = self.model.forward(x)

        # Hitung probs jika diminta
        probs = None
        if use_probs:
            probs = softmax(logits)

        return InferenceOutput(
            logits=logits,
            probs=probs,
            top_k=self.config.top_k,
        )

    def generate(
        self,
        prompt_ids: list[int],
        max_new: int,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> list[int]:
        """Generate token baru secara autoregressive (M-009-T004).

        Args:
            prompt_ids: Token IDs prompt (minimal 1 token).
            max_new: Jumlah token baru yang akan digenerate.
            temperature: Suhu untuk sampling (default dari config).
            top_k: Jumlah token teratas (default dari config).
            seed: Seed untuk RNG (default dari config).

        Returns:
            List token IDs baru (panjang = max_new).
        """
        from naze.lm.mlp_lm import generate as mlp_generate
        from naze.lm.transformer_lm import transformer_generate

        if len(prompt_ids) < 1:
            raise ValueError(f"prompt_ids minimal 1 token, dapat {len(prompt_ids)}")
        if max_new <= 0:
            raise ValueError(f"max_new harus > 0, dapat {max_new}")

        use_temp = temperature if temperature is not None else self.config.temperature
        use_topk = top_k if top_k is not None else self.config.top_k
        use_seed = seed if seed is not None else self.config.seed

        # Get model's block_size
        try:
            model_block_size = self.model.block_size
        except AttributeError:
            model_block_size = self.config.max_sequence_length

        # Pad prompt ke model_block_size jika diperlukan (untuk MLPLM)
        if hasattr(self.model, 'block_size'):
            if len(prompt_ids) < model_block_size:
                # Pad dengan pad_id
                padded = prompt_ids + [self.config.pad_id] * (model_block_size - len(prompt_ids))
                prompt_ids = padded

        # Gunakan transformer_generate jika model TransformerLM
        from naze.lm.transformer_lm import TransformerLM

        if isinstance(self.model, TransformerLM):
            new_ids = transformer_generate(
                self.model,
                prompt_ids,
                max_new,
                temperature=use_temp,
                seed=use_seed,
            )
        else:
            # MLPLM
            new_ids = mlp_generate(
                self.model,
                prompt_ids,
                max_new,
                temperature=use_temp,
                seed=use_seed,
            )

        return new_ids

    def slide_window(
        self,
        token_ids: list[int],
        window_size: int | None = None,
        *,
        overlap: int = 0,
    ) -> Iterator[tuple[Array, int, int]]:
        """Sliding window untuk sequence panjang (M-009-T007).

        Args:
            token_ids: Token IDs input (panjang > window_size).
            window_size: Ukuran jendela (default max_sequence_length).
            overlap: Jumlah token overlap antar window (default 0).

        Yields:
            (chunk, start_pos, end_pos) untuk tiap window.
        """
        ws = window_size if window_size is not None else self.config.max_sequence_length
        if ws <= 0:
            raise ValueError(f"window_size harus > 0, dapat {ws}")
        if overlap < 0 or overlap >= ws:
            raise ValueError(f"overlap harus di [0, window_size), dapat {overlap}")

        n = len(token_ids)
        if n <= ws:
            yield np.array([token_ids], dtype=np.int64), 0, n
            return

        stride = ws - overlap
        for start in range(0, n, stride):
            end = min(start + ws, n)
            chunk = np.array([token_ids[start:end]], dtype=np.int64)
            yield chunk, start, end

    def __repr__(self) -> str:
        model_type = type(self.model).__name__
        return f"InferenceEngine(model={model_type}, config={self.config})"
