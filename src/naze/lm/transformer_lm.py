"""Integrasi pipeline LM Transformer (M007-T013, REQ-010/REQ-006/REQ-011).

TransformerLM: adapter yang menyusun komponen existing tanpa logika baru —
TransformerModel (T011) -> LanguageModelHead (T012): ids (B,T) -> logits
(B,T,256). Loss cross-entropy pada flatten (B*T, 256) dengan target sama di
semua posisi (TD §8: pendekatan default M-007, kompatibel pipeline MLPLM/
SGDTrainer; LM per-posisi penuh = M-008). backward: dlogits =
(p - onehot)/(B*T) -> head.backward -> model.backward; gradien dikumpulkan
menjadi dict datar nama -> array (pola MLPLM) sehingga SGDTrainer dapat
dipakai langsung. transformer_generate: autoregressive greedy/temperature
dengan konteks = token terakhir (<= max_sequence_length), mengikuti pola
naze.lm.mlp_lm.generate. MLPLM dan trainer tidak diubah (AC T013).
"""

from __future__ import annotations

import numpy as np

from naze.core.numeric import Array, seeded_rng
from naze.nn.activations import softmax
from naze.nn.transformer import LanguageModelHead, TransformerConfig, TransformerModel


class TransformerLM:
    """Language model Transformer: adapter model (T011) + head (T012).

    Seed per instance: default config.seed; kwarg seed untuk offset per
    komponen (pola Linear/MLPLM). Model memakai seed s; head memakai
    s+2+num_layers — kelanjutan skema T011 (embedding s, positional s+1,
    block ke-i s+2+i) sehingga head tidak bentrok dengan seed komponen
    model (pola T012). params()/backward() mengembalikan dict datar dengan
    referensi live (bukan salinan) agar mutasi SGDTrainer bekerja.
    """

    def __init__(self, config: TransformerConfig, *, seed: int | None = None) -> None:
        s = config.seed if seed is None else seed
        self.config = config
        self.model = TransformerModel(config, seed=s)
        self.head = LanguageModelHead(config, seed=s + 2 + config.num_layers)
        self._p: Array | None = None
        self._targets: Array | None = None
        self._shape: tuple[int, int] | None = None
        self._p_pos: Array | None = None
        self._rows_pos: Array | None = None
        self._pos_shape: tuple[int, int] | None = None

    def forward(self, ids: Array) -> Array:
        """ids (B,T) -> logits (B,T,256): model -> head (TD §5 18-21)."""
        return self.head.forward(self.model.forward(ids))

    def loss(self, logits: Array, targets: Array) -> float:
        """Cross-entropy flatten (B*T, 256); target sama di semua posisi (TD §8).

        Fail-fast (TD §10): logits harus (B,T,256); targets harus 1D (B,),
        integer, dan dalam rentang byte [0, 256). Cache softmax untuk backward
        (pola MLPLM.loss).
        """
        logits = np.asarray(logits)
        if logits.ndim != 3 or logits.shape[2] != self.config.vocab_size:
            raise ValueError(
                f"logits harus (batch, seq, {self.config.vocab_size}), dapat shape {logits.shape}"
            )
        targets = np.asarray(targets)
        if targets.ndim != 1:
            raise ValueError(f"targets harus 1D (batch,), dapat shape {targets.shape}")
        if targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"targets harus ({logits.shape[0]},) mengikuti batch, dapat {targets.shape[0]}"
            )
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError(f"target ID harus integer, dapat dtype {targets.dtype!r}")
        if targets.min() < 0 or targets.max() >= self.config.vocab_size:
            raise ValueError(
                f"target ID di luar rentang [0, {self.config.vocab_size}) "
                f"(min={targets.min()}, max={targets.max()})"
            )
        b, t = logits.shape[0], logits.shape[1]
        flat = logits.reshape(b * t, self.config.vocab_size)
        p = softmax(flat)
        rows = np.repeat(targets.astype(np.int64, copy=False), t)
        self._p = p
        self._targets = targets.astype(np.int64, copy=True)
        self._shape = (b, t)
        return float(-np.log(p[np.arange(b * t), rows] + 1e-12).mean())

    def backward(self) -> dict[str, Array]:
        """dlogits = (p - onehot)/(B*T) -> head -> model; dict datar (pola MLPLM)."""
        if self._p is None or self._targets is None or self._shape is None:
            raise RuntimeError("backward dipanggil sebelum loss")
        b, t = self._shape
        n = b * t
        dlogits = self._p.copy()
        dlogits[np.arange(n), np.repeat(self._targets, t)] -= 1.0
        dlogits /= n
        d_hidden = self.head.backward(dlogits.reshape(b, t, self.config.vocab_size))
        self.model.backward(d_hidden)
        return self._named("grads")

    def loss_pos(self, logits: Array, targets: Array) -> float:
        """Cross-entropy per-posisi: targets (B, T) beda tiap posisi (M-008).

        Fail-fast seperti loss(): logits (B,T,256); targets (B,T),
        integer, rentang [0, 256). Cache softmax per-posisi untuk
        backward_pos. loss/backward lama tidak berubah (aditif).
        """
        logits = np.asarray(logits)
        if logits.ndim != 3 or logits.shape[2] != self.config.vocab_size:
            raise ValueError(
                f"logits harus (batch, seq, {self.config.vocab_size}), dapat shape {logits.shape}"
            )
        targets = np.asarray(targets)
        if targets.shape != logits.shape[:2]:
            raise ValueError(
                f"targets harus {logits.shape[:2]} (batch, seq), dapat shape {targets.shape}"
            )
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError(f"target ID harus integer, dapat dtype {targets.dtype!r}")
        if targets.min() < 0 or targets.max() >= self.config.vocab_size:
            raise ValueError(
                f"target ID di luar rentang [0, {self.config.vocab_size}) "
                f"(min={targets.min()}, max={targets.max()})"
            )
        b, t = logits.shape[0], logits.shape[1]
        flat = logits.reshape(b * t, self.config.vocab_size)
        p = softmax(flat)
        rows = targets.astype(np.int64, copy=True).reshape(-1)
        self._p_pos = p
        self._rows_pos = rows
        self._pos_shape = (b, t)
        return float(-np.log(p[np.arange(b * t), rows] + 1e-12).mean())

    def backward_pos(self) -> dict[str, Array]:
        """dlogits per-posisi = (p - onehot)/(B*T) -> head -> model (M-008)."""
        if self._p_pos is None or self._rows_pos is None or self._pos_shape is None:
            raise RuntimeError("backward_pos dipanggil sebelum loss_pos")
        b, t = self._pos_shape
        n = b * t
        dlogits = self._p_pos.copy()
        dlogits[np.arange(n), self._rows_pos] -= 1.0
        dlogits /= n
        d_hidden = self.head.backward(dlogits.reshape(b, t, self.config.vocab_size))
        self.model.backward(d_hidden)
        return self._named("grads")

    def params(self) -> dict[str, Array]:
        """Dict datar nama -> parameter (referensi live; pola MLPLM)."""
        return self._named("params")

    def _named(self, attr: str) -> dict[str, Array]:
        """Kumpulkan params/grads seluruh sub-komponen jadi dict datar.

        attr = "params" | "grads". Nama mengikuti struktur model: emb.E,
        pos.P, blk{i}.attn.{q,k,v,o}.{W,b}, blk{i}.attn.norm.{gamma,beta},
        blk{i}.ffn.fc{1,2}.{W,b}, blk{i}.ffn.norm.{gamma,beta},
        final.{gamma,beta}, head.{W,b} — tanpa duplikat (pola Sequential).
        """
        def pick(layer, name: str) -> Array:
            return getattr(layer, attr)[name]

        out: dict[str, Array] = {
            "emb.E": pick(self.model.embedding, "E"),
            "pos.P": pick(self.model.positional, "P"),
        }
        for i, block in enumerate(self.model.blocks):
            attn = block.attention.branch
            out[f"blk{i}.attn.q.W"] = pick(attn.qkv.q_proj, "W")
            out[f"blk{i}.attn.q.b"] = pick(attn.qkv.q_proj, "b")
            out[f"blk{i}.attn.k.W"] = pick(attn.qkv.k_proj, "W")
            out[f"blk{i}.attn.k.b"] = pick(attn.qkv.k_proj, "b")
            out[f"blk{i}.attn.v.W"] = pick(attn.qkv.v_proj, "W")
            out[f"blk{i}.attn.v.b"] = pick(attn.qkv.v_proj, "b")
            out[f"blk{i}.attn.o.W"] = pick(attn.out_proj, "W")
            out[f"blk{i}.attn.o.b"] = pick(attn.out_proj, "b")
            out[f"blk{i}.attn.norm.gamma"] = pick(block.attention.norm, "gamma")
            out[f"blk{i}.attn.norm.beta"] = pick(block.attention.norm, "beta")
            ffn = block.ffn.branch
            out[f"blk{i}.ffn.fc1.W"] = pick(ffn.fc1, "W")
            out[f"blk{i}.ffn.fc1.b"] = pick(ffn.fc1, "b")
            out[f"blk{i}.ffn.fc2.W"] = pick(ffn.fc2, "W")
            out[f"blk{i}.ffn.fc2.b"] = pick(ffn.fc2, "b")
            out[f"blk{i}.ffn.norm.gamma"] = pick(block.ffn.norm, "gamma")
            out[f"blk{i}.ffn.norm.beta"] = pick(block.ffn.norm, "beta")
        out["final.gamma"] = pick(self.model.final_norm, "gamma")
        out["final.beta"] = pick(self.model.final_norm, "beta")
        out["head.W"] = pick(self.head.proj, "W")
        out["head.b"] = pick(self.head.proj, "b")
        return out


def transformer_generate(
    model: TransformerLM,
    prompt_ids: list[int],
    n_new: int,
    *,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[int]:
    """Autoregressive next-token: greedy (temperature <= 0) atau sampling.

    Pola naze.lm.mlp_lm.generate: konteks = token terakhir (maksimal
    max_sequence_length — jendela geser kausal, tanpa padding); logits
    diambil dari posisi terakhir; temperature <= 0 berarti greedy argmax.
    Deterministik per-seed (REQ-101). Prompt minimal 1 token.
    """
    if len(prompt_ids) < 1:
        raise ValueError(f"prompt minimal 1 token, dapat {len(prompt_ids)}")
    rng = seeded_rng(seed)
    ids = list(prompt_ids)
    for _ in range(n_new):
        ctx = np.array([ids[-model.config.max_sequence_length:]], dtype=np.int64)
        logits = model.forward(ctx)[0][-1]  # posisi terakhir -> (256,)
        if temperature <= 0.0:
            nxt = int(np.argmax(logits))
        else:
            probs = softmax(logits / temperature)
            nxt = int(rng.choice(model.config.vocab_size, p=probs))
        ids.append(nxt)
    return ids[len(prompt_ids):]
