"""First language model: MLP over context window (REQ-010, Stage 5, DECISION-110).

Resolusi OPEN DECISION-104: arsitektur Bengio-style MLP LM.
  ids (B, T) -> embedding lookup -> flatten (B, T*d) -> Linear -> tanh -> Linear -> logits (B, V)
Loss: cross-entropy (softmax digabung agar stabil).
Backward: dlogits = (p - onehot)/B, lalu backprop per-layer, scatter-add ke embedding.
Model kecil (parameter ~ ratusan ribu) sesuai REQ-102 hardware terbatas.
"""

from __future__ import annotations

import numpy as np

from naze.core.numeric import Array, as_array, seeded_rng
from naze.nn.activations import softmax, tanh, tanh_grad
from naze.nn.layers import Activation, Linear


def cross_entropy(logits: Array, targets: np.ndarray) -> float:
    p = softmax(logits)
    n = len(targets)
    return float(-np.log(p[np.arange(n), targets] + 1e-12).mean())


class MLPLM:
    def __init__(self, vocab_size: int, block_size: int, d_embed: int, d_hidden: int, *, seed: int = 0) -> None:
        if vocab_size <= 0 or block_size <= 0 or d_embed <= 0 or d_hidden <= 0:
            raise ValueError("semua dimensi harus > 0")
        rng = seeded_rng(seed)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_embed = d_embed
        self.E = rng.normal(0.0, 0.1, (vocab_size, d_embed))
        self.fc1 = Linear(block_size * d_embed, d_hidden, seed=seed + 1)
        self.act = Activation(tanh, tanh_grad, "tanh")
        self.fc2 = Linear(d_hidden, vocab_size, seed=seed + 2)
        self._cache: dict[str, np.ndarray] = {}

    def forward(self, ids: np.ndarray) -> Array:
        """ids (B, T) int -> logits (B, vocab)."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[1] != self.block_size:
            raise ValueError(f"input harus (batch, {self.block_size}), dapat {ids.shape}")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("token ID di luar vocab")
        x = self.E[ids].reshape(ids.shape[0], -1)
        self._cache["ids"] = ids
        self._cache["x"] = x
        h = self.fc1(x)
        a = self.act(h)
        return self.fc2(a)

    def loss(self, logits: Array, targets: np.ndarray) -> float:
        self._cache["p"] = softmax(logits)
        self._cache["targets"] = np.asarray(targets, dtype=np.int64)
        return cross_entropy(logits, self._cache["targets"])

    def backward(self) -> dict[str, Array]:
        """Hitung gradien; return dict param-name -> grad (dipakai optimizer)."""
        p = self._cache["p"]
        targets = self._cache["targets"]
        B = p.shape[0]
        dlogits = p.copy()
        dlogits[np.arange(B), targets] -= 1.0
        dlogits /= B
        da = self.fc2.backward(dlogits)
        dh = self.act.backward(da)
        dx = self.fc1.backward(dh)
        dE = np.zeros_like(self.E)
        np.add.at(dE, self._cache["ids"], dx.reshape(B, self.block_size, self.d_embed))
        return self._collect_grads(dE)

    def params(self) -> dict[str, Array]:
        return {
            "E": self.E,
            "fc1.W": self.fc1.params["W"], "fc1.b": self.fc1.params["b"],
            "fc2.W": self.fc2.params["W"], "fc2.b": self.fc2.params["b"],
        }

    def _collect_grads(self, dE: Array) -> dict[str, Array]:
        return {
            "E": dE,
            "fc1.W": self.fc1.grads["W"], "fc1.b": self.fc1.grads["b"],
            "fc2.W": self.fc2.grads["W"], "fc2.b": self.fc2.grads["b"],
        }


def generate(model: MLPLM, prompt_ids: list[int], n_new: int, *, temperature: float = 1.0, seed: int = 0) -> list[int]:
    """Autoregressive next-token sampling. Temperature 0 = greedy."""
    if len(prompt_ids) < model.block_size:
        raise ValueError(f"prompt minimal {model.block_size} token")
    rng = seeded_rng(seed)
    ids = list(prompt_ids)
    for _ in range(n_new):
        ctx = np.array([ids[-model.block_size:]], dtype=np.int64)
        logits = model.forward(ctx)[0]
        if temperature <= 0.0:
            nxt = int(np.argmax(logits))
        else:
            probs = softmax(logits / temperature)
            nxt = int(rng.choice(model.vocab_size, p=probs))
        ids.append(nxt)
    return ids[len(prompt_ids):]
