# API Reference  Naze 1.0

> **Status:** DRAFT  berdasarkan source code aktual (M-001..M-009 DONE)
> **Version:** 1.0.0 (target)
> **Source of truth:** `src/naze/`

---

## Package Structure

```
naze/
 __init__.py          # Root package, __version__
 core/               # Numerical core
    __init__.py      # Array, DType, seeded_rng, gradcheck
    numeric.py      # Numerical operations
    gradcheck.py     # Gradient check utilities
 token/              # Tokenizer
    __init__.py      # ByteTokenizer
    byte_tokenizer.py
 nn/                  # Neural engine
    __init__.py      # All layers and activations
    layers.py       # Layer, Linear, Sequential, etc.
    activations.py  # softmax, tanh, relu, etc.
    transformer.py  # Transformer components
 data/                # Dataset pipeline
    __init__.py      # TextWindows
    dataset.py      # Dataset utilities
 lm/                  # Language models
    __init__.py      # MLPLM, TransformerLM, cross_entropy, generate
    mlp_lm.py       # MLP Language Model
    transformer_lm.py
 train/               # Training system
    __init__.py      # All training utilities
    config.py       # TrainConfig
    checkpoint.py    # Checkpoint v2
    evaluate.py      # Evaluation
    loop.py         # Training loop
    runlog.py       # RunLog
    trainer.py      # SGDTrainer
 inference/            # Inference engine (M-009)
    __init__.py      # Public inference API
    batch.py        # InferenceBatch
    engine.py       # InferenceEngine
    benchmark.py    # Benchmark utilities
 export.py             # Model export
```

---

## Root Package: `naze`

### Attributes

```python
__version__: str = "0.0.2"  # Will be "1.0.0" for release
```

### Exports

```python
from naze import core, data, lm, nn, token, train

# Available via:
# - naze.core.*
# - naze.data.*
# - naze.lm.*
# - naze.nn.*
# - naze.token.*
# - naze.train.*
```

---

## Module: `naze.core`

Numerical core: array operations, random number generation, gradient checking.

### Classes

#### `Array`
Type alias for `numpy.ndarray`.

#### `DType`
Type alias for numpy dtype.

### Functions

#### `as_array(x, dtype=None) -> Array`
Convert input to numpy array.

**Parameters:**
- `x`: Input (list, tuple, array, scalar)
- `dtype`: Target dtype (optional)

**Returns:** `Array`

**Raises:** `ValueError` if conversion fails.

**Example:**
```python
from naze.core import as_array
arr = as_array([1, 2, 3])  # array([1, 2, 3])
```

---

#### `seeded_rng(seed: int) -> numpy.random.Generator`
Create a seeded random number generator.

**Parameters:**
- `seed`: Random seed (int)

**Returns:** `numpy.random.Generator` 

**Example:**
```python
from naze.core import seeded_rng
rng = seeded_rng(42)
x = rng.random(5)  # Deterministic
```

---

#### `numeric_grad(f, x, h=1e-5) -> Array`
Numerical gradient approximation.

**Parameters:**
- `f`: Function to differentiate
- `x`: Input array
- `h`: Step size (default: 1e-5)

**Returns:** Gradient array

**Example:**
```python
from naze.core import numeric_grad
import numpy as np
def f(x): return x**2
grad = numeric_grad(f, np.array([1.0, 2.0]))
```

---

#### `assert_close(a, b, tol=1e-10)`
Assert arrays are close (for testing).

**Parameters:**
- `a`, `b`: Arrays to compare
- `tol`: Tolerance (default: 1e-10)

**Raises:** `AssertionError` if not close.

---

#### `gradcheck(f, x, tol=1e-5)`
Gradient check: verify analytical gradient matches numerical.

**Parameters:**
- `f`: Function with `.forward()` and `.backward()`
- `x`: Input array
- `tol`: Tolerance (default: 1e-5)

**Raises:** `AssertionError` if gradient check fails.

---

## Module: `naze.token`

Byte-level tokenizer.

### Classes

#### `ByteTokenizer`
Byte-level tokenizer with fixed vocabulary of 256 (DECISION-010, DECISION-015).

**Attributes:**
- `vocab_size: int = 256` (fixed, read-only)

**Methods:**

##### `__init__(self)`
Create a new ByteTokenizer instance.

##### `encode(self, text: str) -> list[int]`
Encode text to token IDs.

**Parameters:**
- `text`: Input text string

**Returns:** List of token IDs (each in [0, 256))

**Example:**
```python
from naze.token import ByteTokenizer
tok = ByteTokenizer()
ids = tok.encode("hello world")  # [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
```

##### `decode(self, ids: list[int]) -> str`
Decode token IDs to text.

**Parameters:**
- `ids`: List of token IDs

**Returns:** Decoded text string

**Raises:** `ValueError` if any ID not in [0, 256)

**Example:**
```python
from naze.token import ByteTokenizer
tok = ByteTokenizer()
text = tok.decode([104, 101, 108, 108, 111])  # "hello"
```

---

## Module: `naze.data`

Dataset pipeline: sliding window, batching.

### Classes

#### `TextWindows`
Sliding window over token IDs for language modeling.

**Parameters (constructor):**
- `ids`: List of token IDs
- `block_size`: Context window size (int)
- `batch_size`: Batch size (int, default: 8)
- `seed`: Random seed for reproducibility (int, default: 0)

**Methods:**

##### `batches(self, seed=None) -> Iterator[tuple[Array, Array]]`
Generate (input, target) batches for next-token prediction.

**Parameters:**
- `seed`: Optional seed override

**Yields:** Tuples of (x, y) where x, y are (B, T) arrays

**Example:**
```python
from naze.token import ByteTokenizer
from naze.data import TextWindows

tok = ByteTokenizer()
ids = tok.encode("hello world hello world")
windows = TextWindows(ids, block_size=8, batch_size=2)
for x, y in windows.batches(seed=42):
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
```

##### `batches_pos(self, seed=None) -> Iterator[tuple[Array, Array]]`
Generate per-position batches (M-008).

**Note:** For full training system, use this method.

---

## Module: `naze.nn`

Neural network layers and activations.

### Submodules

#### `naze.nn.activations`
Activation functions with analytical gradients.

**Functions:**
- `softmax(x: Array) -> Array`
- `tanh(x: Array) -> Array`
- `relu(x: Array) -> Array`
- `sigmoid(x: Array) -> Array`

**Gradient functions:**
- `softmax_grad(dout: Array, out: Array) -> Array`
- `tanh_grad(dout: Array, out: Array) -> Array`
- `relu_grad(dout: Array, out: Array) -> Array`
- `sigmoid_grad(dout: Array, out: Array) -> Array`

---

#### `naze.nn.layers`
Neural network layers.

**Classes:**
- `Layer`: Base layer class
- `Linear`: Dense layer with weights and bias
- `Sequential`: Composition of layers
- `Embedding`: Embedding lookup table
- `LayerNorm`: Layer normalization
- `Activation`: Wrapper for activation functions

**Class: `Linear`**

```python
Linear(in_features: int, out_features: int, *, seed: int = 0)
```

Dense layer: `y = x @ W.T + b`

**Parameters:**
- `in_features`: Input dimension
- `out_features`: Output dimension
- `seed`: Random seed for weight initialization

**Attributes:**
- `W: Array` - Weight matrix (out_features, in_features)
- `b: Array` - Bias vector (out_features,)

**Methods:**
- `forward(x: Array) -> Array`: Forward pass
- `backward(dout: Array) -> Array`: Backward pass
- `params() -> dict[str, Array]`: Get parameters

---

**Class: `Sequential`**

```python
Sequential(layers: list[Layer], *, seed: int = 0)
```

Composition of layers.

**Methods:**
- `forward(x: Array) -> Array`
- `backward(dout: Array) -> Array`
- `params() -> dict[str, Array]`

---

#### `naze.nn.transformer`
Transformer components (DECISION-022: learned positional embedding).

**Classes:**
- `TransformerConfig`: Configuration dataclass
- `PositionalRepr`: Learned positional embeddings
- `CausalAttention`: Causal self-attention
- `MultiHeadAttention`: Multi-head attention
- `FeedForward`: Feed-forward network
- `TransformerBlock`: Single transformer block
- `TransformerModel`: Full transformer model
- `LanguageModelHead`: LM head (DECISION-015: vocab=256)

**Class: `TransformerConfig`**

```python
TransformerConfig(
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 128,
    max_sequence_length: int = 128,
    vocab_size: int = 256,
    seed: int = 0
)
```

**Note:** Config D-018: D=64, H=4, L=2, d_ff=128, T_max=128

---

**Class: `TransformerModel`**

```python
TransformerModel(config: TransformerConfig)
```

Decoder-only causal transformer.

**Methods:**
- `forward(x: Array) -> Array`: (B, T) -> (B, T, D)
- `backward(dout: Array) -> Array`
- `params() -> dict[str, Array]`

---

**Class: `LanguageModelHead`**

```python
LanguageModelHead(config: TransformerConfig, *, seed: int | None = None)
```

LM head: Linear layer from D to vocab_size (256).

---

## Module: `naze.lm`

Language models: MLP and Transformer.

### Functions

#### `cross_entropy(logits: Array, targets: Array) -> float`
Cross-entropy loss for next-token prediction.

**Parameters:**
- `logits`: Model output (B*T, vocab_size)
- `targets`: Target token IDs (B*T,)

**Returns:** Average cross-entropy loss (float)

**Example:**
```python
from naze.lm import cross_entropy
import numpy as np
logits = np.random.randn(10, 256)
targets = np.random.randint(0, 256, 10)
loss = cross_entropy(logits, targets)
```

---

#### `generate(model, prompt_ids: list[int], max_new: int, *, temperature=1.0, seed=0) -> list[int]`
Generate tokens autoregressively.

**Parameters:**
- `model`: Model with `.forward()` method
- `prompt_ids`: Starting token IDs
- `max_new`: Number of tokens to generate
- `temperature`: Sampling temperature (default: 1.0)
- `seed`: Random seed (default: 0)

**Returns:** List of generated token IDs

**Example:**
```python
from naze.lm import MLPLM, generate
model = MLPLM(vocab_size=256, block_size=16, d_embed=32, d_hidden=128, seed=0)
ids = generate(model, [1, 2, 3], max_new=10, temperature=0.8, seed=42)
```

---

#### `transformer_generate(model, prompt_ids, max_new, *, temperature=1.0, seed=0) -> list[int]`
Generate with TransformerLM.

---

### Classes

#### `MLPLM`
Bengio-style MLP language model (DECISION-012).

```python
MLPLM(vocab_size: int, block_size: int, d_embed: int, d_hidden: int, *, seed: int = 0)
```

**Methods:**
- `forward(x: Array) -> Array`: (B, T) -> (B, T, vocab_size)
- `backward(dout: Array) -> Array`
- `params() -> dict[str, Array]`

---

#### `TransformerLM`
Transformer language model.

```python
TransformerLM(config: TransformerConfig, *, seed: int | None = None)
```

**Attributes:**
- `config: TransformerConfig`
- `model: TransformerModel`
- `head: LanguageModelHead`

**Methods:**
- `forward(ids: Array) -> Array`: (B, T) -> (B, T, 256)
- `backward(dout: Array) -> Array`
- `params() -> dict[str, Array]`

---

## Module: `naze.train`

Training system (M-008).

### Classes

#### `TrainConfig`
Training configuration.

```python
TrainConfig(
    model: str = "transformer",  # or "mlp"
    corpus_path: str,
    epochs: int = 10,
    lr: float = 0.08,
    batch_size: int = 8,
    seed: int = 0,
    ...
)
```

---

#### `SGDTrainer`
SGD optimizer (DECISION-013).

```python
SGDTrainer(model, lr: float = 0.08)
```

**Methods:**
- `train_step(x: Array, y: Array) -> float`: Single training step, returns loss

---

#### `TrainingRun`
Full training loop with checkpointing (M-008).

```python
TrainingRun(config: TrainConfig, model, data: TextWindows)
```

**Methods:**
- `run() -> dict`: Run training, returns summary

---

#### `EvalResult`
Evaluation result.

**Attributes:**
- `mean_loss: float`
- `perplexity: float` (exp(mean_loss))
- `n_tokens: int`

---

### Functions

#### `save_checkpoint_v2(model, path: str, step: int, epoch: int, seed: int, meta: dict) -> dict`
Save checkpoint v2 with metadata and checksum (M-008).

#### `load_checkpoint_v2(path: str) -> tuple[dict, dict]`
Load checkpoint v2, returns (params, metadata).

#### `evaluate(model, data: TextWindows) -> EvalResult`
Evaluate model on dataset.

---

## Module: `naze.inference` (M-009)

Inference engine: forward-only, batching, benchmark.

### Classes

#### `InferenceBatch`
Batched input for inference.

```python
InferenceBatch(token_ids: list[list[int]], max_length: int, *, pad_id: int = 0)
```

**Attributes:**
- `x: Array` - Padded input (B, T)
- `mask: Array` - Valid token mask (B, T)
- `lengths: list[int]` - Original sequence lengths

---

#### `InferenceOutput`
Inference output dataclass.

**Attributes:**
- `logits: Array` - Model logits (B, T, 256)
- `probs: Array | None` - Softmax probabilities (optional)
- `top_k: int` - Top-k value used

---

#### `InferenceConfig`
Inference configuration.

```python
InferenceConfig(
    max_sequence_length: int = 128,
    pad_id: int = 0,
    return_probs: bool = False,
    temperature: float = 1.0,
    top_k: int = 5,
    seed: int = 0,
    warmup_runs: int = 3,
    benchmark_runs: int = 5
)
```

---

#### `InferenceEngine`
Forward-only inference engine.

```python
InferenceEngine(model)
```

**Methods:**
- `forward(batch: InferenceBatch, *, return_probs=False) -> InferenceOutput`
- `generate(prompt_ids: list[int], max_new: int, *, temperature=1.0, top_k=5, seed=0) -> list[int]`
- `slide_window(token_ids: list[int], window_size: int, *, overlap=0) -> Iterator[...]`

---

#### `BenchmarkResult`
Benchmark result dataclass.

**Attributes:**
- `tokens_per_second: float`
- `peak_rss_kb: int`
- `input_tokens: int`
- `output_tokens: int`
- `batch_size: int`
- `sequence_length: int`

---

### Functions

#### `benchmark_model(engine: InferenceEngine, batch_sizes, seq_lengths, *, warmup=3, runs=5) -> list[BenchmarkResult]`
Benchmark inference performance.

---

## Module: `naze.export` (M-009)

Model export for integration.

### Functions

#### `export_model_weights(model, path: str, *, version="1.0.0", float32=True) -> dict`
Export model weights to npz format.

**Parameters:**
- `model`: Model to export
- `path`: Output path (.npz)
- `version`: Model version (default: "1.0.0")
- `float32`: Convert to float32 (default: True)

**Returns:** Metadata dict with checksum

**Output format:**
- `params.npz`: float32 weights (D-018: ~0.43 MB)
- `__metadata__`: JSON string with version, config, checksum

---

#### `load_model_weights(path: str) -> tuple[dict, dict]`
Load model weights from npz.

**Returns:** (params, metadata)

---

## Complete Example: End-to-End

```python
import sys
sys.path.insert(0, 'src')

from naze.token import ByteTokenizer
from naze.data import TextWindows
from naze.lm import TransformerLM, TransformerConfig
from naze.train import SGDTrainer, TrainingRun, save_checkpoint_v2, load_checkpoint_v2
from naze.inference import InferenceEngine, InferenceBatch
from naze.export import export_model_weights

# 1. Tokenize
tok = ByteTokenizer()
text = "Hello world. This is Naze."
ids = tok.encode(text)

# 2. Create model
config = TransformerConfig(
    d_model=64, num_heads=4, num_layers=2,
    d_ff=128, max_sequence_length=128, vocab_size=256, seed=0
)
model = TransformerLM(config)

# 3. Training (simplified)
trainer = SGDTrainer(model, lr=0.08)
windows = TextWindows(ids, block_size=16, batch_size=2, seed=0)
for epoch in range(3):
    for x, y in windows.batches_pos(seed=epoch):
        loss = trainer.train_step(x, y)

# 4. Inference
engine = InferenceEngine(model)
prompt = tok.encode("Hello")
generated = engine.generate(prompt, max_new=10, temperature=0.8, seed=0)
output = tok.decode(generated)
print(f"Generated: {output}")

# 5. Export
export_model_weights(model, "naze_v1.npz", version="1.0.0", float32=True)

# 6. Benchmark
from naze.inference import benchmark_model
results = benchmark_model(engine, batch_sizes=[1, 8], seq_lengths=[16, 64])
for r in results:
    print(f"BS={r.batch_size} SL={r.sequence_length}: {r.tokens_per_second:.2f} tok/s")
```

---

## Notes

- **Deterministic:** All operations are deterministic per-seed (DECISION-007)
- **No pretrained:** Built from scratch (DECISION-004)
- **NumPy only:** No deep learning frameworks (DECISION-003)
- **float64 training:** Training uses float64 precision
- **float32 inference:** Inference can use float32 (for nazeio integration)
- **Vocab fixed 256:** Byte-level tokenizer (DECISION-010, DECISION-015)
- **Checkpoint <= 1 MB:** DECISION-018 config
