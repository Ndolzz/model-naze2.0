# M007 TECHNICAL DESIGN — Transformer (Stage 6)

**Milestone:** M-007 — Transformer
**Stage:** Stage 6
**Status:** DRAFT — awaiting project owner approval
**Version:** 0.1.0
**Source of truth:** PROJECT_SPEC v0.4.0, REQUIREMENTS v0.4.0 (REQ-011), DECISION-009/010/015/016/017, ARCHITECTURE Stage 6
**Notation:** B = batch size · T = sequence length · D = model dimension · H = number of attention heads · Dh = head dimension — **invariant: D = H × Dh**

---

## 1. Purpose

Mendefinisikan desain teknis implementasi Transformer Naze dari nol (Stage 6, REQ-011) — dari token embedding hingga logits — menggunakan Python + NumPy dan engine internal Naze yang sudah ada (naze.nn, naze.core). Dokumen ini adalah acuan tunggal implementasi M-007 agar developer berikutnya tidak perlu menebak desain fundamental.

## 2. Scope

- Komponen Transformer decoder-only (causal language model): embedding, positional representation, scaled dot-product attention dengan causal mask, multi-head attention, output projection, residual + LayerNorm, FFN, stacking block, final norm, language-model head → logits (B, T, 256).
- Integrasi dengan pipeline yang sudah ada: ByteTokenizer (vocab 256), TextWindows, MLPLM loss (softmax-CE).
- Konfigurasi model, unit/numerical/integration/regression test plan.
- Forward pass. Backward pass disertakan hanya sejauh engine per-layer (DECISION-009) menopangnya — lihat §5 & §14.

## 3. Non-goals

- Tidak membuat training loop baru / optimizer baru (Stage 7 penuh = M-008).
- Tidak membuat inference engine production / benchmark (Stage 8 = M-009).
- Tidak menetapkan ukuran model final Naze (OD-118 terbuka; konfigurasi §9 adalah konfigurasi dev/test yang dapat diskalakan).
- Tidak membuat tokenizer baru (byte-level tetap, DECISION-010/015).
- Tidak membuat attention efisien untuk konteks panjang (ditangguhkan — tercatat ARCHITECTURE Stage 6).
- Tidak menetapkan ambang loss universal (DECISION-016).
- Tidak menambah dropout bila tidak ada kebutuhan terukur pada milestone ini (keputusan di §9; menambah fitur sebelum dibutuhkan = melanggar REQ-302).

## 4. Requirements

| Req | Relevan untuk M-007 |
|---|---|
| REQ-011 Transformer | Inti milestone: unit test per komponen; dimensi configurable & scalable turun |
| REQ-010 / REQ-006 (parsial) | Terhubung ke pipeline LM yang ada: loss softmax-CE, dataset TextWindows |
| REQ-004 | Input token ID dari ByteTokenizer, vocab 256 fixed (DECISION-015) |
| REQ-003 | Backward per-layer via engine (DECISION-009) sejauh dibutuhkan AC |
| REQ-101 | Deterministik per-seed (seeded_rng) |
| REQ-102 | Resource constraints didokumentasikan (§12) |
| REQ-201/202/203 | Tanpa pretrained / API LLM / framework DL — permanen |
| REQ-301/302 | Spec-driven; no overengineering |

## 5. Architecture

Decoder-only causal Transformer (komponen 1–21 dari instruksi, dengan shape di setiap tahap):

```
input_ids: (B, T)                                  # int64, rentang [0, 256)
  ↓ 1. Token Embedding     E: (256, D)             # lookup → (B, T, D)
  ↓ 2. Positional Representation + input           # → (B, T, D)
  ↓ --- TransformerBlock × num_layers ---
  │   3-4. Q/K/V projection (Linear D→D)           # → Q,K,V: (B, T, D) → reshape (B, H, T, Dh)
  │   5-7. Attention scores QKᵀ/√Dh + causal mask  # scores: (B, H, T, T); masked → -inf atas diagonal
  │   8-9. Softmax (stabil) per baris              # weights: (B, H, T, T)
  │   10. weights @ V                               # context: (B, H, T, Dh)
  │   11-12. Head concat → (B, T, D)
  │   13. Output projection W_O: (D, D)            # attention_out: (B, T, D)
  │   14-15. Residual + LayerNorm                  # → (B, T, D)
  │   16. FFN: Linear D→d_ff, aktivasinya (lihat §9/Risiko), Linear d_ff→D
  │   14-15. Residual + LayerNorm (kedua)          # → (B, T, D)
  ↓ --- akhir blok ---
  ↓ 19. Final LayerNorm (jika diperlukan)          # → (B, T, D)
  ↓ 20. Language-model head (Linear D→256, weight-tied opsional, lihat §9)
  ↓ 21. logits: (B, T, 256)
```

**Causality (kunci, §13):** mask kausal meng-nol-kan bobot posisi > t, sehingga prediksi pada posisi t hanya melihat token ≤ t (softmax pada skor yang di-mask). Implementasi: set skor (t_query > t_key) menjadi −∞ sebelum softmax; softmax stabil (shift-max) sudah ada di naze.nn.activations.

**Aturan engine:** semua operasi dibangun dari Linear, Activation, operasi NumPy di atas float64 (DECISION-007), randomness hanya via seeded_rng. Tidak ada PyTorch/TensorFlow/JAX/HF/Transformer siap pakai (REQ-201/202/203).

## 6. Module Boundaries

Struktur mengikuti repo yang ada — tidak membuat folder baru demi kompleksitas:

| Module | Responsibility | Public interface | Dependency | Input | Output |
|---|---|---|---|---|---|
| naze/nn/layers.py (+ modul baru naze/nn/transformer.py) | Komponen Transformer (Embedding, LayerNorm, Attention, FFN, block, model) | kelas di §7 | naze.core.numeric | ndarray | ndarray |
| naze/nn/transformer.py (baru) | TransformerModel + config | TransformerModel, TransformerConfig | naze.nn, naze.core | ids (B,T) | logits (B,T,256) |
| naze/lm/ | Loss softmax-CE & generate (dipakai ulang; tidak diubah kecuali perlu adapter) | cross_entropy, generate | naze.nn | logits/target | loss / ids |
| naze/token/, naze/data/, naze/train/ | Tidak berubah di M-007 | — | — | — | — |
| tests/ | test plan §11 | — | semua modul | — | — |

Boundary rule: transformer.py tidak mengimpor dari naze.train/naze.lm (model tidak tahu training); arah dependensi: train/lm → nn → core.

## 7. Interfaces

Kontrak publik (signature, bukan implementasi):

```
TransformerConfig: dataclass
  vocab_size=256 (FIXED — DECISION-015; validasi menolak ≠ 256)
  d_model: int; num_heads: int; num_layers: int; d_ff: int
  max_sequence_length: int; seed: int; dtype=float64 (DECISION-007)
  + invariant check: d_model == num_heads * Dh (Dh dihitung = d_model // num_heads)

Embedding.forward(ids: (B,T) int64) -> (B,T,D)          # lookup; backward = scatter-add
PositionalRepr.forward(x: (B,T,D)) -> (B,T,D)           # skema ditentukan di §9/OD
LinearProjection (Q/K/V): memakai naze.nn.Linear, input (B,T,D) → (B,T,D)
Attention.forward(q,k,v: (B,H,T,Dh), mask) -> (B,H,T,Dh) # scores = q @ k^T / sqrt(Dh); softmax(masked)
MultiHeadAttention.forward(x: (B,T,D)) -> (B,T,D)      # Q/K/V/O projections + Attention + concat
LayerNorm.forward(x: (B,T,D)) -> (B,T,D)                # normalisasi per posisi, params gamma/beta
FeedForward.forward(x: (B,T,D)) -> (B,T,D)              # Linear(D,d_ff) → akt → Linear(d_ff,D)
TransformerBlock.forward(x: (B,T,D)) -> (B,T,D)         # MHA + residual+LN; FFN + residual+LN
TransformerModel.forward(ids: (B,T)) -> logits (B,T,256) # embedding → pos → blocks → final LN → head
LanguageModelHead.forward(x: (B,T,D)) -> (B,T,256)      # Linear D→256 (opsional weight-tied ke embedding)
```

Setiap komponen Layer-like mengikuti kontrak engine (DECISION-009): forward meng-cache input; backward(grad_out) menghitung grads — kecuali komponen yang AC M-007 hanya menuntut forward (lihat §11 & §14 catatan backward).

## 8. Data Flow

1. Teks → ByteTokenizer.encode → ids: (B, T) (validasi: 0 ≤ id < 256, T ≤ max_sequence_length).
2. ids → embedding lookup (B, T, D) → + positional → (B, T, D).
3. Per block: MHA (Q/K/V → scores → causal mask → softmax → context → concat → proj) → residual+LN → FFN → residual+LN.
4. Ulangi num_layers × (output tiap blok (B, T, D) — invarian).
5. Final LayerNorm → LM head → logits (B, T, 256).
6. Loss: reshuffle ke (B*T, 256) vs target (B*T,) → cross_entropy yang sudah ada; dataset TextWindows menyediakan (x=(B,T), y=(B,)) — untuk loss per-posisi, M-007 menggunakan y dengan target sama di semua posisi (pendekatan default, kompatibel pipeline; LM per-posisi penuh ditangani di M-008 training system, di luar scope).

**Causal correctness requirement:** untuk input (B,T), logits pada posisi t harus tidak berubah jika token pada posisi > t diganti nilai lain — diuji eksplisit di test plan (§11, mask correctness).

## 9. Configuration

```yaml
vocab_size: 256            # FIXED (DECISION-015)
d_model: 64                # dev/test default — kecil, scalable turun/naik
num_heads: 4               # → Dh = 16 (invariant D = H × Dh: 64 = 4 × 16)
num_layers: 2              # dev/test default
d_ff: 128                  # konvensi 2×d_model (dapat direvisi per eksperimen)
max_sequence_length: 64    # dev/test default (§12 — quadratic)
dropout: tidak digunakan   # Non-goal §3 (REQ-302); ditambah hanya bila overfitting terukur
dtype: float64             # DECISION-007
seed: wajib per instance   # DECISION-007 (seeded_rng)
bias: Linear bias ya (existing engine); LayerNorm gamma+beta
checkpoint configuration: format npz yang sudah ada (DECISION-014) — params dict datar;
                          ukuran & frekuensi → §12; RNG-state untuk resume = M-008 (ISSUE-007)
inference configuration: greedy + temperature via generate() yang sudah ada (adapter §11)
```

Skema positional: learned positional embedding (tabel (T_max, D)) vs sinusoidal — keduanya kompatibel from-scratch; pilihan default diusulkan learned (paling sederhana dengan engine Linear/lookup yang ada) — final ditetapkan saat implementasi M007-T003 atau menjadi catatan risiko §14; keduanya TIDAK melanggar source of truth. Weight-tying LM head ↔ embedding: opsional, default tidak (kesederhanaan).

## 10. Error Handling

Kebijakan (konsisten praktik repo; MISS-002 akan diklarifikasi menyeluruh di review berikut):
- Validasi input (raise ValueError): ids di luar [0,256); shape salah (bukan (B,T)); T > max_sequence_length; konfigurasi invalid (d_model % num_heads ≠ 0; dimensi ≤ 0; vocab_size ≠ 256).
- Guard urutan (raise RuntimeError): backward sebelum forward (kontrak engine).
- Numerik: softmax stabil (shift-max) sudah ada; nilai non-finite di output logits = failure test (bukan di-silent).
- Tidak ada error-swallowing; semua kegagalan fail-fast.

## 11. Testing Strategy

**Unit tests** (per komponen §7): embedding lookup & scatter grad; positional (shape + determinisme); Q/K/V projection (vs hand-computed); attention score (formula QKᵀ/√Dh); causal mask (kebalikan diagonal atas −∞); softmax (sum=1, stabil ekstrem); multi-head (≈ identik dgn H=1 reshaped pada kasus uji); LayerNorm (mean≈0 var≈1, gradient-checkable); FFN (shape + akt kecil); TransformerBlock (shape invarian); stacking (shape invarian × num_layers).

**Numerical tests:** shape semua tahap (§5); finite values (tidak ada NaN/Inf di logits); determinisme per-seed (dua instance seed sama → logits identik); mask correctness (§8: mengubah token masa depan tidak mengubah logits masa lalu); LayerNorm stat; gradient check (engine D-009, tol 1e-5) untuk komponen yang punya backward (MHA, FFN, LayerNorm) — sesuai AC REQ-011 "unit test per komponen".

**Integration tests:** Tokenizer → TextWindows → TransformerModel.forward → cross_entropy loss (finite, menurun pada toy corpus per evaluasi D-016 basis 1–2); generate adapter (greedy + temperature) deterministik per-seed.

**Regression tests:** suite M-001..M-006 tetap pass (pytest penuh, tanpa modifikasi test lama).

## 12. Resource Constraints

Estimasi parameter (P, float64) untuk config default (D=64, H=4, L=2, d_ff=128, T_max=64):
- Embedding: 256×64 = 16,384 · pos (learned): 64×64 = 4,096
- Per block: QKV+O = 4×(64×64+64) = 16,640; FFN = 2×(64×128+128) = 16,640; LN 2×2×64 = 256 → ~33,536
- Head: 64×256+256 = 16,640
- Total ≈ 89k param ≈ 0.7 MB (float64) — sangat ringan; checkpoint npz < 1 MB.
- Attention memory: scores (B,H,T,T) float64 → B×4×T²×8 byte; pada T=64, B=32: ~4 MB — aman; quadratic dalam T → batas praktis T pada hardware owner = OD-118 (jangan dikarang; T=64 dipilih hanya untuk dev/test).
- Activation memory: dominan scores + konteks intermediate ≈ O(B·T·D + B·H·T²) per layer × L; L=2 kecil.
- Training memory: params + grads + aktivasi cache ≈ beberapa MB pada default — realistis.
- Inference memory: params (~0.7 MB) + aktivasi satu forward — kecil.
- Semua angka di atas adalah estimasi desain untuk config dev/test, bukan batas final proyek; batas final (model size, T_max produksi) = OD-118.

## 13. Acceptance Criteria (objective, verifiable — per DECISION-016: tanpa ambang loss universal)

1. Forward pass Transformer berhasil untuk input (B,T) valid.
2. Semua shape §5 terverifikasi di test (embedding (B,T,D); Q/K/V (B,H,T,Dh); scores (B,H,T,T); logits (B,T,256)).
3. Causal mask bekerja — test "mengubah token masa depan tidak mengubah logits posisi sebelumnya" pass.
4. Multi-head attention bekerja (H>1) + unit vs kasus H=1.
5. Residual connection bekerja (test nilai: block dengan residual ≠ tanpa residual; shape invarian).
6. LayerNorm bekerja (output mean≈0, var≈1 per posisi).
7. FFN bekerja (shape + unit hand-computed kecil).
8. num_layers > 1 dapat ditumpuk (invarian (B,T,D)).
9. Logits vocabulary dimension = 256 (DECISION-015).
10. Terhubung pipeline LM: ids → model → loss softmax-CE finite; dataset TextWindows terpakai.
11. Tidak ada pretrained Transformer / bobot pretrained (REQ-201).
12. Tidak ada external LLM API (REQ-202).
13. Tidak ada implementation Transformer siap pakai / framework DL (REQ-203) — verifikasi via review import.
14. Regression: seluruh test M-001..M-006 tetap pass.
15. Determinisme per-seed terverifikasi.
16. Resource constraints (§12) terdokumentasi di dokumen ini.
17. Test plan dijalankan dengan pytest di environment project (satu perintah).
- Evaluasi loss (jika dijalankan sanity training) mengikuti kerangka 5-basis D-016 (baseline, improvement terukur, bukti konvergensi, validasi, config reproducible) — angka hanya per eksperimen.

## 14. Risks

1. Backward MHA/LayerNorm kompleks — backward per-layer MHA rawan bug → mitigasi: gradient check numerik per komponen (tol 1e-5) wajib; jika backward tertentu terbukti membebani M-007, AC backward dapat dinilai ulang ke forward-only + gradient-check pada subset (keputusan owner, jangan asumsikan).
2. Positional scheme (learned vs sinusoidal) belum final — default learned; revisi kecil, tidak mengubah interface.
3. Numerik float64 vs memori pada T besar — mitigasi: T kecil di dev/test; scaling = OD-118/OD-101.
4. Integrasi MLPLM loss per-posisi disederhanakan (§8 catatan) — LM per-posisi penuh = M-008; risiko misalignment rendah, tercatat eksplisit.
5. Perf NumPy pada (B,H,T,T) matmul — diakui; bukan AC performa M-007.

## 15. Open Decisions

- OD-118 — Transformer model size limit (BLOCKER untuk training serius, bukan untuk dev/test config di atas): parameter yang harus ditentukan owner sebelum implementasi skala penuh: d_model, num_layers, T_max produksi, budget RAM/training-time perangkat owner. Konfigurasi §9 hanya dev/test.
- OD-113 — coverage target (DEFERRED untuk technical design; diputuskan sebelum implementasi dianggap selesai penuh).
- OD-114 — CI platform (existing; memengaruhi AC #17 enforcement di masa depan).
- OD-120 tertutup (D-017). Posisi TD ini = docs/architecture/M007_TECHNICAL_DESIGN.md sesuai D-017.
- Baru (kecil, dari §9/§14): OD-121 — skema positional final (default learned; owner dapat mengubah sebelum/awal implementasi T003).
