---
name: turboquant
description: >
  Implement, use, or explain TurboQuant — Google's data-oblivious vector quantization algorithm
  for LLM KV cache compression (ICLR 2026). Use this skill when the user asks about KV cache
  compression, TurboQuant, PolarQuant, QJL (Quantized Johnson-Lindenstrauss), Lloyd-Max
  quantization for high-dimensional vectors, reducing LLM memory usage, compressing attention
  keys/values, or implementing any component of the TurboQuant pipeline. Also trigger when
  the user mentions vector quantization for inference optimization, 3-4 bit KV cache quantization,
  or inner product preserving compression.
---

# TurboQuant — KV Cache Compression Skill

A skill for implementing, using, and explaining Google's TurboQuant algorithm — a data-oblivious
vector quantization framework that achieves 6x memory reduction and up to 8x speedup for LLM
KV caches with zero accuracy loss.

## What TurboQuant Does

TurboQuant compresses the key-value (KV) cache in transformer-based LLMs. During inference,
the KV cache grows linearly with sequence length and becomes the primary memory bottleneck
for long-context generation. TurboQuant compresses each cache vector from 32-bit floats down
to 3-4 bits per coordinate — without retraining, without calibration data, and without
measurable accuracy loss.

## How It Works (Two Stages)

### Stage 1: PolarQuant (MSE Minimization)

1. **Random Orthogonal Rotation**: Multiply the input vector by a random orthogonal matrix `Π`.
   This "isotropizes" the vector — after rotation, each coordinate of a unit vector follows
   a `Beta(d/2, d/2)` distribution (shifted to `[-1, 1]`), regardless of the original
   vector's structure.

2. **Lloyd-Max Scalar Quantization**: Quantize each rotated coordinate independently using
   a Lloyd-Max quantizer optimized for the `Beta(d/2, d/2)` distribution. Because the
   distribution is known analytically (data-oblivious), the codebook is computed once offline
   and reused for all vectors.

### Stage 2: QJL (Unbiased Inner Products)

3. **Residual Computation**: Compute the quantization residual `r = y - ŷ` (difference between
   rotated vector and its quantized reconstruction).

4. **1-bit Sign Quantization**: Project the residual through a random Rademacher matrix `S`
   and store only the signs: `sign(S @ r)`. This uses the Quantized Johnson-Lindenstrauss
   transform to preserve inner product information in just 1 bit per dimension.

5. **Inner Product Estimation**: To compute `<query, key>`, combine the PolarQuant
   reconstruction with a QJL correction term that uses the stored signs to unbias the estimate.

## When to Use Each Variant

| Variant | Use Case | Bits | Accuracy |
|---------|----------|------|----------|
| `TurboQuant_mse` | Reconstruction (nearest neighbor search) | b bits | MSE-optimal |
| `TurboQuant_prod` | Inner products (attention computation) | b + 1 bits | Unbiased IP |

For KV cache compression in transformers, use `TurboQuant_prod` — attention requires inner
products between queries and keys, and the QJL correction ensures these estimates are unbiased.

## Implementation Reference

A complete Python implementation is bundled at `scripts/turboquant.py`. It includes:

- `build_lloyd_max_codebook()` — Offline codebook construction via Lloyd-Max iteration
- `generate_rotation_matrix()` — Random orthogonal matrix via QR decomposition
- `TurboQuant` class — Compress, decompress, and estimate inner products
- `TurboQuantKVCache` class — Simulated KV cache with compressed storage
- `run_self_test()` — Validation suite with MSE, cosine similarity, and IP correlation metrics

### Quick Start

```python
from scripts.turboquant import TurboQuant, TurboQuantConfig

config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
tq = TurboQuant(config)

# Compress a vector
compressed = tq.compress(my_vector)

# Decompress (MSE reconstruction)
reconstructed = tq.decompress(compressed)

# Estimate inner product (unbiased, with QJL correction)
ip_estimate = tq.inner_product(query_vector, compressed)

# Check compression ratio
print(f"Compression: {tq.compression_ratio():.1f}x")
```

### KV Cache Usage

```python
from scripts.turboquant import TurboQuantKVCache, TurboQuantConfig

config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
cache = TurboQuantKVCache(config)

# During generation — append each new KV pair
cache.append(key_vector, value_vector)

# Compute attention scores for a query
scores = cache.attention_scores(query_vector)
```

## Key Mathematical Properties

- **Data-oblivious**: No calibration data needed. The codebook depends only on dimension `d`
  and bit-width `b`, not on the data distribution.
- **Near-optimal distortion**: Achieves rate-distortion performance within constant factors
  of the theoretical optimum for Euclidean vectors.
- **Unbiased inner products**: The QJL stage ensures `E[<q, k̂>] = <q, k>` — critical for
  attention computation where biased estimates shift the softmax distribution.
- **O(d) compression/decompression**: Linear in dimension. No codebook search.

## Performance Benchmarks (from the paper)

| Metric | Result |
|--------|--------|
| KV cache compression | 3 bits/value, 6x reduction |
| Attention speedup (H100) | Up to 8x on 4-bit keys |
| Needle-in-Haystack (104k tokens) | 100% retrieval accuracy at 4x compression |
| Accuracy loss | Zero measurable loss on LongBench, ZeroSCROLLS, RULER, L-Eval |

## Technical Details

For deeper mathematical treatment, see `references/algorithm_details.md`:
- Proof sketch for why random rotation produces Beta-distributed coordinates
- Lloyd-Max convergence properties
- QJL guarantee: the Johnson-Lindenstrauss lemma for quantized projections
- Bit-rate analysis and comparison to Product Quantization

## Dependencies

- Python 3.10+
- NumPy
- SciPy (for `beta` distribution and `minimize_scalar`)

## Paper Reference

Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate,"
ICLR 2026. arXiv: 2504.19874.
