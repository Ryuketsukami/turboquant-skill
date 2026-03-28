# TurboQuant — Algorithm Details

## Why Random Rotation Works

Given a unit vector `x ∈ S^{d-1}`, applying a random orthogonal matrix `Π` produces
`y = Πx` where each coordinate `y_i` follows `Beta(d/2, d/2)` shifted to `[-1, 1]`.

This is a consequence of the rotational invariance of the uniform measure on the sphere.
The marginal distribution of any single coordinate of a uniformly random point on `S^{d-1}`
is `Beta(d/2, d/2)` (after affine shift from `[0,1]` to `[-1,1]`).

Since the quantizer is designed for this exact distribution, the rotation step ensures
that the codebook is near-optimal regardless of the original vector's structure.

## Lloyd-Max Convergence

The Lloyd-Max algorithm alternates between:
1. **Nearest-neighbor assignment**: Given centroids, assign each point to its nearest centroid
2. **Centroid update**: Given assignments, set each centroid to the conditional expectation

For continuous, unimodal distributions like `Beta(d/2, d/2)`, Lloyd-Max converges to the
globally optimal quantizer. The implementation uses 500-point numerical quadrature for
the conditional expectation computation.

## QJL Theory

The Quantized Johnson-Lindenstrauss (QJL) transform from AAAI 2025 shows that for
random vectors `s_1, ..., s_m` with iid Rademacher entries (±1):

```
<a, b> ≈ (√(π/2) / m) Σ_i sign(<s_i, a>) · |<s_i, b>|
```

This estimator is unbiased and has variance `O(||a||² ||b||² / m)`.

In TurboQuant, `a` is the quantization residual (compressed to signs) and `b` is the
rotated query (full precision). The 1-bit storage per projection dimension makes this
extremely memory-efficient.

## Bit-Rate Analysis

| Configuration | Bits per coordinate | Compression ratio (vs float32) |
|---------------|--------------------|-----------------------------|
| PolarQuant 2-bit | 2 | 16x |
| PolarQuant 3-bit | 3 | 10.7x |
| PolarQuant 4-bit | 4 | 8x |
| TurboQuant 3+1 | 4 (3 PQ + 1 QJL) | 8x |
| TurboQuant 4+1 | 5 (4 PQ + 1 QJL) | 6.4x |

## Comparison to Product Quantization (PQ)

| Property | Product Quantization | TurboQuant |
|----------|---------------------|------------|
| Data-dependent | Yes (requires training data) | No (data-oblivious) |
| Codebook training | Expensive (k-means per subspace) | One-time, analytical |
| Compression speed | Requires codebook lookup | O(d) scalar quantization |
| Decompression speed | Codebook lookup | O(d) centroid lookup |
| Accuracy at 4 bits | Good but data-dependent | Near-optimal for any distribution |
| Dynamic data | Requires retraining | No retraining needed |
