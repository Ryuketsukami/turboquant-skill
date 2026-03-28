"""
Evaluation test suite for TurboQuant.

Tests cover:
  - Lloyd-Max codebook construction
  - Random orthogonal rotation
  - PolarQuant compression / decompression
  - QJL inner product estimation
  - KV cache wrapper
  - Compression ratio formula
"""

import sys
import os
import numpy as np
import pytest

# Allow importing from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.turboquant import (
    build_lloyd_max_codebook,
    generate_rotation_matrix,
    TurboQuant,
    TurboQuantConfig,
    TurboQuantKVCache,
)


# ---- Codebook tests ----

class TestCodebook:
    def test_correct_levels_count(self):
        """Codebook should have exactly 2^bits centroids."""
        for bits in [2, 3, 4]:
            centroids, boundaries = build_lloyd_max_codebook(dimension=64, bits=bits)
            assert len(centroids) == 2 ** bits

    def test_monotonic_boundaries(self):
        """Decision boundaries must be strictly increasing."""
        centroids, boundaries = build_lloyd_max_codebook(dimension=128, bits=3)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i + 1], (
                f"Boundary {i} ({boundaries[i]}) >= boundary {i+1} ({boundaries[i+1]})"
            )

    def test_symmetric_centroids(self):
        """For Beta(a,a) on [-1,1], centroids should be symmetric around 0."""
        centroids, _ = build_lloyd_max_codebook(dimension=128, bits=3)
        n = len(centroids)
        for i in range(n // 2):
            assert pytest.approx(centroids[i], abs=1e-6) == -centroids[n - 1 - i]


# ---- Rotation tests ----

class TestRotation:
    def test_orthogonality(self):
        """Rotation matrix should satisfy Q^T Q = I."""
        d = 64
        Q = generate_rotation_matrix(d, seed=42)
        eye = Q.T @ Q
        np.testing.assert_allclose(eye, np.eye(d), atol=1e-10)

    def test_determinism(self):
        """Same seed should produce the same rotation matrix."""
        Q1 = generate_rotation_matrix(64, seed=99)
        Q2 = generate_rotation_matrix(64, seed=99)
        np.testing.assert_array_equal(Q1, Q2)


# ---- Compression tests ----

class TestCompression:
    def test_shape_preserved(self):
        """Decompressed vector should have the same shape as the original."""
        d = 128
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=False)
        tq = TurboQuant(config)
        x = np.random.default_rng(0).standard_normal(d)
        c = tq.compress(x)
        x_hat = tq.decompress(c)
        assert x_hat.shape == x.shape

    def test_mse_below_threshold_3bit(self):
        """MSE should be below 0.01 at 3-bit, d=128 on unit vectors."""
        d = 128
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=False)
        tq = TurboQuant(config)
        rng = np.random.default_rng(42)
        mses = []
        for _ in range(50):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)
            c = tq.compress(x)
            x_hat = tq.decompress(c)
            mses.append(np.mean((x - x_hat) ** 2))
        mean_mse = np.mean(mses)
        assert mean_mse < 0.01, f"Mean MSE {mean_mse:.6f} exceeds 0.01"

    def test_higher_bits_lower_error(self):
        """More bits should yield lower MSE."""
        d = 128
        rng = np.random.default_rng(7)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)

        mse_by_bits = {}
        for bits in [2, 3, 4]:
            config = TurboQuantConfig(dimension=d, bits=bits, qjl_enabled=False)
            tq = TurboQuant(config)
            c = tq.compress(x)
            x_hat = tq.decompress(c)
            mse_by_bits[bits] = np.mean((x - x_hat) ** 2)

        assert mse_by_bits[4] < mse_by_bits[3] < mse_by_bits[2], (
            f"MSE not monotonically decreasing: {mse_by_bits}"
        )


# ---- Inner product tests ----

class TestInnerProduct:
    def test_qjl_produces_signs(self):
        """Compressed vector with QJL enabled should have qjl_signs."""
        d = 64
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=True)
        tq = TurboQuant(config)
        x = np.random.default_rng(0).standard_normal(d)
        c = tq.compress(x)
        assert c.qjl_signs is not None
        assert c.qjl_signs.shape == (d,)

    def test_ip_correlation_above_threshold(self):
        """Inner product correlation should exceed 0.70 on random pairs."""
        d = 128
        n = 50
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=True)
        tq = TurboQuant(config)
        rng = np.random.default_rng(1)

        X = rng.standard_normal((n, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        compressed = [tq.compress(X[i]) for i in range(n)]

        true_ips = []
        est_ips = []
        for _ in range(300):
            qi = rng.integers(0, n)
            ki = rng.integers(0, n)
            true_ips.append(np.dot(X[qi], X[ki]))
            est_ips.append(tq.inner_product(X[qi], compressed[ki]))

        corr = np.corrcoef(true_ips, est_ips)[0, 1]
        assert corr > 0.70, f"IP correlation {corr:.4f} below 0.70"


# ---- KV cache tests ----

class TestKVCache:
    def test_length_tracking(self):
        """Cache length should match number of appended pairs."""
        d = 64
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=True)
        cache = TurboQuantKVCache(config)
        rng = np.random.default_rng(0)
        for _ in range(10):
            k = rng.standard_normal(d)
            v = rng.standard_normal(d)
            cache.append(k, v)
        assert len(cache) == 10

    def test_attention_scores_shape(self):
        """Attention scores shape should match cache length."""
        d = 64
        seq_len = 8
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=True)
        cache = TurboQuantKVCache(config)
        rng = np.random.default_rng(0)
        for _ in range(seq_len):
            cache.append(rng.standard_normal(d), rng.standard_normal(d))
        query = rng.standard_normal(d)
        scores = cache.attention_scores(query)
        assert scores.shape == (seq_len,)


# ---- Compression ratio tests ----

class TestCompressionRatio:
    def test_formula_no_qjl(self):
        """Without QJL: ratio = 32 / bits."""
        d = 128
        for bits in [2, 3, 4]:
            config = TurboQuantConfig(dimension=d, bits=bits, qjl_enabled=False)
            tq = TurboQuant(config)
            expected = 32.0 / bits
            assert pytest.approx(tq.compression_ratio(), rel=1e-6) == expected

    def test_qjl_lowers_ratio(self):
        """Enabling QJL should lower the compression ratio (more bits stored)."""
        d = 128
        config_no = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=False)
        config_yes = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=True)
        tq_no = TurboQuant(config_no)
        tq_yes = TurboQuant(config_yes)
        assert tq_yes.compression_ratio() < tq_no.compression_ratio()
