# TurboQuant Skill — LLM KV Cache Compression for AI Coding Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/Ryuketsukami/turboquant-skill/actions/workflows/tests.yml/badge.svg)](https://github.com/Ryuketsukami/turboquant-skill/actions/workflows/tests.yml)
[![Agent Skills](https://img.shields.io/badge/Agent_Skills-Compatible-blue.svg)](https://github.com/anthropics/skills)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://python.org)
[![ICLR 2026](https://img.shields.io/badge/Paper-ICLR_2026-red.svg)](https://arxiv.org/abs/2504.19874)

> **An AI agent skill implementing Google's TurboQuant compression algorithm** — the data-oblivious vector quantization framework that reduces LLM KV cache memory by **6x** and delivers up to **8x speedup** with **zero accuracy loss**. Compatible with Claude Code, Codex CLI, and all agents supporting the [Agent Skills specification](https://github.com/anthropics/skills).

> **Looking for the standalone Python library?** See [turboquant-compression](https://github.com/Ryuketsukami/turboquant-compression) — pip-installable with 27 tests and full documentation.

---

## What is TurboQuant?

**TurboQuant** is a compression algorithm introduced by Google Research (Zandieh et al.) at **ICLR 2026** that solves the primary memory bottleneck in large language model inference: the key-value (KV) cache.

During text generation, transformers store computed key and value vectors for every token in the context window. For long-context applications (100K+ tokens), this KV cache can consume **tens of gigabytes of GPU memory** — often more than the model weights themselves.

TurboQuant compresses each KV cache vector from 32-bit floats down to **3-4 bits per coordinate** — without retraining, without calibration data, and without measurable accuracy loss.

### Key Results (from the paper)

| Metric | Result |
|--------|--------|
| **Memory reduction** | 6x (3-bit per coordinate) |
| **Attention speedup** | Up to 8x on NVIDIA H100 GPUs |
| **Accuracy loss** | Zero — 100% retrieval on Needle-in-Haystack at 104K tokens |
| **Calibration needed** | None — fully data-oblivious |
| **Retraining needed** | None — works with any pretrained model |

---

## What is This Repo?

This repository packages a **complete Python implementation of TurboQuant** as an [Agent Skill](https://github.com/anthropics/skills) — a self-contained module that AI coding agents (Claude Code, Codex CLI, ChatGPT, Cursor, etc.) can load dynamically to gain expertise in KV cache compression.

### What the Skill Enables

When installed, your AI coding agent can:

- **Implement TurboQuant** from scratch in any language, guided by the bundled reference implementation
- **Explain the algorithm** — PolarQuant rotation, Lloyd-Max quantization, QJL residual correction
- **Integrate TurboQuant** into existing inference pipelines (vLLM, HuggingFace, TGI)
- **Benchmark compression** — measure MSE, cosine similarity, inner product accuracy, and throughput
- **Debug quantization issues** — identify when bit-width is too low, when QJL correction is needed, etc.
- **Compare to alternatives** — Product Quantization, GPTQ, AWQ, SqueezeLLM, and other quantization methods

---

## Installation

### For Claude Code

```bash
# Clone into your skills directory
git clone https://github.com/Ryuketsukami/turboquant-skill.git ~/.claude/skills/turboquant-skill
```

### For Codex CLI

```bash
git clone https://github.com/Ryuketsukami/turboquant-skill.git ~/.codex/skills/turboquant-skill
```

### For Project-Level Installation

```bash
# Add to your project's .claude/skills/ directory
git clone https://github.com/Ryuketsukami/turboquant-skill.git .claude/skills/turboquant-skill
```

The skill is automatically discovered by any compatible agent when relevant tasks arise.

---

## How TurboQuant Works

TurboQuant operates in two stages:

### Stage 1: PolarQuant (MSE Minimization)

```
Input vector x ∈ R^d
    ↓
Random orthogonal rotation: y = Π · x
    ↓
Scalar quantization: each y_i → nearest Lloyd-Max centroid
    ↓
Output: quantization indices (b bits per coordinate)
```

**Why rotation?** After multiplication by a random orthogonal matrix, each coordinate of a unit vector follows a known `Beta(d/2, d/2)` distribution — regardless of the original vector's structure. This makes the quantizer **data-oblivious**: the codebook depends only on dimension and bit-width, never on the data.

### Stage 2: QJL (Unbiased Inner Products)

```
Quantization residual: r = y - ŷ
    ↓
Random Rademacher projection: z = S · r
    ↓
1-bit sign quantization: sign(z)
    ↓
Output: 1 additional bit per dimension
```

The Quantized Johnson-Lindenstrauss (QJL) transform preserves inner product information in the residual using just **1 bit per dimension**. This ensures that attention scores (`<query, key>`) are computed without bias.

---

## Usage

### Quick Start

```python
from scripts.turboquant import TurboQuant, TurboQuantConfig
import numpy as np

# Configure: 128-dimensional vectors, 3-bit quantization + QJL
config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
tq = TurboQuant(config)

# Compress a vector
x = np.random.randn(128)
compressed = tq.compress(x)

# Decompress (MSE reconstruction)
x_hat = tq.decompress(compressed)
print(f"MSE: {np.mean((x - x_hat)**2):.6f}")
print(f"Compression: {tq.compression_ratio():.1f}x")
```

### KV Cache Compression

```python
from scripts.turboquant import TurboQuantKVCache, TurboQuantConfig

# Create compressed KV cache
config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
cache = TurboQuantKVCache(config)

# During generation — append each new KV pair
for key, value in zip(key_vectors, value_vectors):
    cache.append(key, value)

# Compute attention scores
scores = cache.attention_scores(query)
```

### Run Self-Test

```bash
python scripts/turboquant.py
pytest evals/ -v
```

This runs a comprehensive validation suite testing:
- MSE reconstruction quality at 2, 3, and 4 bits
- Cosine similarity preservation
- Inner product estimation accuracy (RMSE and correlation)
- KV cache simulation with attention score comparison

---

## Repository Structure

```
turboquant-skill/
├── SKILL.md                     # Agent skill definition (YAML frontmatter + instructions)
├── README.md                    # This file
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── SECURITY.md                  # Security policy
├── CHANGELOG.md                 # Release history
├── CITATION.cff                 # Citation metadata
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml            # CI pipeline (Python 3.10–3.13)
├── scripts/
│   └── turboquant.py            # Complete Python implementation (420 lines)
├── references/
│   └── algorithm_details.md     # Deep mathematical details and comparisons
└── evals/
    └── test_turboquant.py       # Evaluation test suite (pytest)
```

---

## Configuration Reference

```python
@dataclass
class TurboQuantConfig:
    dimension: int          # Vector dimension (d) — must match your model's head_dim
    bits: int = 4           # Bit-width for PolarQuant (2-4 typical)
    qjl_enabled: bool = True  # Enable QJL residual correction
    qjl_dim: int = 0       # QJL projection dim (0 = auto = dimension)
    seed: int = 42          # Random seed for reproducibility
```

### Recommended Settings

| Use Case | bits | qjl_enabled | Compression | Notes |
|----------|------|-------------|-------------|-------|
| Maximum compression | 2 | False | 16x | Some quality loss, good for non-critical caches |
| Balanced (recommended) | 3 | True | 8x | Best quality/compression tradeoff |
| High quality | 4 | True | 6.4x | Minimal quality loss, matches paper results |
| MSE only (no IP) | 3 | False | 10.7x | Use when you don't need inner product estimation |

---

## Benchmarks

Results from the bundled self-test (`d=256, n=64 vectors`):

| Variant | Bits | MSE | Cosine Sim | IP Correlation | Compression |
|---------|------|-----|------------|----------------|-------------|
| PolarQuant | 2-bit | 0.000461 | 0.9394 | — | 16.0x |
| PolarQuant | 3-bit | 0.000133 | 0.9829 | — | 10.7x |
| PolarQuant | 4-bit | 0.000037 | 0.9952 | — | 8.0x |
| TurboQuant | 3+1 bit | — | — | 0.8605 | 8.0x |
| TurboQuant | 4+1 bit | — | — | 0.8604 | 6.4x |

> **Note:** These benchmarks use `d=256` for fast CI. Production KV caches typically use `d=4096+`, where relative errors are significantly lower due to concentration-of-measure effects.

---

## How This Compares to Other Methods

| Method | Data-Oblivious | Retraining | Bits | Speed |
|--------|---------------|-----------|------|-------|
| **TurboQuant** | Yes | None | 3-4 | O(d) |
| Product Quantization | No | Required | 4-8 | O(d·k) |
| GPTQ | No | Required | 3-4 | O(d) |
| AWQ | No | Required | 4 | O(d) |
| SqueezeLLM | No | Required | 3-4 | O(d) |

TurboQuant's key advantage: **zero calibration data needed**. Other methods require representative data samples to build quantization schemes. TurboQuant works on any vector from any model, immediately.

---

## Dependencies

- **Python** 3.10+
- **NumPy** — array operations and linear algebra
- **SciPy** — Beta distribution for Lloyd-Max codebook construction

```bash
pip install numpy scipy
```

---

## Paper Reference

> Zandieh, A., et al. **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."** International Conference on Learning Representations (ICLR), 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

See also:
- **QJL**: Zandieh & Han, "Quantized Johnson-Lindenstrauss Transform," AAAI 2025
- **PolarQuant**: Zandieh et al., "PolarQuant," AISTATS 2026
- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

---

## Related

- [turboquant-compression](https://github.com/Ryuketsukami/turboquant-compression) — Standalone pip-installable Python library with 27 tests
- [Anthropic Agent Skills](https://github.com/anthropics/skills) — The skill specification this repo follows
- [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — Original TurboQuant paper

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- GPU-accelerated implementation (PyTorch/CuPy)
- Integration with HuggingFace transformers and vLLM
- Benchmarks on real models (Gemma, Llama, Mistral)
- Additional quantization modes

---

## License

[MIT](LICENSE) — free for personal and commercial use.

---

<p align="center">
  <sub>Built as an <a href="https://github.com/anthropics/skills">Agent Skill</a> — works with Claude Code, Codex CLI, and any compatible AI coding agent.</sub>
</p>
