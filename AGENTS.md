# Agent Instructions — TurboQuant Skill

This file provides instructions for AI coding agents working with this repository.

## What This Project Is

An agent skill that teaches AI coding agents (Claude Code, Codex CLI, Cursor, GitHub Copilot, Windsurf) how to implement, explain, and use Google's TurboQuant compression algorithm for LLM KV cache optimization.

## When to Use This Skill

Activate when the user asks about:
- KV cache compression or KV cache memory reduction
- TurboQuant, PolarQuant, or QJL (Quantized Johnson-Lindenstrauss)
- Lloyd-Max quantization for high-dimensional vectors
- 3-4 bit quantization of attention keys/values
- Vector quantization for inference optimization
- Comparing quantization methods (vs GPTQ, AWQ, SqueezeLLM)

## Project Structure

```
SKILL.md                    # Agent skill definition with YAML frontmatter
scripts/turboquant.py       # Complete Python implementation
references/algorithm_details.md  # Mathematical deep-dive
evals/test_turboquant.py    # Evaluation test suite (12 tests)
```

## Key Classes in scripts/turboquant.py

- `TurboQuantConfig(dimension, bits, qjl_enabled, qjl_dim, seed)` — Configuration
- `TurboQuant(config)` — Compressor: `compress()`, `decompress()`, `inner_product()`
- `TurboQuantKVCache(config)` — Cache simulator: `append()`, `attention_scores()`
- `build_lloyd_max_codebook(dimension, bits)` — Offline codebook construction
- `generate_rotation_matrix(dimension, seed)` — Random orthogonal matrix

## How to Install as a Skill

```bash
# Claude Code
git clone https://github.com/Ryuketsukami/turboquant-skill.git ~/.claude/skills/turboquant-skill

# Codex CLI
git clone https://github.com/Ryuketsukami/turboquant-skill.git ~/.codex/skills/turboquant-skill
```

## How to Run Tests

```bash
pip install numpy scipy pytest
pytest evals/ -v             # 12 evaluation tests
python scripts/turboquant.py # self-test with benchmarks
```

## Algorithm Summary

TurboQuant is **data-oblivious** — no calibration data needed. Two stages:

1. **PolarQuant**: Rotate vector by random orthogonal matrix, then Lloyd-Max quantize each coordinate (B bits). The rotation makes coordinates follow Beta(d/2, d/2), enabling an analytically optimal codebook.
2. **QJL**: Project quantization residual through random Rademacher matrix, store 1-bit signs. Enables unbiased inner product estimation for attention computation.

Recommended: 3 bits + QJL for 8x compression with high attention score fidelity.
