# Contributing to turboquant-skill

Thanks for your interest in contributing! This skill brings Google's TurboQuant compression
algorithm to Claude Code and compatible AI coding agents.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/Ryuketsukami/turboquant-skill/issues) tab
- Include your environment (Python version, OS, Claude Code version)
- For accuracy issues, include the dimension, bit-width, and a minimal reproduction

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run the self-test: `python scripts/turboquant.py`
5. Run the eval suite: `pytest evals/ -v`
6. Commit with a descriptive message: `git commit -m "feat: add batch compression"`
7. Push and open a Pull Request

### What We're Looking For

- **Performance improvements**: GPU acceleration (CuPy/PyTorch), vectorized batch operations
- **New quantization modes**: 1-bit extreme compression, mixed-precision per-layer
- **Integration examples**: HuggingFace transformers, vLLM, TGI
- **Documentation**: Better explanations, more examples, benchmarks on real models
- **Tests**: Edge cases, numerical stability, large-dimension behavior

### Code Style

- Follow PEP 8 with a 100-character line limit
- Use type annotations for all public functions
- Docstrings for all public classes and methods
- NumPy-style docstrings preferred

### Skill Format

This project follows the [Anthropic Agent Skills specification](https://github.com/anthropics/skills).
When modifying `SKILL.md`:
- Keep the YAML frontmatter description specific and "pushy" (helps Claude trigger correctly)
- Keep the markdown body under 500 lines
- Move detailed content to `references/` and link from SKILL.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
