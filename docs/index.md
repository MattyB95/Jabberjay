# Jabberjay 🦜

> One API. Every state-of-the-art synthetic voice detector.

[![PyPI](https://img.shields.io/pypi/v/jabberjay)](https://pypi.org/project/Jabberjay/)
[![CI](https://github.com/MattyB95/Jabberjay/actions/workflows/ci.yml/badge.svg)](https://github.com/MattyB95/Jabberjay/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/jabberjay)](https://pypi.org/project/Jabberjay/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Synthetic voice detection is a fragmented landscape — state-of-the-art models are scattered across research repositories, each with its own dependencies, input formats, and output conventions. Jabberjay brings them all under one consistent Python API and CLI so you can detect AI-generated speech without wrestling with model internals.

## Features

- **Seven model families** — ViT, AST, Wav2Vec2, HuBERT, WavLM, RawNet2, and a classical baseline
- **Unified output** — every model returns the same `DetectionResult` with `label`, `confidence`, and `scores`
- **Zero boilerplate** — pass a file path, get a verdict; models are downloaded and cached automatically
- **Flexible** — use strings for quick experiments, enums for IDE autocomplete, or pre-load audio to run multiple models on the same clip

## Quick start

```bash
pip install jabberjay
```

```python
from Jabberjay import Jabberjay

jj = Jabberjay()
result = jj.detect("interview.wav")
print(result)  # Bonafide ✔️ (94.1% confidence, model=VIT)
```

Or from the command line:

```bash
jabberjay interview.wav
jabberjay suspicious.wav -m HuBERT
```

---

[Get started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[Browse models :material-arrow-right:](models.md){ .md-button }
