# Contributing to Jabberjay

Thank you for taking the time to contribute! Jabberjay's primary goal is to be the one-stop shop for synthetic voice detection models, so every addition — whether a new model, a bug fix, or a documentation improvement — directly advances that mission.

## Table of Contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Making changes](#making-changes)
- [Adding a new model](#adding-a-new-model)
- [Submitting a pull request](#submitting-a-pull-request)
- [Code style](#code-style)

---

## Ways to contribute

- **Add a model** — the highest-impact contribution; see [Adding a new model](#adding-a-new-model) below
- **Report a bug** — open a [bug report](https://github.com/MattyB95/Jabberjay/issues/new?template=bug_report.yml)
- **Request a model** — open a [model request](https://github.com/MattyB95/Jabberjay/issues/new?template=model_request.yml)
- **Improve documentation** — fix typos, clarify examples, extend the README
- **Write tests** — increase coverage, especially for edge cases and error paths

---

## Development setup

You will need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/) installed.

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/Jabberjay.git
cd Jabberjay

# 2. Install all dependencies including dev tools
just install

# 3. Install pre-commit hooks (runs lint + format automatically on every commit)
uv run pre-commit install

# 3. Verify everything works
just check   # lint + format check + type check
just test    # run the test suite
```

Work on `develop` — **do not target `main` directly**.

---

## Making changes

1. Create a branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feat/my-change
   ```
2. Make your changes.
3. Run `just fix` to auto-format and lint.
4. Run `just check` — all checks must pass before opening a PR.
5. Run `just test` — all tests must pass.
6. Update `CHANGELOG.md` under `[Unreleased]`.
7. Open a pull request against `develop`.

---

## Adding a new model

New models are the most valuable contribution. The bar for inclusion is:

| Requirement | Detail |
|---|---|
| **Licence** | Apache 2.0 or MIT only |
| **Task** | Binary bonafide / spoof classification |
| **Availability** | Publicly available weights (HuggingFace Hub preferred) |
| **Input** | Raw audio waveform (not pre-extracted features) |

### Step-by-step

**1. Add the model value to the `Model` enum** (`src/Jabberjay/Utilities/enum_handler.py`):

```python
class Model(Enum):
    ...
    MyModel = "MyModel"
```

**2. Create a `run.py`** in a new directory under `src/Jabberjay/Models/MyModel/`:

```python
# src/Jabberjay/Models/MyModel/run.py
import numpy as np
from typing import cast
from loguru import logger
from transformers import pipeline
from Jabberjay.Utilities.label_normalizer import normalize_label
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "author/model-id-on-huggingface"
_TARGET_SR = 16_000

def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    logger.info(f"Loading MyModel: {_MODEL_ID}")
    pipe = pipeline("audio-classification", model=_MODEL_ID, sampling_rate=_TARGET_SR)
    logger.debug(f"Running MyModel inference on {len(y)} samples at {int(sr)}Hz")
    raw = cast(list[dict[str, object]], pipe({"raw": y, "sampling_rate": int(sr)}))
    return [
        PredictionScore(label=normalize_label(str(s["label"])), score=float(str(s["score"])))
        for s in raw
    ]
```

If the model uses non-standard labels, `normalize_label()` handles mapping — check `src/Jabberjay/Utilities/label_normalizer.py` and extend `_BONAFIDE_KEYWORDS` / `_SPOOF_KEYWORDS` if needed.

**3. Add a handler and match case** in `src/Jabberjay/jabberjay.py`:

```python
# In detect():
case Model.MyModel:
    return self._mymodel_handler(y=y, sr=sr)

# New method:
def _mymodel_handler(self, y: np.ndarray, sr: float) -> DetectionResult:
    import Jabberjay.Models.MyModel.run as MyModel
    scores = MyModel.predict(y=y, sr=sr)
    top = scores[0]
    return DetectionResult(
        label=top["label"],
        is_bonafide=top["label"] == "Bonafide",
        confidence=top["score"],
        model=Model.MyModel,
        scores=scores,
    )
```

**4. Update tests** — add the new enum value to `TestEnums.test_model_members` in `tests/test_jabberjay.py`.

**5. Update `CHANGELOG.md`** under `[Unreleased]` with the model name, HuggingFace link, and dataset.

**6. Update `README.md`** — add a row to the relevant models table.

**7. Verify everything passes:**

```bash
just fix
just check
just test
```

---

## Submitting a pull request

- Target the `develop` branch, not `main`
- Fill in the pull request template fully
- Link any related issues with `Closes #<issue>`
- Ensure `just check` and `just test` both pass locally before opening the PR

---

## Code style

- **Formatter**: [black](https://black.readthedocs.io/) (line length 88) — run `just format`
- **Linter**: [ruff](https://docs.astral.sh/ruff/) — run `just lint`
- **Type checker**: [ty](https://docs.astral.sh/ty/) — run `just type-check`
- **All at once**: `just fix` (auto-fix) or `just check` (check only)

Logging uses [loguru](https://loguru.readthedocs.io/): `logger.info()` for model loading, `logger.debug()` for inference details. Do not use `print()` inside library code — only in `main()`.
