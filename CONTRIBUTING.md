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

# 4. Verify everything works
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

| Requirement      | Detail                                                 |
|------------------|--------------------------------------------------------|
| **Licence**      | Apache 2.0 or MIT preferred; others (e.g. non-commercial) accepted if flagged — see below |
| **Task**         | Binary bonafide / spoof classification                 |
| **Availability** | Publicly available weights (HuggingFace Hub preferred) |
| **Input**        | Raw audio waveform (not pre-extracted features)        |

Jabberjay's own code is MIT and never redistributes weights — they are downloaded from the model's own repository on first use. A model under a more restrictive licence (for example DF Arena 1B, which is non-commercial) can still be included, but its licence must be flagged next to it in `README.md` and with a warning admonition in `docs/models.md`.

### Step-by-step

**1. Add the model value to the `Model` enum** (`src/Jabberjay/Utilities/enum_handler.py`):

```python
class Model(Enum):
    ...
    MyModel = "MyModel"
```

**2. Create a `run.py`** in a new directory under `src/Jabberjay/Models/MyModel/`:

If your model fits the standard `transformers` `audio-classification` pipeline, delegate to the shared `run_pipeline()` helper — it loads and caches the pipeline for you (see [`Utilities/pipeline.py`](src/Jabberjay/Utilities/pipeline.py)):

```python
# src/Jabberjay/Models/MyModel/run.py
import numpy as np

from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "author/model-id-on-huggingface"
_TARGET_SR = 16_000


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    return run_pipeline(_MODEL_ID, y, sr, "MyModel", sampling_rate=_TARGET_SR)
```

If your model needs custom loading (a `torch.nn.Module`, a non-pipeline HuggingFace class, a `.joblib`/`.pth` file, etc. — see the `Spectra*`, `RawNet2`, or `Classical` model families for examples), wrap the load in `Jabberjay.Utilities.model_cache.cached_loader` instead of calling `.from_pretrained()`/`load()` directly inside `predict()`. Every model in Jabberjay caches its loaded weights in-process so repeated `detect()` calls for the same model don't pay the reload cost — a new model that skips this would be the one exception and would noticeably slow down multi-call use (e.g. `examples/run_all.py`):

```python
from Jabberjay.Utilities.model_cache import cached_loader

@cached_loader(maxsize=4)
def _load_model(device: str) -> MyModelClass:
    return MyModelClass.from_pretrained(_MODEL_ID, strict=True).eval().to(device)
```

When loading weights yourself:

- load with `strict=True`, so a checkpoint that doesn't match your architecture fails loudly instead of leaving layers randomly initialised
- load pickled checkpoints (`.bin`, `.pth`) with `torch.load(..., weights_only=True)`, and never use `trust_remote_code=True` — port the architecture instead (see `Models/DFArena/`)
- if the checkpoint is pickled and the repository isn't Jabberjay's own, pin a reviewed commit with `download_pretrained_model(..., revision=<commit hash>)`

If the model uses non-standard labels, `normalize_pipeline_scores()` calls `normalize_label()` internally — check `src/Jabberjay/Utilities/label_normalizer.py` and extend `_BONAFIDE_SUBSTR` / `_SPOOF_SUBSTR` (substring matching) or `_BONAFIDE_EXACT` / `_SPOOF_EXACT` (exact matching) if needed.

**3. Add a handler and match case** in `src/Jabberjay/jabberjay.py`:

```python
# In detect():
case Model.MyModel:
    return self._mymodel_handler(y=y, sr=sr)

# New method:
def _mymodel_handler(self, y: np.ndarray, sr: float) -> DetectionResult:
    import Jabberjay.Models.MyModel.run as MyModel
    scores = MyModel.predict(y=y, sr=sr)
    logger.debug(f"MyModel predictions: {scores}")
    return self._result_from_scores(scores, Model.MyModel)
```

**4. Update tests:**

- add the new enum value to `TestEnums.test_model_members` in `tests/test_enums.py`
- add a `predict()` unit test in `tests/test_models.py` (mock all model loading — the suite runs offline)
- add a handler test to `TestDetectHandlers` in `tests/test_jabberjay.py`
- for a custom architecture, add shape and state-dict layout tests in `tests/test_architecture.py`, and an opt-in real-weights test in `tests/test_integration.py` (run with `just test-integration`)

**5. Update `CHANGELOG.md`** under `[Unreleased]` with the model name, HuggingFace link, and dataset.

**6. Update the docs and examples** — the model is listed in several places, so search for an existing model name (e.g. `SpectraAASIST3`) to find them all:

- `README.md` — the model-family count and list, the models table, the CLI usage line and examples, the "Choosing a model" snippet, and the `scores` row
- `docs/models.md` — the overview table, a full section, and the rule of thumb if it changes the recommendation
- `docs/index.md`, `docs/getting-started.md`, and `docs/cli.md`
- `examples/choosing_a_model.py`, `examples/exploring_results.py`, and `examples/run_all.py`
- the abstracts in `CITATION.cff` and `.zenodo.json`

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
