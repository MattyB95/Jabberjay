# Changelog

All notable changes to Jabberjay are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Jabberjay uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.8] — 2026-03-16

### Added
- **Zenodo integration** — repository is now archived on Zenodo; concept DOI
  (`10.5281/zenodo.19056978`) added to `CITATION.cff`, README badge row, and
  BibTeX entry in the README citation section
- **`.zenodo.json`** — explicit Zenodo record metadata (creator ORCID,
  affiliation, keywords, related identifiers for PyPI and docs) so archived
  records are consistent and complete

### Fixed
- **`RawNet2/run.py`** — YAML config loading now raises a descriptive
  `RuntimeError` instead of a raw `OSError`/`YAMLError` if the bundled config
  file is missing or malformed
- **`Utilities/label_normalizer.py`** — removed redundant `float(str(...))`
  double conversion; score is cast directly with `float()`
- **`Models/RawNet2/model.py`** — replaced deprecated `torch.cuda.FloatTensor`
  / `torch.FloatTensor` type-casting pattern with `torch.device` + `.to()`,
  resolving two `ty` type checker errors

---

## [0.0.7] — 2026-03-16

### Added
- **`CITATION.cff`** — machine-readable citation metadata for academic use;
  GitHub surfaces this as a "Cite this repository" button; compatible with
  Zenodo, Zotero, and other reference managers
- **Sample rate validation** — `detect()` now raises `ValueError` immediately
  for zero or negative sample rates when a pre-loaded audio tuple is passed,
  preventing silent failures in downstream model inference
- **Test coverage for new behaviour** — 7 new tests covering sample rate
  validation, `BytesIO` buffer cleanup on both success and error paths,
  figure cleanup on error paths, and `Classical.predict()` bonafide/spoof paths

### Changed
- **Dependency minimum versions pinned** — all runtime dependencies now carry
  lower-bound constraints (`torch>=2.0`, `transformers>=4.30`,
  `huggingface-hub>=0.20`, `librosa>=0.10`, etc.) to prevent silent
  incompatibilities on fresh installs
- **Contact and copyright updated** — maintainer email changed to
  `Matthew.Boakes@Gmail.com` across `CODE_OF_CONDUCT.md`, `SECURITY.md`, and
  `pyproject.toml`; `LICENSE` updated to `2024-2026 Matthew Boakes and The
  Alan Turing Institute`
- **`CONTRIBUTING.md` model template corrected** — example `run.py` now uses
  `cast()` + `normalize_pipeline_scores()` (matching real handlers) and the
  handler example uses `_result_from_scores()` instead of building
  `DetectionResult` manually; label normaliser variable names corrected to
  `_BONAFIDE_SUBSTR` / `_SPOOF_SUBSTR` / `_BONAFIDE_EXACT` / `_SPOOF_EXACT`

### Fixed
- **`VIT/utility.py`** — `BytesIO` buffer is now always closed in the
  `finally` block, preventing a resource leak if `Image.open()` or
  `img.load()` raises
- **`Classical/run.py`** — renamed `model` variable to `model_path` to
  accurately reflect that `download_pretrained_model()` returns a file path,
  not a model object
- **`_result_from_scores()`** — parameter type restored to
  `list[PredictionScore]`, keeping the type contract consistent end-to-end
  from model handlers through to `DetectionResult.scores`

---

## [0.0.6] — 2026-03-16

### Added
- **Documentation site** — MkDocs + Material for MkDocs, hosted on GitHub Pages at
  `https://mattyb95.github.io/Jabberjay`; includes Getting Started, Models, CLI,
  and API Reference pages (auto-generated from docstrings via mkdocstrings)
- **GitHub Release automation** (`release.yml`) — creates a tagged release with
  changelog notes automatically on every push to `main`
- **CodeQL security scanning** (`codeql.yml`) — static analysis on push/PR and
  on a weekly schedule

### Changed
- Updated GitHub Actions: `actions/checkout` v4→v6, `astral-sh/setup-uv` v5→v7,
  `actions/upload-artifact` v4→v7 (via Dependabot)
- Dependabot PRs now target the `develop` branch

---

## [0.0.5] — 2026-03-16

### Added
- **Wav2Vec2 model** — `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`
  (Wav2Vec2-XLSR-300M fine-tuned on ASVspoof2019, EER 4.01%)
- **HuBERT model** — `abhishtagatya/hubert-base-960h-itw-deepfake`
  (HuBERT-base fine-tuned on the In-The-Wild dataset, EER 1.43%)
- **WavLM model** — `DavidCombei/wavLM-base-Deepfake_V2`
  (WavLM-base fine-tuned on a mixed deepfake dataset)
- **Label normaliser** (`Utilities/label_normalizer.py`) — maps arbitrary
  external model labels (`"real"`, `"fake"`, `"LABEL_0"`, etc.) to the
  canonical `"Bonafide"` / `"Spoof"` labels used throughout the API
- `just publish-test` command to publish a dev build to TestPyPI locally
- `just version` command to print the current package version

### Changed
- **CI/CD pipeline consolidated** — three separate workflow files
  (`ruff.yml`, `python-package.yml`, `python-publish.yml`) replaced by a
  single `ci.yml` with a clear lint → test → publish job graph
- **Push to `develop`** now automatically publishes a pre-release
  (`{version}.dev{run_number}`) to TestPyPI after CI passes
- **Push to `main`** now automatically publishes the release version to PyPI
  after CI passes (previously required a manual GitHub release)
- Test matrix extended to Python 3.10–3.14
- Label normalisation now applied consistently across all transformer models
  (AST, VIT, Wav2Vec2, HuBERT, WavLM) via shared `normalize_pipeline_scores()`
- `_result_from_scores()` now sorts scores by confidence descending,
  guaranteeing the top prediction is always the highest-confidence label
- All package sub-directories now include `__init__.py` for reliable imports

### Fixed
- `RawNet2/run.py`: removed `os.chdir()` at module level — was mutating the
  working directory for the entire process
- `label_normalizer`: fixed unsafe substring matching for short digit strings
  (`"0"`, `"1"`) that could produce false-positive label mappings
- `VIT/utility.py`: call `img.load()` after `Image.open()` to force pixel
  data into memory before the buffer is released; `plt.close()` now runs
  in a `try/finally` block
- `jabberjay.load()`: raises `FileNotFoundError` with a clear message for
  missing files; wraps other librosa errors in a descriptive `ValueError`
- `jabberjay.detect()`: raises `ValueError` immediately for empty audio arrays
- `EnumAction`: invalid CLI values now produce a clear error listing valid
  choices instead of a raw `KeyError`
- `hugging_face.download_pretrained_model`: added missing `-> str` return type
- `Classical/run.py`: corrected return type from `tuple[object, float]` to
  `tuple[int, float]`

---

## [0.0.4] — 2024-08-15

### Added
- ASVspoof5 variants for all ViT visualisation types (ConstantQ,
  MelSpectrogram, MFCC) — nine new HuggingFace models in total
- Updated GitHub Actions versions

---

## [0.0.3] — 2024-01-30

### Added
- PyPI publish workflow (`python-publish.yml`) for automated package releases

### Changed
- Core dependencies added to `pyproject.toml`

---

## [0.0.2] — 2024-01-30

### Changed
- Dependencies pinned and added to package metadata

---

## [0.0.1] — 2023-09-06

### Added
- Initial project setup and packaging (`pyproject.toml`, `setup.py`)
- **Classical model** — feature-based KNN classifier for bonafide/spoof detection
- **RawNet2 model** — end-to-end anti-spoofing via
  [rawnet2-antispoofing](https://github.com/eurecom-asp/rawnet2-antispoofing)
  with weights from ASVspoof 2021
- **ViT models** — Vision Transformer image classifiers over three audio
  visualisations (ConstantQ, MelSpectrogram, MFCC) trained on ASVspoof2019
  and VoxCelebSpoof
- **AST model** — Audio Spectrogram Transformer classifier trained on
  ASVspoof2019 and VoxCelebSpoof
- HuggingFace Hub integration for automatic model weight retrieval
- Command-line interface (`jabberjay <audio>`)
- GitHub Actions CI workflow and ruff linting

[0.0.8]: https://github.com/MattyB95/Jabberjay/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/MattyB95/Jabberjay/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/MattyB95/Jabberjay/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/MattyB95/Jabberjay/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/MattyB95/Jabberjay/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/MattyB95/Jabberjay/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/MattyB95/Jabberjay/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/MattyB95/Jabberjay/releases/tag/v0.0.1
