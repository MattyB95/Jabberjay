# Changelog

All notable changes to Jabberjay are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Jabberjay uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.5] — 2026-03-16

### Added
- **Wav2Vec2 model** — `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`
  (Wav2Vec2-XLSR-300M fine-tuned on ASVspoof2019, EER 4.01%)
- **HuBERT model** — `abhishtagatya/hubert-base-960h-itw-deepfake`
  (HuBERT-base fine-tuned on the In-The-Wild dataset, EER 1.43%)
- **WavLM model** — `DavidCombei/wavLM-base-Deepfake_V2`
  (WavLM-base fine-tuned on a mixed deepfake dataset)
- **Label normaliser** — maps arbitrary external model labels (`"real"`, `"fake"`, `"LABEL_0"`, etc.) to the
  canonical `"Bonafide"` / `"Spoof"` labels used throughout the API
- `just publish-test` command to publish a dev build to TestPyPI locally
- `just version` command to print the current package version

### Changed
- CI/CD pipeline consolidated into a single `ci.yml` with a clear lint → test → publish job graph
- Push to `develop` now automatically publishes a pre-release to TestPyPI after CI passes
- Push to `main` now automatically publishes the release version to PyPI after CI passes
- Test matrix extended to Python 3.10–3.14
- Label normalisation applied consistently across all transformer models via shared `normalize_pipeline_scores()`
- `_result_from_scores()` now sorts scores by confidence descending
- All package sub-directories now include `__init__.py` for reliable imports

### Fixed
- `RawNet2/run.py`: removed `os.chdir()` at module level — was mutating the working directory for the entire process
- `label_normalizer`: fixed unsafe substring matching for short digit strings
- `VIT/utility.py`: call `img.load()` after `Image.open()`; `plt.close()` now runs in a `try/finally` block
- `jabberjay.load()`: raises `FileNotFoundError` with a clear message for missing files
- `jabberjay.detect()`: raises `ValueError` immediately for empty audio arrays
- `EnumAction`: invalid CLI values now produce a clear error listing valid choices

---

## [0.0.4] — 2024-08-15

### Added
- ASVspoof5 variants for all ViT visualisation types — nine new HuggingFace models in total
- Updated GitHub Actions versions

---

## [0.0.3] — 2024-01-30

### Added
- PyPI publish workflow for automated package releases

### Changed
- Core dependencies added to `pyproject.toml`

---

## [0.0.2] — 2024-01-30

### Changed
- Dependencies pinned and added to package metadata

---

## [0.0.1] — 2023-09-06

### Added
- Initial project setup
- Classical, RawNet2, ViT, and AST models
- HuggingFace Hub integration for automatic model weight retrieval
- Command-line interface

[0.0.5]: https://github.com/MattyB95/Jabberjay/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/MattyB95/Jabberjay/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/MattyB95/Jabberjay/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/MattyB95/Jabberjay/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/MattyB95/Jabberjay/releases/tag/v0.0.1
