# Changelog

All notable changes to Jabberjay are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Jabberjay uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — 0.0.5

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
- Test matrix extended to Python 3.10, 3.11, and 3.12

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

[Unreleased]: https://github.com/MattyB95/Jabberjay/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/MattyB95/Jabberjay/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/MattyB95/Jabberjay/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/MattyB95/Jabberjay/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/MattyB95/Jabberjay/releases/tag/v0.0.1
