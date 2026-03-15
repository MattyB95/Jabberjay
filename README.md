# Jabberjay

🦜 Synthetic Voice Detection

## Installation

```bash
pip install jabberjay
```

## Models

### Vision Transformer

| **Name**                                                             | **Model** | **Dataset**   | **Visualisation** | **Model**                                                                                                   |
|----------------------------------------------------------------------|-----------|---------------|-------------------|-------------------------------------------------------------------------------------------------------------|
| MattyB95/VIT-ASVspoof2019-ConstantQ-Synthetic-Voice-Detection        | ViT       | ASVspoof2019  | ConstantQ         | [Hugging Face](https://huggingface.co/MattyB95/VIT-ASVspoof2019-ConstantQ-Synthetic-Voice-Detection)        |
| MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection  | ViT       | ASVspoof2019  | MelSpectrogram    | [Hugging Face](https://huggingface.co/MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection)  |
| MattyB95/VIT-ASVspoof2019-MFCC-Synthetic-Voice-Detection             | ViT       | ASVspoof2019  | MFCC              | [Hugging Face](https://huggingface.co/MattyB95/VIT-ASVspoof2019-MFCC-Synthetic-Voice-Detection)             |
| MattyB95/VIT-ASVspoof5-ConstantQ-Synthetic-Voice-Detection           | ViT       | ASVspoof5     | ConstantQ         | [Hugging Face](https://huggingface.co/MattyB95/VIT-ASVspoof5-ConstantQ-Synthetic-Voice-Detection)           |
| MattyB95/VIT-ASVspoof5-Mel_Spectrogram-Synthetic-Voice-Detection     | ViT       | ASVspoof5     | MelSpectrogram    | [Hugging Face](https://huggingface.co/MattyB95/VIT-ASVspoof5-Mel_Spectrogram-Synthetic-Voice-Detection)     |
| MattyB95/VIT-ASVspoof5-MFCC-Synthetic-Voice-Detection                | ViT       | ASVspoof5     | MFCC              | [Hugging Face](https://huggingface.co/MattyB95/VIT-ASVspoof5-MFCC-Synthetic-Voice-Detection)                |
| MattyB95/VIT-VoxCelebSpoof-ConstantQ-Synthetic-Voice-Detection       | ViT       | VoxCelebSpoof | ConstantQ         | [Hugging Face](https://huggingface.co/MattyB95/VIT-VoxCelebSpoof-ConstantQ-Synthetic-Voice-Detection)       |
| MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection | ViT       | VoxCelebSpoof | MelSpectrogram    | [Hugging Face](https://huggingface.co/MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection) |
| MattyB95/VIT-VoxCelebSpoof-MFCC-Synthetic-Voice-Detection            | ViT       | VoxCelebSpoof | MFCC              | [Hugging Face](https://huggingface.co/MattyB95/VIT-VoxCelebSpoof-MFCC-Synthetic-Voice-Detection)            |

### Audio Spectrogram Transformer

| **Name**                                             | **Model** | **Dataset**   | **Model**                                                                                   |
|------------------------------------------------------|-----------|---------------|---------------------------------------------------------------------------------------------|
| MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection  | AST       | ASVspoof2019  | [Hugging Face](https://huggingface.co/MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection)  |
| MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection     | AST       | ASVspoof5     | [Hugging Face](https://huggingface.co/MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection)     |
| MattyB95/AST-VoxCelebSpoof-Synthetic-Voice-Detection | AST       | VoxCelebSpoof | [Hugging Face](https://huggingface.co/MattyB95/AST-VoxCelebSpoof-Synthetic-Voice-Detection) |

### Other

| Name      | Paper                                                                                     | Codebase                                                                    | Model                                                                                          |
|-----------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Classical | Placeholder                                                                               | Placeholder                                                                 | Placeholder                                                                                    |
| RawNet2   | [End-to-End anti-spoofing with RawNet2](https://doi.org/10.1109/ICASSP39728.2021.9414234) | [rawnet2-antispoofing](https://github.com/eurecom-asp/rawnet2-antispoofing) | [pre_trained_DF_RawNet2.zip](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip) |

## Usage

### Command Line Interface

```
usage: Jabberjay [-h] [-m {AST,Classical,RawNet2,VIT}]
                 [-d {ASVspoof2019,ASVspoof5,VoxCelebSpoof}]
                 [-vis {ConstantQ,MelSpectrogram,MFCC}] [-v]
                 audio
```

**Examples**

```bash
# Quickstart — VIT with ConstantQ on VoxCelebSpoof (defaults)
jabberjay audio.wav

# Specify model and dataset
jabberjay audio.wav -m RawNet2
jabberjay audio.wav -m AST -d ASVspoof2019

# VIT with full options
jabberjay audio.wav -m VIT -d ASVspoof5 -vis MelSpectrogram

# Verbose output
jabberjay audio.wav -v
```

### Python API

All public names are importable directly from the top-level package:

```python
from Jabberjay import Jabberjay, DetectionResult, Model, Dataset, Visualisation
```

#### Quickstart

Pass a file path straight to `detect()` — no separate load step required:

```python
from Jabberjay import Jabberjay

jj = Jabberjay()
result = jj.detect("audio.wav")

print(result)           # Bonafide ✔️ (92.3% confidence, model=VIT)
print(result.label)     # "Bonafide"
print(result.is_bonafide)  # True
print(result.confidence)   # 0.923
```

#### Choosing a model

String names and enum values are both accepted:

```python
# Using strings
result = jj.detect("audio.wav", model="Classical")
result = jj.detect("audio.wav", model="RawNet2")
result = jj.detect("audio.wav", model="AST", dataset="VoxCelebSpoof")
result = jj.detect("audio.wav", model="VIT", dataset="ASVspoof5", visualisation="MFCC")

# Using enums (identical behaviour)
from Jabberjay import Model, Dataset, Visualisation

result = jj.detect("audio.wav", model=Model.VIT, dataset=Dataset.ASVspoof5, visualisation=Visualisation.MFCC)
```

#### DetectionResult

Every call to `detect()` returns a `DetectionResult` regardless of the model used:

| Attribute | Type | Description |
|---|---|---|
| `label` | `str` | `"Bonafide"` or `"Spoof"` |
| `is_bonafide` | `bool` | `True` if the audio is classified as genuine |
| `confidence` | `float` | Confidence score for the top prediction (0.0–1.0) |
| `model` | `Model` | The model that produced this result |
| `scores` | `list[dict] \| None` | Full label/score breakdown for VIT and AST models; `None` for Classical and RawNet2 |

```python
result = jj.detect("audio.wav", model="VIT", dataset="VoxCelebSpoof", visualisation="ConstantQ")

if result.is_bonafide:
    print(f"Genuine voice detected with {result.confidence:.1%} confidence")
else:
    print(f"Synthetic voice detected with {result.confidence:.1%} confidence")

# Full scores available for VIT and AST
if result.scores:
    for entry in result.scores:
        print(f"  {entry['label']}: {entry['score']:.3f}")
```

#### Pre-loading audio

Use `load()` when running multiple models on the same clip to avoid re-reading the file:

```python
audio = jj.load("audio.wav")  # returns (samples, sample_rate)

result_classical = jj.detect(audio, model="Classical")
result_rawnet2   = jj.detect(audio, model="RawNet2")
result_vit       = jj.detect(audio, model="VIT", dataset="VoxCelebSpoof", visualisation="ConstantQ")
```

#### Discovering available options

```python
models       = jj.list_models()        # prints and returns list[Model]
datasets     = jj.list_datasets()      # prints and returns list[Dataset]
visualisations = jj.list_visualisations()  # prints and returns list[Visualisation]
```

## Developer Setup

```bash
# Install dependencies (requires uv)
just install

# Run tests
just test

# Lint, format check, and type check
just check

# Auto-fix lint issues and format
just fix
```

See `just --list` for all available commands.

## Contributing to Jabberjay

🌟 We value your contributions!

Whether you're fixing a bug, improving the documentation,
or proposing a new feature, we're delighted to have you as part of the Jabberjay community.
Your efforts help us make Synthetic Voice Detection even better for everyone.

We especially welcome and encourage additional models for speech deepfake (bonafide vs. spoof) detection,
with the aim of making Jabberjay the one-stop shop for state-of-the-art models in the field.

We are truly grateful for your interest in improving Jabberjay.
Your contributions, no matter how big or small, make our open-source community a vibrant place to learn, inspire,
and create.

Let's make Jabberjay the best tool for Synthetic Voice Detection together! 🚀

## Acknowledgement

This work was supported, in whole or in part, by the Bill & Melinda Gates Foundation [INV-001309].
