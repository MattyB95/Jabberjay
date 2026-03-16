# Models

Jabberjay bundles seven model families. Each is downloaded from Hugging Face Hub on first use and cached locally.

---

## Choosing a model

| Model                   | Type                          | Datasets                               | Requires                   |
|-------------------------|-------------------------------|----------------------------------------|----------------------------|
| [VIT](#vit)             | Vision Transformer            | ASVspoof2019, ASVspoof5, VoxCelebSpoof | `dataset`, `visualisation` |
| [AST](#ast)             | Audio Spectrogram Transformer | ASVspoof2019, ASVspoof5, VoxCelebSpoof | `dataset`                  |
| [Wav2Vec2](#wav2vec2)   | Self-supervised transformer   | ASVspoof2019                           | —                          |
| [HuBERT](#hubert)       | Self-supervised transformer   | In-The-Wild                            | —                          |
| [WavLM](#wavlm)         | Self-supervised transformer   | Mixed deepfake                         | —                          |
| [RawNet2](#rawnet2)     | End-to-end CNN                | ASVspoof 2021                          | —                          |
| [Classical](#classical) | KNN classifier                | ASVspoof2019                           | —                          |

**Simple rule of thumb:**

- For a quick, general-purpose result — use **WavLM** or **HuBERT**
- For the lowest error rate on In-The-Wild audio — use **HuBERT** (EER 1.43%)
- For a lightweight baseline with no deep learning — use **Classical**
- To sweep all models and compare — use `jj.load()` once, then call `jj.detect()` for each

---

## VIT

Vision Transformer classifiers that convert the audio to a 2D image (spectrogram) and classify it visually.

**Nine variants**, covering three visualisation types × three training datasets:

| Visualisation  | ASVspoof2019 | ASVspoof5 | VoxCelebSpoof |
|----------------|:------------:|:---------:|:-------------:|
| ConstantQ      |      ✓       |     ✓     |       ✓       |
| MelSpectrogram |      ✓       |     ✓     |       ✓       |
| MFCC           |      ✓       |     ✓     |       ✓       |

```python
jj.detect("audio.wav", model="VIT", dataset="VoxCelebSpoof", visualisation="ConstantQ")
jj.detect("audio.wav", model="VIT", dataset="ASVspoof2019", visualisation="MFCC")
jj.detect("audio.wav", model="VIT", dataset="ASVspoof5", visualisation="MelSpectrogram")
```

**Visualisations:**

- `ConstantQ` — Constant-Q transform; good frequency resolution across the full spectrum
- `MelSpectrogram` — Mel-scaled spectrogram; perceptually motivated, widely used in speech
- `MFCC` — Mel-frequency cepstral coefficients; compact and speech-focused

---

## AST

Audio Spectrogram Transformer. Applies a transformer directly to a patch-based spectrogram without a CNN backbone.

Available for **ASVspoof2019**, **ASVspoof5**, and **VoxCelebSpoof** datasets.

```python
jj.detect("audio.wav", model="AST")  # defaults to VoxCelebSpoof
jj.detect("audio.wav", model="AST", dataset="VoxCelebSpoof")
jj.detect("audio.wav", model="AST", dataset="ASVspoof2019")
```

!!! note
    `dataset` is optional for AST and defaults to `VoxCelebSpoof` if not specified.

---

## Wav2Vec2

`Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`

Wav2Vec2-XLSR-300M fine-tuned on **ASVspoof2019**. EER 4.01%.

```python
jj.detect("audio.wav", model="Wav2Vec2")
```

---

## HuBERT

`abhishtagatya/hubert-base-960h-itw-deepfake`

HuBERT-base fine-tuned on the **In-The-Wild** dataset. EER 1.43% — the lowest of all bundled models on real-world audio.

```python
jj.detect("audio.wav", model="HuBERT")
```

---

## WavLM

`DavidCombei/wavLM-base-Deepfake_V2`

WavLM-base fine-tuned on a **mixed deepfake dataset**.

```python
jj.detect("audio.wav", model="WavLM")
```

---

## RawNet2

End-to-end anti-spoofing network via [rawnet2-antispoofing](https://github.com/eurecom-asp/rawnet2-antispoofing), with weights from **ASVspoof 2021**. Operates directly on raw waveforms — no feature extraction step.

```python
jj.detect("audio.wav", model="RawNet2")
```

!!! note
    `scores` is `None` for RawNet2 — only `label`, `is_bonafide`, and `confidence` are populated.

---

## Classical

Feature-based KNN classifier trained on **ASVspoof2019**. Extracts hand-crafted audio features (MFCCs, spectral features) and classifies with k-nearest neighbours. Fast and dependency-light compared to the transformer models.

```python
jj.detect("audio.wav", model="Classical")
```

!!! note
    `scores` is `None` for Classical — only `label`, `is_bonafide`, and `confidence` are populated.
