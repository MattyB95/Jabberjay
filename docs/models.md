# Models

Jabberjay bundles ten model families. Each is downloaded from Hugging Face Hub on first use and cached locally.

---

## Choosing a model

| Model                     | Type                          | Datasets                               | Requires                   |
|---------------------------|-------------------------------|----------------------------------------|----------------------------|
| [VIT](#vit)               | Vision Transformer            | ASVspoof2019, ASVspoof5, VoxCelebSpoof | `dataset`, `visualisation` |
| [AST](#ast)               | Audio Spectrogram Transformer | ASVspoof2019, ASVspoof5, VoxCelebSpoof | `dataset`                  |
| [Spectra0](#spectra0)         | Wav2Vec2 + ECAPA-TDNN         | ASVspoof19/21, ASVspoof5, In-the-Wild  | —                          |
| [SpectraAASIST](#spectraaasist)   | Wav2Vec2 + AASIST             | ASVspoof19/21, ASVspoof5, In-the-Wild  | —                          |
| [SpectraAASIST3](#spectraaasist3) | Wav2Vec2 + KAN-AASIST         | ASVspoof19/21, ASVspoof5, In-the-Wild  | —                          |
| [Wav2Vec2](#wav2vec2)     | Self-supervised transformer   | ASVspoof2019                           | —                          |
| [HuBERT](#hubert)         | Self-supervised transformer   | In-The-Wild                            | —                          |
| [WavLM](#wavlm)           | Self-supervised transformer   | Mixed deepfake                         | —                          |
| [RawNet2](#rawnet2)       | End-to-end CNN                | ASVspoof 2021                          | —                          |
| [Classical](#classical)   | KNN classifier                | ASVspoof2019                           | —                          |

**Simple rule of thumb:**

- For the lowest error rate on In-the-Wild audio — use **SpectraAASIST3** (EER 0.961%)
- For a quick, general-purpose result — use **WavLM** or **HuBERT**
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

## Spectra0

`lab260/spectra_0`

SSL encoder (Wav2Vec2-XLS-R-300M) → MLP projection → ECAPA-TDNN binary classifier, trained across multiple anti-spoofing benchmarks.

| Benchmark       | EER (%) |
|-----------------|---------|
| ASVspoof19 LA   | 0.181   |
| ASVspoof21 LA   | 6.475   |
| ASVspoof21 DF   | 5.410   |
| ASVspoof5       | 14.426  |
| ADD2022         | 14.716  |
| In-the-Wild     | 1.026   |

```python
jj.detect("audio.wav", model="Spectra0")
```

!!! note
    On first run, Spectra0 downloads both its own weights and the `facebook/wav2vec2-xls-r-300m` SSL encoder (~1.2 GB total). Both are cached by HuggingFace Hub after the first download.

---

## SpectraAASIST

`lab260/Spectra-AASIST`

SSL encoder (Wav2Vec2-XLS-R-300M) → MLP projection → AASIST graph attention network with standard linear layers.

| Benchmark       | EER (%) |
|-----------------|---------|
| ASVspoof19 LA   | 0.159   |
| ASVspoof21 LA   | 5.164   |
| ASVspoof21 DF   | 2.568   |
| ASVspoof5       | 14.056  |
| ADD2022         | 15.205  |
| In-the-Wild     | 1.461   |

```python
jj.detect("audio.wav", model="SpectraAASIST")
```

!!! note
    Downloads `facebook/wav2vec2-xls-r-300m` on first run (~1.2 GB total, cached by HuggingFace Hub).

---

## SpectraAASIST3

`lab260/Spectra-AASIST3`

SSL encoder (Wav2Vec2-XLS-R-300M) → MLP projection → AASIST graph attention network with **KAN (Kolmogorov-Arnold Network)** linear layers — the strongest model in the Spectra family on In-the-Wild audio.

| Benchmark       | EER (%) |
|-----------------|---------|
| ASVspoof19 LA   | 0.723   |
| ASVspoof21 LA   | 4.506   |
| ASVspoof21 DF   | 1.998   |
| ASVspoof5       | 13.820  |
| ADD2022         | 15.187  |
| In-the-Wild     | 0.961   |

```python
jj.detect("audio.wav", model="SpectraAASIST3")
```

!!! note
    Downloads `facebook/wav2vec2-xls-r-300m` on first run (~1.2 GB total, cached by HuggingFace Hub).

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

HuBERT-base fine-tuned on the **In-The-Wild** dataset. EER 1.43%.

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
