# Getting Started

## Installation

```bash
pip install jabberjay
```

Requires **Python ≥ 3.11**. Models are downloaded from Hugging Face Hub on first use and cached locally — no manual setup required.

---

## Your first detection

Pass a file path directly. The default model is VIT with the ConstantQ visualisation on the VoxCelebSpoof dataset.

```python
from Jabberjay import Jabberjay

jj = Jabberjay()
result = jj.detect("interview.wav")

print(result)            # Bonafide ✔️ (94.1% confidence, model=VIT)
print(result.label)      # "Bonafide" or "Spoof"
print(result.is_bonafide)  # True / False
print(result.confidence) # 0.0 – 1.0
```

---

## Choosing a model

Pass a model name as a string, or use the `Model` enum for IDE autocomplete.

=== "String API"

    ```python
    from Jabberjay import Jabberjay

    jj = Jabberjay()

    # Self-contained models — no extra arguments needed
    jj.detect("audio.wav", model="Classical")
    jj.detect("audio.wav", model="RawNet2")
    jj.detect("audio.wav", model="Spectra0")
    jj.detect("audio.wav", model="SpectraAASIST")
    jj.detect("audio.wav", model="SpectraAASIST3")
    jj.detect("audio.wav", model="Wav2Vec2")
    jj.detect("audio.wav", model="HuBERT")
    jj.detect("audio.wav", model="WavLM")

    # AST — uses a dataset (defaults to VoxCelebSpoof if not specified)
    jj.detect("audio.wav", model="AST")
    jj.detect("audio.wav", model="AST", dataset="VoxCelebSpoof")

    # VIT — requires a dataset and a visualisation
    jj.detect("audio.wav", model="VIT", dataset="VoxCelebSpoof", visualisation="ConstantQ")
    ```

=== "Enum API"

    ```python
    from Jabberjay import Dataset, Jabberjay, Model, Visualisation

    jj = Jabberjay()

    result = jj.detect(
        "audio.wav",
        model=Model.VIT,
        dataset=Dataset.VoxCelebSpoof,
        visualisation=Visualisation.ConstantQ,
    )
    ```

---

## Pre-loading audio

If you want to run multiple models against the same clip, load the audio once with `jj.load()` and reuse the result. This avoids redundant disk reads.

```python
from Jabberjay import Dataset, Jabberjay, Model, Visualisation

jj = Jabberjay()

# Load once …
audio = jj.load("interview.wav")

# … run as many models as you like
results = [
    jj.detect(audio, model=Model.Classical),
    jj.detect(audio, model=Model.RawNet2),
    jj.detect(audio, model=Model.Spectra0),
    jj.detect(audio, model=Model.SpectraAASIST),
    jj.detect(audio, model=Model.SpectraAASIST3),
    jj.detect(audio, model=Model.Wav2Vec2),
    jj.detect(audio, model=Model.HuBERT),
    jj.detect(audio, model=Model.WavLM),
    jj.detect(audio, model=Model.AST, dataset=Dataset.VoxCelebSpoof),
    jj.detect(audio, model=Model.VIT, dataset=Dataset.VoxCelebSpoof,
              visualisation=Visualisation.ConstantQ),
]

print(f"{'Model':<12}  {'Label':<10}  {'Confidence':>10}")
print("-" * 38)
for r in results:
    print(f"{r.model.value:<12}  {r.label:<10}  {r.confidence:>10.1%}")
```

---

## Working with results

Every `detect()` call returns a [`DetectionResult`](api-reference.md) with the same fields regardless of model.

```python
result = jj.detect("audio.wav", model="WavLM")

# Branch on the verdict
if result.is_bonafide:
    print(f"Genuine voice ({result.confidence:.1%} confidence)")
else:
    print(f"Synthetic voice detected ({result.confidence:.1%} confidence)")

# Full score breakdown — available for transformer models (VIT, AST, Spectra0, SpectraAASIST, SpectraAASIST3, Wav2Vec2, HuBERT, WavLM)
# None for Classical and RawNet2
if result.scores:
    for entry in result.scores:
        bar = "█" * int(entry["score"] * 30)
        print(f"  {entry['label']:<10} {entry['score']:.3f}  {bar}")
```

---

## Enabling logging

Jabberjay is silent by default. Call `enable_logging()` to see model load and inference messages.

```python
Jabberjay.enable_logging()          # DEBUG and above
Jabberjay.enable_logging("INFO")    # INFO and above only
```

---

## Discovering available options

```python
jj.list_models()         # prints and returns all Model enum values
jj.list_datasets()       # prints and returns all Dataset enum values
jj.list_visualisations() # prints and returns all Visualisation enum values
```
