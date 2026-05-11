# CLI Reference

Jabberjay ships a command-line interface installed as `jabberjay`.

## Usage

```
jabberjay <audio> [-m MODEL] [-d DATASET] [-vis VISUALISATION] [-v]
```

## Arguments

| Argument                  | Description                       | Default         |
|---------------------------|-----------------------------------|-----------------|
| `audio`                   | Path to the audio file to analyse | —               |
| `-m`, `--model`           | Model to use                      | `VIT`           |
| `-d`, `--dataset`         | Dataset the model was trained on  | `VoxCelebSpoof` |
| `-vis`, `--visualisation` | Visualisation type (VIT only)     | `ConstantQ`     |
| `-v`, `--verbose`         | Enable debug logging to stderr    | off             |

## Valid values

**Models:** `VIT`, `AST`, `Spectra0`, `SpectraAASIST`, `SpectraAASIST3`, `Wav2Vec2`, `HuBERT`, `WavLM`, `RawNet2`, `Classical`

**Datasets:** `ASVspoof2019`, `ASVspoof5`, `VoxCelebSpoof`

**Visualisations:** `ConstantQ`, `MelSpectrogram`, `MFCC`

## Examples

```bash
# Default — VIT model, ConstantQ, VoxCelebSpoof
jabberjay interview.wav

# Choose a model
jabberjay interview.wav -m Spectra0
jabberjay interview.wav -m HuBERT
jabberjay interview.wav -m Wav2Vec2
jabberjay interview.wav -m RawNet2

# AST with a specific dataset
jabberjay interview.wav -m AST -d ASVspoof2019

# VIT with a specific visualisation and dataset
jabberjay interview.wav -m VIT -d ASVspoof5 -vis MFCC

# Enable verbose logging
jabberjay interview.wav -m WavLM -v
```

## Output

```
Bonafide ✔️ (94.1% confidence, model=VIT)
```

or

```
Spoof ❌ (97.8% confidence, model=HuBERT)
```
