# CLI Reference

Jabberjay ships a command-line interface installed as `jabberjay`.

## Usage

```
jabberjay [-h] [-m MODEL] [-d DATASET] [-vis VISUALISATION] [-v] audio
```

## Arguments

| Argument                  | Description                                        | Default         |
|---------------------------|---------------------------------------------------|-----------------|
| `audio`                   | Path to the audio file to analyse                  | —               |
| `-m`, `--model`           | Detection model to use                             | `VIT`           |
| `-d`, `--dataset`         | Training dataset for the VIT and AST models        | `VoxCelebSpoof` |
| `-vis`, `--visualisation` | Spectrogram type for the VIT model                 | `ConstantQ`     |
| `-v`, `--verbose`         | Print model load and inference progress to stderr  | off             |

## Valid values

**Models:** `AST`, `Classical`, `HuBERT`, `RawNet2`, `Spectra0`, `SpectraAASIST`, `SpectraAASIST3`, `VIT`, `Wav2Vec2`, `WavLM`

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

## Exit codes

| Code | Meaning                                                                 |
|------|------------------------------------------------------------------------|
| `0`  | Detection succeeded — the verdict is printed to stdout                 |
| `1`  | Expected error (file not found, unreadable audio, invalid options) — a one-line `Error: …` message is printed to stderr |
| `2`  | Invalid command-line arguments (unknown model/dataset/visualisation)  |
