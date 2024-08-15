# Jabberjay

🦜 Synthetic Voice Detection

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

### Python API

```
from Jabberjay.Utilities.enum_handler import Visualisation, Model, Dataset
from Jabberjay.jabberjay import Jabberjay

jabberjay = Jabberjay()

bonafide = jabberjay.load(filename="../res/bonafide/bonafide.flac")
spoof = jabberjay.load(filename="../res/spoof/spoof.flac")

jabberjay.detect(audio=bonafide, model=Model.VIT, visualisation=Visualisation.ConstantQ, dataset=Dataset.VoxCelebSpoof)
jabberjay.detect(audio=spoof, model=Model.VIT, visualisation=Visualisation.ConstantQ, dataset=Dataset.VoxCelebSpoof)
```

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
