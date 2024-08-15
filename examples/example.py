import numpy as np

from Jabberjay.Utilities.enum_handler import Visualisation, Model, Dataset
from Jabberjay.jabberjay import Jabberjay

jabberjay = Jabberjay()

jabberjay.list_models()
jabberjay.list_datasets()
jabberjay.list_visualisations()


def run_models(audio: tuple[np.ndarray, float]) -> None:
    jabberjay.detect(audio=audio, model=Model.Classical)
    jabberjay.detect(audio=audio, model=Model.RawNet2)
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.ConstantQ,
        dataset=Dataset.ASVspoof2019,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.ConstantQ,
        dataset=Dataset.ASVspoof5,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.ConstantQ,
        dataset=Dataset.VoxCelebSpoof,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.MelSpectrogram,
        dataset=Dataset.ASVspoof2019,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.MelSpectrogram,
        dataset=Dataset.ASVspoof5,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.MelSpectrogram,
        dataset=Dataset.VoxCelebSpoof,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.MFCC,
        dataset=Dataset.ASVspoof2019,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.MFCC,
        dataset=Dataset.ASVspoof5,
    )
    jabberjay.detect(
        audio=audio,
        model=Model.VIT,
        visualisation=Visualisation.MFCC,
        dataset=Dataset.VoxCelebSpoof,
    )
    jabberjay.detect(audio=audio, model=Model.AST, dataset=Dataset.ASVspoof2019)
    jabberjay.detect(audio=audio, model=Model.AST, dataset=Dataset.ASVspoof5)
    jabberjay.detect(audio=audio, model=Model.AST, dataset=Dataset.VoxCelebSpoof)


bonafide = jabberjay.load(filename="../res/bonafide/bonafide.flac")
spoof = jabberjay.load(filename="../res/spoof/spoof.flac")

# Bonafide
print("")
print("Bonafide")
run_models(audio=bonafide)

# Spoof
print("")
print("Spoof")
run_models(audio=spoof)
