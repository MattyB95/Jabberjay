from Jabberjay.Utilities.enum_handler import Visualisation, Model, Dataset
from Jabberjay.jabberjay import Jabberjay

jabberjay = Jabberjay()

jabberjay.list_models()
jabberjay.list_datasets()
jabberjay.list_visualisations()

bonafide = jabberjay.load(filename="../res/bonafide/bonafide.flac")
spoof = jabberjay.load(filename="../res/spoof/spoof.flac")

# Bonafide
print("")
print("Bonafide")
jabberjay.detect(audio=bonafide, model=Model.Classical)

jabberjay.detect(audio=bonafide, model=Model.RawNet2)

jabberjay.detect(
    audio=bonafide,
    model=Model.VIT,
    visualisation=Visualisation.ConstantQ,
    dataset=Dataset.ASVspoof2019,
)
jabberjay.detect(
    audio=bonafide,
    model=Model.VIT,
    visualisation=Visualisation.ConstantQ,
    dataset=Dataset.VoxCelebSpoof,
)

jabberjay.detect(
    audio=bonafide,
    model=Model.VIT,
    visualisation=Visualisation.MelSpectrogram,
    dataset=Dataset.ASVspoof2019,
)
jabberjay.detect(
    audio=bonafide,
    model=Model.VIT,
    visualisation=Visualisation.MelSpectrogram,
    dataset=Dataset.VoxCelebSpoof,
)

jabberjay.detect(
    audio=bonafide,
    model=Model.VIT,
    visualisation=Visualisation.MFCC,
    dataset=Dataset.ASVspoof2019,
)
jabberjay.detect(
    audio=bonafide,
    model=Model.VIT,
    visualisation=Visualisation.MFCC,
    dataset=Dataset.VoxCelebSpoof,
)

jabberjay.detect(audio=bonafide, model=Model.AST, dataset=Dataset.ASVspoof2019)
jabberjay.detect(audio=bonafide, model=Model.AST, dataset=Dataset.VoxCelebSpoof)

# Spoof
print("")
print("Spoof")
jabberjay.detect(audio=spoof, model=Model.Classical)

jabberjay.detect(audio=spoof, model=Model.RawNet2)

jabberjay.detect(
    audio=spoof,
    model=Model.VIT,
    visualisation=Visualisation.ConstantQ,
    dataset=Dataset.ASVspoof2019,
)
jabberjay.detect(
    audio=spoof,
    model=Model.VIT,
    visualisation=Visualisation.ConstantQ,
    dataset=Dataset.VoxCelebSpoof,
)

jabberjay.detect(
    audio=spoof,
    model=Model.VIT,
    visualisation=Visualisation.MelSpectrogram,
    dataset=Dataset.ASVspoof2019,
)
jabberjay.detect(
    audio=spoof,
    model=Model.VIT,
    visualisation=Visualisation.MelSpectrogram,
    dataset=Dataset.VoxCelebSpoof,
)

jabberjay.detect(
    audio=spoof,
    model=Model.VIT,
    visualisation=Visualisation.MFCC,
    dataset=Dataset.ASVspoof2019,
)
jabberjay.detect(
    audio=spoof,
    model=Model.VIT,
    visualisation=Visualisation.MFCC,
    dataset=Dataset.VoxCelebSpoof,
)

jabberjay.detect(audio=spoof, model=Model.AST, dataset=Dataset.ASVspoof2019)
jabberjay.detect(audio=spoof, model=Model.AST, dataset=Dataset.VoxCelebSpoof)
