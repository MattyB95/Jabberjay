import soundfile
from transformers import AudioClassificationPipeline, AutoModelForAudioClassification, AutoFeatureExtractor

MODEL_NAME = "checkpoint-68526"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

pipe = AudioClassificationPipeline(model=model, feature_extractor=feature_extractor)
audio = soundfile.read("6-Synthetic.wav")[0]

prediction = pipe(audio, return_all_scores=True)
print(prediction)
