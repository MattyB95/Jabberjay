"""
preloading_audio.py — avoid re-reading the file when running multiple models.

jj.load() returns a (samples, sample_rate) tuple that can be passed directly
to jj.detect(). This saves a librosa.load() call for every extra model you run.

Run:
    uv run python examples/preloading_audio.py
"""

from Jabberjay import Dataset, Jabberjay, Model, Visualisation

jj = Jabberjay()

# Load once …
audio = jj.load("res/bonafide/bonafide.flac")

# … run as many models as you like against the same clip.
results = [
    jj.detect(audio, model=Model.Classical),
    jj.detect(audio, model=Model.RawNet2),
    jj.detect(audio, model=Model.Wav2Vec2),
    jj.detect(audio, model=Model.HuBERT),
    jj.detect(audio, model=Model.WavLM),
    jj.detect(audio, model=Model.AST, dataset=Dataset.VoxCelebSpoof),
    jj.detect(
        audio,
        model=Model.VIT,
        dataset=Dataset.VoxCelebSpoof,
        visualisation=Visualisation.ConstantQ,
    ),
]

print(f"{'Model':<12}  {'Label':<10}  {'Confidence':>10}")
print("-" * 38)
for r in results:
    print(f"{r.model.value:<12}  {r.label:<10}  {r.confidence:>10.1%}")
