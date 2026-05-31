"""
choosing_a_model.py — how to select different models.

Jabberjay bundles seven model families. This file shows how to use each
one, using both plain strings (convenient) and enums (autocomplete-friendly).

Run:
    uv run python examples/choosing_a_model.py
"""

from Jabberjay import Dataset, Jabberjay, Model, Visualisation

jj = Jabberjay()
AUDIO = "res/bonafide/bonafide.flac"

# ── String API ─────────────────────────────────────────────────────────────
# Strings are accepted everywhere — great for scripting and quick experiments.

# Self-contained models (no dataset or visualisation required)
print(jj.detect(AUDIO, model="Classical"))
print(jj.detect(AUDIO, model="RawNet2"))
print(jj.detect(AUDIO, model="Wav2Vec2"))
print(jj.detect(AUDIO, model="HuBERT"))
print(jj.detect(AUDIO, model="WavLM"))

# AST — pick a dataset
print(jj.detect(AUDIO, model="AST", dataset="ASVspoof2019"))
print(jj.detect(AUDIO, model="AST", dataset="ASVspoof5"))
print(jj.detect(AUDIO, model="AST", dataset="VoxCelebSpoof"))

# VIT — pick a dataset and a visualisation
print(jj.detect(AUDIO, model="VIT", dataset="VoxCelebSpoof", visualisation="ConstantQ"))
print(
    jj.detect(AUDIO, model="VIT", dataset="ASVspoof5", visualisation="MelSpectrogram")
)
print(jj.detect(AUDIO, model="VIT", dataset="ASVspoof2019", visualisation="MFCC"))

# ── Enum API ───────────────────────────────────────────────────────────────
# Enums give you IDE autocomplete and catch typos at import time.

result = jj.detect(
    AUDIO,
    model=Model.VIT,
    dataset=Dataset.VoxCelebSpoof,
    visualisation=Visualisation.ConstantQ,
)
print(result)

# ── Discover what's available ──────────────────────────────────────────────
print(jj.list_models())
print(jj.list_datasets())
print(jj.list_visualisations())
