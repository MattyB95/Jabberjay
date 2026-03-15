"""
run_all.py — exhaustive sweep across every model, dataset, and visualisation.

Useful as a smoke test to confirm every combination works end-to-end.
Expect this to take several minutes as each model is downloaded on first run
and cached by HuggingFace Hub locally (~/.cache/huggingface).

Run:
    uv run python examples/run_all.py
    uv run python examples/run_all.py res/spoof/spoof.flac   # custom file
"""

import sys
from pathlib import Path

from Jabberjay import Dataset, Jabberjay, Model, Visualisation

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("res/bonafide/bonafide.flac")

jj = Jabberjay()
audio = jj.load(AUDIO)

print(f"Audio: {AUDIO}\n")
print(f"{'Model':<30}  {'Label':<10}  {'Confidence':>10}")
print("-" * 56)


def row(tag: str, result) -> None:
    print(f"{tag:<30}  {result.label:<10}  {result.confidence:>10.1%}")


# Self-contained models
for model in (
    Model.Classical,
    Model.RawNet2,
    Model.Wav2Vec2,
    Model.HuBERT,
    Model.WavLM,
):
    row(model.value, jj.detect(audio, model=model))

# AST — one model per dataset
for ds in Dataset:
    row(f"AST/{ds.value}", jj.detect(audio, model=Model.AST, dataset=ds))

# VIT — one model per (dataset × visualisation) combination
for vis in Visualisation:
    for ds in Dataset:
        row(
            f"VIT/{ds.value}/{vis.value}",
            jj.detect(audio, model=Model.VIT, dataset=ds, visualisation=vis),
        )
