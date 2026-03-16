"""
exploring_results.py — everything inside a DetectionResult.

Every call to jj.detect() returns the same DetectionResult dataclass,
regardless of which model was used. This file shows all its fields and
how to use them in practice.

Run:
    uv run python examples/exploring_results.py
"""

from Jabberjay import Dataset, Jabberjay, Model, Visualisation

jj = Jabberjay()
AUDIO = "res/bonafide/bonafide.flac"

# ── Core fields (available on every model) ─────────────────────────────────

result = jj.detect(AUDIO, model="RawNet2")

print("label      :", result.label)  # "Bonafide" or "Spoof"
print("is_bonafide:", result.is_bonafide)  # True / False
print("confidence :", result.confidence)  # 0.0 – 1.0
print("model      :", result.model)  # Model.RawNet2
print("str        :", result)  # Bonafide ✔️ (92.3% confidence, model=RawNet2)
print()

# ── Branch on the result ───────────────────────────────────────────────────

if result.is_bonafide:
    print(f"Genuine voice detected ({result.confidence:.1%} confidence)")
else:
    print(f"Synthetic voice detected ({result.confidence:.1%} confidence)")
print()

# ── Full score breakdown (VIT, AST, Wav2Vec2, HuBERT, WavLM) ─────────────
# `scores` is a list[{"label": str, "score": float}] sorted highest-first.
# It is None for Classical and RawNet2.

result = jj.detect(
    AUDIO,
    model=Model.VIT,
    dataset=Dataset.VoxCelebSpoof,
    visualisation=Visualisation.ConstantQ,
)

if result.scores:
    print("Full score breakdown:")
    for entry in result.scores:
        bar = "█" * int(entry["score"] * 30)
        print(f"  {entry['label']:<10} {entry['score']:.3f}  {bar}")
print()

# ── Comparing bonafide vs spoof ────────────────────────────────────────────

for path, label in [
    ("res/bonafide/bonafide.flac", "bonafide sample"),
    ("res/spoof/spoof.flac", "spoof sample"),
]:
    r = jj.detect(path, model="WavLM")
    verdict = "✔️ Bonafide" if r.is_bonafide else "❌ Spoof"
    print(f"{label:<18}  →  {verdict}  ({r.confidence:.1%})")
