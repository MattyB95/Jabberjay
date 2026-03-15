"""
quickstart.py — the fastest way to try Jabberjay.

Run:
    just detect res/bonafide/bonafide.flac
    uv run python examples/quickstart.py
"""

from Jabberjay import Jabberjay

jj = Jabberjay()

# Pass a file path — Jabberjay handles loading automatically.
# Defaults: VIT model, ConstantQ visualisation, VoxCelebSpoof dataset.
result = jj.detect("res/bonafide/bonafide.flac")

print(result)  # Bonafide ✔️ (92.3% confidence, model=VIT)
print(result.label)  # "Bonafide" or "Spoof"
print(result.is_bonafide)  # True / False
print(result.confidence)  # 0.0 – 1.0
