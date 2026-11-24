"""Normalise external model labels to the canonical 'Bonafide' / 'Spoof' labels."""

from collections.abc import Mapping
from typing import cast

from Jabberjay.Utilities.types import Label, PredictionScore

# Exact-match only — short strings that would cause false positives in substring search
_BONAFIDE_EXACT = {"0", "label_0"}
_SPOOF_EXACT = {"1", "label_1"}

# Safe for substring search — long enough to avoid accidental collisions
_BONAFIDE_SUBSTR = {"bonafide", "real", "genuine", "human", "bona"}
_SPOOF_SUBSTR = {"spoof", "fake", "synthetic", "deepfake"}


def normalize_label(raw_label: str) -> Label:
    """Return 'Bonafide' or 'Spoof' for any external model label string."""
    lower = raw_label.lower().replace("-", "_").replace(" ", "_")

    # Exact match first (handles numeric and label_N conventions)
    if lower in _BONAFIDE_EXACT:
        return "Bonafide"
    if lower in _SPOOF_EXACT:
        return "Spoof"

    # Substring search only for long, unambiguous keywords
    for kw in _BONAFIDE_SUBSTR:
        if kw in lower:
            return "Bonafide"
    for kw in _SPOOF_SUBSTR:
        if kw in lower:
            return "Spoof"

    raise ValueError(f"Cannot normalise label: {raw_label!r}")


def normalize_pipeline_scores(
    raw_scores: list[dict[str, str | float]],
    label_map: Mapping[str, Label] | None = None,
) -> list[PredictionScore]:
    """Convert raw transformers pipeline output to normalised PredictionScores.

    ``label_map`` overrides the generic normalisation for models whose raw
    labels don't follow the 0=Bonafide convention (e.g. a checkpoint shipped
    without ``id2label``, which transformers reports as ``LABEL_0``/``LABEL_1``
    regardless of what the indices meant in training).
    """
    label_map = label_map or {}
    return [
        PredictionScore(
            label=label_map.get(str(s["label"])) or normalize_label(str(s["label"])),
            score=float(cast(float, s["score"])),
        )
        for s in raw_scores
    ]
