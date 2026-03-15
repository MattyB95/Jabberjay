"""Normalise external model labels to the canonical 'Bonafide' / 'Spoof' labels."""

from Jabberjay.Utilities.types import PredictionScore

# Exact-match only — short strings that would cause false positives in substring search
_BONAFIDE_EXACT = {"0", "label_0"}
_SPOOF_EXACT = {"1", "label_1"}

# Safe for substring search — long enough to avoid accidental collisions
_BONAFIDE_SUBSTR = {"bonafide", "real", "genuine", "human", "bona"}
_SPOOF_SUBSTR = {"spoof", "fake", "synthetic", "deepfake"}


def normalize_label(raw_label: str) -> str:
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
    raw_scores: list[dict[str, object]],
) -> list[PredictionScore]:
    """Convert raw transformers pipeline output to normalised PredictionScores."""
    return [
        PredictionScore(
            label=normalize_label(str(s["label"])),
            score=float(str(s["score"])),
        )
        for s in raw_scores
    ]
