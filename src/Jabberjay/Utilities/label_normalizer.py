"""Normalise external model labels to the canonical 'Bonafide' / 'Spoof' labels."""

_BONAFIDE_KEYWORDS = {"bonafide", "real", "genuine", "human", "bona", "label_0", "0"}
_SPOOF_KEYWORDS = {"spoof", "fake", "synthetic", "deepfake", "label_1", "1"}


def normalize_label(raw_label: str) -> str:
    """Return 'Bonafide' or 'Spoof' for any external model label string."""
    lower = raw_label.lower().replace("-", "_").replace(" ", "_")
    if lower in _BONAFIDE_KEYWORDS:
        return "Bonafide"
    if lower in _SPOOF_KEYWORDS:
        return "Spoof"
    # Fallback: substring search for common terms
    for kw in _BONAFIDE_KEYWORDS:
        if kw in lower:
            return "Bonafide"
    for kw in _SPOOF_KEYWORDS:
        if kw in lower:
            return "Spoof"
    raise ValueError(f"Cannot normalise label: {raw_label!r}")
