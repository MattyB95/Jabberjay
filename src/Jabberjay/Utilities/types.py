from typing import Literal, TypedDict

Label = Literal["Bonafide", "Spoof"]


class PredictionScore(TypedDict):
    """A single label/score entry; every model returns one per class."""

    label: Label
    score: float
