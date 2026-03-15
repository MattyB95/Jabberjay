from typing import TypedDict


class PredictionScore(TypedDict):
    """A single label/score entry returned by transformer classification pipelines."""

    label: str
    score: float
