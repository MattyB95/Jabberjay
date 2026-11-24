import numpy as np

from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import PredictionScore


def model_id(dataset: Dataset) -> str:
    return f"MattyB95/AST-{dataset.value}-Synthetic-Voice-Detection"


def predict(y: np.ndarray, sr: float, dataset: Dataset) -> list[PredictionScore]:
    return run_pipeline(model_id(dataset), y, sr, "AST")
