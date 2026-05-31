import numpy as np

from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import PredictionScore


def predict(y: np.ndarray, sr: float, dataset: Dataset) -> list[PredictionScore]:
    model_id = f"MattyB95/AST-{dataset.value}-Synthetic-Voice-Detection"
    return run_pipeline(model_id, y, sr, "AST")
