import numpy as np

from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import Label, PredictionScore

MODEL_ID = "DavidCombei/wavLM-base-Deepfake_V2"

# The checkpoint ships without id2label, so transformers reports generic
# LABEL_0/LABEL_1 — and index 0 is *spoof*, the reverse of the convention the
# generic normaliser assumes. Neither the model card nor its paper
# (arXiv:2408.07414) states the mapping; it was established empirically:
# four macOS `say` TTS voices all score LABEL_0 at 99.98% (and are flagged
# spoof by Spectra-AASIST3, DF-Arena-1B and RawNet2), while res/bonafide
# scores LABEL_1 at 99.96%.
LABEL_MAP: dict[str, Label] = {"LABEL_0": "Spoof", "LABEL_1": "Bonafide"}


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    return run_pipeline(MODEL_ID, y, sr, "WavLM", label_map=LABEL_MAP)
