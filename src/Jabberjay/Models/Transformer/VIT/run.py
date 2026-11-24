"""VIT: Vision Transformers that classify a rendered spectrogram image."""

import io
from collections.abc import Callable

import librosa
import librosa.display
import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from PIL import Image

from Jabberjay.Utilities.enum_handler import Dataset, Visualisation
from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.pipeline import load_pipeline
from Jabberjay.Utilities.types import PredictionScore

# The VIT checkpoints were trained on spectrograms of audio decoded by
# librosa.load()'s default 22,050 Hz. Spectrogram images depend on the sample
# rate, so every input is brought to that rate before rendering.
TRAINING_SR = 22_050


def _constant_q(y: np.ndarray, sr: float) -> np.ndarray:
    return librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr)), ref=np.max)


def _mel_spectrogram(y: np.ndarray, sr: float) -> np.ndarray:
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)


def _mfcc(y: np.ndarray, sr: float) -> np.ndarray:
    return librosa.feature.mfcc(y=y, sr=sr)


# visualisation -> (name used in the Hub repo id, spectrogram function)
_VISUALISATIONS: dict[
    Visualisation, tuple[str, Callable[[np.ndarray, float], np.ndarray]]
] = {
    Visualisation.ConstantQ: ("ConstantQ", _constant_q),
    Visualisation.MelSpectrogram: ("Mel_Spectrogram", _mel_spectrogram),
    Visualisation.MFCC: ("MFCC", _mfcc),
}


def model_id(dataset: Dataset, visualisation: Visualisation) -> str:
    repo_name, _ = _VISUALISATIONS[visualisation]
    return f"MattyB95/VIT-{dataset.value}-{repo_name}-Synthetic-Voice-Detection"


def get_image(data: np.ndarray, sr: float) -> Image.Image:
    """Render a spectrogram to a PIL image the way the training images were."""
    logger.debug("Rendering spectrogram image")
    # A bare Figure rather than pyplot: it renders straight to Agg, so there is
    # no GUI backend to start (which fails off the main thread, or on Pythons
    # built without Tk) and no global figure registry to leak into or close
    fig = Figure()
    ax = fig.subplots()
    buf = io.BytesIO()
    try:
        librosa.display.specshow(data=data, sr=sr, ax=ax)
        fig.savefig(buf, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # read all pixel data into memory so the buffer can be released
        return img
    finally:
        buf.close()


def predict(
    y: np.ndarray, sr: float, dataset: Dataset, visualisation: Visualisation
) -> list[PredictionScore]:
    pipe = load_pipeline("image-classification", model_id(dataset, visualisation))
    if sr != TRAINING_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TRAINING_SR)
        sr = TRAINING_SR
    _, spectrogram = _VISUALISATIONS[visualisation]
    logger.debug(f"Computing {visualisation.value} spectrogram")
    image = get_image(data=spectrogram(y, sr), sr=sr)
    return normalize_pipeline_scores(pipe(image))
