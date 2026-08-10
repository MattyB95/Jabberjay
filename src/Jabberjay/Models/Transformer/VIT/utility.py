import io

import librosa
import librosa.display
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray
from PIL import Image
from transformers import Pipeline, pipeline

from Jabberjay.Utilities.model_cache import cached_loader


@cached_loader(maxsize=8)
def load_pipeline(model_id: str) -> Pipeline:
    logger.info(f"Loading VIT model: {model_id}")
    return pipeline(task="image-classification", model=model_id)


def get_image(data: ndarray, sr: float) -> Image.Image:
    logger.debug("Rendering spectrogram image")
    fig, ax = plt.subplots()
    buf = io.BytesIO()
    try:
        librosa.display.specshow(data=data, sr=sr, ax=ax)
        fig.canvas.draw()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # read all pixel data into memory so the buffer can be released
        return img
    finally:
        plt.close(fig)
        buf.close()
