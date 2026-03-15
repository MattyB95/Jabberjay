import io

import librosa
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray
from PIL import Image


def get_image(data: ndarray, sr: float) -> Image.Image:
    logger.debug("Rendering spectrogram image")
    fig, ax = plt.subplots()
    librosa.display.specshow(data=data, sr=sr, ax=ax)
    fig.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
