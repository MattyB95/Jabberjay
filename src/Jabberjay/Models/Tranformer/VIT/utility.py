import io
import logging

import librosa
from PIL import Image
from matplotlib import pyplot as plt
from numpy import ndarray


def get_image(data: ndarray, sr: float) -> Image.Image:
    fig, ax = plt.subplots()
    librosa.display.specshow(data=data, sr=sr, ax=ax)
    fig.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    if logging.getLogger().level == logging.DEBUG:
        plt.show()
    return Image.open(buf)
