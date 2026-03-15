import io
import logging

import librosa
from matplotlib import pyplot as plt
from numpy import ndarray
from PIL import Image


def get_image(data: ndarray, sr: float) -> Image.Image:
    fig, ax = plt.subplots()
    librosa.display.specshow(data=data, sr=sr, ax=ax)
    fig.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    if logging.getLogger().level == logging.DEBUG:
        plt.show()
    plt.close(fig)
    return Image.open(buf)
