"""Audio input validation and the fixed-length framing shared by several models."""

import math

import numpy as np
import torch
import torchaudio
from loguru import logger


def prepare_audio(y: object, sr: object) -> tuple[np.ndarray, float]:
    """Validate pre-loaded audio and normalise it to mono float32 samples.

    Every model assumes a 1-D floating-point waveform; without this check a
    stereo, integer or NaN-filled array either crashes deep inside torch or
    librosa with an unrelated message, or — worse — runs and returns a
    confident but meaningless score.

    Accepts any array-like ``y``. Integer PCM (e.g. ``int16`` straight from
    ``soundfile``/``scipy.io.wavfile``) is scaled to [-1, 1]. Two-dimensional
    input is treated as multi-channel and averaged to mono, in either
    ``(channels, samples)`` (librosa) or ``(samples, channels)`` (soundfile)
    layout — whichever axis is shorter is taken as the channel axis.

    Raises:
        ValueError: If the audio is empty, non-numeric, has more than two
            dimensions, contains NaN/inf, or ``sr`` isn't a positive number.
    """
    if isinstance(sr, bool) or not isinstance(sr, (int, float, np.number)):
        raise ValueError(f"Sample rate must be a number, got {type(sr).__name__}.")
    if not math.isfinite(float(sr)) or sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}. Must be a positive number.")

    samples = np.asarray(y)
    if samples.dtype == np.bool_ or not np.issubdtype(samples.dtype, np.number):
        raise ValueError(f"Audio must be numeric samples, got dtype {samples.dtype}.")
    if np.issubdtype(samples.dtype, np.complexfloating):
        raise ValueError("Audio must be real-valued, got complex samples.")

    if samples.size == 0:
        raise ValueError("Audio array is empty — nothing to classify.")

    # Scale integer PCM before any downmix: mean() would silently promote it
    # to float64 and skip this
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(samples.dtype).max + 1) / 2
        samples = (samples.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / -np.iinfo(samples.dtype).min

    if samples.ndim == 2:
        channel_axis = 0 if samples.shape[0] <= samples.shape[1] else 1
        logger.debug(f"Downmixing {samples.shape[channel_axis]} channels to mono")
        samples = samples.mean(axis=channel_axis)
    elif samples.ndim != 1:
        raise ValueError(
            "Audio must be a 1-D (mono) or 2-D (multi-channel) array, "
            f"got shape {samples.shape}."
        )

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains NaN or infinite samples.")
    if not samples.any():
        logger.warning("Audio is pure silence; any prediction will be meaningless")
    return samples, float(sr)


def resample(y: np.ndarray, sr: float, target_sr: int) -> torch.Tensor:
    """Return ``y`` as a float tensor at ``target_sr``."""
    audio = torch.from_numpy(y).float()
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, int(sr), target_sr)
    return audio


def fixed_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    """Trim ``audio`` to ``length`` samples, or loop it until it fills them.

    The fixed-input anti-spoofing models (RawNet2, Spectra, DF Arena) were
    trained on exactly ``length`` samples framed this way, so only the first
    ``length`` samples of a longer clip are scored.
    """
    n = audio.shape[0]
    if n == 0:
        raise ValueError(
            "Input audio array is empty; cannot run inference on zero samples."
        )
    if n >= length:
        return audio[:length]
    return audio.repeat(length // n + 1)[:length]
