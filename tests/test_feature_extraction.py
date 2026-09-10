"""Tests for the Classical model's hand-crafted feature extraction.

These run real librosa transforms on a short synthetic signal — no network,
no model weights.
"""

import numpy as np

from Jabberjay.Models.Classical.feature_extraction import (
    feature_chromagram,
    feature_melspectrogram,
    feature_mfcc,
    get_features,
)

# 0.5 s of a 220 Hz tone at 22.05 kHz — enough for every transform to run
_SR = 22050.0
_Y = np.sin(2 * np.pi * 220 * np.arange(int(_SR * 0.5)) / _SR).astype(np.float32)


def test_chromagram_has_twelve_pitch_classes():
    out = feature_chromagram(_Y, _SR)
    assert out.shape == (12,)


def test_melspectrogram_has_128_bands():
    out = feature_melspectrogram(_Y, _SR)
    assert out.shape == (128,)


def test_mfcc_has_40_coefficients():
    out = feature_mfcc(_Y, _SR)
    assert out.shape == (40,)


def test_get_features_stacks_into_single_row():
    features = get_features((_Y, _SR))
    assert features.shape == (1, 12 + 128 + 40)
    assert np.isfinite(features).all()
