import librosa
import numpy as np


def feature_chromagram(y: np.ndarray, sr: float) -> np.ndarray:
    stft_spectrogram = np.abs(librosa.stft(y))
    chromagram = np.mean(
        librosa.feature.chroma_stft(S=stft_spectrogram, sr=sr).T, axis=0
    )
    return chromagram


def feature_melspectrogram(y: np.ndarray, sr: float) -> np.ndarray:
    melspectrogram = np.mean(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000).T, axis=0
    )
    return melspectrogram


def feature_mfcc(y: np.ndarray, sr: float) -> np.ndarray:
    mfc_coefficients = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfc_coefficients


def get_features(audio: tuple[np.ndarray, float]) -> np.ndarray:
    y, sr = audio
    chromagram = feature_chromagram(y, sr)
    melspectrogram = feature_melspectrogram(y, sr)
    mfc_coefficients = feature_mfcc(y, sr)
    feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))
    return feature_matrix.reshape(1, -1)
