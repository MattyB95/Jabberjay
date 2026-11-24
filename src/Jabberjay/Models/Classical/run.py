import warnings

import librosa
import numpy as np
from joblib import load
from loguru import logger
from sklearn.exceptions import InconsistentVersionWarning

from Jabberjay.Models.Classical.feature_extraction import get_features
from Jabberjay.Utilities.hugging_face import download_pretrained_model
from Jabberjay.Utilities.label_normalizer import normalize_label
from Jabberjay.Utilities.model_cache import cached_loader
from Jabberjay.Utilities.types import PredictionScore

REPO_ID = "MattyB95/Jabberjay_Classical_Machine_Learning_Models"
FILENAME = "KNeighborsClassifier.joblib"
# Features were extracted from audio decoded at librosa.load()'s default rate;
# chroma/mel/MFCC values all shift with the sample rate, so match it
TRAINING_SR = 22_050


@cached_loader(maxsize=1)
def _load_classifier():
    logger.info(f"Downloading model: {FILENAME} from {REPO_ID}")
    # Pickled — safe only because download_pretrained_model() pins the
    # reviewed revision (see Jabberjay.Utilities.hugging_face.REVISIONS)
    model_path = download_pretrained_model(repo_id=REPO_ID, filename=FILENAME)
    with warnings.catch_warnings():
        # Pickled with scikit-learn 1.3.0; KNeighborsClassifier's state is
        # stable across versions and loads correctly, so don't alarm users
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return load(filename=model_path)


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    clf = _load_classifier()
    if sr != TRAINING_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TRAINING_SR)
        sr = TRAINING_SR
    logger.debug("Extracting audio features")
    features = get_features(audio=(y, sr))
    logger.debug(f"Feature vector shape: {features.shape}")
    # The classifier's classes are the training folder names ("Genuine.wav",
    # "Synthetic.wav"), so map each through the label normaliser rather than
    # assuming an index order
    proba = clf.predict_proba(features)[0]
    return [
        PredictionScore(label=normalize_label(str(cls)), score=float(p))
        for cls, p in zip(clf.classes_, proba, strict=True)
    ]
