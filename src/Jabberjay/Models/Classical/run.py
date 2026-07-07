import numpy as np
from joblib import load
from loguru import logger

from Jabberjay.Models.Classical.feature_extraction import get_features
from Jabberjay.Utilities.hugging_face import download_pretrained_model
from Jabberjay.Utilities.model_cache import cached_loader

_REPO_ID = "MattyB95/Jabberjay_Classical_Machine_Learning_Models"
_FILENAME = "KNeighborsClassifier.joblib"


@cached_loader(maxsize=1)
def _load_classifier():
    logger.info(f"Downloading model: {_FILENAME} from {_REPO_ID}")
    model_path = download_pretrained_model(repo_id=_REPO_ID, filename=_FILENAME)
    return load(filename=model_path)


def predict(audio: tuple[np.ndarray, float]) -> tuple[int, float]:
    clf = _load_classifier()
    logger.debug("Extracting audio features")
    features = get_features(audio=audio)
    logger.debug(f"Feature vector shape: {features.shape}")
    prediction = clf.predict(features)
    proba = clf.predict_proba(features)
    confidence = float(proba[0].max())
    return prediction[0], confidence
