import numpy as np
from joblib import load
from loguru import logger

from Jabberjay.Models.Classical.feature_extraction import get_features
from Jabberjay.Utilities.hugging_face import download_pretrained_model


def predict(audio: tuple[np.ndarray, float]) -> tuple[int, float]:
    repo_id = "MattyB95/Jabberjay_Classical_Machine_Learning_Models"
    filename = "KNeighborsClassifier.joblib"
    logger.info(f"Downloading model: {filename} from {repo_id}")
    model_path = download_pretrained_model(repo_id=repo_id, filename=filename)
    clf = load(filename=model_path)
    logger.debug("Extracting audio features")
    features = get_features(audio=audio)
    logger.debug(f"Feature vector shape: {features.shape}")
    prediction = clf.predict(features)
    proba = clf.predict_proba(features)
    confidence = float(proba[0].max())
    return prediction[0], confidence
