import numpy as np
from joblib import load
from loguru import logger

from Jabberjay.Models.Classical.feature_extraction import get_features
from Jabberjay.Utilities.hugging_face import download_pretrained_model


def predict(audio: tuple[np.ndarray, float]):
    repo_id = "MattyB95/Jabberjay_Classical_Machine_Learning_Models"
    filename = "KNeighborsClassifier.joblib"
    logger.info(f"Downloading model: {filename} from {repo_id}")
    model = download_pretrained_model(repo_id=repo_id, filename=filename)
    clf = load(filename=model)
    logger.debug("Extracting audio features")
    features = get_features(audio=audio)
    logger.debug(f"Feature vector shape: {features.shape}")
    return clf.predict(features)
