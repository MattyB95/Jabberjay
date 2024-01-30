import logging

import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

from Jabberjay.Models.Classical.feature_extraction import get_features
from Jabberjay.Utilities.hugging_face import download_pretrained_model


def predict(audio: tuple[np.ndarray, float]):
    repo_id = "MattyB95/Jabberjay_Classical_Machine_Learning_Models"
    filename = "KNeighborsClassifier.joblib"
    logging.info(f"Using Model: {filename}")
    logging.info(f"Repository: {repo_id}")
    model = download_pretrained_model(repo_id=repo_id, filename=filename)
    clf = load(filename=model)
    features = get_features(audio=audio)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X=features)
    return clf.predict(features_scaled)
