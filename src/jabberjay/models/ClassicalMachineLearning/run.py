from joblib import load

from .Feature_Extraction import get_features

clf = load('SVC.joblib')
features = get_features(args.filename)

scaler = StandardScaler()
# keep our unscaled features just in case we need to process them alternatively
features_scaled = features
features_scaled = scaler.fit_transform(features_scaled)

predict = clf.predict(features_scaled)
print(predict)
