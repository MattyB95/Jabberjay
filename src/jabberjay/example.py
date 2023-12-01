from joblib import load


def detect(filename):
    clf = load('KNeighborsClassifier.joblib')
    features = get_features(filename)
    predict = clf.predict(features)
    print(predict)
