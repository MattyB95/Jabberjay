from joblib import load

from Jabberjay.Feature_Extraction import get_features


def add_one(number):
    return number + 1


def detect(filename):
    clf = load('KNeighborsClassifier.joblib')
    features = get_features(filename)
    predict = clf.predict(features)
    print(predict)
