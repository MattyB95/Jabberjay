import argparse

from joblib import load
from sklearn.preprocessing import StandardScaler

from Feature_Extraction import get_features

parser = argparse.ArgumentParser(
    prog='Jabberjay',
    description='🦜 Synthetic Voice Detection',
    epilog='May The Odds Be Ever In Your Favor.')

parser.add_argument('filename')  # positional argument
parser.add_argument('-c', '--count')  # option that takes a value
parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag

args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")
print(args.filename, args.count, args.verbose)
clf = load('SVC.joblib')
features = get_features(args.filename)

scaler = StandardScaler()
# keep our unscaled features just in case we need to process them alternatively
features_scaled = features
features_scaled = scaler.fit_transform(features_scaled)

predict = clf.predict(features_scaled)
print(predict)
