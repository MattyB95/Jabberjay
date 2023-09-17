import argparse

import librosa

parser = argparse.ArgumentParser(
    prog='Jabberjay',
    description='🦜 Synthetic Voice Detection',
    epilog='May The Odds Be Ever In Your Favor.')

parser.add_argument('filename')
parser.add_argument('-m', '--model', choices=['Classical', 'RawNet2'], default='RawNet2')
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")
    print(args.filename, args.model, args.verbose)


class Jabberjay:
    def __init__(self, filename):
        self.filename = filename

    def detect(self, model='RawNet2'):
        match model:
            case 'Classical':
                pass
            case 'RawNet2':
                import Models.RawNet2AntiSpoofing.Run as RawNet2

                predict = RawNet2.predict(y)
                print(predict)


y, sr = librosa.load(args.filename)
jabberjay = Jabberjay(args.filename)
jabberjay.detect(args.model)
