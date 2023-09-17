import os

import torch
import yaml
from torch import Tensor

from .model import RawNet

GENUINE_WAV = "../../../../DetectionData/Genuine/50-Genuine.wav"
SYNTHETIC_WAV = "../../../../DetectionData/Synthetic/50-Synthetic.wav"

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def predict(y):
    with open('./model_config_RawNet.yaml', 'r') as f_yaml:
        parser = yaml.safe_load(f_yaml)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    model = RawNet(parser['model'], device)
    model.to(device)
    model.load_state_dict(torch.load("pre_trained_DF_RawNet2.pth"))
    model.eval()
    y = Tensor(y).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(y)
        _, predicted = out.max(dim=1)
    return predicted
