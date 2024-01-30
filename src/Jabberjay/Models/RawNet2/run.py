import logging
import os

import numpy as np
import torch
import yaml
from torch import Tensor

from Jabberjay.Models.RawNet2.model import RawNet
from Jabberjay.Utilities.hugging_face import download_pretrained_model

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def predict(y: np.ndarray):
    with open("./model_config_RawNet.yaml", "r") as f_yaml:
        parser = yaml.safe_load(f_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    model = RawNet(parser["model"], device)
    model.to(device)
    repo_id = "MattyB95/pre_trained_DF_RawNet2"
    logging.info(f"Using model: {repo_id}")
    model_file = download_pretrained_model(
        repo_id=repo_id, filename="pre_trained_DF_RawNet2.pth"
    )
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    model.eval()
    y = Tensor(y).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(y)
        _, predicted = out.max(dim=1)
    return predicted
