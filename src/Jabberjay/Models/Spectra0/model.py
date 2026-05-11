import math

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from Jabberjay.Models.Spectra.shared import MLPBridge, Wav2Vec2Encoder


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return input * self.se(input)


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        num_pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = self.se(
            self.bn3(self.relu(self.conv3(torch.cat((out, spx[self.nums]), 1))))
        )
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv1d(128, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = Bottle2neck(C, C, kernel_size=3, dilation=5, scale=8)
        self.layer5 = nn.Conv1d(4 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 2)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x.transpose(1, 2))))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x4 = self.layer4(x + x1 + x2 + x3)
        x = self.relu(self.layer5(torch.cat((x1, x2, x3, x4), dim=1)))
        t = x.size()[-1]
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(
                    1, 1, t
                ),
            ),
            dim=1,
        )
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        return self.fc6(self.bn5(torch.cat((mu, sg), 1)))


class Spectra0Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssl_encoder = Wav2Vec2Encoder("facebook/wav2vec2-xls-r-300m")
        self.bridge = MLPBridge(1024, 128, hidden_dim=128, activation=nn.SELU())
        self.ecapa_tdnn = ECAPA_TDNN(128)

    def forward(self, x):
        return self.ecapa_tdnn(self.bridge(self.ssl_encoder(x)))

    @torch.inference_mode()
    def classify(self, x, threshold: float = -1.0625009):
        return (self.forward(x)[:, 1] > threshold).float().item()
