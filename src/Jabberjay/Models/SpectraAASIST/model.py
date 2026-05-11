import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from Jabberjay.Models.Spectra.shared import MLPBridge, Wav2Vec2Encoder


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.FloatTensor(out_dim, 1))
        for w in (
            self.att_weight11,
            self.att_weight22,
            self.att_weight12,
            self.att_weightM,
        ):
            nn.init.xavier_normal_(w)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1, num_type2 = x1.size(1), x2.size(1)
        x1, x2 = self.proj_type1(x1), self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return x.narrow(1, 0, num_type1), x.narrow(1, num_type1, num_type2), master

    def _update_master(self, x, master):
        att_map = x * master
        att_map = torch.matmul(torch.tanh(self.att_projM(att_map)), self.att_weightM)
        att_map = F.softmax(att_map / self.temp, dim=-2)
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        return x1 + self.proj_without_attM(master)

    def _derive_att_map(self, x, num_type1, num_type2):
        nb = x.size(1)
        pw = x.unsqueeze(2).expand(-1, -1, nb, -1) * x.unsqueeze(1).expand(
            -1, nb, -1, -1
        )
        att_map = torch.tanh(self.att_proj(pw))
        board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )
        return F.softmax(board / self.temp, dim=-2)

    def _project(self, x, att_map):
        return self.proj_with_att(
            torch.matmul(att_map.squeeze(-1), x)
        ) + self.proj_without_att(x)

    def _apply_BN(self, x):
        s = x.size()
        return self.bn(x.view(-1, s[-1])).view(s)


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: float, **kwargs):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        scores = torch.sigmoid(self.proj(self.drop(h)))
        n_nodes = max(int(h.size(1) * self.k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, h.size(-1)))


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        nb = x.size(1)
        pw = x.unsqueeze(2).expand(-1, -1, nb, -1) * x.unsqueeze(1).expand(
            -1, nb, -1, -1
        )
        att_map = F.softmax(
            torch.matmul(torch.tanh(self.att_proj(pw)), self.att_weight) / self.temp,
            dim=-2,
        )
        x = self.proj_with_att(
            torch.matmul(att_map.squeeze(-1), x)
        ) + self.proj_without_att(x)
        s = x.size()
        return self.act(self.bn(x.view(-1, s[-1])).view(s))


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(
            nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1)
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(
            nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1)
        )
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(
                nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1)
            )

    def forward(self, x):
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.selu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


class Encoder(nn.Module):
    def __init__(self, filts):
        super().__init__()
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.selu = nn.SELU(inplace=True)
        self.enc = nn.Sequential(
            *[Residual_block(nb_filts=filts[i], first=(i == 1)) for i in range(1, 5)]
            + [Residual_block(nb_filts=filts[4]), Residual_block(nb_filts=filts[4])]
        )

    def forward(self, x):
        x = self.selu(
            self.first_bn(
                F.max_pool2d(torch.abs(x.transpose(1, 2).unsqueeze(1)), (3, 3))
            )
        )
        x = self.selu(self.first_bn1(self.enc(x)))
        return x


class HSGALBranch_v1(nn.Module):
    def __init__(self, gat_dims, temperatures, pool_ratios):
        super().__init__()
        self.master = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.HtrgGAT_layer_ST1 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST2 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.pool_hS = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.drop_way = nn.Dropout(0.2, inplace=True)

    def forward(self, out_t, out_s):
        out_T, out_S, master = self.HtrgGAT_layer_ST1(out_t, out_s, master=self.master)
        out_S, out_T = self.pool_hS(out_S), self.pool_hT(out_T)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST2(
            out_T, out_S, master=master
        )
        return (
            self.drop_way(out_T + out_T_aug),
            self.drop_way(out_S + out_S_aug),
            self.drop_way(master + master_aug),
        )


class KANAASIST(nn.Module):
    def __init__(self, d_args, n_frames=400):
        super().__init__()
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, filts[0] // 3, filts[-1][-1]))
        self.pos_T = nn.Parameter(torch.randn(1, n_frames, filts[0]))
        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )
        self.branch1 = HSGALBranch_v1(gat_dims, temperatures, pool_ratios)
        self.branch2 = HSGALBranch_v1(gat_dims, temperatures, pool_ratios)
        self.branch3 = HSGALBranch_v1(gat_dims, temperatures, pool_ratios)
        self.branch4 = HSGALBranch_v1(gat_dims, temperatures, pool_ratios)
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        self.enc = Encoder(filts=filts)

    def forward(self, x):
        x = x + self.pos_T[:, : x.size(1), :]
        x = self.enc(x)
        w = self.attention(x)
        e_S = self.GAT_layer_S(
            torch.sum(x * F.softmax(w, dim=-1), dim=-1).transpose(1, 2) + self.pos_S
        )
        out_S = self.pool_S(e_S)
        e_T = self.GAT_layer_T(
            torch.sum(x * F.softmax(w, dim=-2), dim=-2).transpose(1, 2)
        )
        out_T = self.pool_T(e_T)
        out_T1, out_S1, m1 = self.branch1(out_T, out_S)
        out_T2, out_S2, m2 = self.branch2(out_T, out_S)
        out_T3, out_S3, m3 = self.branch3(out_T, out_S)
        out_T4, out_S4, m4 = self.branch4(out_T, out_S)
        out_T = torch.amax(torch.stack([out_T1, out_T2, out_T3, out_T4]), dim=0)
        out_S = torch.amax(torch.stack([out_S1, out_S2, out_S3, out_S4]), dim=0)
        master = torch.amax(torch.stack([m1, m2, m3, m4]), dim=0)
        feat = torch.cat(
            [
                torch.max(torch.abs(out_T), dim=1)[0],
                torch.mean(out_T, dim=1),
                torch.max(torch.abs(out_S), dim=1)[0],
                torch.mean(out_S, dim=1),
                master.squeeze(1),
            ],
            dim=1,
        )
        return self.out_layer(self.drop(feat))


class SpectraAASIST(nn.Module, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssl_encoder = Wav2Vec2Encoder("facebook/wav2vec2-xls-r-300m")
        self.bridge = MLPBridge(1024, 128, hidden_dim=128, activation=nn.SELU())
        self.aasist = KANAASIST(
            d_args={
                "architecture": "AASIST",
                "nb_samp": 64400,
                "filts": [128, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.5, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
            }
        )

    def forward(self, x):
        return self.aasist(self.bridge(self.ssl_encoder(x)))

    @torch.inference_mode()
    def classify(self, x, threshold: float = -1.0625009):
        return (self.forward(x)[:, 1] > threshold).float().item()
