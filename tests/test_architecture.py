"""Architecture-level tests for model components.

Instantiates neural network modules directly with synthetic tensors —
no HuggingFace weights are downloaded.
"""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# ── Shared (Wav2Vec2Encoder, MLPBridge) ───────────────────────────────────────


class TestMLPBridge:
    def test_output_shape(self):
        from Jabberjay.Models.Spectra.shared import MLPBridge

        bridge = MLPBridge(input_dim=16, output_dim=8, hidden_dim=32)
        out = bridge(torch.randn(2, 10, 16))
        assert out.shape == (2, 10, 8)

    def test_custom_activation(self):
        from Jabberjay.Models.Spectra.shared import MLPBridge

        bridge = MLPBridge(
            input_dim=8, output_dim=4, hidden_dim=16, activation=nn.ReLU()
        )
        out = bridge(torch.randn(1, 5, 8))
        assert out.shape == (1, 5, 4)

    def test_default_activation_is_selu(self):
        from Jabberjay.Models.Spectra.shared import MLPBridge

        bridge = MLPBridge(input_dim=4, output_dim=4)
        assert isinstance(bridge.mlp[1], nn.SELU)


class TestWav2Vec2Encoder:
    @staticmethod
    def _make_encoder():
        with patch("Jabberjay.Models.Spectra.shared.Wav2Vec2Model") as mock_cls:
            mock_model = MagicMock()
            mock_model.return_value.last_hidden_state = torch.randn(1, 10, 1024)
            mock_cls.from_pretrained.return_value = mock_model
            from Jabberjay.Models.Spectra.shared import Wav2Vec2Encoder

            return Wav2Vec2Encoder(), mock_model

    def test_forward_2d_input(self):
        encoder, mock_model = self._make_encoder()
        out = encoder(torch.randn(1, 16000))
        assert out.shape == (1, 10, 1024)

    def test_forward_3d_input_squeezed(self):
        encoder, mock_model = self._make_encoder()
        out = encoder(torch.randn(1, 16000, 1))
        assert out.shape == (1, 10, 1024)


# ── Spectra0 architecture ─────────────────────────────────────────────────────


class TestSEModule:
    def test_output_shape_preserved(self):
        from Jabberjay.Models.Spectra0.model import SEModule

        se = SEModule(channels=8)
        x = torch.randn(2, 8, 50)
        assert se(x).shape == x.shape


class TestBottle2neck:
    def test_same_channels(self):
        from Jabberjay.Models.Spectra0.model import Bottle2neck

        block = Bottle2neck(inplanes=64, planes=64, kernel_size=3, dilation=2)
        x = torch.randn(2, 64, 100)
        assert block(x).shape == x.shape

    def test_same_channels_different_dilation(self):
        from Jabberjay.Models.Spectra0.model import Bottle2neck

        block = Bottle2neck(inplanes=64, planes=64, kernel_size=3, dilation=3)
        assert block(torch.randn(2, 64, 100)).shape == (2, 64, 100)


class TestECAPATDNN:
    def test_output_shape(self):
        from Jabberjay.Models.Spectra0.model import ECAPA_TDNN

        model = ECAPA_TDNN(C=16).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 50, 128))
        assert out.shape == (2, 2)


class TestSpectra0Model:
    @staticmethod
    def _make_model():
        with patch("Jabberjay.Models.Spectra0.model.Wav2Vec2Encoder") as mock_enc_cls:
            mock_enc = MagicMock(return_value=torch.randn(1, 50, 1024))
            mock_enc_cls.return_value = mock_enc
            from Jabberjay.Models.Spectra0.model import Spectra0Model

            return Spectra0Model().eval(), mock_enc

    def test_forward_output_shape(self):
        model, _ = self._make_model()
        with torch.no_grad():
            out = model(torch.randn(1, 16000))
        assert out.shape == (1, 2)

    def test_classify_returns_float(self):
        model, _ = self._make_model()
        result = model.classify(torch.randn(1, 16000))
        assert isinstance(result, float)


# ── SpectraAASIST architecture (nn.Linear layers) ─────────────────────────────

_AASIST_D_ARGS = {
    "filts": [128, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.5, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class TestAASISTComponents:
    def test_graph_pool_output_shape(self):
        from Jabberjay.Models.SpectraAASIST.model import GraphPool

        pool = GraphPool(k=0.5, in_dim=32, p=0.1)
        h = torch.randn(2, 10, 32)
        out = pool(h)
        assert out.shape[0] == 2 and out.shape[2] == 32

    def test_graph_attention_layer_output_shape(self):
        from Jabberjay.Models.SpectraAASIST.model import GraphAttentionLayer

        layer = GraphAttentionLayer(in_dim=16, out_dim=8).eval()
        with torch.no_grad():
            out = layer(torch.randn(2, 6, 16))
        assert out.shape == (2, 6, 8)

    def test_htrg_gat_with_explicit_master(self):
        from Jabberjay.Models.SpectraAASIST.model import HtrgGraphAttentionLayer

        layer = HtrgGraphAttentionLayer(in_dim=16, out_dim=8).eval()
        x1, x2 = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        master = torch.randn(2, 1, 16)
        with torch.no_grad():
            out_t, out_s, m = layer(x1, x2, master=master)
        assert out_t.shape == (2, 4, 8)
        assert out_s.shape == (2, 4, 8)
        assert m.shape == (2, 1, 8)

    def test_htrg_gat_without_master_computes_mean(self):
        from Jabberjay.Models.SpectraAASIST.model import HtrgGraphAttentionLayer

        layer = HtrgGraphAttentionLayer(in_dim=16, out_dim=8).eval()
        with torch.no_grad():
            out_t, out_s, m = layer(torch.randn(1, 3, 16), torch.randn(1, 3, 16))
        assert m.shape == (1, 1, 8)

    def test_residual_block_first_no_downsample(self):
        from Jabberjay.Models.SpectraAASIST.model import ResidualBlock

        block = ResidualBlock(nb_filts=[1, 32], first=True)
        x = torch.randn(1, 1, 8, 8)
        assert block(x).shape == (1, 32, 8, 8)

    def test_residual_block_not_first_with_downsample(self):
        from Jabberjay.Models.SpectraAASIST.model import ResidualBlock

        block = ResidualBlock(nb_filts=[32, 64]).eval()
        with torch.no_grad():
            out = block(torch.randn(1, 32, 8, 8))
        assert out.shape == (1, 64, 8, 8)

    def test_encoder_output_shape(self):
        from Jabberjay.Models.SpectraAASIST.model import Encoder

        enc = Encoder(filts=_AASIST_D_ARGS["filts"]).eval()
        with torch.no_grad():
            out = enc(torch.randn(1, 12, 128))
        assert out.shape[1] == 64  # 64 output channels

    def test_hsgal_branch_output_shapes(self):
        from Jabberjay.Models.SpectraAASIST.model import HSGALBranch_v1

        branch = HSGALBranch_v1(
            gat_dims=[64, 32],
            temperatures=[2.0, 2.0, 100.0, 100.0],
            pool_ratios=[0.5, 0.5, 0.5, 0.5],
        ).eval()
        out_t, out_s = torch.randn(1, 2, 64), torch.randn(1, 10, 64)
        with torch.no_grad():
            t, s, m = branch(out_t, out_s)
        assert t.shape[-1] == 32
        assert m.shape == (1, 1, 32)


class TestKANAASSISTForward:
    def test_output_shape(self):
        from Jabberjay.Models.SpectraAASIST.model import KANAASIST

        model = KANAASIST(d_args=_AASIST_D_ARGS, n_frames=12).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 12, 128))
        assert out.shape == (1, 2)


class TestSpectraAASISTModel:
    def test_forward_output_shape(self):
        from Jabberjay.Models.SpectraAASIST.model import SpectraAASIST

        with patch(
            "Jabberjay.Models.SpectraAASIST.model.Wav2Vec2Encoder"
        ) as mock_enc_cls:
            mock_enc = MagicMock(return_value=torch.randn(1, 12, 1024))
            mock_enc_cls.return_value = mock_enc
            model = SpectraAASIST().eval()
        with torch.no_grad():
            out = model(torch.randn(1, 16000))
        assert out.shape == (1, 2)

    def test_classify_returns_float(self):
        from Jabberjay.Models.SpectraAASIST.model import SpectraAASIST

        with patch(
            "Jabberjay.Models.SpectraAASIST.model.Wav2Vec2Encoder"
        ) as mock_enc_cls:
            mock_enc_cls.return_value = MagicMock(return_value=torch.randn(1, 12, 1024))
            model = SpectraAASIST().eval()
        assert isinstance(model.classify(torch.randn(1, 16000)), float)


# ── SpectraAASIST3 architecture (KANLinear layers) ───────────────────────────


class TestKANLinear:
    def test_output_shape(self):
        from Jabberjay.Models.SpectraAASIST3.model import KANLinear

        layer = KANLinear(in_features=8, out_features=4, grid_size=4, spline_order=2)
        out = layer(torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 4)

    def test_2d_input(self):
        from Jabberjay.Models.SpectraAASIST3.model import KANLinear

        layer = KANLinear(in_features=4, out_features=4, grid_size=4, spline_order=2)
        out = layer(torch.randn(3, 4))
        assert out.shape == (3, 4)

    def test_scaled_spline_weight_property(self):
        from Jabberjay.Models.SpectraAASIST3.model import KANLinear

        layer = KANLinear(in_features=4, out_features=4, grid_size=4, spline_order=2)
        w = layer._scaled_spline_weight
        assert w.shape == layer.spline_weight.shape


class TestAASIST3Components:
    def test_graph_pool_output_shape(self):
        from Jabberjay.Models.SpectraAASIST3.model import GraphPool

        pool = GraphPool(k=0.5, in_dim=8, p=0.1)
        out = pool(torch.randn(2, 10, 8))
        assert out.shape[0] == 2 and out.shape[2] == 8

    def test_graph_attention_layer_output_shape(self):
        from Jabberjay.Models.SpectraAASIST3.model import GraphAttentionLayer

        layer = GraphAttentionLayer(in_dim=8, out_dim=4).eval()
        with torch.no_grad():
            out = layer(torch.randn(1, 4, 8))
        assert out.shape == (1, 4, 4)

    def test_htrg_gat_with_master(self):
        from Jabberjay.Models.SpectraAASIST3.model import HtrgGraphAttentionLayer

        layer = HtrgGraphAttentionLayer(in_dim=8, out_dim=4).eval()
        x1, x2 = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
        with torch.no_grad():
            out_t, out_s, m = layer(x1, x2, master=torch.randn(1, 1, 8))
        assert out_t.shape == (1, 3, 4)

    def test_htrg_gat_without_master(self):
        from Jabberjay.Models.SpectraAASIST3.model import HtrgGraphAttentionLayer

        layer = HtrgGraphAttentionLayer(in_dim=8, out_dim=4).eval()
        with torch.no_grad():
            _, _, m = layer(torch.randn(1, 3, 8), torch.randn(1, 3, 8))
        assert m.shape == (1, 1, 4)

    def test_residual_block_first(self):
        from Jabberjay.Models.SpectraAASIST3.model import ResidualBlock

        block = ResidualBlock(nb_filts=[1, 32], first=True)
        assert block(torch.randn(1, 1, 8, 8)).shape == (1, 32, 8, 8)

    def test_residual_block_with_downsample(self):
        from Jabberjay.Models.SpectraAASIST3.model import ResidualBlock

        block = ResidualBlock(nb_filts=[32, 64]).eval()
        with torch.no_grad():
            assert block(torch.randn(1, 32, 8, 8)).shape == (1, 64, 8, 8)

    def test_encoder_output_channels(self):
        from Jabberjay.Models.SpectraAASIST3.model import Encoder

        enc = Encoder(filts=_AASIST_D_ARGS["filts"]).eval()
        with torch.no_grad():
            out = enc(torch.randn(1, 12, 128))
        assert out.shape[1] == 64

    def test_hsgal_branch_output_shapes(self):
        from Jabberjay.Models.SpectraAASIST3.model import HSGALBranch_v1

        branch = HSGALBranch_v1(
            gat_dims=[8, 4],
            temperatures=[2.0, 2.0, 100.0, 100.0],
            pool_ratios=[0.5, 0.5, 0.5, 0.5],
        ).eval()
        with torch.no_grad():
            t, s, m = branch(torch.randn(1, 2, 8), torch.randn(1, 6, 8))
        assert t.shape[-1] == 4
        assert m.shape[-1] == 4


class TestKANAASSIST3Forward:
    def test_output_shape(self):
        from Jabberjay.Models.SpectraAASIST3.model import KANAASIST

        model = KANAASIST(d_args=_AASIST_D_ARGS, n_frames=12).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 12, 128))
        assert out.shape == (1, 2)


class TestSpectraAASIST3Model:
    def test_forward_output_shape(self):
        from Jabberjay.Models.SpectraAASIST3.model import SpectraAASIST3

        with patch(
            "Jabberjay.Models.SpectraAASIST3.model.Wav2Vec2Encoder"
        ) as mock_enc_cls:
            mock_enc = MagicMock(return_value=torch.randn(1, 12, 1024))
            mock_enc_cls.return_value = mock_enc
            model = SpectraAASIST3().eval()
        with torch.no_grad():
            out = model(torch.randn(1, 16000))
        assert out.shape == (1, 2)

    def test_classify_returns_float(self):
        from Jabberjay.Models.SpectraAASIST3.model import SpectraAASIST3

        with patch(
            "Jabberjay.Models.SpectraAASIST3.model.Wav2Vec2Encoder"
        ) as mock_enc_cls:
            mock_enc_cls.return_value = MagicMock(return_value=torch.randn(1, 12, 1024))
            model = SpectraAASIST3().eval()
        assert isinstance(model.classify(torch.randn(1, 16000)), float)
