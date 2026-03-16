"""Unit tests for individual model predict() functions.

All network and model-loading operations are mocked so these tests run
offline without downloading any weights from HuggingFace.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Jabberjay.Utilities.enum_handler import Dataset, Visualisation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AUDIO = (np.zeros(16000, dtype=np.float32), 16000.0)
FAKE_RAW_SCORES = [{"label": "bonafide", "score": 0.9}, {"label": "fake", "score": 0.1}]


def _mock_pipeline(raw_scores=None):
    """Return a mock transformers.pipeline factory whose callable returns raw_scores."""
    if raw_scores is None:
        raw_scores = FAKE_RAW_SCORES
    mock_pipe = MagicMock()
    mock_pipe.return_value = raw_scores
    return mock_pipe


# ---------------------------------------------------------------------------
# Transformer pipeline models (HuBERT, Wav2Vec2, WavLM, AST)
# ---------------------------------------------------------------------------

class TestHuBERTPredict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.HuBERT.run import predict

        with patch("Jabberjay.Models.HuBERT.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["label"] == "Bonafide"
        assert result[1]["label"] == "Spoof"

    def test_confidence_preserved(self):
        from Jabberjay.Models.HuBERT.run import predict

        with patch("Jabberjay.Models.HuBERT.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["score"] == 0.9


class TestWav2Vec2Predict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.Wav2Vec2.run import predict

        with patch("Jabberjay.Models.Wav2Vec2.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["label"] == "Bonafide"

    def test_confidence_preserved(self):
        from Jabberjay.Models.Wav2Vec2.run import predict

        with patch("Jabberjay.Models.Wav2Vec2.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["score"] == 0.9


class TestWavLMPredict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.WavLM.run import predict

        with patch("Jabberjay.Models.WavLM.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["label"] == "Bonafide"

    def test_confidence_preserved(self):
        from Jabberjay.Models.WavLM.run import predict

        with patch("Jabberjay.Models.WavLM.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["score"] == 0.9


class TestASTPRedict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.Transformer.AST.run import predict

        with patch("Jabberjay.Models.Transformer.AST.run.pipeline", return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1], dataset=Dataset.VoxCelebSpoof)

        assert result[0]["label"] == "Bonafide"

    def test_model_id_includes_dataset_name(self):
        from Jabberjay.Models.Transformer.AST.run import predict

        mock_factory = MagicMock(return_value=_mock_pipeline())
        with patch("Jabberjay.Models.Transformer.AST.run.pipeline", mock_factory):
            predict(y=AUDIO[0], sr=AUDIO[1], dataset=Dataset.ASVspoof2019)

        call_kwargs = mock_factory.call_args
        assert "ASVspoof2019" in str(call_kwargs)


# ---------------------------------------------------------------------------
# VIT models
# ---------------------------------------------------------------------------

class TestVITPredict:
    def _run_constantq(self, raw_scores=None):
        from Jabberjay.Models.Transformer.VIT.ConstantQ.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline(raw_scores)
        with patch("Jabberjay.Models.Transformer.VIT.ConstantQ.run.pipeline", return_value=mock_pipe):
            with patch("Jabberjay.Models.Transformer.VIT.ConstantQ.run.get_image", return_value=mock_image):
                return predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

    def test_constantq_returns_normalised_scores(self):
        result = self._run_constantq()
        assert result[0]["label"] == "Bonafide"

    def test_mfcc_returns_normalised_scores(self):
        from Jabberjay.Models.Transformer.VIT.MFCC.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline()
        with patch("Jabberjay.Models.Transformer.VIT.MFCC.run.pipeline", return_value=mock_pipe):
            with patch("Jabberjay.Models.Transformer.VIT.MFCC.run.get_image", return_value=mock_image):
                result = predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

        assert result[0]["label"] == "Bonafide"

    def test_melspectrogram_returns_normalised_scores(self):
        from Jabberjay.Models.Transformer.VIT.MelSpectrogram.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline()
        with patch("Jabberjay.Models.Transformer.VIT.MelSpectrogram.run.pipeline", return_value=mock_pipe):
            with patch("Jabberjay.Models.Transformer.VIT.MelSpectrogram.run.get_image", return_value=mock_image):
                result = predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

        assert result[0]["label"] == "Bonafide"


# ---------------------------------------------------------------------------
# VIT utility
# ---------------------------------------------------------------------------

class TestGetImage:
    def test_returns_pil_image(self):
        import io

        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        def fake_savefig(buf, **kwargs):
            img = Image.new("RGB", (10, 10), color="white")
            img.save(buf, format="PNG")

        with patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"):
            with patch("Jabberjay.Models.Transformer.VIT.utility.plt.savefig", side_effect=fake_savefig):
                result = get_image(data=np.zeros((128, 100)), sr=22050.0)

        assert isinstance(result, Image.Image)

    def test_figure_closed_after_call(self):
        """Ensure plt.close is called even on success (no figure leak)."""
        import io

        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        def fake_savefig(buf, **kwargs):
            img = Image.new("RGB", (10, 10))
            img.save(buf, format="PNG")

        with patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"):
            with patch("Jabberjay.Models.Transformer.VIT.utility.plt.savefig", side_effect=fake_savefig):
                with patch("Jabberjay.Models.Transformer.VIT.utility.plt.close") as mock_close:
                    get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# RawNet2
# ---------------------------------------------------------------------------

class TestRawNet2Predict:
    def test_returns_prediction_and_confidence(self):
        from Jabberjay.Models.RawNet2.run import predict

        mock_predicted = MagicMock()
        mock_predicted.item.return_value = 1

        # probs[0][predicted.item()] must be a float-able value
        mock_inner = MagicMock()
        mock_inner.__getitem__ = MagicMock(return_value=0.85)
        mock_probs = MagicMock()
        mock_probs.__getitem__ = MagicMock(return_value=mock_inner)

        mock_out = MagicMock()
        mock_out.exp.return_value = mock_probs
        mock_out.max.return_value = (MagicMock(), mock_predicted)

        mock_model = MagicMock()
        mock_model.return_value = mock_out

        with patch("Jabberjay.Models.RawNet2.run.RawNet", return_value=mock_model):
            with patch("Jabberjay.Models.RawNet2.run.download_pretrained_model", return_value="/fake/model.pth"):
                with patch("torch.load", return_value={}):
                    with patch("torch.no_grad"):
                        prediction, confidence = predict(y=AUDIO[0])

        assert prediction is mock_predicted
        assert isinstance(confidence, float)
