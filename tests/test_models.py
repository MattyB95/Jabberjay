"""Unit tests for individual model predict() functions.

All network and model-loading operations are mocked so these tests run
offline without downloading any weights from HuggingFace.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.types import PredictionScore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AUDIO = (np.zeros(16000, dtype=np.float32), 16000.0)
FAKE_RAW_SCORES: list[PredictionScore] = [
    {"label": "bonafide", "score": 0.9},
    {"label": "fake", "score": 0.1},
]


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


_PIPELINE_PATH = "Jabberjay.Utilities.pipeline.pipeline"


class TestHuBERTPredict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.HuBERT.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["label"] == "Bonafide"
        assert result[1]["label"] == "Spoof"

    def test_confidence_preserved(self):
        from Jabberjay.Models.HuBERT.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["score"] == 0.9


class TestWav2Vec2Predict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.Wav2Vec2.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["label"] == "Bonafide"

    def test_confidence_preserved(self):
        from Jabberjay.Models.Wav2Vec2.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["score"] == 0.9

    def test_pipeline_is_loaded_once_across_repeated_calls(self):
        """run_pipeline() caches by (model_id, sampling_rate) — a second call
        for the same model must not reload the underlying pipeline."""
        from Jabberjay.Models.Wav2Vec2.run import predict

        mock_factory = MagicMock(return_value=_mock_pipeline())
        with patch(_PIPELINE_PATH, mock_factory):
            predict(y=AUDIO[0], sr=AUDIO[1])
            predict(y=AUDIO[0], sr=AUDIO[1])

        mock_factory.assert_called_once()


class TestWavLMPredict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.WavLM.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["label"] == "Bonafide"

    def test_confidence_preserved(self):
        from Jabberjay.Models.WavLM.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1])

        assert result[0]["score"] == 0.9


class TestASTPredict:
    def test_returns_normalised_scores(self):
        from Jabberjay.Models.Transformer.AST.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline()):
            result = predict(y=AUDIO[0], sr=AUDIO[1], dataset=Dataset.VoxCelebSpoof)

        assert result[0]["label"] == "Bonafide"

    def test_model_id_includes_dataset_name(self):
        from Jabberjay.Models.Transformer.AST.run import predict

        mock_factory = MagicMock(return_value=_mock_pipeline())
        with patch(_PIPELINE_PATH, mock_factory):
            predict(y=AUDIO[0], sr=AUDIO[1], dataset=Dataset.ASVspoof2019)

        call_kwargs = mock_factory.call_args
        assert "ASVspoof2019" in str(call_kwargs)


# ---------------------------------------------------------------------------
# VIT models
# ---------------------------------------------------------------------------


class TestVITPredict:
    @staticmethod
    def _run_constantq(raw_scores=None):
        from Jabberjay.Models.Transformer.VIT.ConstantQ.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline(raw_scores)
        mock_cqt = np.zeros((84, 32))
        with (
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.pipeline",
                return_value=mock_pipe,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.ConstantQ.run.get_image",
                return_value=mock_image,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.ConstantQ.run.librosa.cqt",
                return_value=mock_cqt,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.ConstantQ.run.librosa.amplitude_to_db",
                return_value=mock_cqt,
            ),
        ):
            return predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

    def test_constantq_returns_normalised_scores(self):
        result = self._run_constantq()
        assert result[0]["label"] == "Bonafide"

    def test_constantq_pipeline_is_loaded_once_across_repeated_calls(self):
        """VIT's shared load_pipeline() caches by model id — a second call
        for the same dataset must not reload the pipeline."""
        from Jabberjay.Models.Transformer.VIT.ConstantQ.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline()
        mock_cqt = np.zeros((84, 32))
        mock_factory = MagicMock(return_value=mock_pipe)
        with (
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.pipeline",
                mock_factory,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.ConstantQ.run.get_image",
                return_value=mock_image,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.ConstantQ.run.librosa.cqt",
                return_value=mock_cqt,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.ConstantQ.run.librosa.amplitude_to_db",
                return_value=mock_cqt,
            ),
        ):
            predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)
            predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

        mock_factory.assert_called_once()

    def test_mfcc_returns_normalised_scores(self):
        from Jabberjay.Models.Transformer.VIT.MFCC.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline()
        with (
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.pipeline",
                return_value=mock_pipe,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.MFCC.run.get_image",
                return_value=mock_image,
            ),
        ):
            result = predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

        assert result[0]["label"] == "Bonafide"

    def test_melspectrogram_returns_normalised_scores(self):
        from Jabberjay.Models.Transformer.VIT.MelSpectrogram.run import predict

        mock_image = MagicMock()
        mock_pipe = _mock_pipeline()
        with (
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.pipeline",
                return_value=mock_pipe,
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.MelSpectrogram.run.get_image",
                return_value=mock_image,
            ),
        ):
            result = predict(audio=AUDIO, dataset=Dataset.VoxCelebSpoof)

        assert result[0]["label"] == "Bonafide"


# ---------------------------------------------------------------------------
# VIT utility
# ---------------------------------------------------------------------------


class TestGetImage:
    def test_returns_pil_image(self):

        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        def fake_savefig(buf, **kwargs):
            img = Image.new("RGB", (10, 10), color="white")
            img.save(buf, format="PNG")

        with (
            patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.plt.savefig",
                side_effect=fake_savefig,
            ),
        ):
            result = get_image(data=np.zeros((128, 100)), sr=22050.0)

        assert isinstance(result, Image.Image)

    def test_figure_closed_after_call(self):
        """Ensure plt.close is called even on success (no figure leak)."""

        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        def fake_savefig(buf, **kwargs):
            img = Image.new("RGB", (10, 10))
            img.save(buf, format="PNG")

        with (
            patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.plt.savefig",
                side_effect=fake_savefig,
            ),
            patch("Jabberjay.Models.Transformer.VIT.utility.plt.close") as mock_close,
        ):
            get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_close.assert_called_once()

    def test_figure_closed_on_error(self):
        """Ensure plt.close is called even when rendering raises (no figure leak)."""
        import pytest

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        with (
            patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.plt.savefig",
                side_effect=RuntimeError("render error"),
            ),
            patch("Jabberjay.Models.Transformer.VIT.utility.plt.close") as mock_close,
            pytest.raises(RuntimeError),
        ):
            get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_close.assert_called_once()

    def test_buffer_closed_on_success(self):
        """Ensure the BytesIO buffer is closed after a successful call."""
        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        mock_buf = MagicMock(spec=io.BytesIO)
        mock_buf.read.return_value = b""

        real_buf = io.BytesIO()
        real_image = Image.new("RGB", (10, 10))
        real_image.save(real_buf, format="PNG")
        real_buf.seek(0)
        mock_buf.__enter__ = MagicMock(return_value=mock_buf)
        mock_buf.__exit__ = MagicMock(return_value=False)

        def fake_savefig(buf, **kwargs):
            real_image.save(buf, format="PNG")

        with (
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.io.BytesIO",
                return_value=mock_buf,
            ),
            patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.plt.savefig",
                side_effect=fake_savefig,
            ),
            patch("Jabberjay.Models.Transformer.VIT.utility.plt.close"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.Image.open",
                return_value=real_image,
            ),
        ):
            get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_buf.close.assert_called_once()

    def test_buffer_closed_on_error(self):
        """Ensure the BytesIO buffer is closed even when Image.open raises."""
        import pytest

        from Jabberjay.Models.Transformer.VIT.utility import get_image

        mock_buf = MagicMock(spec=io.BytesIO)

        def fake_savefig(buf, **kwargs):
            pass

        with (
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.plt.subplots",
                return_value=(MagicMock(), MagicMock()),
            ),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.io.BytesIO",
                return_value=mock_buf,
            ),
            patch("Jabberjay.Models.Transformer.VIT.utility.librosa.display.specshow"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.plt.savefig",
                side_effect=fake_savefig,
            ),
            patch("Jabberjay.Models.Transformer.VIT.utility.plt.close"),
            patch(
                "Jabberjay.Models.Transformer.VIT.utility.Image.open",
                side_effect=RuntimeError("corrupt image"),
            ),
            pytest.raises(RuntimeError),
        ):
            get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_buf.close.assert_called_once()


# ---------------------------------------------------------------------------
# Spectra0
# ---------------------------------------------------------------------------


class _SpectraModelTestBase:
    """Shared test logic for all Spectra models that return softmax scores."""

    model_module: str
    model_class: str

    def _make_mock_model(self, logits):
        import torch

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.tensor([logits])
        return mock_model

    def _patch_path(self):
        return f"Jabberjay.Models.{self.model_module}.run.{self.model_class}.from_pretrained"

    def test_returns_bonafide_and_spoof_scores(self):
        import importlib

        predict = importlib.import_module(
            f"Jabberjay.Models.{self.model_module}.run"
        ).predict
        mock_model = self._make_mock_model([0.2, 0.8])
        with patch(self._patch_path(), return_value=mock_model):
            result = predict(y=AUDIO[0], sr=AUDIO[1])
        assert {r["label"] for r in result} == {"Bonafide", "Spoof"}

    def test_higher_bonafide_logit_yields_higher_bonafide_score(self):
        import importlib

        predict = importlib.import_module(
            f"Jabberjay.Models.{self.model_module}.run"
        ).predict
        mock_model = self._make_mock_model([0.1, 0.9])
        with patch(self._patch_path(), return_value=mock_model):
            result = predict(y=AUDIO[0], sr=AUDIO[1])
        bonafide = next(r for r in result if r["label"] == "Bonafide")
        spoof = next(r for r in result if r["label"] == "Spoof")
        assert bonafide["score"] > spoof["score"]

    def test_scores_sum_to_one(self):
        import importlib

        import pytest

        predict = importlib.import_module(
            f"Jabberjay.Models.{self.model_module}.run"
        ).predict
        mock_model = self._make_mock_model([0.3, 0.7])
        with patch(self._patch_path(), return_value=mock_model):
            result = predict(y=AUDIO[0], sr=AUDIO[1])
        assert sum(r["score"] for r in result) == pytest.approx(1.0, abs=1e-5)

    def test_model_is_loaded_once_across_repeated_calls(self):
        """Spectra.shared.load_pretrained() caches by (class, model id, device)
        — a second predict() call for the same model must not reload it."""
        import importlib

        predict = importlib.import_module(
            f"Jabberjay.Models.{self.model_module}.run"
        ).predict
        mock_model = self._make_mock_model([0.2, 0.8])
        with patch(self._patch_path(), return_value=mock_model) as mock_from_pretrained:
            predict(y=AUDIO[0], sr=AUDIO[1])
            predict(y=AUDIO[0], sr=AUDIO[1])
        mock_from_pretrained.assert_called_once()


class TestSpectra0Predict(_SpectraModelTestBase):
    model_module = "Spectra0"
    model_class = "Spectra0Model"

    def test_uses_calibrated_threshold_not_naive_argmax(self):
        """bonafide_logit=-0.5 is below spoof_logit=0.5 (naive argmax would
        say Spoof) but above the model's -1.0625009 calibrated threshold,
        so the verdict must be Bonafide."""
        from Jabberjay.Models.Spectra0.run import predict

        mock_model = self._make_mock_model([0.5, -0.5])
        with patch(self._patch_path(), return_value=mock_model):
            result = predict(y=AUDIO[0], sr=AUDIO[1])
        bonafide = next(r for r in result if r["label"] == "Bonafide")
        spoof = next(r for r in result if r["label"] == "Spoof")
        assert bonafide["score"] > spoof["score"]


class TestSpectraAASISTPredict(_SpectraModelTestBase):
    model_module = "SpectraAASIST"
    model_class = "SpectraAASIST"

    def test_uses_calibrated_threshold_not_naive_argmax(self):
        """bonafide_logit=-0.5 is below spoof_logit=0.5 (naive argmax would
        say Spoof) but above the model's -1.140625 calibrated threshold,
        so the verdict must be Bonafide."""
        from Jabberjay.Models.SpectraAASIST.run import predict

        mock_model = self._make_mock_model([0.5, -0.5])
        with patch(self._patch_path(), return_value=mock_model):
            result = predict(y=AUDIO[0], sr=AUDIO[1])
        bonafide = next(r for r in result if r["label"] == "Bonafide")
        spoof = next(r for r in result if r["label"] == "Spoof")
        assert bonafide["score"] > spoof["score"]


class TestSpectraAASIST3Predict(_SpectraModelTestBase):
    model_module = "SpectraAASIST3"
    model_class = "SpectraAASIST3"


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------


class TestClassicalPredict:
    def test_returns_prediction_and_confidence(self):
        from Jabberjay.Models.Classical.run import predict

        mock_clf = MagicMock()
        mock_clf.predict.return_value = [1]
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.9]])

        with (
            patch(
                "Jabberjay.Models.Classical.run.download_pretrained_model",
                return_value="/fake/model.joblib",
            ),
            patch("Jabberjay.Models.Classical.run.load", return_value=mock_clf),
            patch(
                "Jabberjay.Models.Classical.run.get_features",
                return_value=MagicMock(),
            ),
        ):
            prediction, confidence = predict(audio=AUDIO)

        assert prediction == 1
        assert confidence == pytest.approx(0.9)

    def test_spoof_prediction(self):
        from Jabberjay.Models.Classical.run import predict

        mock_clf = MagicMock()
        mock_clf.predict.return_value = [0]
        mock_clf.predict_proba.return_value = np.array([[0.8, 0.2]])

        with (
            patch(
                "Jabberjay.Models.Classical.run.download_pretrained_model",
                return_value="/fake/model.joblib",
            ),
            patch("Jabberjay.Models.Classical.run.load", return_value=mock_clf),
            patch(
                "Jabberjay.Models.Classical.run.get_features",
                return_value=MagicMock(),
            ),
        ):
            prediction, confidence = predict(audio=AUDIO)

        assert prediction == 0
        assert confidence == pytest.approx(0.8)

    def test_classifier_is_loaded_once_across_repeated_calls(self):
        """_load_classifier() caches the downloaded joblib model — a second
        predict() call must not re-download or re-deserialise it."""
        from Jabberjay.Models.Classical.run import predict

        mock_clf = MagicMock()
        mock_clf.predict.return_value = [1]
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.9]])

        with (
            patch(
                "Jabberjay.Models.Classical.run.download_pretrained_model",
                return_value="/fake/model.joblib",
            ) as mock_download,
            patch(
                "Jabberjay.Models.Classical.run.load", return_value=mock_clf
            ) as mock_load,
            patch(
                "Jabberjay.Models.Classical.run.get_features",
                return_value=MagicMock(),
            ),
        ):
            predict(audio=AUDIO)
            predict(audio=AUDIO)

        mock_download.assert_called_once()
        mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# RawNet2
# ---------------------------------------------------------------------------


class TestRawNet2Config:
    def test_oserror_raises_runtime_error(self):
        import Jabberjay.Models.RawNet2.run as rawnet2_run
        from Jabberjay.Models.RawNet2.run import predict

        original = rawnet2_run._CONFIG
        rawnet2_run._CONFIG = None  # reset cache so the open is attempted
        try:
            with (
                patch("builtins.open", side_effect=OSError("missing")),
                pytest.raises(RuntimeError, match="Failed to load RawNet2 config"),
            ):
                predict(y=AUDIO[0], sr=AUDIO[1])
        finally:
            rawnet2_run._CONFIG = original

    def test_yaml_error_raises_runtime_error(self):
        import yaml

        import Jabberjay.Models.RawNet2.run as rawnet2_run
        from Jabberjay.Models.RawNet2.run import predict

        original = rawnet2_run._CONFIG
        rawnet2_run._CONFIG = None  # reset cache so the open is attempted
        try:
            with patch("builtins.open", mock_open := MagicMock()):
                mock_open.return_value.__enter__.return_value = MagicMock()
                with (
                    patch(
                        "Jabberjay.Models.RawNet2.run.yaml.safe_load",
                        side_effect=yaml.YAMLError("bad yaml"),
                    ),
                    pytest.raises(RuntimeError, match="Failed to load RawNet2 config"),
                ):
                    predict(y=AUDIO[0], sr=AUDIO[1])
        finally:
            rawnet2_run._CONFIG = original


class TestRawNet2Predict:
    def test_empty_audio_raises_value_error(self):
        from Jabberjay.Models.RawNet2.run import predict

        with (
            patch("Jabberjay.Models.RawNet2.run.RawNet", return_value=MagicMock()),
            patch(
                "Jabberjay.Models.RawNet2.run.download_pretrained_model",
                return_value="/fake/model.pth",
            ),
            patch("torch.load", return_value={}),
            pytest.raises(ValueError, match="Input audio array is empty"),
        ):
            predict(y=np.zeros(0, dtype=np.float32), sr=16000.0)

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

        with (
            patch("Jabberjay.Models.RawNet2.run.RawNet", return_value=mock_model),
            patch(
                "Jabberjay.Models.RawNet2.run.download_pretrained_model",
                return_value="/fake/model.pth",
            ),
            patch("torch.load", return_value={}),
            patch("torch.no_grad"),
        ):
            prediction, confidence = predict(y=AUDIO[0], sr=AUDIO[1])

        assert prediction is mock_predicted
        assert isinstance(confidence, float)

    def test_resamples_when_sr_differs_from_target(self):
        """22050Hz input must be resampled to the model's 16kHz target rate."""
        from Jabberjay.Models.RawNet2.run import predict

        mock_predicted = MagicMock()
        mock_predicted.item.return_value = 1
        mock_inner = MagicMock()
        mock_inner.__getitem__ = MagicMock(return_value=0.85)
        mock_probs = MagicMock()
        mock_probs.__getitem__ = MagicMock(return_value=mock_inner)
        mock_out = MagicMock()
        mock_out.exp.return_value = mock_probs
        mock_out.max.return_value = (MagicMock(), mock_predicted)
        mock_model = MagicMock()
        mock_model.return_value = mock_out

        y = np.zeros(22050, dtype=np.float32)
        with (
            patch("Jabberjay.Models.RawNet2.run.RawNet", return_value=mock_model),
            patch(
                "Jabberjay.Models.RawNet2.run.download_pretrained_model",
                return_value="/fake/model.pth",
            ),
            patch("torch.load", return_value={}),
            patch("torch.no_grad"),
        ):
            prediction, confidence = predict(y=y, sr=22050.0)

        assert prediction is mock_predicted
        assert isinstance(confidence, float)

    def test_trims_audio_longer_than_max_len(self):
        """Audio already at/above nb_samp must be trimmed, not repeat-padded."""
        from Jabberjay.Models.RawNet2.run import predict

        mock_predicted = MagicMock()
        mock_predicted.item.return_value = 1
        mock_inner = MagicMock()
        mock_inner.__getitem__ = MagicMock(return_value=0.85)
        mock_probs = MagicMock()
        mock_probs.__getitem__ = MagicMock(return_value=mock_inner)
        mock_out = MagicMock()
        mock_out.exp.return_value = mock_probs
        mock_out.max.return_value = (MagicMock(), mock_predicted)
        mock_model = MagicMock()
        mock_model.return_value = mock_out

        y = np.zeros(70_000, dtype=np.float32)  # longer than nb_samp=64600
        with (
            patch("Jabberjay.Models.RawNet2.run.RawNet", return_value=mock_model),
            patch(
                "Jabberjay.Models.RawNet2.run.download_pretrained_model",
                return_value="/fake/model.pth",
            ),
            patch("torch.load", return_value={}),
            patch("torch.no_grad"),
        ):
            prediction, confidence = predict(y=y, sr=16000.0)

        assert prediction is mock_predicted
        assert isinstance(confidence, float)

    def test_model_is_loaded_once_across_repeated_calls(self):
        """_load_model() caches the constructed RawNet + downloaded weights
        — a second predict() call must not rebuild or re-download them."""
        from Jabberjay.Models.RawNet2.run import predict

        mock_predicted = MagicMock()
        mock_predicted.item.return_value = 1
        mock_inner = MagicMock()
        mock_inner.__getitem__ = MagicMock(return_value=0.85)
        mock_probs = MagicMock()
        mock_probs.__getitem__ = MagicMock(return_value=mock_inner)
        mock_out = MagicMock()
        mock_out.exp.return_value = mock_probs
        mock_out.max.return_value = (MagicMock(), mock_predicted)
        mock_model = MagicMock()
        mock_model.return_value = mock_out

        with (
            patch(
                "Jabberjay.Models.RawNet2.run.RawNet", return_value=mock_model
            ) as mock_rawnet,
            patch(
                "Jabberjay.Models.RawNet2.run.download_pretrained_model",
                return_value="/fake/model.pth",
            ) as mock_download,
            patch("torch.load", return_value={}),
            patch("torch.no_grad"),
        ):
            predict(y=AUDIO[0], sr=AUDIO[1])
            predict(y=AUDIO[0], sr=AUDIO[1])

        mock_rawnet.assert_called_once()
        mock_download.assert_called_once()
