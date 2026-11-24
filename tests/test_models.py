"""Unit tests for individual model predict() functions.

All network and model-loading operations are mocked so these tests run
offline without downloading any weights from HuggingFace.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from Jabberjay.Utilities.enum_handler import Dataset, Visualisation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AUDIO = (np.zeros(16000, dtype=np.float32), 16000.0)
# Raw pipeline output, before label normalisation
FAKE_RAW_SCORES: list[dict[str, str | float]] = [
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

    def test_pipeline_loaded_onto_selected_device(self):
        from Jabberjay.Models.HuBERT.run import predict

        mock_factory = MagicMock(return_value=_mock_pipeline())
        with (
            patch(_PIPELINE_PATH, mock_factory),
            patch("Jabberjay.Utilities.pipeline.get_device", return_value="cuda"),
        ):
            predict(y=AUDIO[0], sr=AUDIO[1])

        assert str(mock_factory.call_args.kwargs["device"]) == "cuda"


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


class TestWavLMLabelMapping:
    """The WavLM checkpoint has no id2label; index 0 is spoof (see WavLM/run.py)."""

    def _scores(self, raw):
        from Jabberjay.Models.WavLM.run import predict

        with patch(_PIPELINE_PATH, return_value=_mock_pipeline(raw)):
            return {r["label"]: r["score"] for r in predict(y=AUDIO[0], sr=AUDIO[1])}

    def test_label_0_is_spoof(self):
        scores = self._scores(
            [{"label": "LABEL_0", "score": 0.98}, {"label": "LABEL_1", "score": 0.02}]
        )
        assert scores == {"Spoof": 0.98, "Bonafide": 0.02}

    def test_label_1_is_bonafide(self):
        scores = self._scores(
            [{"label": "LABEL_1", "score": 0.7}, {"label": "LABEL_0", "score": 0.3}]
        )
        assert scores == {"Bonafide": 0.7, "Spoof": 0.3}


class TestPipelineRevisions:
    def test_pipeline_is_loaded_at_the_pinned_revision(self):
        from Jabberjay.Models.HuBERT.run import MODEL_ID, predict
        from Jabberjay.Utilities.hugging_face import REVISIONS

        mock_factory = MagicMock(return_value=_mock_pipeline())
        with patch(_PIPELINE_PATH, mock_factory):
            predict(y=AUDIO[0], sr=AUDIO[1])
        assert mock_factory.call_args.kwargs["revision"] == REVISIONS[MODEL_ID]


def _every_repo_jabberjay_loads() -> list[str]:
    from Jabberjay.Models.Classical import run as classical
    from Jabberjay.Models.DFArena import run as df_arena
    from Jabberjay.Models.HuBERT import run as hubert
    from Jabberjay.Models.RawNet2 import run as rawnet2
    from Jabberjay.Models.Spectra0 import run as spectra0
    from Jabberjay.Models.SpectraAASIST import run as spectra_aasist
    from Jabberjay.Models.SpectraAASIST3 import run as spectra_aasist3
    from Jabberjay.Models.Transformer.AST import run as ast
    from Jabberjay.Models.Transformer.VIT import run as vit
    from Jabberjay.Models.Wav2Vec2 import run as wav2vec2
    from Jabberjay.Models.WavLM import run as wavlm

    return [
        classical.REPO_ID,
        rawnet2.REPO_ID,
        df_arena.MODEL_ID,
        "facebook/wav2vec2-xls-r-1b",  # DFArena1B's encoder config
        "facebook/wav2vec2-xls-r-300m",  # the Spectra encoders' config
        *(m.MODEL_ID for m in (hubert, wav2vec2, wavlm)),
        *(m.MODEL_ID for m in (spectra0, spectra_aasist, spectra_aasist3)),
        *(ast.model_id(d) for d in Dataset),
        *(vit.model_id(d, v) for d in Dataset for v in Visualisation),
    ]


def test_every_repo_jabberjay_loads_is_pinned():
    from Jabberjay.Utilities.hugging_face import REVISIONS

    repos = _every_repo_jabberjay_loads()
    assert set(repos) == set(REVISIONS)


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

_VIT = "Jabberjay.Models.Transformer.VIT.run"


class TestVITPredict:
    @staticmethod
    def _predict(visualisation, *, sr=22050.0, factory=None, **patches):
        from Jabberjay.Models.Transformer.VIT.run import predict

        factory = factory or MagicMock(return_value=_mock_pipeline())
        with (
            patch(_PIPELINE_PATH, factory),
            patch(f"{_VIT}.get_image", return_value=MagicMock()) as mock_image,
        ):
            result = predict(
                y=np.ones(22050, dtype=np.float32),
                sr=sr,
                dataset=Dataset.VoxCelebSpoof,
                visualisation=visualisation,
            )
        return result, factory, mock_image

    @pytest.mark.parametrize("visualisation", list(Visualisation))
    def test_returns_normalised_scores(self, visualisation):
        result, *_ = self._predict(visualisation)
        assert result[0]["label"] == "Bonafide"

    @pytest.mark.parametrize(
        ("visualisation", "repo_name"),
        [
            (Visualisation.ConstantQ, "ConstantQ"),
            (Visualisation.MelSpectrogram, "Mel_Spectrogram"),
            (Visualisation.MFCC, "MFCC"),
        ],
    )
    def test_model_id_matches_hub_repo_name(self, visualisation, repo_name):
        from Jabberjay.Models.Transformer.VIT.run import model_id

        assert model_id(Dataset.ASVspoof5, visualisation) == (
            f"MattyB95/VIT-ASVspoof5-{repo_name}-Synthetic-Voice-Detection"
        )

    def test_pipeline_loaded_at_pinned_revision_on_selected_device(self):
        from Jabberjay.Models.Transformer.VIT.run import model_id
        from Jabberjay.Utilities.hugging_face import REVISIONS

        with patch("Jabberjay.Utilities.pipeline.get_device", return_value="mps"):
            _, factory, _ = self._predict(Visualisation.ConstantQ)
        kwargs = factory.call_args.kwargs
        assert factory.call_args.args[0] == "image-classification"
        assert kwargs["device"].type == "mps"
        repo = model_id(Dataset.VoxCelebSpoof, Visualisation.ConstantQ)
        assert kwargs["revision"] == REVISIONS[repo]

    def test_pipeline_is_loaded_once_across_repeated_calls(self):
        factory = MagicMock(return_value=_mock_pipeline())
        self._predict(Visualisation.ConstantQ, factory=factory)
        self._predict(Visualisation.ConstantQ, factory=factory)
        factory.assert_called_once()

    def test_audio_is_rendered_at_the_training_sample_rate(self):
        from Jabberjay.Models.Transformer.VIT.run import TRAINING_SR

        _, _, mock_image = self._predict(Visualisation.MFCC, sr=16000.0)
        assert mock_image.call_args.kwargs["sr"] == TRAINING_SR

    def test_training_rate_audio_is_not_resampled(self):
        with patch(f"{_VIT}.librosa.resample") as mock_resample:
            self._predict(Visualisation.MFCC, sr=22050.0)
        mock_resample.assert_not_called()


class TestGetImage:
    def test_returns_pil_image(self):
        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.run import get_image

        def fake_savefig(buf, **kwargs):
            img = Image.new("RGB", (10, 10), color="white")
            img.save(buf, format="PNG")

        with (
            patch(f"{_VIT}.librosa.display.specshow"),
            patch(f"{_VIT}.Figure.savefig", side_effect=fake_savefig),
        ):
            result = get_image(data=np.zeros((128, 100)), sr=22050.0)

        assert isinstance(result, Image.Image)

    def test_does_not_touch_pyplot(self):
        """Rendering must bypass pyplot entirely — no GUI backend start-up and
        no figures left registered in pyplot's global state."""
        from matplotlib import pyplot as plt

        from Jabberjay.Models.Transformer.VIT.run import get_image

        plt.close("all")
        with patch("matplotlib.pyplot.figure") as mock_plt_figure:
            get_image(data=np.zeros((128, 100)), sr=22050.0)
        mock_plt_figure.assert_not_called()
        assert plt.get_fignums() == []

    def test_renders_off_the_main_thread(self):
        import threading

        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.run import get_image

        result = {}

        def render():
            result["img"] = get_image(data=np.zeros((128, 100)), sr=22050.0)

        thread = threading.Thread(target=render)
        thread.start()
        thread.join()
        assert isinstance(result["img"], Image.Image)

    def test_buffer_closed_on_success(self):
        from PIL import Image

        from Jabberjay.Models.Transformer.VIT.run import get_image

        mock_buf = MagicMock(spec=io.BytesIO)
        real_image = Image.new("RGB", (10, 10))

        with (
            patch(f"{_VIT}.io.BytesIO", return_value=mock_buf),
            patch(f"{_VIT}.librosa.display.specshow"),
            patch(f"{_VIT}.Figure.savefig"),
            patch(f"{_VIT}.Image.open", return_value=real_image),
        ):
            get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_buf.close.assert_called_once()

    def test_buffer_closed_on_error(self):
        from Jabberjay.Models.Transformer.VIT.run import get_image

        mock_buf = MagicMock(spec=io.BytesIO)

        with (
            patch(f"{_VIT}.Figure"),
            patch(f"{_VIT}.io.BytesIO", return_value=mock_buf),
            patch(f"{_VIT}.librosa.display.specshow"),
            patch(f"{_VIT}.Image.open", side_effect=RuntimeError("corrupt image")),
            pytest.raises(RuntimeError),
        ):
            get_image(data=np.zeros((128, 100)), sr=22050.0)

        mock_buf.close.assert_called_once()


# ---------------------------------------------------------------------------
# Spectra family
# ---------------------------------------------------------------------------


class TestSpectraPreprocess:
    def test_rejects_empty_audio(self):
        from Jabberjay.Models.Spectra.shared import preprocess

        with pytest.raises(ValueError, match="empty"):
            preprocess(np.array([], dtype=np.float32), sr=16000.0)

    def test_pads_short_audio_to_max_len(self):
        from Jabberjay.Models.Spectra.shared import _MAX_LEN, preprocess

        out = preprocess(np.ones(1000, dtype=np.float32), sr=16000.0)
        assert out.shape == (1, _MAX_LEN)

    def test_truncates_long_audio_to_max_len(self):
        from Jabberjay.Models.Spectra.shared import _MAX_LEN, preprocess

        out = preprocess(np.ones(_MAX_LEN * 2, dtype=np.float32), sr=16000.0)
        assert out.shape == (1, _MAX_LEN)

    def test_resamples_when_sample_rate_differs(self):
        from Jabberjay.Models.Spectra.shared import _MAX_LEN, preprocess

        out = preprocess(np.ones(8000, dtype=np.float32), sr=8000.0)
        assert out.shape == (1, _MAX_LEN)

    def test_preemphasis_can_be_disabled(self):
        from Jabberjay.Models.Spectra.shared import preprocess

        y = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        raw = preprocess(y, sr=16000.0, preemphasis=False)
        emphasised = preprocess(y, sr=16000.0)
        assert np.allclose(raw[0, :1000].numpy(), y)
        assert not np.allclose(emphasised[0, :1000].numpy(), y)


class _TinySpectra(torch.nn.Module):
    """Stand-in for a Spectra model: one parameter and one persistent buffer."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("scale", torch.ones(2))


def _tiny_state_dict(**overrides):
    state = {
        "linear.weight": torch.eye(2),
        "linear.bias": torch.tensor([0.5, -0.5]),
        "scale": torch.full((2,), 3.0),
    }
    state.update(overrides)
    return {k: v for k, v in state.items() if v is not None}


class TestSpectraLoadPretrained:
    _SHARED = "Jabberjay.Models.Spectra.shared"

    def _load(self, state_dict, repo="lab260/spectra_0"):
        from Jabberjay.Models.Spectra.shared import load_pretrained

        with (
            patch(
                f"{self._SHARED}.download_pretrained_model",
                return_value="/fake/model.safetensors",
            ) as mock_download,
            patch(f"{self._SHARED}.load_file", return_value=state_dict),
        ):
            model = load_pretrained(_TinySpectra, repo, "cpu")
        return model, mock_download

    def test_adopts_checkpoint_tensors(self):
        model, _ = self._load(_tiny_state_dict())
        assert torch.equal(model.linear.weight, torch.eye(2))
        assert torch.equal(model.scale, torch.full((2,), 3.0))
        assert not any(p.is_meta for p in model.parameters())
        assert not model.training

    def test_downloads_safetensors_from_the_pinned_revision(self):
        _, mock_download = self._load(_tiny_state_dict())
        mock_download.assert_called_once_with(
            repo_id="lab260/spectra_0", filename="model.safetensors"
        )

    def test_missing_checkpoint_key_raises(self):
        with pytest.raises(RuntimeError, match="Missing key"):
            self._load(_tiny_state_dict(scale=None))

    def test_unexpected_checkpoint_key_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected key"):
            self._load(_tiny_state_dict(extra=torch.zeros(1)))

    def test_tensor_left_on_meta_device_raises(self):
        from Jabberjay.Models.Spectra.shared import load_pretrained

        class WithNonPersistentBuffer(_TinySpectra):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2), persistent=False)

        with (
            patch(f"{self._SHARED}.download_pretrained_model"),
            patch(f"{self._SHARED}.load_file", return_value=_tiny_state_dict()),
            pytest.raises(RuntimeError, match="not provided by the checkpoint"),
        ):
            load_pretrained(WithNonPersistentBuffer, "lab260/spectra_0", "cpu")

    def test_model_is_loaded_once_across_repeated_calls(self):
        from Jabberjay.Models.Spectra.shared import load_pretrained

        with (
            patch(f"{self._SHARED}.download_pretrained_model") as mock_download,
            patch(f"{self._SHARED}.load_file", return_value=_tiny_state_dict()),
        ):
            first = load_pretrained(_TinySpectra, "lab260/spectra_0", "cpu")
            second = load_pretrained(_TinySpectra, "lab260/spectra_0", "cpu")
        assert first is second
        mock_download.assert_called_once()


@pytest.mark.parametrize(
    ("model_module", "model_class"),
    [
        ("Spectra0", "Spectra0Model"),
        ("SpectraAASIST", "SpectraAASIST"),
        ("SpectraAASIST3", "SpectraAASIST3"),
    ],
)
def test_spectra_architecture_builds_on_meta_device(model_module, model_class):
    """Every Spectra architecture must construct under torch.device("meta")
    (AASIST3's KANLinear init uses lstsq, which has no meta kernel)."""
    import importlib

    cls = getattr(
        importlib.import_module(f"Jabberjay.Models.{model_module}.model"), model_class
    )
    with (
        patch("Jabberjay.Models.Spectra.shared.Wav2Vec2Config.from_pretrained") as cfg,
        torch.device("meta"),
    ):
        from transformers import Wav2Vec2Config

        cfg.return_value = Wav2Vec2Config(
            hidden_size=1024,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            conv_dim=(8, 8),
            conv_kernel=(10, 3),
            conv_stride=(5, 2),
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=4,
        )
        model = cls()
    assert all(p.is_meta for p in model.parameters())


class _SpectraPredictBase:
    """Shared predict() tests for the Spectra models (loader mocked)."""

    model_module: str

    def _predict(self, logits):
        import importlib

        run = importlib.import_module(f"Jabberjay.Models.{self.model_module}.run")
        mock_model = MagicMock(return_value=torch.tensor([logits]))
        with patch.object(run, "load_pretrained", return_value=mock_model) as loader:
            result = run.predict(y=AUDIO[0], sr=AUDIO[1])
        return {r["label"]: r["score"] for r in result}, loader

    def test_returns_bonafide_and_spoof_scores(self):
        scores, _ = self._predict([0.2, 0.8])
        assert set(scores) == {"Bonafide", "Spoof"}

    def test_higher_bonafide_logit_yields_higher_bonafide_score(self):
        scores, _ = self._predict([0.1, 0.9])
        assert scores["Bonafide"] > scores["Spoof"]

    def test_scores_sum_to_one(self):
        scores, _ = self._predict([0.3, 0.7])
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)

    def test_loads_its_own_model_id(self):
        import importlib

        run = importlib.import_module(f"Jabberjay.Models.{self.model_module}.run")
        _, loader = self._predict([0.2, 0.8])
        assert loader.call_args.args[1] == run.MODEL_ID


class TestSpectra0Predict(_SpectraPredictBase):
    model_module = "Spectra0"

    def test_uses_calibrated_threshold_not_naive_argmax(self):
        """bonafide_logit=-0.5 is below spoof_logit=0.5 (naive argmax would
        say Spoof) but above the model's -1.0625009 calibrated threshold,
        so the verdict must be Bonafide."""
        scores, _ = self._predict([0.5, -0.5])
        assert scores["Bonafide"] > scores["Spoof"]


class TestSpectraAASISTPredict(_SpectraPredictBase):
    model_module = "SpectraAASIST"

    def test_uses_calibrated_threshold_not_naive_argmax(self):
        """bonafide_logit=-0.5 is below spoof_logit=0.5 (naive argmax would
        say Spoof) but above the model's -1.140625 calibrated threshold,
        so the verdict must be Bonafide."""
        scores, _ = self._predict([0.5, -0.5])
        assert scores["Bonafide"] > scores["Spoof"]


class TestSpectraAASIST3Predict(_SpectraPredictBase):
    model_module = "SpectraAASIST3"


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------


class TestClassicalPredict:
    _RUN = "Jabberjay.Models.Classical.run"

    def _predict(self, proba, *, sr=22050.0, classes=("Genuine.wav", "Synthetic.wav")):
        from Jabberjay.Models.Classical.run import predict

        mock_clf = MagicMock()
        mock_clf.classes_ = np.array(classes)
        mock_clf.predict_proba.return_value = np.array([proba])
        with (
            patch(
                f"{self._RUN}.download_pretrained_model",
                return_value="/fake/model.joblib",
            ) as mock_download,
            patch(f"{self._RUN}.load", return_value=mock_clf) as mock_load,
            patch(f"{self._RUN}.get_features", return_value=MagicMock()) as mock_feat,
        ):
            result = predict(y=np.ones(22050, dtype=np.float32), sr=sr)
            result_again = predict(y=np.ones(22050, dtype=np.float32), sr=sr)
        assert result == result_again
        return (
            {r["label"]: r["score"] for r in result},
            mock_download,
            mock_load,
            mock_feat,
        )

    def test_synthetic_class_maps_to_spoof(self):
        """Regression: the classifier's classes are the strings
        'Genuine.wav'/'Synthetic.wav', and bool() of either is True — so
        earlier versions labelled every clip Bonafide."""
        scores, *_ = self._predict([0.2, 0.8])
        assert scores == {"Bonafide": pytest.approx(0.2), "Spoof": pytest.approx(0.8)}

    def test_genuine_class_maps_to_bonafide(self):
        scores, *_ = self._predict([0.9, 0.1])
        assert scores["Bonafide"] == pytest.approx(0.9)

    def test_scores_follow_classes_order_not_index_convention(self):
        scores, *_ = self._predict([0.7, 0.3], classes=("Synthetic.wav", "Genuine.wav"))
        assert scores == {"Spoof": pytest.approx(0.7), "Bonafide": pytest.approx(0.3)}

    def test_classifier_is_loaded_once_across_repeated_calls(self):
        _, mock_download, mock_load, _ = self._predict([0.5, 0.5])
        mock_download.assert_called_once()
        mock_load.assert_called_once()

    def test_features_extracted_at_training_sample_rate(self):
        from Jabberjay.Models.Classical.run import TRAINING_SR

        *_, mock_feat = self._predict([0.5, 0.5], sr=16000.0)
        y, sr = mock_feat.call_args.kwargs["audio"]
        assert sr == TRAINING_SR
        assert len(y) == pytest.approx(22050 * TRAINING_SR / 16000, abs=2)


# ---------------------------------------------------------------------------
# DF Arena 1B
# ---------------------------------------------------------------------------


class TestDFArenaPredict:
    def _predict(self, logits, state_dict=None):
        """Run predict() with the model, download, and torch.load all mocked."""
        from Jabberjay.Models.DFArena.run import predict

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.tensor([logits])
        with (
            patch(
                "Jabberjay.Models.DFArena.run.DFArena1B", return_value=mock_model
            ) as mock_cls,
            patch(
                "Jabberjay.Models.DFArena.run.download_pretrained_model",
                return_value="/fake/pytorch_model.bin",
            ) as mock_download,
            patch("torch.load", return_value=state_dict or {}) as mock_load,
        ):
            result = predict(y=AUDIO[0], sr=AUDIO[1])
        return result, mock_model, mock_cls, mock_download, mock_load

    def test_returns_bonafide_and_spoof_scores(self):
        result, *_ = self._predict([0.2, 0.8])
        assert {r["label"] for r in result} == {"Bonafide", "Spoof"}

    def test_logit_order_is_spoof_then_bonafide(self):
        """config.json maps id 0 -> spoof and id 1 -> bonafide."""
        result, *_ = self._predict([-2.0, 3.0])
        bonafide = next(r for r in result if r["label"] == "Bonafide")
        spoof = next(r for r in result if r["label"] == "Spoof")
        assert bonafide["score"] > spoof["score"]

    def test_scores_sum_to_one(self):
        result, *_ = self._predict([0.3, 0.7])
        assert sum(r["score"] for r in result) == pytest.approx(1.0, abs=1e-5)

    def test_downloads_the_checkpoint_through_the_pinned_wrapper(self):
        from Jabberjay.Models.DFArena.run import MODEL_ID, WEIGHTS_FILE

        *_, mock_download, _ = self._predict([0.2, 0.8])
        mock_download.assert_called_once_with(repo_id=MODEL_ID, filename=WEIGHTS_FILE)

    def test_loads_weights_safely_and_strictly(self):
        *_, mock_model, _, _, mock_load = self._predict([0.2, 0.8])
        assert mock_load.call_args.kwargs["weights_only"] is True
        assert mock_model.load_state_dict.call_args.kwargs["strict"] is True

    def test_strips_wrapper_prefix_from_state_dict_keys(self):
        state_dict = {
            "backbone.fc0.weight": torch.zeros(1),
            "backbone.conformer.fc5.bias": torch.zeros(2),
        }
        _, mock_model, *_ = self._predict([0.2, 0.8], state_dict=state_dict)
        loaded = mock_model.load_state_dict.call_args.args[0]
        assert set(loaded) == {"fc0.weight", "conformer.fc5.bias"}

    def test_model_is_loaded_once_across_repeated_calls(self):
        from Jabberjay.Models.DFArena.run import predict

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.tensor([[0.2, 0.8]])
        with (
            patch(
                "Jabberjay.Models.DFArena.run.DFArena1B", return_value=mock_model
            ) as mock_cls,
            patch("Jabberjay.Models.DFArena.run.download_pretrained_model"),
            patch("torch.load", return_value={}),
        ):
            predict(y=AUDIO[0], sr=AUDIO[1])
            predict(y=AUDIO[0], sr=AUDIO[1])
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# RawNet2
# ---------------------------------------------------------------------------

_RAWNET = "Jabberjay.Models.RawNet2.run"


class TestRawNet2Config:
    def test_loads_bundled_config(self):
        from Jabberjay.Models.RawNet2.run import _load_config

        assert _load_config()["nb_samp"] == 64600

    def test_oserror_raises_runtime_error(self):
        from Jabberjay.Models.RawNet2.run import _load_config

        with (
            patch(f"{_RAWNET}._CONFIG_PATH") as mock_path,
            pytest.raises(RuntimeError, match="Failed to load RawNet2 config"),
        ):
            mock_path.open.side_effect = OSError("missing")
            _load_config()

    def test_yaml_error_raises_runtime_error(self):
        import yaml

        from Jabberjay.Models.RawNet2.run import _load_config

        with (
            patch(f"{_RAWNET}.yaml.safe_load", side_effect=yaml.YAMLError("bad")),
            pytest.raises(RuntimeError, match="Failed to load RawNet2 config"),
        ):
            _load_config()

    def test_config_without_model_section_raises_runtime_error(self):
        from Jabberjay.Models.RawNet2.run import _load_config

        with (
            patch(f"{_RAWNET}.yaml.safe_load", return_value={"optimizer": "Adam"}),
            pytest.raises(RuntimeError, match="Failed to load RawNet2 config"),
        ):
            _load_config()


class TestRawNet2Predict:
    @staticmethod
    def _predict(y=None, sr=16000.0, log_probs=(-2.3, -0.1), state_dict=None):
        from Jabberjay.Models.RawNet2.run import predict

        mock_model = MagicMock(return_value=torch.tensor([log_probs]))
        with (
            patch(f"{_RAWNET}.RawNet", return_value=mock_model) as mock_rawnet,
            patch(
                f"{_RAWNET}.download_pretrained_model",
                return_value="/fake/model.pth",
            ) as mock_download,
            patch("torch.load", return_value={} if state_dict is None else state_dict),
        ):
            y = AUDIO[0] if y is None else y
            result = predict(y=y, sr=sr)
            predict(y=y, sr=sr)
        return result, mock_model, mock_rawnet, mock_download

    def test_sincconv_filters_buffer_is_not_persisted_in_state_dict(self):
        from Jabberjay.Models.RawNet2.model import SincConv

        sinc_conv = SincConv(device="cpu", out_channels=2, kernel_size=31)
        assert "filters" not in sinc_conv.state_dict()

    def test_ignores_legacy_sinc_filters_state_dict_key(self):
        state_dict = {"Sinc_conv.filters": torch.tensor([1.0]), "foo": torch.ones(1)}
        _, mock_model, *_ = self._predict(state_dict=state_dict)
        loaded = mock_model.load_state_dict.call_args.args[0]
        assert "Sinc_conv.filters" not in loaded
        assert loaded["foo"] is state_dict["foo"]

    def test_empty_audio_raises_value_error(self):
        with pytest.raises(ValueError, match="Input audio array is empty"):
            self._predict(y=np.zeros(0, dtype=np.float32))

    def test_returns_both_class_probabilities_from_log_softmax(self):
        import math

        result, *_ = self._predict(log_probs=(math.log(0.2), math.log(0.8)))
        scores = {r["label"]: r["score"] for r in result}
        assert scores == {"Spoof": pytest.approx(0.2), "Bonafide": pytest.approx(0.8)}

    @pytest.mark.parametrize(
        ("length", "sr"), [(1000, 16000.0), (70_000, 16000.0), (22050, 22050.0)]
    )
    def test_model_always_sees_exactly_nb_samp_samples(self, length, sr):
        _, mock_model, *_ = self._predict(y=np.ones(length, dtype=np.float32), sr=sr)
        assert mock_model.call_args.args[0].shape == (1, 64600)

    def test_model_is_loaded_once_across_repeated_calls(self):
        *_, mock_rawnet, mock_download = self._predict()
        mock_rawnet.assert_called_once()
        mock_download.assert_called_once()
