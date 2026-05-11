from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Jabberjay import Dataset, DetectionResult, Jabberjay, Model, Visualisation
from Jabberjay.jabberjay import main
from Jabberjay.Utilities.types import PredictionScore

RES_DIR = Path(__file__).parent.parent / "res"


class TestLoad:
    def test_load_returns_array_and_sr(self):
        jj = Jabberjay()
        y, sr = jj.load(RES_DIR / "bonafide" / "bonafide.flac")
        assert isinstance(y, np.ndarray)
        assert len(y) > 0
        assert sr > 0

    def test_load_accepts_string_path(self):
        jj = Jabberjay()
        y, sr = jj.load(str(RES_DIR / "spoof" / "spoof.flac"))
        assert isinstance(y, np.ndarray)
        assert len(y) > 0

    def test_load_missing_file_raises(self):
        jj = Jabberjay()
        with patch("librosa.load", side_effect=FileNotFoundError("No such file")):
            with pytest.raises(FileNotFoundError, match="not found"):
                jj.load("nonexistent_file.wav")


class TestListMethods:
    def test_list_models_returns_list(self, capsys):
        models = Jabberjay.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, Model) for m in models)
        assert len(models) == len(Model)
        assert "Models:" in capsys.readouterr().out

    def test_list_datasets_returns_list(self, capsys):
        datasets = Jabberjay.list_datasets()
        assert isinstance(datasets, list)
        assert all(isinstance(d, Dataset) for d in datasets)
        assert len(datasets) == len(Dataset)

    def test_list_visualisations_returns_list(self, capsys):
        vis = Jabberjay.list_visualisations()
        assert isinstance(vis, list)
        assert all(isinstance(v, Visualisation) for v in vis)
        assert len(vis) == len(Visualisation)


class TestDetectDefaults:
    """Verify that detect() works with minimal arguments using its built-in defaults."""

    def setup_method(self):
        self.jj = Jabberjay()
        self.audio = (np.zeros(16000, dtype=np.float32), 16000.0)

    def test_vit_default_visualisation_is_constantq(self):
        import inspect

        sig = inspect.signature(Jabberjay.detect)
        assert sig.parameters["visualisation"].default == Visualisation.ConstantQ

    def test_vit_default_dataset_is_voxcelebspoof(self):
        import inspect

        sig = inspect.signature(Jabberjay.detect)
        assert sig.parameters["dataset"].default == Dataset.VoxCelebSpoof

    def test_empty_audio_still_raises_before_model_dispatch(self):
        """Default params must not mask the empty-audio guard."""
        empty = (np.array([], dtype=np.float32), 16000.0)
        with pytest.raises(ValueError, match="empty"):
            self.jj.detect(empty)


class TestEnableLogging:
    def setup_method(self, method=None):
        # Reset the guard before each test so tests are independent.
        Jabberjay._logging_enabled = False

    def test_enable_logging_sets_flag(self):
        assert Jabberjay._logging_enabled is False
        Jabberjay.enable_logging()
        assert Jabberjay._logging_enabled is True

    def test_enable_logging_is_idempotent(self, capsys):
        Jabberjay.enable_logging()
        Jabberjay.enable_logging()
        Jabberjay.enable_logging()
        # Flag should still be True and no exception raised
        assert Jabberjay._logging_enabled is True

    def test_enable_logging_only_adds_one_handler(self):
        from loguru import logger as _logger

        before = len(_logger._core.handlers)
        Jabberjay.enable_logging()
        Jabberjay.enable_logging()
        Jabberjay.enable_logging()
        after = len(_logger._core.handlers)
        assert after - before == 1


class TestDetectValidation:
    def setup_method(self):
        self.jj = Jabberjay()
        self.audio = (np.zeros(16000, dtype=np.float32), 16000.0)

    def test_vit_requires_visualisation(self):
        with pytest.raises(ValueError, match="Visualisation"):
            self.jj.detect(
                self.audio,
                model=Model.VIT,
                visualisation=None,
                dataset=Dataset.VoxCelebSpoof,
            )

    def test_vit_requires_dataset(self):
        with pytest.raises(ValueError, match="Dataset"):
            self.jj.detect(
                self.audio,
                model=Model.VIT,
                visualisation=Visualisation.ConstantQ,
                dataset=None,
            )

    def test_ast_requires_dataset(self):
        with pytest.raises(ValueError, match="Dataset"):
            self.jj.detect(self.audio, model=Model.AST, dataset=None)

    def test_empty_audio_raises(self):
        empty = (np.array([], dtype=np.float32), 16000.0)
        with pytest.raises(ValueError, match="empty"):
            self.jj.detect(empty, model=Model.Classical)

    def test_zero_sample_rate_raises(self):
        bad_audio = (np.zeros(16000, dtype=np.float32), 0)
        with pytest.raises(ValueError, match="sample rate"):
            self.jj.detect(bad_audio, model=Model.Classical)

    def test_negative_sample_rate_raises(self):
        bad_audio = (np.zeros(16000, dtype=np.float32), -1.0)
        with pytest.raises(ValueError, match="sample rate"):
            self.jj.detect(bad_audio, model=Model.Classical)


class TestDetectPathInput:
    def setup_method(self):
        self.jj = Jabberjay()
        self.audio_path = RES_DIR / "bonafide" / "bonafide.flac"

    def test_detect_accepts_str_path(self):
        result = self.jj.detect(str(self.audio_path), model=Model.Classical)
        assert isinstance(result, DetectionResult)

    def test_detect_accepts_path_object(self):
        result = self.jj.detect(self.audio_path, model=Model.Classical)
        assert isinstance(result, DetectionResult)


class TestStringCoercion:
    def setup_method(self):
        self.jj = Jabberjay()
        self.audio = (np.zeros(16000, dtype=np.float32), 16000.0)

    def test_invalid_model_string_raises(self):
        with pytest.raises(KeyError):
            self.jj.detect(self.audio, model="NotAModel")

    def test_invalid_dataset_string_raises(self):
        with pytest.raises(KeyError):
            self.jj.detect(
                self.audio,
                model=Model.AST,
                dataset="NotADataset",
            )

    def test_invalid_visualisation_string_raises(self):
        with pytest.raises(KeyError):
            self.jj.detect(
                self.audio,
                model=Model.VIT,
                visualisation="NotAVis",
                dataset=Dataset.VoxCelebSpoof,
            )


class TestResultFromScores:
    def test_top_score_wins(self):
        scores: list[PredictionScore] = [
            {"label": "Spoof", "score": 0.7},
            {"label": "Bonafide", "score": 0.3},
        ]
        result = Jabberjay._result_from_scores(scores, Model.VIT)
        assert result.label == "Spoof"
        assert result.is_bonafide is False
        assert result.confidence == 0.7
        assert result.model == Model.VIT

    def test_bonafide_sets_is_bonafide_true(self):
        scores: list[PredictionScore] = [
            {"label": "Bonafide", "score": 0.95},
            {"label": "Spoof", "score": 0.05},
        ]
        result = Jabberjay._result_from_scores(scores, Model.Classical)
        assert result.is_bonafide is True

    def test_scores_sorted_descending(self):
        scores: list[PredictionScore] = [
            {"label": "Bonafide", "score": 0.1},
            {"label": "Spoof", "score": 0.9},
        ]
        result = Jabberjay._result_from_scores(scores, Model.AST)
        assert result.scores[0]["score"] == 0.9
        assert result.scores[1]["score"] == 0.1

    def test_full_scores_attached_to_result(self):
        scores: list[PredictionScore] = [
            {"label": "Bonafide", "score": 0.6},
            {"label": "Spoof", "score": 0.4},
        ]
        result = Jabberjay._result_from_scores(scores, Model.HuBERT)
        assert len(result.scores) == 2


class TestLoadErrors:
    def test_load_raises_value_error_on_corrupt_file(self):
        jj = Jabberjay()
        with patch("librosa.load", side_effect=Exception("codec error")):
            with pytest.raises(ValueError, match="Failed to load"):
                jj.load("corrupt.wav")


class TestDetectHandlers:
    def setup_method(self):
        self.jj = Jabberjay()
        self.audio = (np.zeros(16000, dtype=np.float32), 16000.0)
        self.scores: list[PredictionScore] = [
            {"label": "Bonafide", "score": 0.9},
            {"label": "Spoof", "score": 0.1},
        ]

    def test_ast_handler(self):
        with patch(
            "Jabberjay.Models.Transformer.AST.run.predict", return_value=self.scores
        ):
            result = self.jj.detect(
                self.audio, model=Model.AST, dataset=Dataset.VoxCelebSpoof
            )
        assert isinstance(result, DetectionResult)
        assert result.model == Model.AST

    def test_hubert_handler(self):
        with patch("Jabberjay.Models.HuBERT.run.predict", return_value=self.scores):
            result = self.jj.detect(self.audio, model=Model.HuBERT)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.HuBERT

    def test_rawnet2_handler_bonafide(self):
        mock_pred = MagicMock()
        mock_pred.item.return_value = True
        with patch(
            "Jabberjay.Models.RawNet2.run.predict", return_value=(mock_pred, 0.85)
        ):
            result = self.jj.detect(self.audio, model=Model.RawNet2)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.RawNet2
        assert result.is_bonafide is True

    def test_rawnet2_handler_spoof(self):
        mock_pred = MagicMock()
        mock_pred.item.return_value = False
        with patch(
            "Jabberjay.Models.RawNet2.run.predict", return_value=(mock_pred, 0.75)
        ):
            result = self.jj.detect(self.audio, model=Model.RawNet2)
        assert result.is_bonafide is False

    def test_vit_handler(self):
        mock_module = MagicMock()
        mock_module.predict.return_value = self.scores
        with patch(
            "Jabberjay.jabberjay.importlib.import_module", return_value=mock_module
        ):
            result = self.jj.detect(
                self.audio,
                model=Model.VIT,
                visualisation=Visualisation.ConstantQ,
                dataset=Dataset.VoxCelebSpoof,
            )
        assert isinstance(result, DetectionResult)
        assert result.model == Model.VIT

    def test_vit_invalid_module_raises(self):
        with patch(
            "Jabberjay.jabberjay.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ):
            with pytest.raises(ValueError, match="No VIT module"):
                self.jj._vit_handler(
                    self.audio, Visualisation.ConstantQ, Dataset.VoxCelebSpoof
                )

    def test_spectra0_handler(self):
        with patch("Jabberjay.Models.Spectra0.run.predict", return_value=self.scores):
            result = self.jj.detect(self.audio, model=Model.Spectra0)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.Spectra0

    def test_spectra_aasist_handler(self):
        with patch(
            "Jabberjay.Models.SpectraAASIST.run.predict", return_value=self.scores
        ):
            result = self.jj.detect(self.audio, model=Model.SpectraAASIST)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.SpectraAASIST

    def test_spectra_aasist3_handler(self):
        with patch(
            "Jabberjay.Models.SpectraAASIST3.run.predict", return_value=self.scores
        ):
            result = self.jj.detect(self.audio, model=Model.SpectraAASIST3)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.SpectraAASIST3

    def test_wav2vec2_handler(self):
        with patch("Jabberjay.Models.Wav2Vec2.run.predict", return_value=self.scores):
            result = self.jj.detect(self.audio, model=Model.Wav2Vec2)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.Wav2Vec2

    def test_wavlm_handler(self):
        with patch("Jabberjay.Models.WavLM.run.predict", return_value=self.scores):
            result = self.jj.detect(self.audio, model=Model.WavLM)
        assert isinstance(result, DetectionResult)
        assert result.model == Model.WavLM


class TestCLI:
    _BONAFIDE = DetectionResult(
        label="Bonafide", is_bonafide=True, confidence=0.95, model=Model.VIT
    )
    _SPOOF = DetectionResult(
        label="Spoof", is_bonafide=False, confidence=0.8, model=Model.Classical
    )

    def test_main_prints_result(self, capsys):
        with patch("sys.argv", ["jabberjay", "audio.flac"]):
            with patch.object(Jabberjay, "detect", return_value=self._BONAFIDE):
                main()
        assert "Bonafide" in capsys.readouterr().out

    def test_main_classical_model(self, capsys):
        with patch("sys.argv", ["jabberjay", "audio.flac", "-m", "Classical"]):
            with patch.object(Jabberjay, "detect", return_value=self._SPOOF):
                main()
        assert "Spoof" in capsys.readouterr().out

    def test_main_verbose_flag(self, capsys):
        with patch("sys.argv", ["jabberjay", "audio.flac", "-v"]):
            with patch.object(Jabberjay, "detect", return_value=self._BONAFIDE):
                main()
        assert "Bonafide" in capsys.readouterr().out

    def test_main_passes_dataset_and_visualisation(self):
        with patch(
            "sys.argv",
            ["jabberjay", "audio.flac", "-d", "ASVspoof2019", "-vis", "MFCC"],
        ):
            with patch.object(
                Jabberjay, "detect", return_value=self._BONAFIDE
            ) as mock_detect:
                main()
        mock_detect.assert_called_once()
        _, kwargs = mock_detect.call_args
        assert kwargs["dataset"] == Dataset.ASVspoof2019
        assert kwargs["visualisation"] == Visualisation.MFCC
