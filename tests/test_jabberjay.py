from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from Jabberjay import Dataset, DetectionResult, Jabberjay, Model, Visualisation
from Jabberjay.jabberjay import _MODELS, main
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
        y, _sr = jj.load(str(RES_DIR / "spoof" / "spoof.flac"))
        assert isinstance(y, np.ndarray)
        assert len(y) > 0

    def test_load_missing_file_raises(self):
        jj = Jabberjay()
        with pytest.raises(FileNotFoundError, match="not found"):
            jj.load("nonexistent_file.wav")


class TestListMethods:
    def test_list_models_returns_list(self):
        models = Jabberjay.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, Model) for m in models)
        assert len(models) == len(Model)

    def test_list_datasets_returns_list(self):
        datasets = Jabberjay.list_datasets()
        assert isinstance(datasets, list)
        assert all(isinstance(d, Dataset) for d in datasets)
        assert len(datasets) == len(Dataset)

    def test_list_visualisations_returns_list(self):
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

        # loguru exposes no public handler count, so read its private core
        before = len(_logger._core.handlers)  # ty: ignore[missing-attribute]
        Jabberjay.enable_logging()
        Jabberjay.enable_logging()
        Jabberjay.enable_logging()
        after = len(_logger._core.handlers)  # ty: ignore[missing-attribute]
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
        self.scores: list[PredictionScore] = [
            {"label": "Bonafide", "score": 0.9},
            {"label": "Spoof", "score": 0.1},
        ]

    def test_detect_accepts_str_path(self):
        with patch(
            "Jabberjay.Models.Classical.run.predict", return_value=self.scores
        ) as mock_predict:
            result = self.jj.detect(str(self.audio_path), model=Model.Classical)
        assert isinstance(result, DetectionResult)
        # detect() loaded the file and passed real samples through
        kwargs = mock_predict.call_args.kwargs
        assert isinstance(kwargs["y"], np.ndarray) and len(kwargs["y"]) > 0
        assert kwargs["sr"] > 0

    def test_detect_accepts_path_object(self):
        with patch("Jabberjay.Models.Classical.run.predict", return_value=self.scores):
            result = self.jj.detect(self.audio_path, model=Model.Classical)
        assert isinstance(result, DetectionResult)

    def test_path_input_keeps_native_sample_rate(self):
        # bonafide.flac is 16 kHz; load() must no longer resample to 22.05 kHz
        with patch(
            "Jabberjay.Models.HuBERT.run.predict", return_value=self.scores
        ) as mock_predict:
            self.jj.detect(self.audio_path, model=Model.HuBERT)
        assert mock_predict.call_args.kwargs["sr"] == 16_000


class TestDetectAudioValidation:
    """detect() normalises or rejects pre-loaded audio before any model runs."""

    def setup_method(self):
        self.jj = Jabberjay()
        self.scores: list[PredictionScore] = [
            {"label": "Bonafide", "score": 0.9},
            {"label": "Spoof", "score": 0.1},
        ]

    def _passed_samples(self, audio) -> np.ndarray:
        with patch(
            "Jabberjay.Models.HuBERT.run.predict", return_value=self.scores
        ) as mock_predict:
            self.jj.detect(audio, model=Model.HuBERT)
        return mock_predict.call_args.kwargs["y"]

    def test_nan_audio_is_rejected(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            self.jj.detect((np.full(1000, np.nan), 16000), model=Model.HuBERT)

    def test_non_tuple_audio_is_rejected(self):
        with pytest.raises(ValueError, match="file path or a"):
            # Deliberately the wrong type: a bare array, not (samples, sr)
            self.jj.detect(cast(Any, np.zeros(1000)), model=Model.HuBERT)

    def test_int16_audio_is_scaled_to_float(self):
        y = self._passed_samples((np.array([0, 16384, -32768], dtype=np.int16), 16000))
        assert y.dtype == np.float32
        np.testing.assert_allclose(y, [0.0, 0.5, -1.0])

    def test_stereo_audio_is_downmixed(self):
        stereo = np.stack([np.ones(1000), np.zeros(1000)])
        y = self._passed_samples((stereo, 16000))
        assert y.shape == (1000,)
        np.testing.assert_allclose(y, 0.5)

    def test_unrecognised_names_raise_value_error(self):
        with pytest.raises(ValueError, match="choose from"):
            self.jj.detect((np.ones(1000), 16000), model="NotAModel")


class TestClearCache:
    def test_clears_every_loader_and_releases_memory(self):
        with (
            patch("Jabberjay.jabberjay.model_cache.clear_all") as mock_clear,
            patch("Jabberjay.jabberjay.model_cache.release_memory") as mock_release,
        ):
            Jabberjay.clear_cache()
        mock_clear.assert_called_once_with()
        mock_release.assert_called_once_with()


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
        with (
            patch("librosa.load", side_effect=RuntimeError("codec error")),
            pytest.raises(ValueError, match="Failed to load"),
        ):
            jj.load(RES_DIR / "bonafide" / "bonafide.flac")


class TestDetectDispatch:
    def setup_method(self):
        self.jj = Jabberjay()
        self.audio = (np.ones(16000, dtype=np.float32), 16000.0)
        self.scores: list[PredictionScore] = [
            {"label": "Spoof", "score": 0.1},
            {"label": "Bonafide", "score": 0.9},
        ]

    @pytest.mark.parametrize("model", list(Model))
    def test_every_model_dispatches_to_its_module(self, model):
        spec = _MODELS[model]
        with patch(f"{spec.module}.predict", return_value=self.scores) as mock_predict:
            result = self.jj.detect(self.audio, model=model)
        assert isinstance(result, DetectionResult)
        assert result.model == model
        assert result.label == "Bonafide"
        assert result.is_bonafide is True
        assert result.confidence == 0.9
        assert [s["label"] for s in result.scores] == ["Bonafide", "Spoof"]
        kwargs = mock_predict.call_args.kwargs
        assert ("dataset" in kwargs) == spec.needs_dataset
        assert ("visualisation" in kwargs) == spec.needs_visualisation

    def test_every_model_has_a_spec(self):
        assert set(_MODELS) == set(Model)

    def test_vit_receives_dataset_and_visualisation(self):
        with patch(
            "Jabberjay.Models.Transformer.VIT.run.predict", return_value=self.scores
        ) as mock_predict:
            self.jj.detect(
                self.audio, model="vit", visualisation="mfcc", dataset="asvspoof5"
            )
        kwargs = mock_predict.call_args.kwargs
        assert kwargs["visualisation"] is Visualisation.MFCC
        assert kwargs["dataset"] is Dataset.ASVspoof5

    def test_empty_predictions_raise(self):
        with (
            patch("Jabberjay.Models.HuBERT.run.predict", return_value=[]),
            pytest.raises(RuntimeError, match="no predictions"),
        ):
            self.jj.detect(self.audio, model=Model.HuBERT)


class TestCLI:
    _BONAFIDE = DetectionResult(
        label="Bonafide", is_bonafide=True, confidence=0.95, model=Model.VIT
    )
    _SPOOF = DetectionResult(
        label="Spoof", is_bonafide=False, confidence=0.8, model=Model.Classical
    )

    def test_main_prints_result(self, capsys):
        with (
            patch("sys.argv", ["jabberjay", "audio.flac"]),
            patch.object(Jabberjay, "detect", return_value=self._BONAFIDE),
        ):
            main()
        assert "Bonafide" in capsys.readouterr().out

    def test_main_classical_model(self, capsys):
        with (
            patch("sys.argv", ["jabberjay", "audio.flac", "-m", "Classical"]),
            patch.object(Jabberjay, "detect", return_value=self._SPOOF),
        ):
            main()
        assert "Spoof" in capsys.readouterr().out

    def test_main_prints_result_without_crashing_on_non_utf8_stdout(self):
        """Regression test: DetectionResult.__str__ embeds emoji (✔️/❌) that
        used to raise UnicodeEncodeError when stdout uses a non-UTF-8 encoding
        (e.g. Windows cp1252, the default outside Windows Terminal/UTF-8 mode)."""
        import io

        buffer = io.BytesIO()
        fake_stdout = io.TextIOWrapper(buffer, encoding="cp1252")
        fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with (
            patch("sys.argv", ["jabberjay", "audio.flac"]),
            patch.object(Jabberjay, "detect", return_value=self._BONAFIDE),
            patch("sys.stdout", fake_stdout),
            patch("sys.stderr", fake_stderr),
        ):
            main()

        fake_stdout.flush()
        assert "Bonafide" in buffer.getvalue().decode("utf-8")

    def test_main_verbose_flag(self, capsys):
        with (
            patch("sys.argv", ["jabberjay", "audio.flac", "-v"]),
            patch.object(Jabberjay, "detect", return_value=self._BONAFIDE),
        ):
            main()
        assert "Bonafide" in capsys.readouterr().out

    def test_main_reports_missing_file_without_traceback(self, capsys):
        # No mock on detect() — exercises the real load() -> main() error path.
        with (
            patch("sys.argv", ["jabberjay", "/no/such/file.flac"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
        assert "Error: Audio file not found" in capsys.readouterr().err

    def test_main_reports_value_error_without_traceback(self, capsys):
        with (
            patch("sys.argv", ["jabberjay", "audio.flac"]),
            patch.object(
                Jabberjay, "detect", side_effect=ValueError("Audio array is empty")
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
        assert "Error: Audio array is empty" in capsys.readouterr().err

    def test_main_passes_dataset_and_visualisation(self):
        with (
            patch(
                "sys.argv",
                ["jabberjay", "audio.flac", "-d", "ASVspoof2019", "-vis", "MFCC"],
            ),
            patch.object(
                Jabberjay, "detect", return_value=self._BONAFIDE
            ) as mock_detect,
        ):
            main()
        mock_detect.assert_called_once()
        _, kwargs = mock_detect.call_args
        assert kwargs["dataset"] == Dataset.ASVspoof2019
        assert kwargs["visualisation"] == Visualisation.MFCC
