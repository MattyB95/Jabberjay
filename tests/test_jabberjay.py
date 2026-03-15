from pathlib import Path

import numpy as np
import pytest

from Jabberjay import Dataset, DetectionResult, Jabberjay, Model, Visualisation

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


class TestEnums:
    def test_model_members(self):
        assert {m.value for m in Model} == {
            "AST",
            "Classical",
            "HuBERT",
            "RawNet2",
            "VIT",
            "Wav2Vec2",
            "WavLM",
        }

    def test_dataset_members(self):
        assert {d.value for d in Dataset} == {
            "ASVspoof2019",
            "ASVspoof5",
            "VoxCelebSpoof",
        }

    def test_visualisation_members(self):
        assert {v.value for v in Visualisation} == {
            "ConstantQ",
            "MelSpectrogram",
            "MFCC",
        }


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


class TestDetectionResult:
    def test_str_bonafide(self):
        r = DetectionResult(
            label="Bonafide",
            is_bonafide=True,
            confidence=0.95,
            model=Model.VIT,
        )
        assert "Bonafide" in str(r)
        assert "95.0%" in str(r)
        assert "VIT" in str(r)

    def test_str_spoof(self):
        r = DetectionResult(
            label="Spoof",
            is_bonafide=False,
            confidence=0.80,
            model=Model.Classical,
        )
        assert "Spoof" in str(r)
        assert "80.0%" in str(r)

    def test_scores_defaults_to_none(self):
        r = DetectionResult(
            label="Bonafide", is_bonafide=True, confidence=1.0, model=Model.RawNet2
        )
        assert r.scores is None


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
