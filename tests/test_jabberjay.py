from pathlib import Path

import numpy as np
import pytest

from Jabberjay.jabberjay import Jabberjay
from Jabberjay.Utilities.enum_handler import Dataset, Model, Visualisation

RES_DIR = Path(__file__).parent.parent / "res"


class TestLoad:
    def test_load_bonafide_returns_array_and_sr(self):
        jj = Jabberjay()
        y, sr = jj.load(str(RES_DIR / "bonafide" / "bonafide.flac"))
        assert isinstance(y, np.ndarray)
        assert len(y) > 0
        assert sr > 0

    def test_load_spoof_returns_array_and_sr(self):
        jj = Jabberjay()
        y, sr = jj.load(str(RES_DIR / "spoof" / "spoof.flac"))
        assert isinstance(y, np.ndarray)
        assert len(y) > 0
        assert sr > 0


class TestEnums:
    def test_model_members(self):
        assert {m.value for m in Model} == {"AST", "Classical", "RawNet2", "VIT"}

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


class TestDetectValidation:
    def setup_method(self):
        self.jj = Jabberjay()
        self.audio = (np.zeros(16000, dtype=np.float32), 16000.0)

    def test_vit_requires_visualisation(self):
        with pytest.raises(ValueError, match="Visualisation"):
            self.jj.detect(
                audio=self.audio,
                model=Model.VIT,
                visualisation=None,
                dataset=Dataset.VoxCelebSpoof,
            )

    def test_vit_requires_dataset(self):
        with pytest.raises(ValueError, match="Dataset"):
            self.jj.detect(
                audio=self.audio,
                model=Model.VIT,
                visualisation=Visualisation.ConstantQ,
                dataset=None,
            )

    def test_ast_requires_dataset(self):
        with pytest.raises(ValueError, match="Dataset"):
            self.jj.detect(
                audio=self.audio,
                model=Model.AST,
                visualisation=None,
                dataset=None,
            )
