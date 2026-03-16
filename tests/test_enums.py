from Jabberjay import Dataset, Model, Visualisation


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
