from Jabberjay import DetectionResult, Model


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
