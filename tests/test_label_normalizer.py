import pytest

from Jabberjay.Utilities.label_normalizer import normalize_label, normalize_pipeline_scores


class TestLabelNormalizer:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("real", "Bonafide"),
            ("REAL", "Bonafide"),
            ("bonafide", "Bonafide"),
            ("Bonafide", "Bonafide"),
            ("genuine", "Bonafide"),
            ("human", "Bonafide"),
            ("bona-fide", "Bonafide"),
            ("label_0", "Bonafide"),
            ("LABEL_0", "Bonafide"),
            ("0", "Bonafide"),
            ("fake", "Spoof"),
            ("FAKE", "Spoof"),
            ("spoof", "Spoof"),
            ("Spoof", "Spoof"),
            ("synthetic", "Spoof"),
            ("deepfake", "Spoof"),
            ("label_1", "Spoof"),
            ("LABEL_1", "Spoof"),
            ("1", "Spoof"),
        ],
    )
    def test_known_labels(self, raw, expected):
        assert normalize_label(raw) == expected

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Cannot normalise"):
            normalize_label("unknown_label_xyz")

    def test_digit_one_no_false_positive(self):
        # "1" should only match exactly — not any string containing "1"
        with pytest.raises(ValueError):
            normalize_label("Speech_Quality_1_Normal")

    def test_digit_zero_no_false_positive(self):
        # "0" should only match exactly — not strings like "label_0_variant"
        with pytest.raises(ValueError):
            normalize_label("label_0_variant")


class TestNormalizePipelineScores:
    def test_converts_raw_labels_and_scores(self):
        raw = [{"label": "bonafide", "score": 0.9}, {"label": "fake", "score": 0.1}]
        result = normalize_pipeline_scores(raw)
        assert result[0]["label"] == "Bonafide"
        assert result[0]["score"] == 0.9
        assert result[1]["label"] == "Spoof"
        assert result[1]["score"] == 0.1

    def test_returns_correct_length(self):
        raw = [{"label": "real", "score": 0.7}, {"label": "spoof", "score": 0.3}]
        assert len(normalize_pipeline_scores(raw)) == 2

    def test_empty_input_returns_empty(self):
        assert normalize_pipeline_scores([]) == []
