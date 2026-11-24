"""Unit tests for audio input validation and fixed-length framing."""

import numpy as np
import pytest
import torch

from Jabberjay.Utilities.audio import fixed_length, prepare_audio, resample


class TestPrepareAudio:
    def test_float_mono_passes_through_as_float32(self):
        y, sr = prepare_audio(np.array([0.1, -0.2], dtype=np.float64), 16000)
        assert y.dtype == np.float32
        np.testing.assert_allclose(y, [0.1, -0.2], rtol=1e-6)
        assert sr == 16000.0
        assert isinstance(sr, float)

    def test_accepts_python_lists(self):
        y, _ = prepare_audio([0.0, 0.5], 8000)
        assert isinstance(y, np.ndarray)

    def test_accepts_numpy_scalar_sample_rate(self):
        _, sr = prepare_audio(np.ones(4), np.int64(22050))
        assert sr == 22050.0

    @pytest.mark.parametrize(
        ("dtype", "raw", "expected"),
        [
            (np.int16, [0, 16384, -32768], [0.0, 0.5, -1.0]),
            (np.int32, [0, 2**30, -(2**31)], [0.0, 0.5, -1.0]),
            (np.int8, [0, 64, -128], [0.0, 0.5, -1.0]),
            (np.uint8, [128, 192, 0], [0.0, 0.5, -1.0]),
        ],
    )
    def test_integer_pcm_is_scaled_to_unit_range(self, dtype, raw, expected):
        y, _ = prepare_audio(np.array(raw, dtype=dtype), 16000)
        np.testing.assert_allclose(y, expected)

    def test_channels_first_is_downmixed(self):
        stereo = np.stack([np.full(100, 0.2), np.full(100, 0.4)])  # (2, 100)
        y, _ = prepare_audio(stereo, 16000)
        assert y.shape == (100,)
        np.testing.assert_allclose(y, 0.3)

    def test_integer_stereo_is_scaled_then_downmixed(self):
        stereo = np.array([[16384, 16384, 16384], [0, 0, 0]], dtype=np.int16)
        y, _ = prepare_audio(stereo, 16000)
        np.testing.assert_allclose(y, 0.25)

    def test_channels_last_is_downmixed(self):
        stereo = np.stack([np.full(100, 0.2), np.full(100, 0.4)], axis=1)  # (100, 2)
        y, _ = prepare_audio(stereo, 16000)
        assert y.shape == (100,)
        np.testing.assert_allclose(y, 0.3)

    @pytest.mark.parametrize("sr", [0, -1, float("nan"), float("inf")])
    def test_invalid_sample_rate_rejected(self, sr):
        with pytest.raises(ValueError, match="sample rate"):
            prepare_audio(np.ones(4), sr)

    @pytest.mark.parametrize("sr", ["16000", None, True])
    def test_non_numeric_sample_rate_rejected(self, sr):
        with pytest.raises(ValueError, match="Sample rate must be a number"):
            prepare_audio(np.ones(4), sr)

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            prepare_audio(np.array([], dtype=np.float32), 16000)

    def test_empty_multichannel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            prepare_audio(np.zeros((2, 0)), 16000)

    def test_three_dimensional_rejected(self):
        with pytest.raises(ValueError, match=r"1-D .* or 2-D"):
            prepare_audio(np.zeros((2, 2, 2)), 16000)

    def test_scalar_rejected(self):
        with pytest.raises(ValueError, match=r"1-D .* or 2-D"):
            prepare_audio(np.float32(0.5), 16000)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(ValueError, match="NaN or infinite"):
            prepare_audio(np.array([0.0, bad]), 16000)

    def test_boolean_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            prepare_audio(np.array([True, False]), 16000)

    def test_strings_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            prepare_audio(np.array(["a", "b"]), 16000)

    def test_complex_rejected(self):
        with pytest.raises(ValueError, match="real-valued"):
            prepare_audio(np.array([1 + 1j, 0j]), 16000)

    def test_silence_is_accepted_with_a_warning(self):
        from loguru import logger

        messages: list[str] = []
        logger.enable("Jabberjay")
        handler = logger.add(messages.append, level="WARNING")
        try:
            y, _ = prepare_audio(np.zeros(100), 16000)
        finally:
            logger.remove(handler)
            logger.disable("Jabberjay")
        assert not y.any()
        assert any("silence" in m for m in messages)


class TestResample:
    def test_same_rate_is_a_float_tensor_copy(self):
        out = resample(np.ones(10, dtype=np.float64), 16000, 16000)
        assert out.dtype == torch.float32
        assert out.shape == (10,)

    def test_changes_length_by_rate_ratio(self):
        out = resample(np.ones(8000, dtype=np.float32), 8000, 16000)
        assert out.shape == (16000,)


class TestFixedLength:
    def test_trims_long_audio(self):
        out = fixed_length(torch.arange(10.0), 4)
        assert out.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_loops_short_audio(self):
        out = fixed_length(torch.tensor([1.0, 2.0, 3.0]), 7)
        assert out.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]

    def test_exact_length_unchanged(self):
        out = fixed_length(torch.ones(5), 5)
        assert out.shape == (5,)

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            fixed_length(torch.zeros(0), 5)
