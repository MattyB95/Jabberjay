"""Tests for the torch device selection helper."""

from unittest.mock import patch

from Jabberjay.Utilities.device import get_device


class TestGetDevice:
    def test_prefers_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert get_device() == "cuda"

    def test_falls_back_to_mps_when_no_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert get_device() == "mps"

    def test_falls_back_to_cpu_when_no_accelerator(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert get_device() == "cpu"
