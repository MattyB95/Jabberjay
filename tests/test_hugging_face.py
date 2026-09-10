"""Test the thin Hugging Face Hub download wrapper."""

from unittest.mock import patch

from Jabberjay.Utilities.hugging_face import download_pretrained_model


def test_forwards_repo_and_filename_and_returns_path():
    with patch(
        "Jabberjay.Utilities.hugging_face.hf_hub_download",
        return_value="/cache/model.pth",
    ) as mock_dl:
        path = download_pretrained_model("owner/repo", "model.pth")

    assert path == "/cache/model.pth"
    mock_dl.assert_called_once_with(repo_id="owner/repo", filename="model.pth")
