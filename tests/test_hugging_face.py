"""Test the pinned Hugging Face Hub download wrapper."""

from unittest.mock import patch

import pytest

from Jabberjay.Utilities.hugging_face import (
    REVISIONS,
    download_pretrained_model,
    revision_for,
)


def test_downloads_the_pinned_revision_and_returns_path():
    repo = "MattyB95/pre_trained_DF_RawNet2"
    with patch(
        "Jabberjay.Utilities.hugging_face.hf_hub_download",
        return_value="/cache/model.pth",
    ) as mock_dl:
        path = download_pretrained_model(repo, "model.pth")

    assert path == "/cache/model.pth"
    mock_dl.assert_called_once_with(
        repo_id=repo, filename="model.pth", revision=REVISIONS[repo]
    )


def test_unpinned_repo_is_refused():
    with (
        patch("Jabberjay.Utilities.hugging_face.hf_hub_download") as mock_dl,
        pytest.raises(KeyError, match=r"No pinned revision for .*owner/repo"),
    ):
        download_pretrained_model("owner/repo", "model.pth")
    mock_dl.assert_not_called()


@pytest.mark.parametrize("repo", sorted(REVISIONS))
def test_every_revision_is_a_full_commit_hash(repo):
    revision = revision_for(repo)
    assert len(revision) == 40
    assert set(revision) <= set("0123456789abcdef")
