from huggingface_hub import hf_hub_download


def download_pretrained_model(
    repo_id: str, filename: str, revision: str | None = None
) -> str:
    """Download a file from a HuggingFace Hub repository and return its local path.

    Pass ``revision`` (a commit hash) to pin the exact file version rather
    than following the repository's default branch.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
