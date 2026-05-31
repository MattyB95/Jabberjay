from huggingface_hub import hf_hub_download


def download_pretrained_model(repo_id: str, filename: str) -> str:
    """Download a file from a HuggingFace Hub repository and return its local path."""
    return hf_hub_download(repo_id=repo_id, filename=filename)
