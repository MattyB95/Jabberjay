from huggingface_hub import hf_hub_download


def download_pretrained_model(repo_id="MattyB95/pre_trained_DF_RawNet2", filename="pre_trained_DF_RawNet2.pth"):
    # login(token="")
    return hf_hub_download(repo_id=repo_id, filename=filename)
