from huggingface_hub import hf_hub_download

# Every Hugging Face repo Jabberjay loads from, pinned to the commit that was
# reviewed and tested. Following a repo's default branch instead would let an
# upstream push silently change results (or, for pickled checkpoints such as
# the Classical model's .joblib, run arbitrary code), so a repo missing from
# this table is refused rather than loaded unpinned. To move to a newer
# upstream commit, update the hash here and re-run `just test-integration`.
REVISIONS: dict[str, str] = {
    # Classical (pickled scikit-learn model — pinning matters most here)
    "MattyB95/Jabberjay_Classical_Machine_Learning_Models": (
        "4d7e4aaaf0bdecb219ed81de4526b2fe68494121"
    ),
    # RawNet2
    "MattyB95/pre_trained_DF_RawNet2": "7853bd7b723aeed2ada7ce92684c4c5a3c75627f",
    # DF Arena (pickled .bin, always loaded with weights_only=True)
    "Speech-Arena-2025/DF_Arena_1B_V_1": "fb6ce85de12c2c5a509d89114adaf827dd75f49f",
    "facebook/wav2vec2-xls-r-1b": "35eaea9a0ed0f97592277d79208e40ab8917d1e3",
    # Spectra family, plus the base config their encoder is built from
    "lab260/spectra_0": "b3fb6d5c9d8ce069996f100cbe10117fd74165f3",
    "lab260/Spectra-AASIST": "eb65c2662d9e646d72557b3f4bdd08b000068c7f",
    "lab260/Spectra-AASIST3": "bc0ded888080ddad493177bb53aa6f5b95219d7c",
    "facebook/wav2vec2-xls-r-300m": "1a640f32ac3e39899438a2931f9924c02f080a54",
    # Third-party transformers pipelines
    "abhishtagatya/hubert-base-960h-itw-deepfake": (
        "965d3063b43e05849f677740ecda3d4cd6b18131"
    ),
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification": (
        "f7050b586236dc910d1157f430def2d0647b02b4"
    ),
    "DavidCombei/wavLM-base-Deepfake_V2": "03396be9160ddb0dd38f1718428b595707577641",
    # AST — one per training dataset
    "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection": (
        "afd7436a7205ebd7c9ae22d46bbbc6f4f04f85b5"
    ),
    "MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection": (
        "f8187e7a7e8885df82c855ae833d4cf9d6f0d816"
    ),
    "MattyB95/AST-VoxCelebSpoof-Synthetic-Voice-Detection": (
        "507e45b07e5bfe2fa505800070f9d2df361b1b50"
    ),
    # VIT — one per (training dataset x visualisation)
    "MattyB95/VIT-ASVspoof2019-ConstantQ-Synthetic-Voice-Detection": (
        "7f9a3195fd0fbefdf5f47b101f929bfde46f5049"
    ),
    "MattyB95/VIT-ASVspoof2019-MFCC-Synthetic-Voice-Detection": (
        "16d073132766b064d50c2f4eae689a7a543516e9"
    ),
    "MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection": (
        "503839eb5dd09d11ee3eb782ea9d669f00f13d8f"
    ),
    "MattyB95/VIT-ASVspoof5-ConstantQ-Synthetic-Voice-Detection": (
        "3333a085893e0a4747a46d562ce6c1ecea0e775f"
    ),
    "MattyB95/VIT-ASVspoof5-MFCC-Synthetic-Voice-Detection": (
        "701d4c53bd047167f9883fdeb7350568b6aebab0"
    ),
    "MattyB95/VIT-ASVspoof5-Mel_Spectrogram-Synthetic-Voice-Detection": (
        "8902e6e897ff86fb101bf9438985a8773fa38425"
    ),
    "MattyB95/VIT-VoxCelebSpoof-ConstantQ-Synthetic-Voice-Detection": (
        "ce8f2d32645f5adf16b0801d929fd3ee050f00c0"
    ),
    "MattyB95/VIT-VoxCelebSpoof-MFCC-Synthetic-Voice-Detection": (
        "703fabe592cfbc019682035f7907b5dd397b8e72"
    ),
    "MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection": (
        "31323edb1e15868d1426338126a61374e0c398e6"
    ),
}


def revision_for(repo_id: str) -> str:
    """Return the pinned commit hash for ``repo_id``.

    Raises:
        KeyError: If the repo has no pinned revision in ``REVISIONS``.
    """
    try:
        return REVISIONS[repo_id]
    except KeyError:
        raise KeyError(
            f"No pinned revision for Hugging Face repo {repo_id!r}; "
            "add one to Jabberjay.Utilities.hugging_face.REVISIONS"
        ) from None


def download_pretrained_model(repo_id: str, filename: str) -> str:
    """Download a file from a pinned Hugging Face Hub revision; return its path."""
    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision_for(repo_id)
    )
