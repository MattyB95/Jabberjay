"""Opt-in integration tests against real model weights.

Deselected by default; run with ``pytest -m integration``. Each test also
skips unless its checkpoint is already in the local HuggingFace cache, so
an opt-in run never triggers a multi-GB download. Set
``JABBERJAY_REQUIRE_WEIGHTS=1`` (as CI does) to fail instead of skipping.

Every model / dataset / visualisation combination is run on both reference
clips and its bonafide score compared against the value recorded when its
pinned revision was reviewed. The unit tests mock every model, so this is
what catches a dependency upgrade (torch, transformers, librosa, ...) or a
refactor that silently changes real predictions.
"""

import os
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

from Jabberjay import Dataset, Jabberjay, Model, Visualisation
from Jabberjay.Utilities.hugging_face import revision_for

RES_DIR = Path(__file__).parent.parent / "res"
CLIPS = {
    "bonafide": RES_DIR / "bonafide/bonafide.flac",
    "spoof": RES_DIR / "spoof/spoof.flac",
}

# Bonafide-class score per clip, recorded on Apple Silicon (MPS). CPU/CUDA
# kernels differ in the last few digits, hence the tolerance. A model being
# confidently wrong on a reference clip (e.g. HuBERT on spoof.flac) is
# recorded as-is: this guards against regressions, not model accuracy.
TOLERANCE = 0.05
REFERENCE_SCORES: dict[str, tuple[float, float]] = {
    "Classical": (1.0, 0.0),
    "DFArena1B": (0.9356, 0.0009),
    "HuBERT": (0.9983, 0.9983),
    "RawNet2": (1.0, 0.0001),
    "Spectra0": (0.9989, 0.0033),
    "SpectraAASIST": (0.993, 0.0206),
    "SpectraAASIST3": (1.0, 0.0507),
    "Wav2Vec2": (0.9346, 0.9086),
    "WavLM": (0.9996, 0.9996),
    "AST/ASVspoof2019": (1.0, 0.0),
    "AST/ASVspoof5": (0.0, 0.0),
    "AST/VoxCelebSpoof": (1.0, 0.0),
    "VIT/ASVspoof2019/ConstantQ": (0.9994, 0.0003),
    "VIT/ASVspoof2019/MelSpectrogram": (0.9908, 0.0001),
    "VIT/ASVspoof2019/MFCC": (0.9957, 0.9914),
    "VIT/ASVspoof5/ConstantQ": (0.9856, 0.0002),
    "VIT/ASVspoof5/MelSpectrogram": (0.9997, 0.996),
    "VIT/ASVspoof5/MFCC": (0.9834, 0.9994),
    "VIT/VoxCelebSpoof/ConstantQ": (1.0, 0.0),
    "VIT/VoxCelebSpoof/MelSpectrogram": (1.0, 0.0),
    "VIT/VoxCelebSpoof/MFCC": (0.0, 0.0),
}

pytestmark = pytest.mark.integration


def _checkpoint(model: Model, dataset: Dataset, vis: Visualisation) -> tuple[str, str]:
    """(repo id, a file whose presence means the weights are cached)."""
    from Jabberjay.Models.Classical import run as classical
    from Jabberjay.Models.DFArena import run as df_arena
    from Jabberjay.Models.HuBERT import run as hubert
    from Jabberjay.Models.RawNet2 import run as rawnet2
    from Jabberjay.Models.Spectra0 import run as spectra0
    from Jabberjay.Models.SpectraAASIST import run as spectra_aasist
    from Jabberjay.Models.SpectraAASIST3 import run as spectra_aasist3
    from Jabberjay.Models.Transformer.AST import run as ast
    from Jabberjay.Models.Transformer.VIT import run as vit
    from Jabberjay.Models.Wav2Vec2 import run as wav2vec2
    from Jabberjay.Models.WavLM import run as wavlm

    match model:
        case Model.Classical:
            return classical.REPO_ID, classical.FILENAME
        case Model.RawNet2:
            return rawnet2.REPO_ID, rawnet2.FILENAME
        case Model.DFArena1B:
            return df_arena.MODEL_ID, df_arena.WEIGHTS_FILE
        case Model.Spectra0 | Model.SpectraAASIST | Model.SpectraAASIST3:
            run = {
                Model.Spectra0: spectra0,
                Model.SpectraAASIST: spectra_aasist,
                Model.SpectraAASIST3: spectra_aasist3,
            }[model]
            return run.MODEL_ID, "model.safetensors"
        case Model.AST:
            return ast.model_id(dataset), "config.json"
        case Model.VIT:
            return vit.model_id(dataset, vis), "config.json"
        case _:
            run = {Model.HuBERT: hubert, Model.Wav2Vec2: wav2vec2, Model.WavLM: wavlm}
            return run[model].MODEL_ID, "config.json"


def _cases():
    for model in Model:
        if model is Model.AST:
            for ds in Dataset:
                yield model, ds, Visualisation.ConstantQ, f"AST/{ds.value}"
        elif model is Model.VIT:
            for ds in Dataset:
                for vis in Visualisation:
                    yield model, ds, vis, f"VIT/{ds.value}/{vis.value}"
        else:
            yield model, Dataset.VoxCelebSpoof, Visualisation.ConstantQ, model.value


def _require_cached(repo_id: str, filename: str) -> None:
    revision = revision_for(repo_id)
    if not isinstance(
        try_to_load_from_cache(repo_id, filename, revision=revision), str
    ):
        reason = f"{repo_id}@{revision[:7]} not in the HuggingFace cache"
        if os.environ.get("JABBERJAY_REQUIRE_WEIGHTS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture(scope="module")
def jj():
    return Jabberjay()


@pytest.fixture(scope="module")
def clips(jj):
    return {name: jj.load(path) for name, path in CLIPS.items()}


def test_every_combination_has_a_reference_score():
    assert {tag for *_, tag in _cases()} == set(REFERENCE_SCORES)


@pytest.mark.parametrize(
    ("model", "dataset", "visualisation", "tag"),
    list(_cases()),
    ids=[tag for *_, tag in _cases()],
)
def test_matches_reference_scores(jj, clips, model, dataset, visualisation, tag):
    _require_cached(*_checkpoint(model, dataset, visualisation))
    try:
        for (clip, audio), expected in zip(
            clips.items(), REFERENCE_SCORES[tag], strict=True
        ):
            result = jj.detect(
                audio, model=model, dataset=dataset, visualisation=visualisation
            )
            bonafide = next(
                s["score"] for s in result.scores if s["label"] == "Bonafide"
            )
            assert bonafide == pytest.approx(expected, abs=TOLERANCE), (
                f"{tag} on {clip}.flac: bonafide score {bonafide:.4f}, "
                f"reference {expected:.4f}"
            )
            assert sum(s["score"] for s in result.scores) == pytest.approx(
                1.0, abs=1e-4
            )
    finally:
        # One model at a time: holding all 21 at once needs well over 16 GB
        Jabberjay.clear_cache()


def test_path_input_matches_preloaded_input(jj, clips):
    """detect(path) and detect(load(path)) must agree exactly."""
    _require_cached(
        *_checkpoint(Model.Spectra0, Dataset.VoxCelebSpoof, Visualisation.ConstantQ)
    )
    try:
        from_path = jj.detect(CLIPS["spoof"], model=Model.Spectra0)
        preloaded = jj.detect(clips["spoof"], model=Model.Spectra0)
        assert from_path.scores == preloaded.scores
    finally:
        Jabberjay.clear_cache()
