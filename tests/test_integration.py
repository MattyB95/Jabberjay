"""Opt-in integration tests against real model weights.

Deselected by default; run with ``pytest -m integration``. Each test also
skips unless its checkpoint is already in the local HuggingFace cache, so
an opt-in run never triggers a multi-GB download. Set
``JABBERJAY_REQUIRE_WEIGHTS=1`` (as CI does) to fail instead of skipping.
"""

import os
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

from Jabberjay import Jabberjay, Model

RES_DIR = Path(__file__).parent.parent / "res"

pytestmark = pytest.mark.integration


def _require_cached(repo_id: str, filename: str, revision: str) -> None:
    if not isinstance(
        try_to_load_from_cache(repo_id, filename, revision=revision), str
    ):
        reason = f"{repo_id}@{revision[:7]} not in the HuggingFace cache"
        if os.environ.get("JABBERJAY_REQUIRE_WEIGHTS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture(scope="module")
def df_arena_jj():
    from Jabberjay.Models.DFArena.run import _MODEL_ID, _REVISION, _WEIGHTS_FILE

    _require_cached(_MODEL_ID, _WEIGHTS_FILE, _REVISION)
    return Jabberjay()


class TestDFArena1BWeights:
    @pytest.mark.parametrize(
        ("clip", "expected"),
        [("bonafide/bonafide.flac", "Bonafide"), ("spoof/spoof.flac", "Spoof")],
    )
    def test_classifies_reference_clip(self, df_arena_jj, clip, expected):
        # A strict=True load of the real checkpoint is the full proof that the
        # ported architecture matches upstream; the label checks the logit order
        result = df_arena_jj.detect(RES_DIR / clip, model=Model.DFArena1B)
        assert result.label == expected
        assert result.confidence > 0.9
