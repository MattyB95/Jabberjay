import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from loguru import logger

from Jabberjay.Utilities import model_cache
from Jabberjay.Utilities.audio import prepare_audio
from Jabberjay.Utilities.enum_handler import (
    Dataset,
    EnumAction,
    Model,
    Visualisation,
    parse_enum,
)
from Jabberjay.Utilities.types import Label, PredictionScore

# Silence loguru by default — callers and the CLI configure output
logger.disable("Jabberjay")

Audio = tuple[np.ndarray, float]


@dataclass
class DetectionResult:
    """Structured result returned by every call to Jabberjay.detect()."""

    label: Label
    """'Bonafide' or 'Spoof'."""

    is_bonafide: bool
    """True if the audio was classified as genuine."""

    confidence: float
    """Confidence score for the top prediction (0.0-1.0)."""

    model: Model
    """The model used to produce this result."""

    scores: list[PredictionScore] = field(default_factory=list)
    """Full label/score breakdown for both classes, sorted highest-first."""

    def __str__(self) -> str:
        verdict = "Bonafide ✔️" if self.is_bonafide else "Spoof ❌"
        return f"{verdict} ({self.confidence:.1%} confidence, model={self.model.value})"


@dataclass(frozen=True)
class _ModelSpec:
    """Where a model's ``predict(y, sr, ...)`` lives and which options it takes.

    Modules are imported on first use, so importing Jabberjay stays cheap and
    a model's heavy dependencies only load when that model is run.
    """

    module: str
    needs_dataset: bool = False
    needs_visualisation: bool = False


_MODELS: dict[Model, _ModelSpec] = {
    Model.AST: _ModelSpec("Jabberjay.Models.Transformer.AST.run", needs_dataset=True),
    Model.Classical: _ModelSpec("Jabberjay.Models.Classical.run"),
    Model.DFArena1B: _ModelSpec("Jabberjay.Models.DFArena.run"),
    Model.HuBERT: _ModelSpec("Jabberjay.Models.HuBERT.run"),
    Model.RawNet2: _ModelSpec("Jabberjay.Models.RawNet2.run"),
    Model.Spectra0: _ModelSpec("Jabberjay.Models.Spectra0.run"),
    Model.SpectraAASIST: _ModelSpec("Jabberjay.Models.SpectraAASIST.run"),
    Model.SpectraAASIST3: _ModelSpec("Jabberjay.Models.SpectraAASIST3.run"),
    Model.VIT: _ModelSpec(
        "Jabberjay.Models.Transformer.VIT.run",
        needs_dataset=True,
        needs_visualisation=True,
    ),
    Model.Wav2Vec2: _ModelSpec("Jabberjay.Models.Wav2Vec2.run"),
    Model.WavLM: _ModelSpec("Jabberjay.Models.WavLM.run"),
}


class Jabberjay:
    """Unified entry point for every synthetic voice detection model.

    Construct once and call ``detect()`` with an audio file path or a
    pre-loaded ``(samples, sample_rate)`` tuple. Model weights are downloaded
    from Hugging Face Hub on first use and cached in memory for the lifetime
    of the process, so repeated calls for the same model skip both the
    download and the reload. Call ``clear_cache()`` to release them.

    Example::

        jj = Jabberjay()
        result = jj.detect("audio.wav")            # default VIT model
        result = jj.detect("audio.wav", model="HuBERT")
    """

    _logging_enabled: bool = False

    @classmethod
    def enable_logging(cls, level: str = "DEBUG") -> None:
        """Enable Jabberjay's built-in logging output to stderr.

        Safe to call multiple times — the handler is only registered once.
        Subsequent calls with a different level are silently ignored.

        Args:
            level: Minimum log level to emit. One of ``"DEBUG"``, ``"INFO"``,
                   ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``. Defaults to ``"DEBUG"``.

        Example::

            Jabberjay.enable_logging()          # all messages
            Jabberjay.enable_logging("INFO")    # skip debug messages
        """
        if cls._logging_enabled:
            return
        logger.enable("Jabberjay")
        logger.add(sys.stderr, format="{level}: {message}", level=level)
        cls._logging_enabled = True

    @staticmethod
    def clear_cache() -> None:
        """Release every loaded model and return its memory.

        Models stay loaded after their first ``detect()`` call so later calls
        are fast. Running many different models in one process adds up —
        several GB each for DFArena1B and the Spectra family — so call this
        once you're done with a model (or between models in a sweep) to free
        RAM and GPU memory. The next ``detect()`` reloads from the local
        Hugging Face cache without re-downloading.
        """
        model_cache.clear_all()
        model_cache.release_memory()

    @staticmethod
    def list_models() -> list[Model]:
        """Return all available models."""
        return list(Model)

    @staticmethod
    def list_datasets() -> list[Dataset]:
        """Return all available datasets."""
        return list(Dataset)

    @staticmethod
    def list_visualisations() -> list[Visualisation]:
        """Return all available visualisations."""
        return list(Visualisation)

    @staticmethod
    def load(path: str | os.PathLike) -> Audio:
        """Load an audio file as mono at its native rate: (samples, sample_rate).

        No resampling happens here; each model resamples to the rate it was
        trained at, so audio is only ever resampled once.
        """
        path = os.fspath(path)
        logger.debug(f"Loading audio file: {path}")
        # librosa 1.0 decodes via soundfile only, which raises an opaque
        # LibsndfileError ("System error") for a missing path — catch that
        # here so callers still get a clear FileNotFoundError. A path that
        # exists but isn't a readable audio file (a directory, a non-audio
        # file) falls through to librosa and surfaces as ValueError below.
        if not Path(path).exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            y, sr = librosa.load(path, sr=None)
        except (OSError, RuntimeError, EOFError) as exc:
            raise ValueError(f"Failed to load audio from '{path}': {exc}") from exc
        logger.info(f"Loaded {len(y) / sr:.2f}s of audio at {int(sr)}Hz")
        return y, sr

    def detect(
        self,
        audio: str | os.PathLike | Audio,
        model: Model | str = Model.VIT,
        visualisation: Visualisation | str | None = Visualisation.ConstantQ,
        dataset: Dataset | str | None = Dataset.VoxCelebSpoof,
    ) -> DetectionResult:
        """
        Detect whether audio is bonafide or spoofed.

        Args:
            audio: Path to an audio file, or a pre-loaded ``(samples,
                sample_rate)`` tuple at any sample rate. Samples may be float
                or integer PCM, mono or multi-channel (downmixed to mono).
            model: Model to use — a Model member or its name, case-insensitive
                (e.g. ``"VIT"`` or ``"vit"``).
            visualisation: Spectrogram type for VIT. Member or name.
            dataset: Training dataset for VIT and AST. Member or name.

        Returns:
            DetectionResult with the label, its confidence, and the scores
            for both classes.

        Raises:
            FileNotFoundError: If ``audio`` is a path that doesn't exist.
            ValueError: If the audio is empty, non-finite, wrongly shaped or
                unreadable, a required argument is ``None``, or a name is
                unrecognised (``InvalidChoiceError``, also a ``KeyError``).

        Note:
            VIT requires both ``visualisation`` and ``dataset``; both default to
            ``ConstantQ`` and ``VoxCelebSpoof`` respectively so a bare
            ``jj.detect("audio.flac")`` works out of the box.
            AST requires ``dataset``; defaults to ``VoxCelebSpoof``.
            All other models (Classical, DFArena1B, RawNet2, Spectra0,
            SpectraAASIST, SpectraAASIST3, HuBERT, Wav2Vec2, WavLM) ignore both.
        """
        model = parse_enum(Model, model)
        if visualisation is not None:
            visualisation = parse_enum(Visualisation, visualisation)
        if dataset is not None:
            dataset = parse_enum(Dataset, dataset)

        if isinstance(audio, (str, os.PathLike)):
            audio = self.load(audio)
        if not isinstance(audio, tuple) or len(audio) != 2:
            raise ValueError(
                "audio must be a file path or a (samples, sample_rate) tuple."
            )
        y, sr = prepare_audio(*audio)

        spec = _MODELS[model]
        options: dict[str, Dataset | Visualisation] = {}
        if spec.needs_dataset:
            if dataset is None:
                raise ValueError(f"Dataset is required for the {model.value} model.")
            options["dataset"] = dataset
        if spec.needs_visualisation:
            if visualisation is None:
                raise ValueError(
                    f"Visualisation is required for the {model.value} model."
                )
            options["visualisation"] = visualisation

        logger.info(
            f"Running detection — model={model.value}, "
            + ", ".join(f"{k}={v.value}" for k, v in options.items())
        )
        scores = importlib.import_module(spec.module).predict(y=y, sr=sr, **options)
        logger.debug(f"{model.value} predictions: {scores}")
        return self._result_from_scores(scores, model)

    @staticmethod
    def _result_from_scores(
        scores: list[PredictionScore], model: Model
    ) -> DetectionResult:
        """Build a DetectionResult from a list of PredictionScores."""
        if not scores:
            raise RuntimeError(f"{model.value} returned no predictions.")
        scores = sorted(scores, key=lambda s: s["score"], reverse=True)
        top = scores[0]
        return DetectionResult(
            label=top["label"],
            is_bonafide=top["label"] == "Bonafide",
            confidence=top["score"],
            model=model,
            scores=scores,
        )


def main():
    # DetectionResult.__str__ and the CLI description embed emoji; force
    # UTF-8 output so this doesn't crash on non-UTF-8 consoles (e.g. Windows
    # cp1252, which is still the default outside Windows Terminal/UTF-8 mode)
    if reconfigure := getattr(sys.stdout, "reconfigure", None):
        reconfigure(encoding="utf-8", errors="replace")
    if reconfigure := getattr(sys.stderr, "reconfigure", None):
        reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="Jabberjay",
        description="🦜 Synthetic Voice Detection",
        epilog="May The Odds Be Ever In Your Favor.",
    )

    parser.add_argument("audio", type=str, help="path to the audio file to analyse")
    parser.add_argument(
        "-m",
        "--model",
        type=Model,
        action=EnumAction,
        default=Model.VIT,
        help="detection model to use (default: VIT)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Dataset,
        action=EnumAction,
        default=Dataset.VoxCelebSpoof,
        help="training dataset for the VIT and AST models (default: VoxCelebSpoof)",
    )
    parser.add_argument(
        "-vis",
        "--visualisation",
        type=Visualisation,
        action=EnumAction,
        default=Visualisation.ConstantQ,
        help="spectrogram type for the VIT model (default: ConstantQ)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print model load and inference progress to stderr",
    )

    args = parser.parse_args()

    # Configure loguru for CLI use
    logger.enable("Jabberjay")
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, format="{level}: {message}", level="DEBUG")
        logger.info("Verbose logging enabled")
    else:
        logger.add(sys.stderr, format="{level}: {message}", level="WARNING")

    logger.debug(f"audio={args.audio}")
    logger.debug(f"model={args.model}")
    logger.debug(f"dataset={args.dataset}")
    logger.debug(f"visualisation={args.visualisation}")

    jabberjay = Jabberjay()
    try:
        result = jabberjay.detect(
            audio=args.audio,
            model=args.model,
            visualisation=args.visualisation,
            dataset=args.dataset,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Expected user-facing errors (missing/unreadable file, bad argument
        # combination) — show a clean message instead of a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(result)


if __name__ == "__main__":
    main()
