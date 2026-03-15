import argparse
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from loguru import logger

from Jabberjay.Utilities.enum_handler import Dataset, EnumAction, Model, Visualisation
from Jabberjay.Utilities.types import PredictionScore

# Silence loguru by default — callers and the CLI configure output
logger.disable("Jabberjay")

Audio = tuple[np.ndarray, float]


@dataclass
class DetectionResult:
    """Structured result returned by every call to Jabberjay.detect()."""

    label: str
    """'Bonafide' or 'Spoof'."""

    is_bonafide: bool
    """True if the audio was classified as genuine."""

    confidence: float
    """Confidence score for the top prediction (0.0–1.0)."""

    model: Model
    """The model used to produce this result."""

    scores: list[PredictionScore] | None = field(default=None)
    """Full label/score breakdown from transformer models; None for Classical and RawNet2."""

    def __str__(self) -> str:
        verdict = "Bonafide ✔️" if self.is_bonafide else "Spoof ❌"
        return f"{verdict} ({self.confidence:.1%} confidence, model={self.model.value})"


class Jabberjay:
    @staticmethod
    def list_models() -> list[Model]:
        """Print and return all available models."""
        models = list(Model)
        print("Models:", ", ".join(m.value for m in models))
        return models

    @staticmethod
    def list_datasets() -> list[Dataset]:
        """Print and return all available datasets."""
        datasets = list(Dataset)
        print("Datasets:", ", ".join(d.value for d in datasets))
        return datasets

    @staticmethod
    def list_visualisations() -> list[Visualisation]:
        """Print and return all available visualisations."""
        visualisations = list(Visualisation)
        print("Visualisations:", ", ".join(v.value for v in visualisations))
        return visualisations

    def load(self, path: str | Path) -> Audio:
        """Load an audio file and return (samples, sample_rate)."""
        path = str(path)
        logger.debug(f"Loading audio file: {path}")
        try:
            y, sr = librosa.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {path}")
        except Exception as exc:
            raise ValueError(f"Failed to load audio from '{path}': {exc}") from exc
        logger.info(f"Loaded {len(y) / sr:.2f}s of audio at {int(sr)}Hz")
        return y, sr

    def detect(
        self,
        audio: str | Path | Audio,
        model: Model | str = Model.VIT,
        visualisation: Visualisation | str | None = None,
        dataset: Dataset | str | None = None,
    ) -> DetectionResult:
        """
        Detect whether audio is bonafide or spoofed.

        Args:
            audio: Path to an audio file, or a pre-loaded (samples, sample_rate) tuple.
            model: Model to use. Accepts a Model enum value or its string name (e.g. "VIT").
            visualisation: Visualisation for VIT models. Accepts Visualisation enum or string.
            dataset: Dataset the model was trained on. Accepts Dataset enum or string.

        Returns:
            DetectionResult with label, confidence, and full scores where available.

        Raises:
            ValueError: If required arguments are missing, or audio is empty.
            KeyError: If an unrecognised string is passed for model, dataset, or visualisation.
        """
        # Coerce strings to enums
        if isinstance(model, str):
            model = Model[model]
        if isinstance(visualisation, str):
            visualisation = Visualisation[visualisation]
        if isinstance(dataset, str):
            dataset = Dataset[dataset]

        # Accept a file path directly
        if isinstance(audio, (str, Path)):
            audio = self.load(audio)

        y, sr = audio
        if len(y) == 0:
            raise ValueError("Audio array is empty — nothing to classify.")

        logger.info(
            f"Running detection — model={model.value}, "
            f"dataset={dataset.value if dataset else None}, "
            f"visualisation={visualisation.value if visualisation else None}"
        )
        match model:
            case Model.AST:
                if dataset is None:
                    raise ValueError("Dataset is required for the AST model.")
                return self._ast_handler(y=y, sr=sr, dataset=dataset)
            case Model.Classical:
                return self._classical_handler(audio=audio)
            case Model.HuBERT:
                return self._hubert_handler(y=y, sr=sr)
            case Model.RawNet2:
                return self._rawnet2_handler(y=y)
            case Model.VIT:
                if visualisation is None:
                    raise ValueError("Visualisation is required for the VIT model.")
                if dataset is None:
                    raise ValueError("Dataset is required for the VIT model.")
                return self._vit_handler(
                    audio=audio, visualisation=visualisation, dataset=dataset
                )
            case Model.Wav2Vec2:
                return self._wav2vec2_handler(y=y, sr=sr)
            case Model.WavLM:
                return self._wavlm_handler(y=y, sr=sr)
            case _:
                raise ValueError(f"Unknown model: {model}")

    @staticmethod
    def _result_from_scores(
        scores: list[PredictionScore], model: Model
    ) -> DetectionResult:
        """Build a DetectionResult from a sorted list of PredictionScores."""
        top = scores[0]
        return DetectionResult(
            label=top["label"],
            is_bonafide=top["label"] == "Bonafide",
            confidence=top["score"],
            model=model,
            scores=scores,
        )

    def _ast_handler(
        self, y: np.ndarray, sr: float, dataset: Dataset
    ) -> DetectionResult:
        import Jabberjay.Models.Transformer.AST.run as AST

        scores = AST.predict(y=y, sr=sr, dataset=dataset)
        logger.debug(f"AST predictions: {scores}")
        return self._result_from_scores(scores, Model.AST)

    def _classical_handler(self, audio: Audio) -> DetectionResult:
        import Jabberjay.Models.Classical.run as Classical

        prediction, confidence = Classical.predict(audio=audio)
        is_bonafide = bool(prediction)
        logger.debug(
            f"Classical prediction: {is_bonafide} (confidence={confidence:.3f})"
        )
        return DetectionResult(
            label="Bonafide" if is_bonafide else "Spoof",
            is_bonafide=is_bonafide,
            confidence=confidence,
            model=Model.Classical,
        )

    def _rawnet2_handler(self, y: np.ndarray) -> DetectionResult:
        import Jabberjay.Models.RawNet2.run as RawNet2

        prediction, confidence = RawNet2.predict(y=y)
        is_bonafide = bool(prediction.item())
        logger.debug(f"RawNet2 prediction: {is_bonafide} (confidence={confidence:.3f})")
        return DetectionResult(
            label="Bonafide" if is_bonafide else "Spoof",
            is_bonafide=is_bonafide,
            confidence=confidence,
            model=Model.RawNet2,
        )

    def _vit_handler(
        self,
        audio: Audio,
        visualisation: Visualisation,
        dataset: Dataset,
    ) -> DetectionResult:
        try:
            vit_module = importlib.import_module(
                f"Jabberjay.Models.Transformer.VIT.{visualisation.value}.run"
            )
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"No VIT module found for visualisation '{visualisation.value}'. "
                f"Expected: Jabberjay.Models.Transformer.VIT.{visualisation.value}.run"
            ) from exc
        scores = vit_module.predict(audio=audio, dataset=dataset)
        logger.debug(f"VIT predictions: {scores}")
        return self._result_from_scores(scores, Model.VIT)

    def _wav2vec2_handler(self, y: np.ndarray, sr: float) -> DetectionResult:
        import Jabberjay.Models.Wav2Vec2.run as Wav2Vec2

        scores = Wav2Vec2.predict(y=y, sr=sr)
        logger.debug(f"Wav2Vec2 predictions: {scores}")
        return self._result_from_scores(scores, Model.Wav2Vec2)

    def _hubert_handler(self, y: np.ndarray, sr: float) -> DetectionResult:
        import Jabberjay.Models.HuBERT.run as HuBERT

        scores = HuBERT.predict(y=y, sr=sr)
        logger.debug(f"HuBERT predictions: {scores}")
        return self._result_from_scores(scores, Model.HuBERT)

    def _wavlm_handler(self, y: np.ndarray, sr: float) -> DetectionResult:
        import Jabberjay.Models.WavLM.run as WavLM

        scores = WavLM.predict(y=y, sr=sr)
        logger.debug(f"WavLM predictions: {scores}")
        return self._result_from_scores(scores, Model.WavLM)


def main():
    parser = argparse.ArgumentParser(
        prog="Jabberjay",
        description="🦜 Synthetic Voice Detection",
        epilog="May The Odds Be Ever In Your Favor.",
    )

    parser.add_argument("audio", type=str)
    parser.add_argument(
        "-m", "--model", type=Model, action=EnumAction, default=Model.VIT
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Dataset,
        action=EnumAction,
        default=Dataset.VoxCelebSpoof,
    )
    parser.add_argument(
        "-vis",
        "--visualisation",
        type=Visualisation,
        action=EnumAction,
        default=Visualisation.ConstantQ,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

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
    result = jabberjay.detect(
        audio=args.audio,
        model=args.model,
        visualisation=args.visualisation,
        dataset=args.dataset,
    )
    print(result)


if __name__ == "__main__":
    main()
