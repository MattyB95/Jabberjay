import argparse
import importlib
import sys

import librosa
import numpy as np
from loguru import logger

from Jabberjay.Utilities.enum_handler import Dataset, EnumAction, Model, Visualisation

# Silence loguru by default — callers and the CLI configure output
logger.disable("Jabberjay")


class Jabberjay:
    @staticmethod
    def list_models():
        print("Models: ", *Model)

    @staticmethod
    def list_datasets():
        print("Datasets: ", *Dataset)

    @staticmethod
    def list_visualisations():
        print("Visualisations: ", *Visualisation)

    def load(self, filename: str) -> tuple[np.ndarray, float]:
        logger.debug(f"Loading audio file: {filename}")
        y, sr = librosa.load(filename)
        logger.info(f"Loaded {len(y) / sr:.2f}s of audio at {int(sr)}Hz")
        return y, sr

    def detect(
        self,
        audio: tuple[np.ndarray, float],
        model: Model,
        visualisation: Visualisation | None = None,
        dataset: Dataset | None = None,
    ):
        logger.info(
            f"Running detection — model={model.value}, "
            f"dataset={dataset.value if dataset else None}, "
            f"visualisation={visualisation.value if visualisation else None}"
        )
        y, sr = audio
        match model:
            case Model.AST:
                if dataset is None:
                    raise ValueError("Dataset Is Required For AST Model!")
                return self.ast_handler(y=y, sr=sr, dataset=dataset)
            case Model.Classical:
                return self.classical_handler(audio=audio)
            case Model.RawNet2:
                return self.rawnet2_handler(y=y)
            case Model.VIT:
                if visualisation is None:
                    raise ValueError("Visualisation Is Required For VIT Model!")
                if dataset is None:
                    raise ValueError("Dataset Is Required For VIT Model!")
                return self.vit_handler(
                    audio=audio, visualisation=visualisation, dataset=dataset
                )

    def ast_handler(
        self, y: np.ndarray, sr: float, dataset: Dataset
    ) -> list[dict[str, float]]:
        import Jabberjay.Models.Transformer.AST.run as AST

        predict = AST.predict(y=y, sr=sr, dataset=dataset)
        logger.debug(f"AST predictions: {predict}")
        print("Bonafide ✔️" if predict[0].get("label") == "Bonafide" else "Spoof ❌")
        return predict

    def classical_handler(self, audio: tuple[np.ndarray, float]) -> bool:
        import Jabberjay.Models.Classical.run as Classical

        predict = Classical.predict(audio=audio)
        result = bool(predict[0])
        logger.debug(f"Classical prediction: {result}")
        print("Bonafide ✔️" if result else "Spoof ❌")
        return result

    def rawnet2_handler(self, y: np.ndarray) -> bool:
        import Jabberjay.Models.RawNet2.run as RawNet2

        predict = RawNet2.predict(y=y)
        result = bool(predict.item())
        logger.debug(f"RawNet2 prediction: {result}")
        print("Bonafide ✔️" if result else "Spoof ❌")
        return result

    def vit_handler(
        self,
        audio: tuple[np.ndarray, float],
        visualisation: Visualisation,
        dataset: Dataset,
    ) -> list[dict[str, float]]:
        vit_module = importlib.import_module(
            f"Jabberjay.Models.Transformer.VIT.{visualisation.value}.run"
        )
        predict = vit_module.predict(audio=audio, dataset=dataset)
        logger.debug(f"VIT predictions: {predict}")
        print("Bonafide ✔️" if predict[0].get("label") == "Bonafide" else "Spoof ❌")
        return predict


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
    jabberjay.detect(
        audio=jabberjay.load(args.audio),
        model=args.model,
        visualisation=args.visualisation,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
