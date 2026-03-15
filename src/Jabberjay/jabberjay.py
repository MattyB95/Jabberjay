import argparse
import importlib
import logging

import librosa
import numpy as np

from Jabberjay.Utilities.enum_handler import EnumAction, Visualisation, Model, Dataset


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
        y, sr = librosa.load(filename)
        return y, sr

    def detect(
        self,
        audio: tuple[np.ndarray, float],
        model: Model,
        visualisation: Visualisation | None = None,
        dataset: Dataset | None = None,
    ):
        y, sr = audio
        match model:
            case Model.AST:
                return self.ast_handler(y=y, sr=sr, dataset=dataset)
            case Model.Classical:
                return self.classical_handler(audio=audio)
            case Model.RawNet2:
                return self.rawnet2_handler(y=y)
            case Model.VIT:
                return self.vit_handler(
                    audio=audio, visualisation=visualisation, dataset=dataset
                )

    def ast_handler(self, y: np.ndarray, sr: float, dataset: Dataset) -> list[dict[str, float]]:
        if dataset is None:
            raise ValueError("Dataset Is Required For AST Model!")
        import Jabberjay.Models.Transformer.AST.run as AST

        predict = AST.predict(y=y, sr=sr, dataset=dataset)
        logging.info(predict)
        print("Bonafide ✔️" if predict[0].get("label") == "Bonafide" else "Spoof ❌")
        return predict

    def classical_handler(self, audio: tuple[np.ndarray, float]) -> bool:
        import Jabberjay.Models.Classical.run as Classical

        predict = Classical.predict(audio=audio)
        logging.info(predict)
        result = bool(predict[0])
        print("Bonafide ✔️" if result else "Spoof ❌")
        return result

    def rawnet2_handler(self, y: np.ndarray) -> bool:
        import Jabberjay.Models.RawNet2.run as RawNet2

        predict = RawNet2.predict(y=y)
        result = bool(predict.item())
        logging.info(result)
        print("Bonafide ✔️" if result else "Spoof ❌")
        return result

    def vit_handler(
        self,
        audio: tuple[np.ndarray, float],
        visualisation: Visualisation,
        dataset: Dataset,
    ) -> list[dict[str, float]]:
        if visualisation is None:
            raise ValueError("Visualisation Is Required For VIT Model!")
        if dataset is None:
            raise ValueError("Dataset Is Required For VIT Model!")
        vit_module = importlib.import_module(
            f"Jabberjay.Models.Transformer.VIT.{visualisation.value}.run"
        )
        predict = vit_module.predict(audio=audio, dataset=dataset)
        logging.info(predict)
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
    if args.verbose:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
        logging.info("Verbosity Turned On!")
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    logging.info(f"Filename: {args.audio}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Visualisation: {args.visualisation}")
    logging.info(f"Verbose: {args.verbose}")

    jabberjay = Jabberjay()
    jabberjay.detect(
        audio=jabberjay.load(args.audio),
        model=args.model,
        visualisation=args.visualisation,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
