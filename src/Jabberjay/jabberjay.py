import argparse
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
        filename = filename
        y, sr = librosa.load(filename)
        return y, sr

    def detect(
        self,
        audio: tuple[np.ndarray, float],
        model: Model,
        visualisation: Visualisation | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        y, sr = audio
        match model:
            case Model.AST:
                self.ast_handler(y=y, dataset=dataset)
            case Model.Classical:
                self.classical_handler(audio=audio)
            case Model.RawNet2:
                self.rawnet2_handler(y=y)
            case Model.VIT:
                self.vit_handler(
                    audio=audio, visualisation=visualisation, dataset=dataset
                )

    def ast_handler(self, y, dataset: Dataset) -> None:
        if dataset is None:
            raise ValueError("Dataset Is Required For AST Model!")
        import Jabberjay.Models.Tranformer.AST.run as AST

        predict = AST.predict(y=y, dataset=dataset)
        logging.info(predict)
        print("Bonafide ✔️" if predict[0].get("label") == "Bonafide" else "Spoof ❌")

    def classical_handler(self, audio: tuple[np.ndarray, float]) -> None:
        import Jabberjay.Models.Classical.run as Classical

        predict = Classical.predict(audio=audio)
        logging.info(predict)
        print("Bonafide ✔️" if predict else "Spoof ❌")
        pass

    def rawnet2_handler(self, y) -> None:
        import Jabberjay.Models.RawNet2.run as RawNet2

        predict = RawNet2.predict(y=y)
        predict = predict.item()
        logging.info(predict)
        print("Bonafide ✔️" if predict else "Spoof ❌")

    def vit_handler(
        self,
        audio: tuple[np.ndarray, float],
        visualisation: Visualisation,
        dataset: Dataset,
    ) -> None:
        if visualisation is None:
            raise ValueError("Visualisation Is Required For VIT Model!")
        if dataset is None:
            raise ValueError("Dataset Is Required For VIT Model!")
        match visualisation:
            case Visualisation.ConstantQ:
                import Jabberjay.Models.Tranformer.VIT.ConstantQ.run as VITConstantQ

                predict = VITConstantQ.predict(audio=audio, dataset=dataset)
                logging.info(predict)
                print(
                    "Bonafide ✔️"
                    if predict[0].get("label") == "Bonafide"
                    else "Spoof ❌"
                )
            case Visualisation.MelSpectrogram:
                import Jabberjay.Models.Tranformer.VIT.MelSpectrogram.run as VITMelSpectrogram

                predict = VITMelSpectrogram.predict(audio=audio, dataset=dataset)
                logging.info(predict)
                print(
                    "Bonafide ✔️"
                    if predict[0].get("label") == "Bonafide"
                    else "Spoof ❌"
                )
            case Visualisation.MFCC:
                import Jabberjay.Models.Tranformer.VIT.MFCC.run as VITMFCC

                predict = VITMFCC.predict(audio=audio, dataset=dataset)
                logging.info(predict)
                print(
                    "Bonafide ✔️"
                    if predict[0].get("label") == "Bonafide"
                    else "Spoof ❌"
                )


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

    logging.info(f"Filename: {args.filename}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Visualisation: {args.visualisation}")
    logging.info(f"Verbose: {args.verbose}")

    jabberjay = Jabberjay()
    jabberjay.detect(
        audio=jabberjay.load(args.filename),
        model=args.model,
        visualisation=args.visualisation,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
