import argparse
from enum import Enum


class InvalidChoiceError(ValueError, KeyError):
    """An unrecognised model, dataset or visualisation name.

    Subclasses ValueError (the natural type for a bad argument value) and,
    for backwards compatibility, KeyError, which earlier versions raised.
    """

    def __str__(self) -> str:
        # KeyError.__str__ repr()-quotes its message; show it plainly
        return str(self.args[0]) if self.args else ""


def parse_enum[E: Enum](enum_type: type[E], value: E | str) -> E:
    """Return the ``enum_type`` member named ``value``, ignoring case.

    Members pass through unchanged, so callers can accept either form.

    Raises:
        InvalidChoiceError: If ``value`` names no member.
    """
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        folded = value.casefold()
        for member in enum_type:
            if member.name.casefold() == folded:
                return member
    choices = ", ".join(member.name for member in enum_type)
    raise InvalidChoiceError(
        f"Unknown {enum_type.__name__.lower()} {value!r} — choose from: {choices}"
    )


class Model(Enum):
    AST = "AST"
    Classical = "Classical"
    DFArena1B = "DFArena1B"
    HuBERT = "HuBERT"
    RawNet2 = "RawNet2"
    Spectra0 = "Spectra0"
    SpectraAASIST = "SpectraAASIST"
    SpectraAASIST3 = "SpectraAASIST3"
    VIT = "VIT"
    Wav2Vec2 = "Wav2Vec2"
    WavLM = "WavLM"


class Dataset(Enum):
    ASVspoof2019 = "ASVspoof2019"
    ASVspoof5 = "ASVspoof5"
    VoxCelebSpoof = "VoxCelebSpoof"


class Visualisation(Enum):
    ConstantQ = "ConstantQ"
    MelSpectrogram = "MelSpectrogram"
    MFCC = "MFCC"


class EnumAction(argparse.Action):
    def __init__(self, **kwargs):
        enum_type = kwargs.pop("type", None)
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, Enum):
            raise TypeError("type must be an Enum when using EnumAction")
        # Shown in --help like argparse's `choices`, but validated in
        # __call__ instead so names match case-insensitively (-m vit)
        kwargs.setdefault("metavar", "{" + ",".join(e.name for e in enum_type) + "}")

        super().__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            value = parse_enum(self._enum, values)
        except InvalidChoiceError as exc:
            parser.error(str(exc))
            return  # pragma: no cover — parser.error() always raises SystemExit
        setattr(namespace, self.dest, value)
