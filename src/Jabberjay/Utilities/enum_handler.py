import argparse
from enum import Enum


class Model(Enum):
    AST = "AST"
    Classical = "Classical"
    RawNet2 = "RawNet2"
    VIT = "VIT"


class Dataset(Enum):
    ASVspoof2019 = "ASVspoof2019"
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
        kwargs.setdefault("choices", tuple(e.name for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        value = self._enum[values]
        setattr(namespace, self.dest, value)
