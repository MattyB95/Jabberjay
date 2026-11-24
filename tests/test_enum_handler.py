import argparse

import pytest

from Jabberjay.Utilities.enum_handler import (
    Dataset,
    EnumAction,
    InvalidChoiceError,
    Model,
    Visualisation,
    parse_enum,
)


class TestEnumAction:
    def test_init_raises_without_type(self):
        with pytest.raises(ValueError, match="type must be assigned"):
            EnumAction(option_strings=["--model"], dest="model")

    def test_init_raises_with_non_enum_type(self):
        with pytest.raises(TypeError, match="type must be an Enum"):
            EnumAction(option_strings=["--model"], dest="model", type=str)

    def test_init_lists_enum_names_in_metavar(self):
        action = EnumAction(option_strings=["--model"], dest="model", type=Model)
        assert action.metavar == "{" + ",".join(m.name for m in Model) + "}"

    def test_call_matches_names_case_insensitively(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=Model, action=EnumAction)
        assert parser.parse_args(["--model", "vit"]).model == Model.VIT
        assert parser.parse_args(["--model", "HUBERT"]).model == Model.HuBERT

    def test_invalid_value_error_lists_choices(self, capsys):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=Model, action=EnumAction)
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "nope"])
        err = capsys.readouterr().err
        assert "Unknown model 'nope'" in err
        assert "Classical" in err

    def test_call_sets_enum_value_via_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=Model, action=EnumAction)
        args = parser.parse_args(["--model", "Classical"])
        assert args.model == Model.Classical

    def test_call_invalid_value_triggers_parser_error(self):
        """Call __call__ directly to hit the InvalidChoiceError path."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=Dataset, action=EnumAction)
        action = next(a for a in parser._actions if a.dest == "dataset")
        with pytest.raises(SystemExit):
            action(parser, argparse.Namespace(), "NotADataset")


class TestParseEnum:
    def test_member_passes_through(self):
        assert parse_enum(Model, Model.AST) is Model.AST

    @pytest.mark.parametrize("name", ["MFCC", "mfcc", "Mfcc"])
    def test_name_matches_case_insensitively(self, name):
        assert parse_enum(Visualisation, name) is Visualisation.MFCC

    def test_unknown_name_raises_value_error_listing_choices(self):
        with pytest.raises(ValueError, match="choose from: ASVspoof2019"):
            parse_enum(Dataset, "LibriSpeech")

    def test_unknown_name_is_still_a_key_error(self):
        # Earlier versions raised a bare KeyError; keep `except KeyError` working
        with pytest.raises(KeyError):
            parse_enum(Model, "nope")

    def test_error_message_is_not_repr_quoted(self):
        exc = InvalidChoiceError("Unknown model 'x'")
        assert str(exc) == "Unknown model 'x'"
        assert str(InvalidChoiceError()) == ""

    def test_member_of_another_enum_is_rejected(self):
        with pytest.raises(InvalidChoiceError):
            parse_enum(Model, Dataset.ASVspoof5)
