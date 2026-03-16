import argparse

import pytest

from Jabberjay.Utilities.enum_handler import Dataset, EnumAction, Model


class TestEnumAction:
    def test_init_raises_without_type(self):
        with pytest.raises(ValueError, match="type must be assigned"):
            EnumAction(option_strings=["--model"], dest="model")

    def test_init_raises_with_non_enum_type(self):
        with pytest.raises(TypeError, match="type must be an Enum"):
            EnumAction(option_strings=["--model"], dest="model", type=str)

    def test_init_sets_choices_from_enum_names(self):
        action = EnumAction(option_strings=["--model"], dest="model", type=Model)
        assert set(action.choices) == {m.name for m in Model}

    def test_call_sets_enum_value_via_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=Model, action=EnumAction)
        args = parser.parse_args(["--model", "Classical"])
        assert args.model == Model.Classical

    def test_call_invalid_value_triggers_parser_error(self):
        """Call __call__ directly to hit the KeyError path, bypassing argparse choice validation."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=Dataset, action=EnumAction)
        action = next(a for a in parser._actions if a.dest == "dataset")
        with pytest.raises(SystemExit):
            action(parser, argparse.Namespace(), "NotADataset")
