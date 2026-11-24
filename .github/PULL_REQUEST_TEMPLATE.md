## Summary

<!-- What does this PR do? Link any related issues with "Closes #<number>" -->

## Type of change

- [ ] Bug fix
- [ ] New model
- [ ] New feature
- [ ] Documentation update
- [ ] Refactor / maintenance

## Checklist

- [ ] `just fix` run (auto-format and lint)
- [ ] `just check` passes (lint + format + type check)
- [ ] `just test` passes
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] `README.md` updated if applicable (new model added to table, new feature documented)
- [ ] PR targets the `develop` branch, not `main`

## For new models

- [ ] Licence checked — Apache 2.0 / MIT, or any other licence flagged in `README.md` and `docs/models.md`
- [ ] Model added to `Model` enum in `enum_handler.py`
- [ ] `run.py` created under `src/Jabberjay/Models/<Name>/`, with `predict()` returning a score for both `Bonafide` and `Spoof`
- [ ] Hugging Face repo pinned to a reviewed commit in `REVISIONS` (`Utilities/hugging_face.py`)
- [ ] Label mapping verified on known-genuine and known-synthetic audio (pass a `label_map` if the checkpoint has no `id2label`)
- [ ] Model registered in `_MODELS` in `jabberjay.py`
- [ ] `test_model_members` updated in `tests/test_enums.py`, and the repo added to `_every_repo_jabberjay_loads()` in `tests/test_models.py`
- [ ] Reference scores added to `REFERENCE_SCORES` (and the files it loads to `_required_files()`) in `tests/test_integration.py`, and `just test-integration` passes
- [ ] Model card link added to README models table, and every other model list updated (see CONTRIBUTING.md, step 6)
