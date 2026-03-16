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

- [ ] Licence confirmed as Apache 2.0 or MIT
- [ ] Model added to `Model` enum in `enum_handler.py`
- [ ] `run.py` created under `src/Jabberjay/Models/<Name>/`
- [ ] Handler and match case added to `jabberjay.py`
- [ ] `test_model_members` updated in `tests/test_jabberjay.py`
- [ ] Model card link added to README models table
