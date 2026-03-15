default:
    @just --list

# Install all dependencies including dev
install:
    uv sync --group dev

# Run the test suite
test:
    uv run pytest

# Run tests with verbose output
test-verbose:
    uv run pytest -v

# Lint with ruff
lint:
    uv run ruff check .

# Format code with black
format:
    uv run black .

# Check formatting without making changes
format-check:
    uv run black --check .

# Type check with ty
type-check:
    uv run ty check src/

# Run lint, format, and type checks (used in CI)
check: lint format-check type-check

# Auto-fix lint issues then format with black
fix:
    uv run ruff check --fix .
    uv run black .

# Build the package
build:
    uv build

# Publish the package to PyPI
publish: build
    uv publish

# Publish a dev build to TestPyPI (mirrors what CI does on the develop branch)
publish-test: build
    uv publish --publish-url https://test.pypi.org/legacy/

# Show the current package version
version:
    @grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'

# Remove build artifacts
clean:
    rm -rf dist/ .pytest_cache/ __pycache__/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Run the CLI against a file (usage: just detect path/to/audio.wav)
detect audio model="VIT" dataset="VoxCelebSpoof" vis="ConstantQ":
    uv run jabberjay {{ audio }} -m {{ model }} -d {{ dataset }} -vis {{ vis }}

# Run the quickstart example (recommended first step)
example:
    uv run python examples/quickstart.py

# Run a specific example (usage: just run-example preloading_audio)
run-example name:
    uv run python examples/{{ name }}.py

# Run the exhaustive model sweep (slow — downloads all models)
run-all:
    uv run python examples/run_all.py
