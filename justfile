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

# Run lint and format checks (used in CI)
check: lint format-check

# Auto-fix lint issues then format with black
fix:
    uv run ruff check --fix .
    uv run black .

# Build the package
build:
    uv build

# Publish the package to PyPI
publish:
    uv publish

# Remove build artifacts
clean:
    rm -rf dist/ .pytest_cache/ __pycache__/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Run the CLI against a file (usage: just detect path/to/audio.wav)
detect audio model="VIT" dataset="VoxCelebSpoof" vis="ConstantQ":
    uv run jabberjay {{ audio }} -m {{ model }} -d {{ dataset }} -vis {{ vis }}

# Run the example script
example:
    uv run python examples/example.py
