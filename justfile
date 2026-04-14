set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

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

# Run tests and open the HTML coverage report
[unix]
coverage:
    uv run pytest --cov-report=html
    open htmlcov/index.html

[windows]
coverage:
    uv run pytest --cov-report=html
    Invoke-Item htmlcov/index.html

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
    @uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"

# Serve docs locally with live reload
docs:
    uv run mkdocs serve

# Build docs site to site/ directory
docs-build:
    uv run mkdocs build

# Remove build artifacts
[unix]
clean:
    rm -rf dist/ .pytest_cache/ htmlcov/ .coverage coverage.xml
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

[windows]
clean:
    Remove-Item -Recurse -Force dist, .pytest_cache, htmlcov, .coverage, coverage.xml -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

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
