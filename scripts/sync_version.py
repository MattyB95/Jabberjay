#!/usr/bin/env python3
"""Sync the version from pyproject.toml into CITATION.cff and README.md.

Run automatically as a pre-commit hook, or manually via:
    python scripts/sync_version.py
"""

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def get_version() -> str:
    text = _read(ROOT / "pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print("ERROR: Could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def sync_citation(version: str) -> bool:
    path = ROOT / "CITATION.cff"
    original = _read(path)
    version_match = re.search(r"^version:\s*(.+)$", original, flags=re.MULTILINE)
    if version_match and version_match.group(1).strip() == version:
        return False
    updated = re.sub(
        r"^version:\s*.+$", f"version: {version}", original, flags=re.MULTILINE
    )
    updated = re.sub(
        r"^date-released:\s*.+$",
        f'date-released: "{date.today()}"',
        updated,
        flags=re.MULTILINE,
    )
    if updated != original:
        _write(path, updated)
        print(f"  updated CITATION.cff → version {version}, date {date.today()}")
        return True
    return False


def sync_readme(version: str) -> bool:
    path = ROOT / "README.md"
    original = _read(path)
    updated = re.sub(r"(  version\s*=\s*\{)[^}]+(\})", rf"\g<1>{version}\2", original)
    if updated != original:
        _write(path, updated)
        print(f"  updated README.md → BibTeX version {version}")
        return True
    return False


if __name__ == "__main__":
    version = get_version()
    print(f"Syncing version {version} from pyproject.toml...")
    changed = any([sync_citation(version), sync_readme(version)])
    if changed:
        print("Version sync complete — stage the modified files and re-run.")
        sys.exit(1)  # non-zero tells pre-commit files were modified
    else:
        print("All version references already in sync.")
