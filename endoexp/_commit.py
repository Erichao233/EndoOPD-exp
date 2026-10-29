"""Commit of this source tree.

`git archive` substitutes the placeholder below (see `.gitattributes`), so a
bundle extracted without `.git` still knows which commit it came from.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_ARCHIVED = "$Format:%H$"


def code_commit() -> str:
    if not _ARCHIVED.startswith("$"):
        return _ARCHIVED
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return result.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
