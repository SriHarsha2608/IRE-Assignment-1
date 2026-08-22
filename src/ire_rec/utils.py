from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


def unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)


def find_file(root: Path, rel: str) -> Path | None:
    # Sort candidates by path string so the match is deterministic even when
    # multiple same-named files exist under different subdirectories.
    candidates = sorted(
        (c for c in root.rglob(rel) if "__MACOSX" not in str(c)),
        key=str,
    )
    return candidates[0] if candidates else None


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)