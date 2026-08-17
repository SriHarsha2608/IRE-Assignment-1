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
    for candidate in root.rglob(rel):
        if "__MACOSX" not in str(candidate):
            return candidate
    return None


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)