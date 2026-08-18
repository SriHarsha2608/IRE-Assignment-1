from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(dest)


def ensure_file(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.info("raw file already present (skip download): %s", dest)
        return dest
    _fetch(url, dest)
    return dest


def fingerprint(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"bytes": st.st_size, "mtime": int(st.st_mtime)}