from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# Files larger than this are fingerprinted from a head+tail sample instead of
# being fully hashed, to keep re-fingerprinting fast on the 11GB-laptop setup.
LARGE_THRESHOLD = 256 << 20  # 256 MiB
HEAD_TAIL_BYTES = 64 << 20  # 64 MiB sampled from each end


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_head_tail(path: Path, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(HEAD_TAIL_BYTES))
        if size > 2 * HEAD_TAIL_BYTES:
            f.seek(-HEAD_TAIL_BYTES, 2)
            h.update(f.read(HEAD_TAIL_BYTES))
    return h.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    """Content-based fingerprint so a re-downloaded identical file is NOT
    mistaken for 'changed' just because its mtime differs (the previous
    bytes+mtime fingerprint triggered needless rebuilds on mtime change).

    Small files (< ``LARGE_THRESHOLD``) are fully sha256-hashed. Very large
    files are hashed from their first+last ``HEAD_TAIL_BYTES`` plus their byte
    size, which is fast and still catches content drift / re-downloads.
    """
    path = Path(path)
    st = path.stat()
    size = st.st_size
    if size <= LARGE_THRESHOLD:
        return {"size": size, "sha256": _sha256_of_file(path)}
    return {
        "size": size,
        "sha256": _sha256_head_tail(path, size),
        "partial": True,
    }


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