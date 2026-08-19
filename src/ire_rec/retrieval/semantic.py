from __future__ import annotations

from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import polars as pl

from ..dataio import ARTICLE_ID


def load_embeddings(
    emb_dir: Path, name: str, catalog: set[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load ``{name}.npy`` + ``{name}_ids.parquet``, optionally restricted to a catalog.

    ``catalog`` is the set of article ids in the dataset's articles frame; rows
    whose id is not in the catalog are dropped so the returned matrix is aligned
    with ``ids`` (row i == article ``ids[i]``) and covers only retrievable
    articles.

    The store is validated: the id list must match the matrix row count and the
    ids must be unique (duplicates would silently corrupt ``id_to_row`` lookups).
    """
    mat = np.load(emb_dir / f"{name}.npy")
    ids = pl.read_parquet(emb_dir / f"{name}_ids.parquet")[ARTICLE_ID].to_list()
    if len(ids) != mat.shape[0]:
        raise ValueError(
            f"{name}: {len(ids)} article ids but embedding matrix has "
            f"{mat.shape[0]} rows (mismatched {name}_ids.parquet / {name}.npy)"
        )
    counts = Counter(ids)
    dups = {i: c for i, c in counts.items() if c > 1}
    if dups:
        raise ValueError(
            f"{name}: {len(dups)} duplicate article ids in embedding file "
            f"(e.g. {sorted(dups)[:5]}); ids must be unique"
        )
    if catalog is not None:
        keep = np.asarray(
            [i for i, aid in enumerate(ids) if aid in catalog], dtype=np.int64
        )
        if keep.size < len(ids):
            mat = mat[keep]
            ids = [ids[int(i)] for i in keep]
    return np.asarray(mat, dtype=np.float32), ids


def build_ann_index(mat: np.ndarray, normalize: bool = True) -> "faiss.IndexFlatIP":
    """Build a FAISS flat inner-product index over ``mat`` without mutating it.

    Documents: ``docs = raw.copy()`` (the caller's ``mat`` is never modified —
    it is reused for mean-pooling user representations).  When ``normalize`` is
    true, ``docs`` are L2-normalized row-wise so ``IndexFlatIP`` scores equal
    cosine similarity; otherwise the raw inner product is used.

    The index rows stay aligned with the caller's ``ids`` list (row ``i`` is
    article ``ids[i]``). Exact flat search is fine for the MIND/EB-NeRD
    catalogue sizes (~65k x <=768); HNSW/IVF are the 10x-scale alternatives.
    """
    docs = np.ascontiguousarray(mat, dtype=np.float32).copy()
    if normalize:
        faiss.normalize_L2(docs)
    index = faiss.IndexFlatIP(int(docs.shape[1]))
    index.add(docs)
    return index


def mean_pool_user_vector(
    history: list[str],
    id_to_row: dict[str, int],
    mat: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray | None, int]:
    """Mean-pool the *raw* embeddings of ``history`` articles in the index.

    Returns ``(None, 0)`` for an empty / fully-uncovered history (cold start).
    A pooled mean with zero norm (e.g. mutually cancelling clicks) also returns
    ``None`` so an undefined zero query is never sent to FAISS; the impression
    then gets empty candidates (recall 0), the same well-defined behaviour as a
    cold start.

    When ``normalize`` is true the pooled mean is L2-normalized (unit-norm
    query against a cosine index); otherwise the raw mean is returned (raw
    inner-product retrieval), keeping document and query normalization
    semantics consistent.
    """
    rows = [id_to_row[aid] for aid in history if aid in id_to_row]
    if not rows:
        return None, 0
    vec = mat[rows].mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return None, len(rows)
    if normalize:
        vec = vec / norm
    return np.asarray(vec, dtype=np.float32), len(rows)


def search(
    user_vec: np.ndarray, index: "faiss.IndexFlatIP", top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, row_indices)`` for the top-``top_k`` index rows.

    FAISS pads missing results with -1; those are dropped so ``indices`` maps
    cleanly back to article ids.
    """
    scores, idx = index.search(user_vec[None, :], top_k)
    valid = idx[0] >= 0
    return scores[0][valid].astype(np.float64), idx[0][valid]