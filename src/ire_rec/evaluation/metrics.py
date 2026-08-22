from __future__ import annotations

from collections import Counter

import numpy as np


def auc_score(scores: list[float], labels: list[int]):
    """Binary AUC via the Mann-Whitney / rank formulation with average ranks
    for tied scores. Returns ``None`` when there are no positives or no
    negatives (metric undefined)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = scores.argsort(kind="mergesort")  # ascending
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    n = len(sorted_scores)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mrr_from_ranked(ranked_labels: list[int]):
    """MRR from labels in deterministic ranked order (1-based rank of the
    first clicked article). Returns ``None`` when there is no click."""
    rl = np.asarray(ranked_labels, dtype=int)
    idx = np.where(rl == 1)[0]
    if len(idx) == 0:
        return None
    return 1.0 / (int(idx[0]) + 1)


def ndcg_at_k_from_ranked(ranked_labels: list[int], k: int):
    """Binary-gain nDCG@k over the ranked inview list. Returns ``None`` when
    there are zero positives. Handles ``len < k`` naturally."""
    rl = np.asarray(ranked_labels, dtype=int)
    if int(rl.sum()) == 0:
        return None
    L = len(rl)
    eff = min(k, L)
    gains = rl[:eff].astype(float)
    disc = np.log2(np.arange(1, eff + 1) + 1.0)
    dcg = float(np.sum(gains / disc))
    n_pos = int(rl.sum())
    ideal_pos = min(n_pos, k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(1, ideal_pos + 1) + 1.0)))
    return dcg / idcg


def intra_list_diversity(categories: list[str]) -> float:
    """Intra-list diversity over the recommended (candidate) list.

    ``1 - sum_c n_c(n_c-1) / (L(L-1))`` where ``L`` is the list length.
    ``L < 2`` -> ``0.0``. Null/missing categories must be passed in as a
    deterministic bucket (``"__UNKNOWN__"``) by the caller; they are kept.
    """
    cats = list(categories)
    L = len(cats)
    if L < 2:
        return 0.0
    counts = Counter(cats)
    same = sum(c * (c - 1) for c in counts.values())
    denom = L * (L - 1)
    return 1.0 - same / denom


def novelty_for_ids(ids: list[str], p_lookup: dict[str, float]) -> float:
    """Mean self-information novelty ``-log2(p(a))`` over a recommended list.

    ``p_lookup[a]`` must already include Laplace smoothing
    ``(pop(a)+1) / sum_b(pop(b)+1)``. Articles absent from the lookup (should
    not happen for validated ids) are skipped. An empty list returns ``0.0``
    (no recommendation -> no novelty contribution)."""
    if not ids:
        return 0.0
    vals = []
    for a in ids:
        p = p_lookup.get(a)
        if p is None or p <= 0:
            continue
        vals.append(-np.log2(p))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def bootstrap_mean_ci(values, bootstrap_runs: int, seed: int, chunk_size: int = 100) -> dict:
    """Bootstrap 95% CI for the mean of per-impression metric values.

    Bootstrap sample means are computed in CHUNKS of ``chunk_size`` runs rather
    than by allocating the full ``(bootstrap_runs, n)`` index matrix at once.
    This keeps memory bounded at ~``chunk_size * n`` int64 + float64 instead of
    ``bootstrap_runs * n``, which matters at MIND/EB-NeRD scale (n in the
    hundreds of thousands, ~15-20 such allocations per dataset/split/method).

    The RNG is the same ``np.random.default_rng(seed)`` and the integer indices
    are drawn in the same sequential order as a single full ``size=(runs, n)``
    call, so results are bit-for-bit identical to the unchunked implementation.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    n = values.size
    sample_means = np.empty(bootstrap_runs, dtype=float)
    offset = 0
    remaining = bootstrap_runs
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        idx = rng.integers(0, n, size=(chunk, n))
        sample_means[offset : offset + chunk] = values[idx].mean(axis=1)
        offset += chunk
        remaining -= chunk
    lo = float(np.percentile(sample_means, 2.5))
    hi = float(np.percentile(sample_means, 97.5))
    return {"value": float(values.mean()), "ci_low": lo, "ci_high": hi}


def bootstrap_coverage_ci(
    recommended_sets: list[set],
    catalog_size: int,
    bootstrap_runs: int,
    seed: int,
) -> dict:
    """Bootstrap 95% CI for catalog coverage (set-level, NOT per-impression
    averaged). Each bootstrap sample resamples impressions with replacement,
    recomputes the union of recommended ids, and takes the ratio over the
    catalog size."""
    if not recommended_sets or catalog_size == 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    n = len(recommended_sets)
    ratios = np.empty(bootstrap_runs, dtype=float)
    for b in range(bootstrap_runs):
        sel = rng.integers(0, n, size=n)
        u = set().union(*(recommended_sets[i] for i in sel))
        ratios[b] = len(u) / catalog_size
    base = len(set().union(*recommended_sets)) / catalog_size
    return {
        "value": float(base),
        "ci_low": float(np.percentile(ratios, 2.5)),
        "ci_high": float(np.percentile(ratios, 97.5)),
    }


def bootstrap_coverage_ci_cats(
    recommended_cat_sets: list[set],
    catalog_cat_size: int,
    bootstrap_runs: int,
    seed: int,
) -> dict:
    if not recommended_cat_sets or catalog_cat_size == 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    n = len(recommended_cat_sets)
    ratios = np.empty(bootstrap_runs, dtype=float)
    for b in range(bootstrap_runs):
        sel = rng.integers(0, n, size=n)
        u = set().union(*(recommended_cat_sets[i] for i in sel))
        ratios[b] = len(u) / catalog_cat_size
    base = len(set().union(*recommended_cat_sets)) / catalog_cat_size
    return {
        "value": float(base),
        "ci_low": float(np.percentile(ratios, 2.5)),
        "ci_high": float(np.percentile(ratios, 97.5)),
    }
