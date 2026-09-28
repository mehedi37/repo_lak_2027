"""Shared, label-free scoring and shortlist operations for all experiments."""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd


def structural_signal(graph):
    """Compute both terms on the full catalogue, without rounding."""
    between = pd.Series(nx.betweenness_centrality(graph), dtype=float)
    blocking = pd.Series({c: len(nx.descendants(graph, c)) for c in graph}, dtype=float)
    normalized = blocking / blocking.max() if blocking.max() > 0 else blocking * 0
    return pd.DataFrame({"S": (between + normalized) / 2,
                         "blocking": blocking, "betweenness": between})


def posterior_failures(failures, enrolments):
    """Jeffreys Beta(1/2, 1/2) posterior mean; missing evidence is not zero."""
    failures = np.asarray(failures, dtype=float)
    enrolments = np.asarray(enrolments, dtype=float)
    if np.any(enrolments <= 0) or np.any(failures < 0) or np.any(failures > enrolments):
        raise ValueError("Failure counts must lie between zero and positive enrolment counts")
    return (failures + 0.5) / (enrolments + 1)


def percentiles(signals):
    """Average percentile ranks on the same candidate universe for all sources.

    Constant sources are excluded explicitly by the caller. Dividing all ranks
    by n gives equal rank increments and avoids a tie-dependent min/max rescale.
    """
    if signals.index.has_duplicates or not np.isfinite(signals.to_numpy()).all():
        raise ValueError("Signals need unique course identifiers and finite complete values")
    return (signals.rank(method="average") - 0.5) / len(signals)


def ranking(score, tie_order=None):
    """Descending score with a declared, score-independent secondary ordering."""
    if tie_order is None:
        tie_order = sorted(score.index)
    if set(tie_order) != set(score.index) or len(tie_order) != len(score):
        raise ValueError("Tie order must be a permutation of the candidate universe")
    return score.reindex(tie_order).sort_values(ascending=False, kind="stable").index.tolist()


def merged_list(ranks, k, source_order=None, tie_order=None):
    columns = list(ranks) if source_order is None else list(source_order)
    lists = [ranking(ranks[c], tie_order) for c in columns]
    picked = []
    for position in range(len(ranks)):
        for ordered in lists:
            course = ordered[position]
            if course not in picked:
                picked.append(course)
                if len(picked) == k:
                    return picked
    raise ValueError("Insufficient candidates")


def gain_bound(fused, singles):
    """Upper bound on P@k(fused) minus the best single source's P@k."""
    fused = set(fused)
    if not fused or not singles or any(len(set(s)) != len(fused) for s in singles):
        raise ValueError("All shortlists must have the same positive size")
    return min(len(fused ^ set(s)) for s in singles) / (2 * len(fused))


def shortlist_audit(signals, k, source_order=None, tie_order=None):
    if not 0 < k <= len(signals):
        raise ValueError("Invalid shortlist size")
    dead = [c for c in signals if signals[c].nunique() <= 1]
    live = signals.drop(columns=dead)
    if live.shape[1] < 2:
        raise ValueError("At least two nonconstant sources are needed for combination")
    ranks = percentiles(live)
    singles = {c: ranking(ranks[c], tie_order)[:k] for c in ranks}
    overlap = np.mean([len(set(singles[a]) & set(singles[b])) / k
                       for a, b in combinations(ranks, 2)])
    average = ranking(ranks.mean(axis=1), tie_order)[:k]
    merge = merged_list(ranks, k, source_order, tie_order)
    descending = len(ranks) + 1 - live.rank(method="average")
    rrf_score = (1 / (60 + descending)).sum(axis=1)
    rrf = ranking(rrf_score, tie_order)[:k]
    choice = "average" if overlap > k / len(signals) else "merge"
    lists = {**singles, "average": average, "merge": merge, "rrf": rrf,
             "heuristic": average if choice == "average" else merge}
    return {"n": len(signals), "k": k, "overlap": float(overlap),
            "chance_overlap": k / len(signals), "heuristic_choice": choice,
            "excluded_constant_sources": dead, "lists": lists,
            "bounds": {m: gain_bound(lists[m], list(singles.values()))
                       for m in ["average", "merge", "rrf", "heuristic"]}}


def precision(shortlist, reference):
    return len(set(shortlist) & set(reference)) / len(shortlist)


def bootstrap_mean(values, draws=5000, seed=71019):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975))}
