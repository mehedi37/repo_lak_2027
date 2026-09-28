"""Validation of the mathematical bound, scoring, and real-data protocol."""

from itertools import combinations, product
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from core import gain_bound, posterior_failures, precision, shortlist_audit, structural_signal

PAPER = Path(__file__).resolve().parents[1]


def test_structural_signal_includes_blocking_even_when_betweenness_zero():
    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    graph.add_node("isolated")
    scores = structural_signal(graph)
    assert scores.loc["a", "blocking"] == 2
    assert scores.loc["a", "betweenness"] == 0
    assert scores.loc["a", "S"] == .5
    assert scores.loc["isolated", "S"] == 0


def test_precision_bound_exhaustively_for_small_universe():
    universe = range(4)
    sets = [set(s) for s in combinations(universe, 2)]
    labels = [set(c for c, y in zip(universe, bits) if y)
              for bits in product([0, 1], repeat=4)]
    for fused, first, second in product(sets, repeat=3):
        bound = gain_bound(fused, [first, second])
        for gold in labels:
            assert precision(fused, gold) - max(precision(first, gold), precision(second, gold)) <= bound + 1e-12


def test_zero_bound_can_still_mean_worse_than_another_source():
    assert gain_bound({1, 2}, [{1, 2}, {3, 4}]) == 0
    assert precision({1, 2}, {3, 4}) < precision({3, 4}, {3, 4})


def test_scores_use_jeffreys_prior_and_reject_missing_enrolments():
    assert posterior_failures([0, 10], [10, 10]).tolist() == [.5 / 11, 10.5 / 11]
    with pytest.raises(ValueError):
        posterior_failures([0], [0])


def test_rankings_are_invariant_to_input_row_order():
    signals = pd.DataFrame({"S": [1, 1, 0, 0], "P": [.2, .4, .4, .1]}, index=list("abcd"))
    assert shortlist_audit(signals, 2) == shortlist_audit(signals.iloc[::-1], 2)


def test_constant_signal_is_explicitly_excluded():
    signals = pd.DataFrame({"S": [1, 2, 3], "P": [3, 2, 1], "E": [0, 0, 0]}, index=list("abc"))
    audit = shortlist_audit(signals, 1)
    assert audit["excluded_constant_sources"] == ["E"]
    assert "E" not in audit["lists"]


def test_local_candidates_use_all_eligible_courses_and_blocking_is_present():
    report = json.loads((PAPER / "results/real_audits.json").read_text())
    for cohort, expected in [("A", 69), ("B", 47)]:
        scores = pd.read_csv(PAPER / f"results/local_{cohort}_signals.csv")
        assert len(scores) == expected == report["local"][cohort]["n"]
        assert (scores.enrolments >= 30).all()
        assert scores.blocking.max() > 0
        assert scores.S.max() == .5
        assert scores[["S", "P"]].notna().all().all()


def test_oulad_keeps_active_dispersion_and_cohort_specific_candidate_sets():
    report = json.loads((PAPER / "results/real_audits.json").read_text())["oulad"]
    assert report["2013"]["n"] == 6
    assert report["2014"]["n"] == 7
    assert "CCC" not in report["2013"]["module_list"]
    assert np.isclose(report["2013"]["tau_weighted_vs_performance"], 13 / 15)
    assert np.isclose(report["2014"]["tau_weighted_vs_performance"], 17 / 21)
    assert set(report["2014"]["lists"]["average"]) == set(report["2014"]["lists"]["D"])
    assert set(report["2014"]["lists"]["average"]) != set(report["2014"]["lists"]["P"])
