"""Reproduce the short-paper experiments from anonymous aggregate inputs.

python analysis/run_study.py --trials 30 --workers 3
python analysis/run_study.py --real-only
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from core import (bootstrap_mean, gain_bound, percentiles, posterior_failures,
                  precision, ranking, shortlist_audit, structural_signal)
from synth.universe import CANONICAL_ASPECTS, UniverseConfig, build_universe
from synth.grades import simulate_grades
from synth.reviews import simulate_reviews

PAPER = Path(__file__).resolve().parents[1]
CONDITIONS = ["Coverage", "Mixed", "Shared", "Correlated perturbations",
              "Shuffled feedback", "Missing edges", "Shared bias"]
METHODS = ["S", "P", "E", "average", "merge", "rrf", "heuristic"]


def metrics(signals, k, gold, tie_order, source_order=None):
    audit = shortlist_audit(signals, k, source_order, tie_order)
    scores = {m: precision(v, gold) for m, v in audit["lists"].items()}
    best = max(scores[c] for c in signals)
    for method, bound in audit["bounds"].items():
        if scores[method] - best > bound + 1e-12:
            raise AssertionError("Set-theoretic precision bound violated")
    return {**scores, "overlap": audit["overlap"], "best_single": best,
            "average_chosen": int(audit["heuristic_choice"] == "average"),
            "heuristic_regret": max(scores["average"], scores["merge"]) - scores["heuristic"],
            "average_bound": audit["bounds"]["average"],
            "merge_bound": audit["bounds"]["merge"]}


def one_simulation(task):
    seed, overlap, rho = task
    config = UniverseConfig(seed=seed, n_courses=60, n_students=600,
                            n_bottlenecks=8, modality_overlap=overlap,
                            noise_correlation=rho, measurement_noise=1.0)
    universe = build_universe(config)
    graph = nx.DiGraph()
    graph.add_nodes_from(c.code for c in universe.courses)
    graph.add_edges_from(universe.prereq_edges())
    grades = simulate_grades(universe)
    reviews = simulate_reviews(universe)
    counts = grades.assign(failed=grades.status.eq("failed")).groupby("course_code").failed.agg(["sum", "count"])
    ratings = reviews.groupby("course_code")[[f"likert_{a}" for a in CANONICAL_ASPECTS]].mean().mean(axis=1)
    signals = pd.DataFrame({"S": structural_signal(graph).S,
                            "P": pd.Series(posterior_failures(counts["sum"], counts["count"]), index=counts.index),
                            "E": (5 - ratings) / 4}).sort_index()
    assert len(signals) == 60 and signals.notna().all().all()
    gold = set(universe.bottleneck_codes())
    condition = "Correlated perturbations" if rho else {0.: "Coverage", .5: "Mixed", 1.: "Shared"}[overlap]
    variants = {condition: signals}
    if overlap == 1 and rho == 0:
        rng = np.random.default_rng(seed + 900000)
        shuffled = signals.copy()
        shuffled["E"] = rng.permutation(shuffled.E.to_numpy())
        variants["Shuffled feedback"] = shuffled
        recorded = graph.copy()
        edges = list(recorded.edges())
        removed = rng.choice(len(edges), size=round(.25 * len(edges)), replace=False)
        recorded.remove_edges_from(edges[int(i)] for i in removed)
        missing = signals.copy()
        missing["S"] = structural_signal(recorded).S.reindex(signals.index)
        variants["Missing edges"] = missing
        # A score-level sensitivity condition, not a generative curriculum claim.
        noise = rng.normal(size=len(signals))
        variants["Shared bias"] = percentiles(signals).add(2 * noise, axis=0)
    rows = []
    for name, data in variants.items():
        tie_order = np.random.default_rng(seed + 800000).permutation(data.index).tolist()
        for k in [5, 8, 12]:
            row = {"condition": name, "seed": seed, "k": k,
                   **metrics(data, k, gold, tie_order)}
            if k == 8:
                order_values = [metrics(data, k, gold, tie_order, order)["merge"]
                                for order in permutations(data.columns)]
                row["merge_order_min"] = min(order_values)
                row["merge_order_max"] = max(order_values)
                tie_rng = np.random.default_rng(seed + 700000)
                tie_metrics = [metrics(data, k, gold, tie_rng.permutation(data.index).tolist())
                               for _ in range(20)]
                row["average_tie_range"] = max(t["average"] for t in tie_metrics) - min(t["average"] for t in tie_metrics)
                row["merge_tie_range"] = max(t["merge"] for t in tie_metrics) - min(t["merge"] for t in tie_metrics)
            rows.append(row)
    return rows


def real_study():
    graph = nx.DiGraph(nx.read_graphml(PAPER / "data/local_catalogue.graphml"))
    structure = structural_signal(graph)
    provenance = json.loads((PAPER / "data/local_provenance.json").read_text())
    reference = pd.read_csv(PAPER / "data/local_reference.csv").set_index("course").reference
    gold = set(reference[reference.eq(1)].index)
    report = {"local": {}, "oulad": {}, "provenance": provenance}
    local_rows = []
    for cohort in ["A", "B"]:
        counts = pd.read_csv(PAPER / f"data/local_cohort_{cohort}.csv").set_index("course")
        eligible = counts.index[(counts.enrolments >= 30) & counts.index.isin(graph)]
        counts = counts.loc[eligible].sort_index()
        signals = pd.DataFrame({"S": structure.S.reindex(counts.index),
            "P": pd.Series(posterior_failures(counts.failures, counts.enrolments), index=counts.index)})
        assert signals.notna().all().all()
        gold_here = gold & set(signals.index)
        studies = {}
        for k in [3, 5, 10]:
            audit = shortlist_audit(signals, k)
            raw = signals.mean(axis=1)
            audit["lists"]["raw_mean"] = ranking(raw)[:k]
            audit["bounds"]["raw_mean"] = gain_bound(audit["lists"]["raw_mean"], [audit["lists"][m] for m in signals])
            audit["lists"]["blocking"] = ranking(structure.blocking.reindex(signals.index))[:k]
            audit["lists"]["betweenness"] = ranking(structure.betweenness.reindex(signals.index))[:k]
            audit["precision_reference"] = {m: precision(v, gold_here) for m, v in audit["lists"].items()}
            audit["eligible_reference_courses"] = len(gold_here)
            for m in ["average", "merge", "rrf", "heuristic", "raw_mean"]:
                local_rows.append({"cohort": cohort, "k": k, "method": m,
                                   "n": len(signals), "bound": audit["bounds"][m],
                                   "p_reference": audit["precision_reference"][m]})
            studies[str(k)] = audit
        # Sensitivity to near-tied structural scores without using the reference.
        rng = np.random.default_rng(981)
        default = shortlist_audit(signals, 5)
        tie_changes = {m: [] for m in ["S", "P", "average", "merge"]}
        tie_precision = {m: [] for m in tie_changes}
        for _ in range(200):
            audit = shortlist_audit(signals, 5, tie_order=rng.permutation(signals.index).tolist())
            for m in tie_changes:
                tie_changes[m].append(len(set(audit["lists"][m]) ^ set(default["lists"][m])) / 10)
                tie_precision[m].append(precision(audit["lists"][m], gold_here))
        order_precision = [precision(shortlist_audit(signals, 5, source_order=order)["lists"]["merge"], gold_here)
                           for order in permutations(signals.columns)]
        report["local"][cohort] = {"n": len(signals), "students": provenance["cohorts"][cohort]["students"],
            "excluded_no_eligible_grades": len(graph) - len(signals), "analyses": studies,
            "max_tie_replacement_fraction": {m: max(v) for m, v in tie_changes.items()},
            "tie_precision_range": {m: [min(v), max(v)] for m, v in tie_precision.items()},
            "merge_source_order_precision_range": [min(order_precision), max(order_precision)]}
        output = signals.join(structure[["blocking", "betweenness"]]).join(counts)
        output["reference"] = output.index.isin(gold).astype(int)
        output.to_csv(PAPER / f"results/local_{cohort}_signals.csv")
    for year in [2013, 2014]:
        aggregate = pd.read_csv(PAPER / f"data/oulad_{year}.csv").set_index("course")
        signals = pd.DataFrame({"P": pd.Series(posterior_failures(aggregate.failures, aggregate.enrolments), index=aggregate.index),
                               "D": aggregate.sd_mark / aggregate.mean_mark}).dropna().sort_index()
        ranks = percentiles(signals)
        scaled = (signals - signals.min()) / (signals.max() - signals.min())
        weighted = (2 * scaled.P + scaled.D) / 3
        audit = shortlist_audit(signals, 2)
        audit["lists"]["weighted"] = ranking(weighted)[:2]
        audit["bounds"]["weighted"] = gain_bound(audit["lists"]["weighted"], [audit["lists"][m] for m in signals])
        audit["tau_rank_average_vs_performance"] = float(kendalltau(ranks.mean(axis=1), signals.P).statistic)
        audit["tau_weighted_vs_performance"] = float(kendalltau(weighted, signals.P).statistic)
        audit["module_list"] = signals.index.tolist()
        audit["registrations"] = int(aggregate.enrolments.sum())
        report["oulad"][str(year)] = audit
        signals.assign(rank_average=ranks.mean(axis=1), weighted=weighted).to_csv(PAPER / f"results/oulad_{year}_signals.csv")
    (PAPER / "results/real_audits.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(local_rows).to_csv(PAPER / "results/local_sensitivity.csv", index=False)
    print("Real-data audits regenerated from unfiltered aggregate inputs.", flush=True)


def summarize_simulations(data):
    rows = []
    for (condition, k), frame in data.groupby(["condition", "k"], sort=False):
        for method in METHODS + ["best_single", "heuristic_regret"]:
            rows.append({"condition": condition, "k": int(k), "statistic": method,
                         **bootstrap_mean(frame[method])})
        for first, second in [("merge", "average"), ("heuristic", "average"), ("heuristic", "merge")]:
            rows.append({"condition": condition, "k": int(k), "statistic": f"{first}_minus_{second}",
                         **bootstrap_mean(frame[first] - frame[second])})
    pd.DataFrame(rows).to_csv(PAPER / "results/simulation_summary.csv", index=False)
    primary = data[data.k.eq(8)]
    display = primary.groupby("condition").agg(
        n=("seed", "size"), average=("average", "mean"), merge=("merge", "mean"),
        rrf=("rrf", "mean"), heuristic=("heuristic", "mean"), best_single=("best_single", "mean"),
        overlap=("overlap", "mean"), average_chosen=("average_chosen", "mean"), regret=("heuristic_regret", "mean"))
    print(display.reindex(CONDITIONS).round(3).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--simulation-only", action="store_true")
    args = parser.parse_args()
    (PAPER / "results").mkdir(exist_ok=True)
    if not args.simulation_only:
        real_study()
    if not args.real_only:
        tasks = [(43000 + trial, overlap, rho) for trial in range(args.trials)
                 for overlap, rho in [(0., 0.), (.5, 0.), (1., 0.), (1., .8)]]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, results in enumerate(pool.map(one_simulation, tasks)):
                rows.extend(results)
                if (i + 1) % 10 == 0:
                    print(f"Simulated {i + 1}/{len(tasks)} base curricula", flush=True)
        data = pd.DataFrame(rows)
        data.to_csv(PAPER / "results/simulation_trials.csv", index=False)
        summarize_simulations(data)
        config = {"trials_per_condition": args.trials, "seeds": [43000, 43000 + args.trials - 1],
                  "base_curricula": len(tasks), "conditions": CONDITIONS, "n_courses": 60,
                  "n_students": 600, "n_planted": 8, "k": [5, 8, 12],
                  "reviews_per_course": 30, "bootstrap_replicates": 5000,
                  "source_order": ["S", "P", "E"], "source_order_sensitivity": 6,
                  "tie_order_sensitivity": 20, "numpy": np.__version__, "pandas": pd.__version__,
                  "networkx": nx.__version__}
        (PAPER / "results/simulation_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
