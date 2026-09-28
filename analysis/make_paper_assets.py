"""Generate every numerical table, headline macro, and figure from study outputs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_study import CONDITIONS

PAPER = Path(__file__).resolve().parents[1]


def main():
    trials = pd.read_csv(PAPER / "results/simulation_trials.csv")
    summary = pd.read_csv(PAPER / "results/simulation_summary.csv")
    real = json.loads((PAPER / "results/real_audits.json").read_text())
    table = trials[trials.k.eq(8)].groupby("condition").mean(numeric_only=True).reindex(CONDITIONS)
    rows = []
    for condition, row in table.iterrows():
        cells = [condition] + [f"{row[m]:.2f}" for m in ["overlap", "average", "merge", "rrf", "heuristic", "best_single"]]
        rows.append(" & ".join(cells) + r" \\")
    (PAPER / "generated/simulation_rows.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for method, label in [("blocking", "Blocking factor"), ("betweenness", "Betweenness"),
                           ("S", "Structural composite (S)"), ("P", "Failure estimate (P)"),
                           ("raw_mean", "Mean of raw S and P"), ("average", "Rank average"),
                           ("merge", "Round-robin merge"), ("rrf", "Reciprocal rank fusion"),
                           ("heuristic", "Overlap heuristic")]:
        values = [real["local"][c]["analyses"]["5"]["precision_reference"][method] for c in ["A", "B"]]
        rows.append(f"{label} & {values[0]:.2f} & {values[1]:.2f}" + r" \\")
    (PAPER / "generated/local_rows.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for year, row in real["oulad"].items():
        lists = row["lists"]
        cells = [year, str(row["n"])] + [", ".join(sorted(lists[m])) for m in ["P", "D", "weighted", "average"]]
        rows.append(" & ".join(cells) + f" & {row['tau_weighted_vs_performance']:.2f}" + r" \\")
    (PAPER / "generated/oulad_rows.tex").write_text("\n".join(rows) + "\n")
    macros = []
    for prefix, condition in [("Coverage", "Coverage"), ("Shared", "Shared"), ("Bias", "Shared bias")]:
        for suffix, metric in [("Average", "average"), ("Merge", "merge"), ("Heuristic", "heuristic"), ("Overlap", "overlap")]:
            macros.append("\\newcommand{\\" + prefix + suffix + "}{" + f"{table.loc[condition, metric]:.2f}" + "}")
    gap = summary[(summary.k.eq(8)) & summary.statistic.eq("merge_minus_average")].set_index("condition")
    for prefix, condition in [("Coverage", "Coverage"), ("Shared", "Shared")]:
        row = gap.loc[condition]
        macros.append("\\newcommand{\\" + prefix + "Gap}{" + f"{row['mean']:+.2f}" + "}")
        macros.append("\\newcommand{\\" + prefix + "GapCI}{" + f"[{row['low']:+.2f},\\,{row['high']:+.2f}]" + "}")
    macros.append(r"\newcommand{\CoverageWrong}{" + f"{100 * table.loc['Coverage', 'average_chosen']:.0f}" + "}")
    macros.append(r"\newcommand{\SharedWrong}{" + f"{100 * (1 - table.loc['Shared', 'average_chosen']):.0f}" + "}")
    (PAPER / "generated/statistics.tex").write_text("% Automatically generated; do not edit.\n" + "\n".join(macros) + "\n")

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 3.05), constrained_layout=True)
    colors = ["#6d6d6d", "#126a82", "#b05621"]
    for offset, k, color, marker in zip([-.20, 0, .20], [5, 8, 12], colors, ["s", "o", "^"]):
        data = summary[(summary.k.eq(k)) & summary.statistic.eq("merge_minus_average")].set_index("condition").loc[CONDITIONS]
        mean = data["mean"].to_numpy()
        ax.errorbar(mean, np.arange(len(data)) + offset,
                    xerr=np.stack([mean - data.low, data.high - mean]),
                    fmt=marker, markersize=4, capsize=2, color=color, label=f"k = {k}", linewidth=1)
    ax.axvline(0, color="#555555", lw=.8, ls="--")
    ax.set_yticks(np.arange(len(CONDITIONS)), CONDITIONS)
    ax.invert_yaxis()
    ax.set_xlabel("Precision difference: merge minus rank average")
    ax.set_xlim(-.42, .52)
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=8)
    fig.savefig(PAPER / "figures/combiner_sensitivity.pdf", metadata={"Creator": "Matplotlib", "Author": "Anonymous", "CreationDate": None})
    fig.savefig(PAPER / "figures/combiner_sensitivity.png", dpi=200)
    plt.close(fig)
    print("Generated three tables, numerical macros, and the paired-interval figure.")


if __name__ == "__main__":
    main()
