"""Aggregate public OULAD files, without exporting student identifiers."""

import argparse
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing the official OULAD CSVs")
    args = parser.parse_args()
    info = pd.read_csv(args.input / "studentInfo.csv")
    marks = pd.read_csv(args.input / "studentAssessment.csv")
    assessments = pd.read_csv(args.input / "assessments.csv")
    marks = marks.merge(assessments[["id_assessment", "code_module", "code_presentation"]],
                        on="id_assessment", validate="many_to_one")
    marks["score"] = pd.to_numeric(marks.score, errors="coerce")
    for year in [2013, 2014]:
        cohort = info[info.code_presentation.str.startswith(str(year))]
        counts = cohort.assign(failed=cohort.final_result.isin(["Fail", "Withdrawn"])).groupby("code_module").agg(
            failures=("failed", "sum"), enrolments=("failed", "size"))
        assessed = marks[marks.code_presentation.str.startswith(str(year))].groupby("code_module").score.agg(["mean", "std", "count"])
        data = counts.join(assessed).rename(columns={"count": "assessment_records", "mean": "mean_mark", "std": "sd_mark"})
        data.index.name = "course"
        data.to_csv(PAPER / f"data/oulad_{year}.csv")
        print(f"OULAD {year}: {len(data)} modules, {int(counts.enrolments.sum())} registrations")


if __name__ == "__main__":
    main()
