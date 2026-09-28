"""
Grade simulation for the synthetic curriculum benchmark.

Each (student, course) enrolment is turned into a numeric mark and a letter
grade using a noisy latent-trait model:

    latent = w_ability * ability
           + w_prereq  * prereq_performance
           + w_consc   * conscientiousness
           - w_diff    * course_difficulty
           - w_bneck   * bottleneck_penalty
           + noise

``prereq_performance`` is the student's mean normalised mark on the course's
prerequisites, so planted ``prerequisite_gap`` bottlenecks genuinely depress
downstream outcomes. This produces a real causal signal that the OULAD-style
causal validation can recover.

Output is a tidy enrolment table in the local-university grade schema so it
drops straight into ``LocalGradeLoader``/dashboard pipelines.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .universe import Course, LatentUniverse, Student

# Letter-grade thresholds on a 0-100 mark scale (a common university grading scheme).
# (letter, grade_point, lower_bound_inclusive)
GRADE_BANDS: List[Tuple[str, float, float]] = [
    ("A+", 4.00, 80.0),
    ("A", 3.75, 75.0),
    ("A-", 3.50, 70.0),
    ("B+", 3.25, 65.0),
    ("B", 3.00, 60.0),
    ("B-", 2.75, 55.0),
    ("C+", 2.50, 50.0),
    ("C", 2.25, 45.0),
    ("D", 2.00, 40.0),
    ("F", 0.00, 0.0),
]

PASS_MARK = 40.0

# Latent-model weights.
_W_ABILITY = 42.0
_W_PREREQ = 24.0
_W_CONSC = 14.0
_W_DIFF = 30.0
_W_BNECK = 26.0
_INTERCEPT = 48.0
_NOISE_SD = 7.0


def _mark_to_grade(mark: float) -> Tuple[str, float]:
    for letter, gp, lo in GRADE_BANDS:
        if mark >= lo:
            return letter, gp
    return "F", 0.0


def _topological_order(universe: LatentUniverse) -> List[Course]:
    """Courses ordered by semester then code (semester order is a valid topo
    order because prerequisites only point backward in time)."""
    return sorted(universe.courses, key=lambda c: (c.semester, c.code))


def simulate_grades(universe: LatentUniverse) -> pd.DataFrame:
    """Simulate a full grade table for every (student, course) enrolment.

    Returns a DataFrame with the local-university grade schema columns:
        student_id, course_code, course_name, letter_grade, grade_points,
        credits, semester_num, status, is_backlog, cohort, mark
    """
    rng = np.random.default_rng(universe.config.seed + 1001)
    ordered = _topological_order(universe)
    by_code = universe.course_by_code

    rows: List[Dict[str, object]] = []
    # student_id -> course_code -> normalised mark in [0, 1]
    norm_marks: Dict[str, Dict[str, float]] = {s.student_id: {} for s in universe.students}

    for student in universe.students:
        smarks = norm_marks[student.student_id]
        for course in ordered:
            # Prerequisite performance: mean of already-taken prereq marks.
            if course.prerequisites:
                prior = [smarks[p] for p in course.prerequisites if p in smarks]
                prereq_perf = float(np.mean(prior)) if prior else 0.5
            else:
                prereq_perf = 0.5  # neutral when no prerequisites

            bottleneck_penalty = 0.0
            e_perf = course.expresses("performance")
            if e_perf > 0.0:
                # Grades are depressed in proportion to how strongly this
                # bottleneck expresses through the performance modality. At
                # modality_overlap = 0 only performance-type bottlenecks have
                # e_perf > 0, which keeps the modalities independent and
                # reproduces the original controlled benchmark.
                if course.bottleneck_aspect == "prerequisite_gap":
                    bottleneck_penalty = (
                        course.bottleneck_severity * (1.0 - prereq_perf) * e_perf
                    )
                else:
                    bottleneck_penalty = course.bottleneck_severity * 0.6 * e_perf

            latent = (
                _INTERCEPT
                + _W_ABILITY * (student.ability - 0.5)
                + _W_PREREQ * (prereq_perf - 0.5)
                + _W_CONSC * (student.conscientiousness - 0.5)
                - _W_DIFF * course.difficulty
                - _W_BNECK * bottleneck_penalty
            )
            mark = float(latent + rng.normal(0.0, _NOISE_SD))
            mark = float(np.clip(mark, 0.0, 100.0))

            letter, gp = _mark_to_grade(mark)
            status = "passed" if mark >= PASS_MARK else "failed"
            smarks[course.code] = mark / 100.0

            rows.append(
                {
                    "student_id": student.student_id,
                    "course_code": course.code,
                    "course_name": course.title,
                    "letter_grade": letter,
                    "grade_points": gp,
                    "credits": course.credits,
                    "semester_num": course.semester,
                    "status": status,
                    "is_backlog": status == "failed",
                    "cohort": student.cohort,
                    "mark": round(mark, 2),
                }
            )

    df = pd.DataFrame(rows)
    return df


def fail_rates(grades: pd.DataFrame) -> pd.DataFrame:
    """Per-course fail rate, useful for quick validation and logging."""
    g = grades.copy()
    g["failed"] = (g["status"] == "failed").astype(int)
    out = (
        g.groupby("course_code")
        .agg(n=("student_id", "size"), fail_rate=("failed", "mean"))
        .reset_index()
        .sort_values("fail_rate", ascending=False)
    )
    return out
