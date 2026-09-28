"""
Review / feedback generation for the synthetic benchmark, tuned for
Aspect-Based Sentiment Analysis (ABSA).

CRITICAL DESIGN POINT: reviews are generated from each course's *planted*
latent aspect profile (bottleneck aspect + severity), NOT from computed fail
rates. This makes the sentiment modality genuinely independent of the
performance modality, avoiding the circularity present in earlier feedback
generation. Fusion experiments can therefore fairly test whether combining
independent modalities helps.

Output schema:
    student_id, course_code,
    likert_prerequisite_gap, likert_pacing, likert_instruction_quality,
    likert_assessment, likert_content_relevance,
    text, sentiment_label, source
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .universe import CANONICAL_ASPECTS, LatentUniverse

# Per-aspect sentence banks indexed by tone bucket: negative / neutral / positive.
_PHRASES: Dict[str, Dict[str, List[str]]] = {
    "prerequisite_gap": {
        "neg": [
            "I felt completely unprepared; the course assumed background I never had.",
            "The prerequisites were not enough — I was lost from the first week.",
            "Too much prior knowledge was assumed and it was never revisited.",
        ],
        "neu": [
            "Some prior topics were assumed but mostly recoverable with effort.",
            "A quick refresher on the basics would have helped me settle in.",
        ],
        "pos": [
            "The course built smoothly on what we already knew.",
            "Prior material was reviewed well, so the start felt comfortable.",
        ],
    },
    "pacing": {
        "neg": [
            "The pace was brutal; deadlines piled up faster than I could keep up.",
            "We rushed through everything and there was no time to absorb it.",
            "The workload was relentless and poorly spaced across the term.",
        ],
        "neu": [
            "The pace was a bit fast at times but generally manageable.",
            "A few weeks were heavy, though most of the term was reasonable.",
        ],
        "pos": [
            "The pacing was steady and the workload felt well distributed.",
            "Deadlines were spaced sensibly, leaving time to learn.",
        ],
    },
    "instruction_quality": {
        "neg": [
            "The teaching was unclear and I mostly had to learn on my own.",
            "Lectures were disorganised and hard to follow.",
            "Explanations were rushed and rarely answered our questions.",
        ],
        "neu": [
            "Instruction was acceptable but could have used clearer examples.",
            "The teaching was fine, though some topics needed more depth.",
        ],
        "pos": [
            "The instructor explained difficult ideas clearly and patiently.",
            "Teaching was excellent — concepts were made genuinely intuitive.",
        ],
    },
    "assessment": {
        "neg": [
            "The exams were unfair and tested material we never covered.",
            "Grading felt arbitrary and the assessments were disconnected from class.",
            "Assignments were ambiguous and marking was inconsistent.",
        ],
        "neu": [
            "Assessments were okay, though a couple of questions were unclear.",
            "Grading was reasonable but the rubric could be more transparent.",
        ],
        "pos": [
            "Assessments were fair and clearly aligned with what we learned.",
            "The exams were well designed and the grading was transparent.",
        ],
    },
    "content_relevance": {
        "neg": [
            "Much of the content felt outdated and irrelevant to practice.",
            "The material seemed disconnected from anything useful.",
        ],
        "neu": [
            "The content was mostly relevant with a few dated sections.",
            "Some topics were useful, others felt like filler.",
        ],
        "pos": [
            "The syllabus felt current and connected to real-world systems.",
            "Content relevance was high and mapped to modern practice.",
        ],
    },
}


def _likert_from_score(score: float) -> int:
    """Map a [-1, 1] aspect sentiment to a 1-5 Likert rating."""
    # -1 -> 1, 0 -> 3, +1 -> 5
    return int(np.clip(round((score + 1.0) * 2.0 + 1.0), 1, 5))


def _tone_bucket(score: float) -> str:
    if score <= -0.34:
        return "neg"
    if score >= 0.34:
        return "pos"
    return "neu"


def _overall_label(mean_score: float) -> str:
    if mean_score <= -0.2:
        return "negative"
    if mean_score >= 0.2:
        return "positive"
    return "neutral"


def simulate_reviews(
    universe: LatentUniverse,
    reviews_per_course: int = 30,
) -> pd.DataFrame:
    """Generate ``reviews_per_course`` reviews per course.

    Each course has a latent per-aspect sentiment baseline. The planted
    bottleneck aspect is pushed strongly negative in proportion to severity;
    all other aspects stay mildly positive with per-review noise.
    """
    rng = np.random.default_rng(universe.config.seed + 3003)
    students_by_cohort = {}
    for s in universe.students:
        students_by_cohort.setdefault(s.cohort, []).append(s.student_id)
    all_students = [s.student_id for s in universe.students]

    rows: List[Dict[str, object]] = []
    for course in sorted(universe.courses, key=lambda c: c.code):
        # Baseline per-aspect sentiment in [-1, 1]; mildly positive by
        # default, shifted by this course's experiential measurement noise.
        # Without that shift every non-bottleneck course would share one
        # identical baseline (sd 0.05 Likert after averaging 30 reviews), so
        # any course with negative sentiment at all would be perfectly
        # separable and the sentiment modality would be a noiseless oracle.
        # The shift is what gives the modality a realistic noise floor.
        #
        # Sign: the noise vector is oriented so that a positive draw means
        # "looks more bottleneck-like", matching the structural and
        # performance signals. Sentiment runs the other way, since a
        # bottleneck reads as *lower* satisfaction, so the draw is subtracted
        # here. Without this the shared factor would raise two signals and
        # lower the third, and the cross-modal error correlation the config
        # asks for would come out negative.
        level = 0.35 - course.noise("experiential")
        baseline: Dict[str, float] = {a: level for a in CANONICAL_ASPECTS}
        # Sentiment degrades in proportion to how strongly the course
        # expresses through the experiential modality: broadly across aspects,
        # and hardest on the planted aspect. At modality_overlap = 0 only
        # experiential-type bottlenecks have e_exp > 0, so performance and
        # structural bottlenecks keep neutral-positive feedback and the
        # sentiment modality does NOT detect them.
        e_exp = course.expresses("experiential")
        if e_exp > 0.0:
            sev = course.bottleneck_severity
            for a in CANONICAL_ASPECTS:
                baseline[a] = level + e_exp * (-0.80 - 0.25 * sev)
            if course.bottleneck_aspect:
                baseline[course.bottleneck_aspect] = (
                    level + e_exp * (-0.95 - 0.35 * sev)
                )

        reviewers = rng.choice(
            all_students,
            size=min(reviews_per_course, len(all_students)),
            replace=False,
        )
        for sid in reviewers:
            aspect_scores: Dict[str, float] = {}
            for aspect in CANONICAL_ASPECTS:
                noise = rng.normal(0.0, 0.25)
                score = float(np.clip(baseline[aspect] + noise, -1.0, 1.0))
                aspect_scores[aspect] = score

            # Compose a short review: focus a sentence on the most extreme aspect.
            worst_aspect = min(aspect_scores, key=aspect_scores.get)
            best_aspect = max(aspect_scores, key=aspect_scores.get)
            sentences = [
                _pick(rng, worst_aspect, aspect_scores[worst_aspect]),
                _pick(rng, best_aspect, aspect_scores[best_aspect]),
            ]
            text = " ".join(sentences)
            mean_score = float(np.mean(list(aspect_scores.values())))

            row: Dict[str, object] = {
                "student_id": sid,
                "course_code": course.code,
            }
            for aspect in CANONICAL_ASPECTS:
                row[f"likert_{aspect}"] = _likert_from_score(aspect_scores[aspect])
            row["text"] = text
            row["sentiment_label"] = _overall_label(mean_score)
            row["source"] = "synthetic_template"
            rows.append(row)

    cols = (
        ["student_id", "course_code"]
        + [f"likert_{a}" for a in CANONICAL_ASPECTS]
        + ["text", "sentiment_label", "source"]
    )
    return pd.DataFrame(rows, columns=cols)


def _pick(rng: np.random.Generator, aspect: str, score: float) -> str:
    bucket = _tone_bucket(score)
    options = _PHRASES[aspect][bucket]
    return options[int(rng.integers(0, len(options)))]
