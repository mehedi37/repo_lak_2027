"""
Latent universe construction for the synthetic curriculum benchmark.

Everything downstream (grades, VLE clicks, reviews) is derived from the
structures sampled here, so this module is the single source of ground truth.

Latent objects produced:
  * A prerequisite DAG over ``n_courses`` courses laid out across ``n_semesters``.
  * A set of planted bottleneck courses, each tagged with a difficulty
    *aspect* (one of the five canonical ABSA aspects) and a severity.
  * A primary ACM Knowledge Unit per course (drawn from the real ACM mapping
    when available, otherwise a deterministic fallback).
  * A population of students, each assigned a latent archetype with ability,
    conscientiousness and engagement traits, split into disjoint cohorts.

All randomness flows from a single seeded ``numpy.random.Generator`` so that
a given (seed, config) pair reproduces byte-identical output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Canonical ABSA aspects used as planted bottleneck difficulty profiles.
# Kept literal here to avoid importing the (heavy) NLP package at generation
# time; validated against src/nlp/absa_engine.CANONICAL_ASPECTS in tests.
CANONICAL_ASPECTS: List[str] = [
    "prerequisite_gap",
    "pacing",
    "instruction_quality",
    "assessment",
    "content_relevance",
]

# Student archetypes: (name, ability_mean, conscientiousness_mean, engagement_mean, weight)
# Traits are sampled per-student from Beta distributions centred on these means.
ARCHETYPES: List[Tuple[str, float, float, float, float]] = [
    ("striver", 0.80, 0.85, 0.85, 0.18),
    ("average", 0.55, 0.55, 0.55, 0.42),
    ("struggler", 0.35, 0.45, 0.45, 0.22),
    ("sporadic", 0.50, 0.30, 0.35, 0.12),
    ("dropout", 0.30, 0.20, 0.20, 0.06),
]

# Planted-bottleneck primary-cause types:
#   structural   -> high downstream blocking
#   performance  -> high fail rate
#   experiential -> broadly negative sentiment
# How exclusively a bottleneck expresses through its primary type is set by
# ``UniverseConfig.modality_overlap``. At 0 the assignment is one-hot, which is
# the original type-disjoint benchmark: each bottleneck is visible to exactly
# one modality, so no single modality can exceed a coverage ceiling of roughly
# 1/3. That ceiling is an artifact of the design, not a property of curriculum
# data, which is why the overlap knob exists.
BOTTLENECK_TYPES: List[str] = ["performance", "structural", "experiential"]

# Calibrated noise floor for each modality, in that modality's own units, at
# ``measurement_noise = 1.0``. These are measurement errors, not signal: they
# apply to every course, bottleneck or not.
#   performance  -> additive shift on latent difficulty, which is drawn with
#                   sd ~0.15, so 0.10 is a comparable second source of spread;
#   experiential -> additive shift on the [-1, 1] per-course sentiment
#                   baseline; 0.30 here is 0.60 Likert points after the
#                   score-to-Likert map, against a bottleneck gap of ~2.0;
#   structural   -> expected count of spurious or missing prerequisite edges,
#                   the registry errors that perturb observed centrality.
#
# The three values are calibrated so that no single modality dominates: at
# ``measurement_noise = 1.0`` and full overlap each recovers the planted set at
# an AUC in the 0.75-0.86 band. Balance matters because the clean form of the
# fusion-gain law assumes comparable signal quality. Left uncalibrated the
# performance modality reaches AUC 0.94 while the others sit near 0.75, and
# equal-weight fusion then *loses* to the best single signal no matter what the
# error correlation is, which would confound unequal quality with redundancy.
_NOISE_SD: Dict[str, float] = {
    "performance": 0.32,
    "experiential": 0.24,
    "structural": 1.3,
}

# Department buckets used for catalog cosmetics and same-department grouping.
DEPARTMENTS: List[str] = [
    "Computer Science and Engineering",
    "Mathematics",
    "Physics",
    "Electrical Engineering",
    "Humanities",
]


@dataclass
class UniverseConfig:
    """Tunable parameters for the synthetic universe."""

    n_students: int = 600
    n_courses: int = 60
    n_semesters: int = 8
    n_bottlenecks: int = 8
    n_cohorts: int = 2
    seed: int = 42
    # Cohort year labels (arbitrary identifiers).
    cohort_labels: Tuple[str, ...] = ("2003", "2103")
    min_courses_per_sem: int = 4
    max_courses_per_sem: int = 7
    # Probability mass governing how many prereq edges a non-intro course gets.
    prereq_edge_lambda: float = 1.6
    # Cross-modal overlap in [0, 1] for planted bottlenecks. At 0 each
    # bottleneck expresses through exactly one modality (the original
    # type-disjoint benchmark, reproduced byte-for-byte). At 1 every
    # bottleneck expresses through all three, so the modalities become
    # near-redundant. Sweeping this parameter traces the fusion-gain curve.
    modality_overlap: float = 0.0
    # Scale of the per-course, per-modality measurement-noise random effect,
    # expressed as a multiple of each modality's calibrated noise floor (see
    # ``_NOISE_SD``). This is the epsilon term of the fusion model
    # ``s_jc = alpha_j * b_c + eps_jc``. Without it the sentiment modality is a
    # noiseless oracle: every non-bottleneck course shares one constant
    # baseline, so any course that expresses experientially at all is
    # perfectly separable. Set to 0.0 to reproduce that earlier behaviour
    # exactly (no random numbers are drawn), which the regression test uses.
    measurement_noise: float = 1.0
    # Correlation of the measurement-noise draws across modalities, in [0, 1].
    # This is the rho of the fusion-gain law G = J / (1 + (J - 1) * rho).
    # At 0 the modalities' errors are independent and fusion gains the most;
    # at 1 they share one error and fusion cannot help at all.
    noise_correlation: float = 0.0
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.n_courses < self.n_semesters * self.min_courses_per_sem:
            raise ValueError(
                "n_courses too small for n_semesters * min_courses_per_sem"
            )
        if self.n_bottlenecks >= self.n_courses:
            raise ValueError("n_bottlenecks must be < n_courses")
        if len(self.cohort_labels) < self.n_cohorts:
            raise ValueError("Need at least n_cohorts cohort_labels")
        if not 0.0 <= self.modality_overlap <= 1.0:
            raise ValueError("modality_overlap must lie in [0, 1]")
        if self.measurement_noise < 0.0:
            raise ValueError("measurement_noise must be >= 0")
        if not 0.0 <= self.noise_correlation <= 1.0:
            raise ValueError("noise_correlation must lie in [0, 1]")


@dataclass
class Course:
    code: str
    title: str
    credits: float
    semester: int  # 1-indexed
    department: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    acm_ku: str = ""
    # Latent difficulty in [0, 1]; elevated for planted bottlenecks.
    difficulty: float = 0.0
    is_bottleneck: bool = False
    bottleneck_aspect: Optional[str] = None
    bottleneck_severity: float = 0.0
    # Primary-cause type for planted bottlenecks (see BOTTLENECK_TYPES).
    bottleneck_type: Optional[str] = None
    # Intensity in [0, 1] with which this bottleneck expresses through each
    # modality. Empty for non-bottlenecks and for universes generated before
    # this field existed, in which case ``expresses`` falls back to the
    # one-hot primary type.
    modality_expression: Dict[str, float] = field(default_factory=dict)
    # Per-modality measurement-noise draw for this course, in each modality's
    # own units. Independent of ``is_bottleneck``: measurement error applies to
    # every course, which is what gives non-bottlenecks a realistic spread.
    # Empty when ``measurement_noise`` is 0 or for universes built before this
    # field existed, in which case ``noise`` returns 0.
    measurement_noise: Dict[str, float] = field(default_factory=dict)

    def noise(self, modality: str) -> float:
        """Return this course's measurement-noise draw for ``modality``."""
        return float(self.measurement_noise.get(modality, 0.0))

    def expresses(self, modality: str) -> float:
        """Return how strongly this course expresses ``modality``, in [0, 1].

        Non-bottlenecks express nothing. When no expression vector was
        sampled the course expresses only its primary cause type, which is
        the original type-disjoint behaviour.
        """
        if not self.is_bottleneck:
            return 0.0
        if self.modality_expression:
            return float(self.modality_expression.get(modality, 0.0))
        return 1.0 if modality == self.bottleneck_type else 0.0


@dataclass
class Student:
    student_id: str
    cohort: str
    archetype: str
    ability: float
    conscientiousness: float
    engagement: float


@dataclass
class LatentUniverse:
    config: UniverseConfig
    courses: List[Course]
    students: List[Student]
    # course_code -> list of semester indices in which it is offered/taken
    rng_seed: int = 0

    # ── Convenience accessors ────────────────────────────────────────────
    @property
    def course_by_code(self) -> Dict[str, Course]:
        return {c.code: c for c in self.courses}

    @property
    def courses_in_semester(self) -> Dict[int, List[Course]]:
        out: Dict[int, List[Course]] = {}
        for c in self.courses:
            out.setdefault(c.semester, []).append(c)
        return out

    def bottleneck_codes(self) -> List[str]:
        return [c.code for c in self.courses if c.is_bottleneck]

    def prereq_edges(self) -> List[Tuple[str, str]]:
        """Directed edges (prerequisite -> dependent)."""
        edges: List[Tuple[str, str]] = []
        for c in self.courses:
            for p in c.prerequisites:
                edges.append((p, c.code))
        return edges


# ── ACM knowledge-unit loading ──────────────────────────────────────────────


def _load_acm_kus(acm_mapping_path: Optional[Path]) -> List[str]:
    """Return the list of ACM KU codes, or a deterministic fallback."""
    if acm_mapping_path is not None and acm_mapping_path.exists():
        try:
            with open(acm_mapping_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            kus = payload.get("ku_codes", [])
            if kus:
                return list(kus)
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback: synthetic KU codes so generation never hard-depends on ACM.
    return [f"SYN/Knowledge Unit {i:03d}" for i in range(60)]


# ── Sampling helpers ────────────────────────────────────────────────────────


def _beta_from_mean(rng: np.random.Generator, mean: float, concentration: float = 8.0) -> float:
    """Sample a Beta value centred on ``mean`` (clamped to a safe range)."""
    mean = float(np.clip(mean, 0.05, 0.95))
    a = mean * concentration
    b = (1.0 - mean) * concentration
    return float(rng.beta(a, b))


def _course_code(semester: int, seq: int) -> str:
    return f"CSE{semester}{seq:02d}"


def _course_title(rng: np.random.Generator, semester: int, seq: int) -> str:
    stems = [
        "Programming", "Data Structures", "Algorithms", "Discrete Mathematics",
        "Calculus", "Linear Algebra", "Digital Logic", "Computer Architecture",
        "Operating Systems", "Databases", "Networks", "Software Engineering",
        "Theory of Computation", "Machine Learning", "Compilers", "Graphics",
        "Signals and Systems", "Probability", "Numerical Methods", "Security",
        "Distributed Systems", "Embedded Systems", "Web Systems", "Optimization",
    ]
    level = ["Introduction to", "Fundamentals of", "Principles of", "Advanced",
             "Applied", "Topics in"]
    stem = stems[(semester * 7 + seq) % len(stems)]
    lvl = level[(semester + seq) % len(level)]
    return f"{lvl} {stem}"


def _description(title: str, ku: str, aspect: Optional[str]) -> str:
    base = (
        f"This course covers {title.lower()}. It develops core competencies "
        f"aligned with the {ku} knowledge unit and builds the foundations "
        f"required for subsequent coursework."
    )
    if aspect == "prerequisite_gap":
        base += " Strong prior preparation is assumed."
    elif aspect == "pacing":
        base += " The course proceeds at a demanding pace with frequent deliverables."
    elif aspect == "instruction_quality":
        base += " Content is delivered primarily through self-directed study."
    return base


# ── Universe construction ────────────────────────────────────────────────────


def build_universe(
    config: UniverseConfig,
    acm_mapping_path: Optional[Path] = None,
) -> LatentUniverse:
    """Construct the full latent universe deterministically from ``config``."""
    rng = np.random.default_rng(config.seed)
    acm_kus = _load_acm_kus(acm_mapping_path)

    # 1. Distribute courses across semesters.
    counts = _distribute_courses(rng, config)

    courses: List[Course] = []
    for sem in range(1, config.n_semesters + 1):
        for seq in range(1, counts[sem - 1] + 1):
            code = _course_code(sem, seq)
            title = _course_title(rng, sem, seq)
            ku = acm_kus[(len(courses)) % len(acm_kus)]
            credits = float(rng.choice([0.75, 1.5, 3.0, 3.0, 4.0]))
            dept = DEPARTMENTS[(sem + seq) % len(DEPARTMENTS)]
            courses.append(
                Course(
                    code=code,
                    title=title,
                    credits=credits,
                    semester=sem,
                    department=dept,
                    description="",  # filled after aspect assignment
                    acm_ku=ku,
                    difficulty=_beta_from_mean(rng, 0.22, concentration=7.0),
                )
            )

    code_to_course = {c.code: c for c in courses}

    # 2. Draw per-course measurement noise for all three modalities, correlated
    #    across modalities at ``config.noise_correlation``. This is the error
    #    term the redundancy law is stated over. The performance component
    #    lands on latent difficulty here; the structural component is applied
    #    to the registry in step 5 and the experiential component is read by
    #    ``synth.reviews``.
    _sample_measurement_noise(rng, courses, config)
    for c in courses:
        if c.measurement_noise:
            c.difficulty = float(
                np.clip(c.difficulty + c.noise("performance"), 0.0, 1.0)
            )

    # 3. Sample prerequisite edges (forward in time only -> guaranteed DAG).
    _sample_prerequisites(rng, courses, config)

    # 4. Plant bottlenecks (avoid first semester so downstream impact exists).
    _plant_bottlenecks(rng, courses, config)

    # 5. Perturb the recorded prerequisite graph. Applied after planting so
    #    that registry error degrades planted hubs as well, which is what gives
    #    the structural signal a noise floor rather than a clean read.
    by_sem_all: Dict[int, List[Course]] = {}
    for c in courses:
        by_sem_all.setdefault(c.semester, []).append(c)
    _apply_structural_noise(rng, courses, by_sem_all, config)

    # 6. Fill descriptions now that aspects are known.
    for c in courses:
        c.description = _description(c.title, c.acm_ku, c.bottleneck_aspect)

    # 7. Sample students across disjoint cohorts.
    students = _sample_students(rng, config)

    _ = code_to_course  # retained for clarity / potential validation
    return LatentUniverse(
        config=config,
        courses=courses,
        students=students,
        rng_seed=config.seed,
    )


def _distribute_courses(rng: np.random.Generator, config: UniverseConfig) -> List[int]:
    """Assign a course count to each semester summing to n_courses."""
    lo, hi = config.min_courses_per_sem, config.max_courses_per_sem
    counts = [lo] * config.n_semesters
    remaining = config.n_courses - sum(counts)
    # Distribute the remainder one course at a time to random semesters that
    # still have headroom below ``hi``.
    while remaining > 0:
        eligible = [i for i in range(config.n_semesters) if counts[i] < hi]
        if not eligible:
            # Allow exceeding hi as a last resort to honour n_courses.
            eligible = list(range(config.n_semesters))
        idx = int(rng.choice(eligible))
        counts[idx] += 1
        remaining -= 1
    return counts


def _sample_prerequisites(
    rng: np.random.Generator,
    courses: List[Course],
    config: UniverseConfig,
) -> None:
    """Add forward-in-time prerequisite edges (earlier semester -> later)."""
    by_sem: Dict[int, List[Course]] = {}
    for c in courses:
        by_sem.setdefault(c.semester, []).append(c)

    for c in courses:
        if c.semester == 1:
            continue  # intro courses have no prerequisites
        # Candidate prereqs: courses from strictly earlier semesters, biased
        # toward the immediately preceding semester.
        candidates: List[Course] = []
        weights: List[float] = []
        for prior_sem in range(1, c.semester):
            for pc in by_sem.get(prior_sem, []):
                candidates.append(pc)
                # weight decays with semester distance
                weights.append(1.0 / (c.semester - prior_sem))
        if not candidates:
            continue
        n_edges = int(rng.poisson(config.prereq_edge_lambda))
        n_edges = max(0, min(n_edges, 3, len(candidates)))
        if n_edges == 0:
            continue
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        chosen_idx = rng.choice(
            len(candidates), size=n_edges, replace=False, p=w
        )
        prereqs = sorted({candidates[int(i)].code for i in np.atleast_1d(chosen_idx)})
        c.prerequisites = prereqs


def _plant_bottlenecks(
    rng: np.random.Generator,
    courses: List[Course],
    config: UniverseConfig,
) -> None:
    """Tag a subset of non-intro courses as planted bottlenecks.

    Bottlenecks are assigned heterogeneous primary-cause *types* in a balanced
    round-robin so that each is detectable by exactly one modality:
      * ``performance``  -> elevated difficulty (high fail rate);
      * ``structural``   -> turned into a prerequisite hub (high blocking);
      * ``experiential`` -> broadly negative student sentiment only.
    """
    # Structural hubs must leave room for downstream dependents, so prefer
    # early-but-not-intro semesters for them; all types avoid semester 1.
    eligible = [c for c in courses if c.semester >= 2]
    if len(eligible) < config.n_bottlenecks:
        eligible = list(courses)
    chosen_idx = rng.choice(
        len(eligible), size=config.n_bottlenecks, replace=False
    )
    by_sem: Dict[int, List[Course]] = {}
    for c in courses:
        by_sem.setdefault(c.semester, []).append(c)
    max_sem = max(by_sem)

    # Assign types by semester order so structural hubs sit in early semesters
    # (maximal downstream reach), performance in the middle, experiential late.
    # Counts follow a balanced round-robin over BOTTLENECK_TYPES.
    chosen = [eligible[int(i)] for i in np.atleast_1d(chosen_idx)]
    chosen.sort(key=lambda c: c.semester)
    type_sequence: List[str] = [
        BOTTLENECK_TYPES[(rank % len(BOTTLENECK_TYPES))]
        for rank in range(len(chosen))
    ]
    # Map the round-robin counts onto semester-sorted courses: structural first.
    n_struct = type_sequence.count("structural")
    n_perf = type_sequence.count("performance")
    ordered_types = (
        ["structural"] * n_struct
        + ["performance"] * n_perf
        + ["experiential"] * (len(chosen) - n_struct - n_perf)
    )

    for c, btype in zip(chosen, ordered_types):
        severity = _beta_from_mean(rng, 0.75, concentration=10.0)
        c.is_bottleneck = True
        c.bottleneck_severity = severity
        c.bottleneck_type = btype
        c.modality_expression = _sample_expression(
            rng, btype, config.modality_overlap
        )

        # The ABSA aspect still follows the primary cause type.
        if btype == "performance":
            c.bottleneck_aspect = "prerequisite_gap" if rng.random() < 0.5 else "assessment"
        elif btype == "structural":
            c.bottleneck_aspect = "content_relevance"
        else:
            c.bottleneck_aspect = "instruction_quality" if rng.random() < 0.5 else "pacing"

        # Each modality expresses at its own intensity. At overlap 0 the
        # expression vector is one-hot and these reduce to the original
        # per-type effects exactly: +0.30*sev+0.40 difficulty for a
        # performance bottleneck, +0.05 otherwise, hub rewiring only for a
        # structural one.
        e_perf = c.modality_expression["performance"]
        e_struct = c.modality_expression["structural"]
        c.difficulty = float(
            np.clip(
                c.difficulty + 0.05 + (0.30 * severity + 0.35) * e_perf,
                0.0,
                1.0,
            )
        )
        if e_struct > 0.0:
            _make_structural_hub(
                rng, c, by_sem, max_sem, severity, strength=e_struct
            )


def _sample_measurement_noise(
    rng: np.random.Generator,
    courses: List[Course],
    config: UniverseConfig,
) -> None:
    """Draw each course's per-modality measurement-noise vector.

    This is the ``eps_jc`` of the fusion model ``s_jc = alpha_j * b_c + eps_jc``
    that the redundancy law is stated over. The draws use a one-factor
    construction::

        z_j = sqrt(rho) * common + sqrt(1 - rho) * private_j

    which gives every pair of modalities exactly ``Corr(eps_j, eps_k) = rho``
    while keeping each marginal standard normal. ``rho = 0`` leaves the three
    errors independent, the regime in which fusion gains the most;
    ``rho = 1`` collapses them onto a single shared error, in which fusion
    cannot help at all. Sweeping it traces the gain curve.

    At ``measurement_noise = 0`` no random numbers are drawn, so the RNG stream
    and therefore every previously published artifact is unchanged.
    """
    if config.measurement_noise <= 0.0:
        return
    rho = config.noise_correlation
    w_common = float(np.sqrt(rho))
    w_private = float(np.sqrt(1.0 - rho))
    for c in courses:
        common = float(rng.normal())
        c.measurement_noise = {
            m: config.measurement_noise
            * _NOISE_SD[m]
            * (w_common * common + w_private * float(rng.normal()))
            for m in BOTTLENECK_TYPES
        }


def _apply_structural_noise(
    rng: np.random.Generator,
    courses: List[Course],
    by_sem: Dict[int, List[Course]],
    config: UniverseConfig,
) -> None:
    """Perturb the prerequisite registry to give the structural signal a floor.

    Real prerequisite registries are recorded by hand. They carry informal
    dependencies that were never written down and stale entries that no longer
    apply, and both distort observed centrality. A positive noise draw adds
    spurious forward edges out of a course, making it look more central than it
    is; a negative draw removes recorded ones, hiding centrality it has. The
    perturbation is applied after planting, so it degrades planted hubs too.

    Edges are only ever added forward in time, so the graph stays a DAG.
    """
    if config.measurement_noise <= 0.0:
        return
    max_sem = max(by_sem) if by_sem else 0
    dependents: Dict[str, List[Course]] = {}
    for c in courses:
        for pre in c.prerequisites:
            dependents.setdefault(pre, []).append(c)

    for c in courses:
        n = int(round(c.noise("structural")))
        if n > 0:
            later = [
                t
                for sem in range(c.semester + 1, max_sem + 1)
                for t in by_sem.get(sem, [])
                if c.code not in t.prerequisites
            ]
            if not later:
                continue
            idx = rng.choice(len(later), size=min(n, len(later)), replace=False)
            for i in np.atleast_1d(idx):
                target = later[int(i)]
                target.prerequisites = sorted(
                    set(target.prerequisites) | {c.code}
                )
        elif n < 0:
            existing = dependents.get(c.code, [])
            if not existing:
                continue
            idx = rng.choice(
                len(existing), size=min(-n, len(existing)), replace=False
            )
            for i in np.atleast_1d(idx):
                target = existing[int(i)]
                target.prerequisites = [
                    pre for pre in target.prerequisites if pre != c.code
                ]


def _sample_expression(
    rng: np.random.Generator,
    primary: str,
    overlap: float,
) -> Dict[str, float]:
    """Draw the per-modality expression vector for one planted bottleneck.

    The primary cause type always expresses fully. ``overlap`` scales how
    strongly the other two modalities also express it, which is what drives
    the cross-modal redundancy of the resulting benchmark. At ``overlap = 0``
    no random numbers are drawn, so the generator's RNG stream (and therefore
    every downstream artifact) is unchanged from the type-disjoint design.
    """
    expr = {t: 0.0 for t in BOTTLENECK_TYPES}
    expr[primary] = 1.0
    if overlap <= 0.0:
        return expr
    for t in BOTTLENECK_TYPES:
        if t != primary:
            expr[t] = float(overlap * rng.uniform(0.7, 1.0))
    # Conserve total expressed severity. Without this the knob would make
    # bottlenecks strictly *more* detectable rather than more redundant:
    # every modality would gain signal, any single modality would recover the
    # full planted set, and fusion would have no headroom left to measure.
    # Normalising redistributes one bottleneck's severity across modalities
    # instead of duplicating it.
    total = sum(expr.values())
    return {t: v / total for t, v in expr.items()}


def _make_structural_hub(
    rng: np.random.Generator,
    hub: Course,
    by_sem: Dict[int, List[Course]],
    max_sem: int,
    severity: float,
    strength: float = 1.0,
) -> None:
    """Add forward edges hub -> later courses so the hub gains high blocking.

    Targets are drawn preferentially from the next two semesters: those
    courses themselves have downstream dependents, so the hub accumulates a
    large transitive descendant set (high blocking factor) and lies on many
    prerequisite paths (high betweenness).
    """
    near: List[Course] = []
    for sem in (hub.semester + 1, hub.semester + 2):
        near.extend(by_sem.get(sem, []))
    far: List[Course] = []
    for sem in range(hub.semester + 3, max_sem + 1):
        far.extend(by_sem.get(sem, []))
    if not near and not far:
        return
    # Connect to most of the next-two-semester courses (strong hub) plus a few
    # farther ones to extend reach.
    n_near = (
        int(np.clip(round(len(near) * (0.6 + 0.3 * severity) * strength), 1, len(near)))
        if near
        else 0
    )
    n_far = int(np.clip(round((2 + 3 * severity) * strength), 0, len(far))) if far else 0
    targets: List[Course] = []
    if n_near:
        idx = rng.choice(len(near), size=n_near, replace=False)
        targets.extend(near[int(i)] for i in np.atleast_1d(idx))
    if n_far:
        idx = rng.choice(len(far), size=n_far, replace=False)
        targets.extend(far[int(i)] for i in np.atleast_1d(idx))
    for target in targets:
        if hub.code not in target.prerequisites:
            target.prerequisites = sorted(set(target.prerequisites) | {hub.code})


def _sample_students(
    rng: np.random.Generator,
    config: UniverseConfig,
) -> List[Student]:
    """Sample students partitioned into disjoint cohorts."""
    names = [a[0] for a in ARCHETYPES]
    means = {a[0]: (a[1], a[2], a[3]) for a in ARCHETYPES}
    probs = np.asarray([a[4] for a in ARCHETYPES], dtype=float)
    probs /= probs.sum()

    per_cohort = config.n_students // config.n_cohorts
    students: List[Student] = []
    counter = 0
    for ci in range(config.n_cohorts):
        cohort = config.cohort_labels[ci]
        # Last cohort absorbs any remainder.
        n = per_cohort if ci < config.n_cohorts - 1 else (
            config.n_students - per_cohort * (config.n_cohorts - 1)
        )
        for _ in range(n):
            arch = names[int(rng.choice(len(names), p=probs))]
            am, cm, em = means[arch]
            students.append(
                Student(
                    student_id=f"S{cohort}{counter:05d}",
                    cohort=cohort,
                    archetype=arch,
                    ability=_beta_from_mean(rng, am),
                    conscientiousness=_beta_from_mean(rng, cm),
                    engagement=_beta_from_mean(rng, em),
                )
            )
            counter += 1
    return students


def universe_summary(u: LatentUniverse) -> Dict[str, object]:
    """Return a small JSON-serialisable summary for logging/manifests."""
    return {
        "n_courses": len(u.courses),
        "n_students": len(u.students),
        "n_semesters": u.config.n_semesters,
        "n_prereq_edges": len(u.prereq_edges()),
        "n_bottlenecks": len(u.bottleneck_codes()),
        "bottleneck_codes": u.bottleneck_codes(),
        "modality_overlap": u.config.modality_overlap,
        "measurement_noise": u.config.measurement_noise,
        "noise_correlation": u.config.noise_correlation,
        "modality_expression": {
            c.code: c.modality_expression for c in u.courses if c.is_bottleneck
        },
        "bottleneck_types": {
            c.code: c.bottleneck_type for c in u.courses if c.is_bottleneck
        },
        "cohorts": list(u.config.cohort_labels[: u.config.n_cohorts]),
        "seed": u.config.seed,
        "version": u.config.version,
        "config": asdict(u.config),
    }
