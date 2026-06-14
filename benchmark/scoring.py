"""Scoring shared by both systems (spec §8 M5).

A *finding* reduces to a ``(id, direction)`` key plus a magnitude. Both systems are
scored identically against the injected ground truth:

- **precision / recall** — exact ``(id, direction)`` match against the labels.
- **localization** — fraction of injected findings whose *feature* (the id, any
  direction) the system named: "did it point at the right thing", separate from
  getting the direction right.
- **magnitude error** — median ``|reported − injected| / injected`` over matched
  findings (the same fractional metric the M1 harness reports).
- **consistency** — over the 3 repeats, the mean pairwise Jaccard of each case's
  reported key set, averaged across cases. A deterministic system scores 1.0; this
  is the axis the spec calls out as the VLM's structural weakness (§1).

Per-repeat metrics are micro-averaged over cases, then averaged over the repeats, so
a system that is right on a big case and wrong on a small one is scored on totals,
not on a mean of ratios.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FindingKey = tuple[str, str]


@dataclass(frozen=True)
class ReportedFinding:
    """A finding as produced by either system, normalized for scoring."""

    id: str
    direction: str
    magnitude: float

    @property
    def key(self) -> FindingKey:
        return (self.id, self.direction)


@dataclass(frozen=True)
class GroundTruthFinding:
    id: str
    direction: str
    magnitude: float

    @property
    def key(self) -> FindingKey:
        return (self.id, self.direction)


@dataclass(frozen=True)
class SystemScore:
    """Headline metrics for one system over the whole benchmark."""

    precision: float
    recall: float
    localization: float
    median_magnitude_error: float
    consistency: float
    n_cases: int
    n_repeats: int
    total_expected: int
    total_reported: float          # averaged over repeats → may be fractional
    total_true_positive: float


def _jaccard(a: set[FindingKey], b: set[FindingKey]) -> float:
    if not a and not b:
        return 1.0  # both correctly reported nothing → perfectly consistent
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _repeat_counts(
    expected_by_case: Sequence[Sequence[GroundTruthFinding]],
    reported_by_case: Sequence[Sequence[ReportedFinding]],
) -> tuple[int, int, int, int, list[float]]:
    """One repeat: (tp, n_reported, n_expected, loc_hits, magnitude_errors)."""
    tp = n_reported = n_expected = loc_hits = 0
    mag_errors: list[float] = []
    for expected, reported in zip(expected_by_case, reported_by_case):
        exp_by_key = {f.key: f for f in expected}
        rep_by_key = {f.key: f for f in reported}
        exp_ids = {f.id for f in expected}
        rep_ids = {f.id for f in reported}

        matches = set(exp_by_key) & set(rep_by_key)
        tp += len(matches)
        n_reported += len(rep_by_key)
        n_expected += len(exp_by_key)
        loc_hits += len(exp_ids & rep_ids)
        for key in matches:
            inj = exp_by_key[key].magnitude
            mag_errors.append(abs(rep_by_key[key].magnitude - inj) / max(abs(inj), 1e-9))
    return tp, n_reported, n_expected, loc_hits, mag_errors


def score_system(
    expected_by_case: Sequence[Sequence[GroundTruthFinding]],
    reported_by_repeat: Sequence[Sequence[Sequence[ReportedFinding]]],
) -> SystemScore:
    """Score one system.

    Args:
        expected_by_case: ground-truth findings per case (length ``n_cases``).
        reported_by_repeat: ``reported_by_repeat[r][c]`` is the findings the system
            reported for case ``c`` on repeat ``r``.
    """
    n_repeats = len(reported_by_repeat)
    n_cases = len(expected_by_case)
    if n_repeats == 0:
        raise ValueError("need at least one repeat to score")

    precisions: list[float] = []
    recalls: list[float] = []
    locs: list[float] = []
    all_mag_errors: list[float] = []
    sum_reported = sum_tp = 0.0
    total_expected = sum(len(e) for e in expected_by_case)

    for reported_by_case in reported_by_repeat:
        tp, n_rep, n_exp, loc_hits, mag_errors = _repeat_counts(expected_by_case, reported_by_case)
        precisions.append(tp / n_rep if n_rep else 1.0)
        recalls.append(tp / n_exp if n_exp else 1.0)
        locs.append(loc_hits / n_exp if n_exp else 1.0)
        all_mag_errors.extend(mag_errors)
        sum_reported += n_rep
        sum_tp += tp

    # Consistency: mean pairwise Jaccard of key sets across repeats, per case.
    consistencies: list[float] = []
    for c in range(n_cases):
        key_sets = [{f.key for f in reported_by_repeat[r][c]} for r in range(n_repeats)]
        if n_repeats < 2:
            consistencies.append(1.0)
            continue
        pairwise = [
            _jaccard(key_sets[i], key_sets[j])
            for i in range(n_repeats) for j in range(i + 1, n_repeats)
        ]
        consistencies.append(float(np.mean(pairwise)))

    median_mag = float(np.median(all_mag_errors)) if all_mag_errors else 0.0
    return SystemScore(
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        localization=float(np.mean(locs)),
        median_magnitude_error=median_mag,
        consistency=float(np.mean(consistencies)) if consistencies else 1.0,
        n_cases=n_cases,
        n_repeats=n_repeats,
        total_expected=total_expected,
        total_reported=sum_reported / n_repeats,
        total_true_positive=sum_tp / n_repeats,
    )
