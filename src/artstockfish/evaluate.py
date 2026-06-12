"""Evaluation — weighting, ranking, and the aggregate accuracy score (spec §9.5).

The measurement modules (``measure/landmarks.py``, ``angles.py``, ``proportions.py``)
already attach a ``score`` to every :class:`~artstockfish.schema.Finding` using the
§9.5 formula::

    score = importance_weight[feature] × (magnitude / severity_unit)

where ``severity_unit`` is that axis's OK noise floor (so a finding that just crosses
the floor contributes ≈ its feature weight). This module does **not** re-measure or
re-weight — doing so would duplicate the per-axis ``severity_unit`` choices and risk
diverging from them. It consumes the findings, **ranks** them coarse-to-fine, rolls
their scores into the aggregate **accuracy score** ("eval bar"), and packages a
:class:`~artstockfish.schema.Report`.

Ranking (spec §9.5, principle #5): sort by ``(Level asc, score desc)`` — global
pose/proportion errors outrank feature placement, which outranks local shape; within a
tier the highest-leverage correction comes first ("best move first").

Accuracy (spec §9.5): ``accuracy_score = 100 · exp(-k · Σ score)`` over the *surfaced*
findings, with ``k = config.ACCURACY_K`` chosen so a typical first attempt lands ~55–70.
A perfect match (no findings) scores 100. All functions are pure.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .config import ACCURACY_K
from .schema import Finding, Report


def rank_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Sort findings coarse-to-fine: ``(Level asc, score desc)`` (spec §9.5).

    The sort is stable, so findings that tie on both keys keep their input order
    (which keeps the report deterministic — principle #7).
    """
    return tuple(sorted(findings, key=lambda f: (int(f.level), -f.score)))


def total_score(findings: Iterable[Finding]) -> float:
    """Sum of the surfaced findings' scores — the penalty the accuracy bar decays on."""
    return float(sum(f.score for f in findings))


def accuracy_score(findings: Iterable[Finding], k: float = ACCURACY_K) -> float:
    """Aggregate accuracy in ``[0, 100]`` (spec §9.5).

    ``100 · exp(-k · Σ score)``. No findings → ``Σ score = 0`` → exactly ``100``.
    More/severer findings drive it monotonically toward 0. ``k`` is calibrated in
    ``config.ACCURACY_K`` so a typical first-attempt sketch lands ~55–70.
    """
    return 100.0 * math.exp(-k * total_score(findings))


def build_report(
    findings: Sequence[Finding],
    transform: dict,
    pose: dict | None = None,
    *,
    k: float = ACCURACY_K,
) -> Report:
    """Assemble a ranked :class:`Report` from raw findings + the fitted transform.

    Args:
        findings: the surfaced findings (each already ≥ OK and scored by its
            measurement module). They are ranked here; the caller need not pre-sort.
        transform: the fitted similarity-transform parameters (from ``align``),
            stored verbatim on the report for annotation/inspection.
        pose: per-image pose estimates, or ``None`` until the M1.5 pose stage exists.
        k: accuracy-decay constant (defaults to the calibrated ``config.ACCURACY_K``).

    Returns:
        A :class:`Report` whose ``findings`` are sorted ``(Level asc, score desc)``
        and whose ``accuracy_score`` decays on the same surfaced findings.
    """
    ranked = rank_findings(findings)
    return Report(
        findings=ranked,
        accuracy_score=accuracy_score(ranked, k=k),
        transform=transform,
        pose=pose,
    )
