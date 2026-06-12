"""Wave 2 unit tests — evaluation: ranking, accuracy score, report assembly (§9.5).

Run: ``pytest tests/test_evaluate.py -q``

These pin the three contracts of ``evaluate.py``:
- ranking is coarse-to-fine: ``(Level asc, score desc)`` (spec §9.5, principle #5);
- the accuracy "eval bar" is ``100·exp(-k·Σ score)`` — 100 for a clean sketch and
  monotonically decreasing as findings pile up (spec §9.5);
- ``build_report`` returns a ranked :class:`Report` carrying the transform/pose.
"""

from __future__ import annotations

import math

import pytest

from artstockfish.config import ACCURACY_K
from artstockfish.evaluate import (
    accuracy_score,
    build_report,
    rank_findings,
    total_score,
)
from artstockfish.schema import Finding, Level, Severity


def _finding(id: str, level: Level, score: float, severity=Severity.MISTAKE) -> Finding:
    """A minimal Finding for ranking/scoring tests (geometry fields are inert here)."""
    return Finding(
        id=id,
        level=level,
        severity=severity,
        feature=id,
        axis="vertical",
        direction="too high",
        magnitude=score,  # arbitrary; ranking/accuracy use `score`
        units="%head_height",
        score=score,
        evidence={},
    )


# --- ranking ------------------------------------------------------------------

def test_rank_orders_by_level_then_score():
    a = _finding("global_lo", Level.GLOBAL, 0.5)
    b = _finding("placement_hi", Level.PLACEMENT, 9.0)
    c = _finding("placement_lo", Level.PLACEMENT, 1.0)
    d = _finding("shape_hi", Level.SHAPE, 99.0)
    ranked = rank_findings([b, d, c, a])
    # GLOBAL first regardless of its (small) score; SHAPE last despite huge score.
    assert [f.id for f in ranked] == ["global_lo", "placement_hi", "placement_lo", "shape_hi"]


def test_rank_is_stable_for_exact_ties():
    f1 = _finding("first", Level.PLACEMENT, 2.0)
    f2 = _finding("second", Level.PLACEMENT, 2.0)
    assert [f.id for f in rank_findings([f1, f2])] == ["first", "second"]
    assert [f.id for f in rank_findings([f2, f1])] == ["second", "first"]


# --- accuracy score -----------------------------------------------------------

def test_accuracy_is_100_with_no_findings():
    assert accuracy_score([]) == 100.0


def test_accuracy_matches_formula():
    findings = [
        _finding("a", Level.PLACEMENT, 2.5),
        _finding("b", Level.GLOBAL, 4.0),
    ]
    expected = 100.0 * math.exp(-ACCURACY_K * 6.5)
    assert total_score(findings) == pytest.approx(6.5)
    assert accuracy_score(findings) == pytest.approx(expected)


def test_accuracy_is_monotonic_decreasing_and_bounded():
    one = [_finding("a", Level.PLACEMENT, 3.0)]
    two = one + [_finding("b", Level.PLACEMENT, 3.0)]
    assert 0.0 < accuracy_score(two) < accuracy_score(one) < 100.0


def test_typical_first_attempt_lands_in_target_band():
    """k is calibrated (spec §9.5) so a typical first attempt scores ~55–70.

    A typical first-attempt total score is ~9–14 with the project's weights; assert
    that band maps in-range so a future k change that breaks the calibration trips
    this test. The authoritative anchor is the realistic demo (checked below).
    """
    assert 55.0 <= accuracy_score([_finding("x", Level.GLOBAL, 9.0)]) <= 70.0
    assert 55.0 <= accuracy_score([_finding("x", Level.GLOBAL, 14.0)]) <= 70.0


def test_demo_accuracy_is_in_calibration_band():
    """The realistic demo face — the spec's "typical first attempt" — scores ~55–70."""
    from artstockfish.pipeline import critique_pair, demo_synthetic_pair

    reference, sketch = demo_synthetic_pair()
    accuracy = critique_pair(reference, sketch).report.accuracy_score
    assert 55.0 <= accuracy <= 70.0, f"demo accuracy {accuracy:.1f} outside 55–70 band"


# --- report assembly ----------------------------------------------------------

def test_build_report_sorts_and_carries_transform_and_pose():
    findings = [
        _finding("placement", Level.PLACEMENT, 1.0),
        _finding("global", Level.GLOBAL, 0.2),
    ]
    transform = {"scale": 1.0, "rotation": [[1.0, 0.0], [0.0, 1.0]]}
    report = build_report(findings, transform=transform, pose=None)

    assert [f.id for f in report.findings] == ["global", "placement"]
    assert report.transform is transform
    assert report.pose is None
    assert report.accuracy_score == pytest.approx(accuracy_score(findings))
