"""M0 acceptance tests — the integrated pipeline (spec §8 M0).

Run: ``pytest tests/test_m0_acceptance.py -q``

- **M0-T1:** shift the left-eye landmarks up 5% of head height → the report contains
  **exactly one** PLACEMENT finding ``left_eye_vertical`` / "too high", magnitude
  5% ± 0.5%, and nothing else above OK. Through the *full* pipeline this is non-trivial:
  moving the eye also tilts the eye line (the eye-line fit reads the eye corners), so a
  naive wiring would surface a second ``eye_line_angle`` finding. The pipeline's
  ``suppress_explained_angles`` recognizes that tilt as a symptom of the placement and
  removes it (spec §2 principle #5, pitfall §12; see DECISIONS.md).
- **M0-T2:** identical sets → zero findings, accuracy_score = 100.

The alignment-spine tests M0-T3/M0-T4 live in ``tests/test_align.py``.
"""

from __future__ import annotations

import numpy as np

from artstockfish.frame import SEMANTIC_GROUPS, build_face_frame
from artstockfish.measure.angles import measure_angles
from artstockfish.pipeline import critique_pair
from artstockfish.schema import Landmarks, Level, Severity

from fixtures import canonical_face_landmarks, canonical_face_points


def _shift_group_up(points: np.ndarray, group: str, frac_head_height: float) -> np.ndarray:
    """Return a copy with ``group`` moved ``frac_head_height`` *up the face frame*."""
    frame = build_face_frame(points)
    out = points.copy()
    out[list(SEMANTIC_GROUPS[group])] += frac_head_height * frame.head_height * frame.y_axis
    return out


def _sketch_landmarks(points: np.ndarray) -> Landmarks:
    base = canonical_face_landmarks()
    return Landmarks(points=points, names=base.names, image_size=base.image_size)


# --- M0-T1: one 5% eye shift → exactly one finding ----------------------------

def test_m0_t1_single_eye_shift_one_finding():
    ref = canonical_face_landmarks()
    sketch = _sketch_landmarks(_shift_group_up(canonical_face_points(), "left_eye", 0.05))

    result = critique_pair(ref, sketch)
    findings = result.report.findings

    assert len(findings) == 1, f"expected exactly one finding, got {[f.id for f in findings]}"
    (f,) = findings
    assert f.id == "left_eye_vertical"
    assert f.feature == "left eye"
    assert f.axis == "vertical"
    assert f.direction == "too high"
    assert f.level is Level.PLACEMENT
    assert f.units == "%head_height"
    assert f.severity is Severity.MISTAKE          # 5% is in the 4–8% tier (§6)
    assert abs(f.magnitude - 5.0) <= 0.5
    # A teacher-voiced sentence is produced for the finding (spec §11).
    assert len(result.sentences) == 1
    assert "left eye" in result.sentences[0]


def test_m0_t1_eye_line_angle_is_explained_away():
    """Sanity: the *raw* angle pass DOES flag the eye line; the pipeline suppresses it.

    Confirms the single-finding result above is genuine suppression of a redundant
    symptom — not the angle module quietly missing the tilt (Ground Rule 5).
    """
    ref = canonical_face_landmarks()
    sketch = _sketch_landmarks(_shift_group_up(canonical_face_points(), "left_eye", 0.05))

    raw_angle_ids = {f.id for f in measure_angles(ref, sketch)}
    assert "eye_line_angle" in raw_angle_ids  # the tilt is real before suppression

    surfaced_ids = {f.id for f in critique_pair(ref, sketch).report.findings}
    assert "eye_line_angle" not in surfaced_ids  # …and explained away in the report


# --- M0-T2: identical inputs → zero findings, accuracy 100 --------------------

def test_m0_t2_identical_inputs_clean_report():
    ref = canonical_face_landmarks()
    result = critique_pair(ref, canonical_face_landmarks())

    assert result.report.findings == ()
    assert result.report.accuracy_score == 100.0
    assert result.sentences == ()


# --- the M0 demo path renders an overlay end-to-end ---------------------------

def test_demo_overlay_renders(tmp_path):
    from artstockfish.pipeline import demo_synthetic_pair

    out = tmp_path / "overlay.png"
    reference, sketch = demo_synthetic_pair()
    result = critique_pair(reference, sketch, overlay_path=str(out))

    assert result.overlay_path == str(out)
    assert out.exists() and out.stat().st_size > 0
    # The realistic demo surfaces a ranked, non-empty critique led coarse-to-fine.
    assert len(result.report.findings) >= 3
    assert result.report.findings[0].level <= result.report.findings[-1].level
