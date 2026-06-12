"""Wave 1A acceptance tests — per-group landmark residual decomposition (§9.3).

Run: ``pytest tests/test_measure_landmarks.py -q``

The prompt's two gates:
- Shift the left-eye group up 5% of head height → **exactly one** Finding,
  ``left_eye_vertical`` / "too high", magnitude 5% ± 0.5%, and nothing else
  above OK.
- Identical inputs → **zero** findings.

These exercise spec §9.3 on the hardcoded canonical 68-point face. The sketch is
shifted in image space along the reference face frame's up-axis so the test does
not assume the fixture is perfectly axis-aligned.
"""

from __future__ import annotations

import numpy as np

from artstockfish.frame import SEMANTIC_GROUPS, build_face_frame
from artstockfish.measure.landmarks import MEASURE_GROUPS, measure_landmarks
from artstockfish.schema import Finding, Level, Severity

from fixtures import canonical_face_landmarks, canonical_face_points


def _shift_group_up(points: np.ndarray, group: str, frac_head_height: float) -> np.ndarray:
    """Return a copy with ``group`` moved ``frac_head_height`` *up the face*."""
    frame = build_face_frame(points)
    out = points.copy()
    out[list(SEMANTIC_GROUPS[group])] += frac_head_height * frame.head_height * frame.y_axis
    return out


# --- identical inputs → zero findings ----------------------------------------

def test_identical_inputs_zero_findings():
    ref = canonical_face_landmarks()
    findings = measure_landmarks(ref, canonical_face_landmarks())
    assert findings == []


# --- 5% left-eye shift → exactly one finding ---------------------------------

def test_left_eye_up_5pct_single_finding():
    ref_pts = canonical_face_points()
    sketch_pts = _shift_group_up(ref_pts, "left_eye", 0.05)

    ref = canonical_face_landmarks()
    sketch = canonical_face_landmarks()
    object.__setattr__(sketch, "points", sketch_pts)  # frozen dataclass

    findings = measure_landmarks(ref, sketch)

    assert len(findings) == 1, f"expected exactly one finding, got {findings}"
    (f,) = findings
    assert isinstance(f, Finding)
    assert f.id == "left_eye_vertical"
    assert f.feature == "left eye"
    assert f.axis == "vertical"
    assert f.direction == "too high"
    assert f.level is Level.PLACEMENT
    assert f.units == "%head_height"
    assert f.severity is not Severity.OK
    assert f.magnitude == 5.0 or abs(f.magnitude - 5.0) <= 0.5
    # 5% displacement is in the MISTAKE tier (4–8%) per §6.
    assert f.severity is Severity.MISTAKE
    # Evidence carries the raw geometry for annotate.py.
    assert set(f.evidence) >= {"group", "indices", "ref_points", "sketch_points", "mean_residual"}


def test_left_eye_shift_no_other_groups_flagged():
    """Every group except the shifted one stays silent (alignment not dragged)."""
    ref_pts = canonical_face_points()
    sketch_pts = _shift_group_up(ref_pts, "left_eye", 0.05)

    ref = canonical_face_landmarks()
    sketch = canonical_face_landmarks()
    object.__setattr__(sketch, "points", sketch_pts)

    findings = measure_landmarks(ref, sketch)
    flagged_groups = {f.evidence["group"] for f in findings}
    assert flagged_groups == {"left_eye"}


def test_measure_groups_cover_expected_features():
    """Sanity: the documented teacher-named groups exist."""
    assert {"left_eye", "right_eye", "nose", "mouth", "jaw"} <= set(MEASURE_GROUPS)
