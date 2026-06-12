"""Wave 1C acceptance tests — feature angle comparisons (spec §9.4).

Run: ``pytest tests/test_measure_angles.py -q``

Exercises ``measure.angles`` on the hardcoded canonical 68-point face
(``tests/fixtures.py``):

- Rotating the **eye-line landmarks** by a known angle yields exactly one
  ``eye_line_angle`` Finding, magnitude within ±0.5° and the correct direction.
- A pure **page tilt** (whole face rotated) is absorbed by the similarity
  alignment (§9.1), so it yields **zero** findings — tilt of the page is not a
  drawing error (principle #2).
"""

from __future__ import annotations

import numpy as np
import pytest

from artstockfish.config import ANGLE_LINES
from artstockfish.frame import LANDMARK_NAMES_68
from artstockfish.measure.angles import measure_angles
from artstockfish.schema import Landmarks, Level, Severity

from fixtures import canonical_face_points

EYE_LINE_INDICES = list(ANGLE_LINES["eye_line"]["indices"])


def _landmarks(points: np.ndarray) -> Landmarks:
    return Landmarks(points=points, names=LANDMARK_NAMES_68, image_size=(500, 500))


def _rotate_about_centroid(points: np.ndarray, indices: list[int], deg: float) -> np.ndarray:
    """Return a copy of ``points`` with ``indices`` rotated by ``deg`` about their
    own centroid. In image coordinates (y down) a positive ``deg`` rotates the
    points **clockwise** on screen."""
    theta = np.radians(deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    out = points.copy()
    sub = out[indices]
    center = sub.mean(axis=0)
    out[indices] = (rot @ (sub - center).T).T + center
    return out


# --- eye-line rotation → one eye_line_angle finding ---------------------------

def test_eye_line_clockwise_rotation():
    """Rotate the eye-line landmarks +6° (clockwise on screen) → exactly one
    eye_line_angle Finding, magnitude 6° ± 0.5°, direction 'tilted clockwise'."""
    ref = canonical_face_points()
    sketch = _rotate_about_centroid(ref, EYE_LINE_INDICES, 6.0)

    findings = measure_angles(_landmarks(ref), _landmarks(sketch))

    assert len(findings) == 1, f"expected one finding, got {[f.id for f in findings]}"
    f = findings[0]
    assert f.id == "eye_line_angle"
    assert f.axis == "angle"
    assert f.units == "deg"
    assert f.level is Level.PLACEMENT
    assert f.direction == "tilted clockwise"
    assert f.magnitude == pytest.approx(6.0, abs=0.5)
    assert f.severity is Severity.MISTAKE  # 5–10° tier (spec §6)


def test_eye_line_counterclockwise_rotation():
    """Rotate the eye-line landmarks -6° → direction 'tilted counterclockwise',
    same magnitude. Confirms the sign convention is correct in both directions."""
    ref = canonical_face_points()
    sketch = _rotate_about_centroid(ref, EYE_LINE_INDICES, -6.0)

    findings = measure_angles(_landmarks(ref), _landmarks(sketch))

    assert len(findings) == 1
    f = findings[0]
    assert f.id == "eye_line_angle"
    assert f.direction == "tilted counterclockwise"
    assert f.magnitude == pytest.approx(6.0, abs=0.5)


# --- page tilt is absorbed → zero findings ------------------------------------

def test_page_tilt_yields_zero():
    """Rotate the WHOLE face 7° → the similarity alignment absorbs the page tilt,
    so every line angle matches and no findings are emitted (principle #2)."""
    ref = canonical_face_points()
    sketch = _rotate_about_centroid(ref, list(range(68)), 7.0)

    findings = measure_angles(_landmarks(ref), _landmarks(sketch))

    assert findings == [], f"page tilt should yield zero findings, got {[f.id for f in findings]}"


def test_identical_yields_zero():
    """Identical landmark sets → zero angle findings (no error to report)."""
    ref = canonical_face_points()
    findings = measure_angles(_landmarks(ref), _landmarks(ref.copy()))
    assert findings == []
