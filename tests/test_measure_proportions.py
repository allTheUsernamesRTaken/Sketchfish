"""Wave 1B acceptance tests — canon-ratio proportions (spec §9.4).

Run: ``pytest tests/test_measure_proportions.py -q``

These exercise ``measure/proportions.py`` on the hardcoded canonical 68-point
face (``tests/fixtures.py``):

- **Matched ratios → zero findings.** A sketch whose canon ratios equal the
  reference's produces no findings (the system critiques the *difference*, and a
  difference of zero is not a finding — spec §9.4).
- **A known interocular widening → one proportion finding** of the right id, sign
  and magnitude. To keep the injected error a *single* canon-ratio error we widen
  the eye spacing AND widen the mouth in the same proportion, so that the only
  ratio that departs from the reference is interocular/eye-width — the
  mouth/interocular ratio (which also reads the interocular distance) stays
  matched. This isolates the rule under test; it is test-input construction, not
  a special case in the implementation (Ground Rule 5).
"""

from __future__ import annotations

import numpy as np
import pytest

from artstockfish.config import PROPORTION_OK_MAX
from artstockfish.measure.proportions import proportion_findings
from artstockfish.schema import Level, Severity

from fixtures import canonical_face_points

# iBUG indices the perturbations touch.
_RIGHT_EYE = list(range(36, 42))
_LEFT_EYE = list(range(42, 48))
_MOUTH = list(range(48, 68))
_MOUTH_CENTER_X = 250.0  # the canonical face's mouth is symmetric about x = 250


# --- matched ratios → zero findings -------------------------------------------

def test_matched_ratios_yield_no_findings():
    """Identical ratio set (an exact copy) → no proportion findings."""
    ref = canonical_face_points()
    assert proportion_findings(ref, ref.copy()) == []


def test_sub_threshold_widening_stays_silent():
    """A widening below the OK floor is matched enough → still zero findings."""
    ref = canonical_face_points()
    sketch = _widen_interocular(ref, factor=1.0 + (PROPORTION_OK_MAX / 100.0) * 0.5)
    assert proportion_findings(ref, sketch) == []


# --- a known interocular widening → exactly one proportion finding ------------

def _widen_interocular(ref: np.ndarray, factor: float) -> np.ndarray:
    """Widen the inner-corner gap by ``factor`` while holding mouth/interocular.

    Each eye group is rigidly translated outward (eye *width* is unchanged, so
    interocular/eye-width scales by exactly ``factor``); the mouth is scaled about
    its centre by the same ``factor`` so mouth-width/interocular is unchanged.
    Both moves are symmetric about the face midline, so the centroid, the midline
    axis and every vertical ratio are untouched.
    """
    sketch = ref.copy()
    gap = ref[42, 0] - ref[39, 0]            # inner-corner gap (eyes share a row)
    delta = 0.5 * (factor - 1.0) * gap        # per-eye outward shift
    sketch[_RIGHT_EYE, 0] -= delta            # subject's right eye = image left
    sketch[_LEFT_EYE, 0] += delta
    sketch[_MOUTH, 0] = _MOUTH_CENTER_X + (ref[_MOUTH, 0] - _MOUTH_CENTER_X) * factor
    return sketch


def test_interocular_widening_one_finding_correct_sign_and_magnitude():
    """Widen interocular by a known 15% → exactly one finding, the right one."""
    ref = canonical_face_points()
    injected_pct = 15.0
    sketch = _widen_interocular(ref, factor=1.0 + injected_pct / 100.0)

    findings = proportion_findings(ref, sketch)

    # Exactly one canon ratio departed from the reference.
    assert len(findings) == 1, [f.id for f in findings]
    f = findings[0]

    assert f.id == "interocular_eye_width"
    assert f.axis == "proportion"
    assert f.level is Level.PLACEMENT
    assert f.units == "%ratio"

    # Correct sign: the eyes are too far apart for their width.
    assert f.direction == "too wide"
    assert f.evidence["deviation_pct"] > 0

    # Correct magnitude: 15% ratio deviation (geometry is exact) → MISTAKE tier.
    assert f.magnitude == pytest.approx(injected_pct, abs=0.5)
    assert f.severity is Severity.MISTAKE


def test_interocular_narrowing_flips_direction():
    """Narrowing the eye spacing flips the sign/direction but keeps the id."""
    ref = canonical_face_points()
    sketch = _widen_interocular(ref, factor=1.0 - 0.15)

    findings = proportion_findings(ref, sketch)

    assert len(findings) == 1, [f.id for f in findings]
    f = findings[0]
    assert f.id == "interocular_eye_width"
    assert f.direction == "too narrow"
    assert f.evidence["deviation_pct"] < 0
    assert f.magnitude == pytest.approx(15.0, abs=0.5)


def test_widening_without_mouth_comp_also_flags_mouth():
    """Sanity: the widening genuinely couples two ratios.

    Without the proportional mouth compensation, translating the eyes apart pulls
    *both* interocular/eye-width (too wide) and mouth/interocular (too narrow) off
    the reference — confirming the single-finding test isolates the rule honestly
    rather than the implementation simply missing the second ratio.
    """
    ref = canonical_face_points()
    sketch = ref.copy()
    gap = ref[42, 0] - ref[39, 0]
    delta = 0.5 * 0.15 * gap
    sketch[_RIGHT_EYE, 0] -= delta
    sketch[_LEFT_EYE, 0] += delta            # eyes apart, mouth left untouched

    ids = {f.id for f in proportion_findings(ref, sketch)}
    assert ids == {"interocular_eye_width", "mouth_interocular"}
