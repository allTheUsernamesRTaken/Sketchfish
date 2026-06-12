"""M1.5 acceptance tests — head-pose attribution (spec §8 M1.5, principle #4).

Run: ``pytest tests/test_pose.py -q``

300W-LP (the same faces at labelled yaw angles) is not downloaded in this repo, so
— exactly as the task permits — we synthesize the pose pair by projecting the layer's
own canonical 3D model to 2D at two rotations (``cv2.projectPoints``). Using the same
camera model and 3D points that ``measure.pose`` solves against means the recovered
poses are exact up to the drawing errors we deliberately inject, which is precisely
what the tests need to isolate.

- **M1.5-T1:** frontal reference vs a sketch at +10° yaw → **exactly one** GLOBAL
  pose finding and **zero** PLACEMENT findings (the head turn is attributed once, not
  smeared into a storm of correlated local errors — spec §2 #5, pitfall §12).
- **M1.5-T2:** +10° yaw **and** the left eye shifted up 5% of head height → the pose
  finding **and** the eye finding both surface, and the eye magnitude is still
  5% ± 1% *after* pose conditioning (the reprojection removed the pose gap, leaving
  the genuine local error intact).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from artstockfish.frame import LANDMARK_NAMES_68, SEMANTIC_GROUPS
from artstockfish.measure.pose import CANONICAL_FACE_3D, camera_matrix, condition_on_pose
from artstockfish.pipeline import critique_pair
from artstockfish.schema import Landmarks, Level, Severity

IMG_SIZE = (500, 500)
_K = camera_matrix(IMG_SIZE)
# Place the (origin-centred) model in front of the camera; a depth ≈ focal length
# keeps the frontal projection life-sized and only mildly perspective.
_TVEC = np.array([0.0, 0.0, _K[0, 0]], dtype=np.float64)
_YAW_DEG = 10.0


def _project(rotation: np.ndarray) -> np.ndarray:
    """Project the canonical 3D face to 2D under ``rotation`` (model→camera)."""
    cam = (rotation @ CANONICAL_FACE_3D.T).T + _TVEC
    u = _K[0, 0] * cam[:, 0] / cam[:, 2] + _K[0, 2]
    v = _K[1, 1] * cam[:, 1] / cam[:, 2] + _K[1, 2]
    return np.column_stack([u, v])


def _landmarks(points: np.ndarray) -> Landmarks:
    return Landmarks(points=points, names=LANDMARK_NAMES_68, image_size=IMG_SIZE)


def _frontal_reference() -> Landmarks:
    return _landmarks(_project(np.eye(3)))


def _yawed_sketch(deg: float = _YAW_DEG) -> np.ndarray:
    """2D landmarks of the canonical face turned ``deg`` about the vertical axis."""
    rot, _ = cv2.Rodrigues(np.array([0.0, np.radians(deg), 0.0]))
    return _project(rot)


# --- M1.5-T1: pure pose difference → one GLOBAL finding, no placement storm ----

def test_m1_5_t1_pure_yaw_one_global_finding():
    reference = _frontal_reference()
    sketch = _landmarks(_yawed_sketch())

    result = critique_pair(reference, sketch)
    findings = result.report.findings

    global_findings = [f for f in findings if f.level is Level.GLOBAL]
    placement_findings = [f for f in findings if f.level is Level.PLACEMENT]

    assert len(global_findings) == 1, (
        f"expected exactly one GLOBAL pose finding, got {[f.id for f in global_findings]}"
    )
    pose = global_findings[0]
    assert pose.id == "pose_yaw"
    assert pose.axis == "pose"
    assert pose.units == "deg"
    assert pose.feature == "head"
    assert pose.direction == "rotated further right"   # +yaw → image-right
    assert pose.magnitude == pytest.approx(_YAW_DEG, abs=1.0)

    # The head turn is attributed once; it does NOT smear into local placement errors.
    assert placement_findings == [], (
        f"pose turn leaked into placement findings: {[f.id for f in placement_findings]}"
    )
    # And the whole report is just that one finding (no SHAPE/proportion noise either).
    assert len(findings) == 1

    # The pose estimates are recorded on the report for inspection.
    assert result.report.pose is not None
    assert result.report.pose["difference"]["yaw"] == pytest.approx(_YAW_DEG, abs=1.0)


def test_m1_5_t1_reprojection_matches_sketch():
    """The reference reprojected at the student's pose lands on the sketch.

    Confirms the single-finding result is genuine pose conditioning — the reference
    really is re-posed onto the student's angle — not the residual stage quietly
    missing everything (Ground Rule 5).
    """
    reference = _frontal_reference()
    sketch = _landmarks(_yawed_sketch())

    cond = condition_on_pose(reference, sketch)
    assert cond.applied is True
    max_px_err = np.abs(cond.reference.points - sketch.points).max()
    assert max_px_err < 1.0, f"reprojection should match the sketch; off by {max_px_err:.3f}px"


# --- M1.5-T2: pose + a real local error → both, eye magnitude preserved --------

def test_m1_5_t2_yaw_plus_eye_shift_both_present():
    reference = _frontal_reference()
    sketch_pts = _yawed_sketch()

    # Shift the left-eye group up 5% of head height (image y is down → up is −y),
    # measured in the projected sketch image so the injected magnitude is well-defined.
    head_height = sketch_pts[:, 1].max() - sketch_pts[:, 1].min()
    sketch_pts = sketch_pts.copy()
    sketch_pts[list(SEMANTIC_GROUPS["left_eye"])] += np.array([0.0, -0.05 * head_height])
    sketch = _landmarks(sketch_pts)

    result = critique_pair(reference, sketch)
    findings = result.report.findings
    by_id = {f.id: f for f in findings}

    # Both the pose finding AND the eye placement finding are present.
    assert "pose_yaw" in by_id, f"missing pose finding; got {sorted(by_id)}"
    assert "left_eye_vertical" in by_id, f"missing eye finding; got {sorted(by_id)}"

    pose = by_id["pose_yaw"]
    assert pose.level is Level.GLOBAL
    assert pose.magnitude == pytest.approx(_YAW_DEG, abs=1.5)

    eye = by_id["left_eye_vertical"]
    assert eye.level is Level.PLACEMENT
    assert eye.direction == "too high"
    assert eye.units == "%head_height"
    # The eye magnitude survives pose conditioning at 5% ± 1% (spec §8 M1.5-T2).
    assert eye.magnitude == pytest.approx(5.0, abs=1.0)

    # Coarse-to-fine: the GLOBAL pose finding outranks the PLACEMENT eye finding.
    assert findings[0].id == "pose_yaw"
    assert findings[0].level <= findings[-1].level


def test_m1_5_t2_eye_finding_not_a_pose_artifact():
    """Without the eye shift the same yawed sketch yields no eye finding.

    Pins that the T2 eye finding is the injected error, not something pose
    conditioning manufactures from the turn alone.
    """
    reference = _frontal_reference()
    clean = critique_pair(reference, _landmarks(_yawed_sketch()))
    assert "left_eye_vertical" not in {f.id for f in clean.report.findings}


# --- a frontal pair (no pose difference) is a clean no-op ----------------------

def test_no_pose_difference_is_a_noop():
    """Reference vs. an identical frontal sketch → no pose finding at all.

    The pose stage must not fire (or perturb the report) when the heads face the
    same way; identical inputs stay a perfect score (guards the M0 path).
    """
    reference = _frontal_reference()
    result = critique_pair(reference, _frontal_reference())

    assert result.report.findings == ()
    assert result.report.accuracy_score == 100.0
    # Poses were still estimated and recorded, and they agree.
    assert result.report.pose is not None
    assert abs(result.report.pose["difference"]["yaw"]) < 4.0
