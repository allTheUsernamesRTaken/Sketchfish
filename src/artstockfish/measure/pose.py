"""Head-pose attribution layer (spec §8 M1.5, design principle #4).

A beginner's drawing is **not** a valid projection of any 3D scene — its
inconsistencies *are* the signal — so we never fit 3D geometry to the sketch
(principle #4). The single permitted 3D operation is *attribution*:

1. Estimate head pose for the reference **and** the sketch **independently** with
   :func:`cv2.solvePnP` on a fixed canonical 3D 68-point model (the rigid model is
   the same for both images; only the recovered rotation differs).
2. Compare the rotations. If the heads face meaningfully different ways
   (yaw/pitch difference past :data:`config.POSE_DIFF_OK_MAX`), emit **one**
   ``Level.GLOBAL`` pose finding — "the head is rotated N° further right than the
   reference" — instead of letting that one structural difference smear into a
   storm of correlated local placement errors (spec §2 #5, pitfall §12).
3. **Reproject the reference at the student's pose** and hand that reprojection
   downstream, so the local residuals (placement, angle, proportion) measure only
   what the student drew wrong *given the angle they chose*, not the pose gap.

The reprojection preserves the reference's own identity: each reference landmark is
back-projected to the depth its canonical counterpart sits at under the reference's
estimated pose, re-expressed in the head frame, then re-posed to the student's
rotation and projected again. The canonical 3D model supplies only *depth* — it is
attribution scaffolding, never a reconstruction of the sketch (principle #4). At the
reference's own pose this round-trip reproduces the reference landmarks exactly.

In-plane roll is deliberately **not** a pose finding: a tilted page is absorbed by
the similarity alignment (principle #2), so only yaw/pitch — a genuine turn of the
head in depth — can surface here. All numbers come from deterministic geometry
(principle #1); ``cv2.solvePnP`` is correspondence/attribution, not measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import (
    ANGLE_TIERS,
    POSE_DIFF_OK_MAX,
    POSE_FOCAL_LENGTH_FACTOR,
    POSE_SEVERITY_UNIT,
    POSE_TRIM,
    POSE_WEIGHT,
)
from ..schema import Finding, Landmarks, Level, Severity

_EPS = 1e-9

# Canonical 68-point face in image coordinates (iBUG / 300-W ordering) — the same
# plausible frontal face the project uses elsewhere. pose.py keeps its **own** copy
# (rather than importing one) so the canonical 3D model that defines pose attribution
# is independent of the frozen-contract fixtures and the demo data. Only the 2D shape
# matters here; the depth profile is synthesized below.
_CANONICAL_FACE_2D = np.array(
    [
        # jaw (0–16)
        [102.18, 278.90], [108.91, 307.72], [119.89, 334.59], [134.87, 358.96],
        [153.37, 380.02], [174.83, 397.12], [198.57, 409.69], [223.88, 417.40],
        [250.00, 420.00], [276.12, 417.40], [301.43, 409.69], [325.17, 397.12],
        [346.63, 380.02], [365.13, 358.96], [380.11, 334.59], [391.09, 307.72],
        [397.82, 278.90],
        # right brow (17–21)
        [120.00, 152.00], [142.00, 142.00], [166.00, 139.00], [190.00, 142.00],
        [212.00, 150.00],
        # left brow (22–26)
        [288.00, 150.00], [310.00, 142.00], [334.00, 139.00], [358.00, 142.00],
        [380.00, 152.00],
        # nose bridge (27–30)
        [250.00, 165.00], [250.00, 200.00], [250.00, 235.00], [250.00, 270.00],
        # nose bottom (31–35)
        [228.00, 285.00], [239.00, 290.00], [250.00, 293.00], [261.00, 290.00],
        [272.00, 285.00],
        # right eye (36–41)
        [135.00, 200.00], [150.00, 190.00], [180.00, 190.00], [195.00, 200.00],
        [180.00, 210.00], [150.00, 210.00],
        # left eye (42–47)
        [305.00, 200.00], [320.00, 190.00], [350.00, 190.00], [365.00, 200.00],
        [350.00, 210.00], [320.00, 210.00],
        # mouth outer (48–59)
        [213.00, 345.00], [225.00, 335.00], [238.00, 330.00], [250.00, 332.00],
        [262.00, 330.00], [275.00, 335.00], [287.00, 345.00], [275.00, 357.00],
        [262.00, 362.00], [250.00, 364.00], [238.00, 362.00], [225.00, 357.00],
        # mouth inner (60–67)
        [220.00, 345.00], [235.00, 340.00], [250.00, 341.00], [265.00, 340.00],
        [280.00, 345.00], [265.00, 350.00], [250.00, 351.00], [235.00, 350.00],
    ],
    dtype=np.float64,
)

# Indices that protrude toward the camera (the nose): bridge + nostril base + tip.
_NOSE_INDICES = tuple(range(27, 36))


def _build_canonical_3d() -> np.ndarray:
    """Synthesize a non-degenerate canonical 3D 68-point model from the 2D face.

    The depth (Z) profile is an **ellipsoidal head**: the face centre bulges toward
    the camera and the rim (temples, jaw sides, brow, chin) recedes, which is what
    makes a yaw/pitch turn foreshorten the way a real head does — i.e. what gives
    ``solvePnP`` the signal to recover the rotation. The nose is pushed forward on
    top of that. Coordinates share the image convention (x right, y **down**); the
    model is centred at its own centroid so it is a proper object-frame model.

    This is attribution scaffolding (a rigid model used for both images), not a fit
    to anyone's drawing — principle #4. Absolute anatomical depth is irrelevant; only
    a consistent, non-planar shape is needed for a well-posed PnP.
    """
    xy = _CANONICAL_FACE_2D - _CANONICAL_FACE_2D.mean(axis=0)
    rx = (xy[:, 0].max() - xy[:, 0].min()) / 2.0
    ry = (xy[:, 1].max() - xy[:, 1].min()) / 2.0

    # Ellipsoid: norm 0 at centre → 1 at the rim; depth 0 at centre → +radius at rim
    # (rim recedes from the camera, larger Z = farther under the solvePnP convention).
    norm = np.clip((xy[:, 0] / rx) ** 2 + (xy[:, 1] / ry) ** 2, 0.0, 1.0)
    depth_radius = 0.55 * rx                       # head half-depth ≈ a bit over half width
    z = depth_radius * (1.0 - np.sqrt(1.0 - norm))

    # Nose protrudes toward the camera (smaller Z), graded by how far down the nose.
    protrusion = np.zeros(len(xy))
    protrusion[list(_NOSE_INDICES)] = 0.25 * depth_radius
    protrusion[30] = 0.45 * depth_radius           # nose tip protrudes most
    z = z - protrusion

    model = np.column_stack([xy[:, 0], xy[:, 1], z]).astype(np.float64)
    model -= model.mean(axis=0)                    # centre the object frame
    return model


# The fixed canonical 3D model. Built once at import — it is a constant of the layer.
CANONICAL_FACE_3D = _build_canonical_3d()


@dataclass(frozen=True)
class Pose:
    """A head pose recovered by ``solvePnP``: model→camera rotation plus angles.

    ``rvec``/``tvec`` are the raw ``solvePnP`` outputs; ``R`` is the ``(3, 3)``
    rotation. ``yaw``/``pitch``/``roll`` (degrees) are a convenience read-out of the
    *absolute* model→camera rotation and are mainly for inspection — pose
    *differences* are computed from the relative rotation in :func:`pose_difference`,
    which avoids Euler-wrap pitfalls.
    """

    rvec: np.ndarray            # (3, 1) Rodrigues rotation vector (model→camera)
    tvec: np.ndarray            # (3, 1) translation (model→camera)
    R: np.ndarray               # (3, 3) rotation matrix
    yaw: float                  # deg, about the vertical (Y) axis
    pitch: float                # deg, about the horizontal (X) axis
    roll: float                 # deg, in-plane (Z) axis


def camera_matrix(image_size: tuple[int, int]) -> np.ndarray:
    """Pinhole intrinsics ``K`` for an image of ``image_size`` (spec §4 stack).

    Long focal length (``POSE_FOCAL_LENGTH_FACTOR × max(side)``) → mild perspective.
    The principal point is the image centre; any constant offset between it and the
    face centroid is absorbed by ``solvePnP``'s translation, so the exact value only
    has to be *consistent* between the pose solve and the reprojection (it is).
    """
    w, h = float(image_size[0]), float(image_size[1])
    f = POSE_FOCAL_LENGTH_FACTOR * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_pose(points_2d: np.ndarray, image_size: tuple[int, int]) -> Pose:
    """Estimate head pose from 2D landmarks via ``cv2.solvePnP`` on the canonical 3D model.

    **Robust and deterministic** (principles #3, #7). A first iterative-PnP solve over
    all 68 points is refined by a *trimmed* re-solve: the worst ``POSE_TRIM`` fraction
    of points by reprojection error are dropped and the pose is re-estimated on the
    inliers. This keeps a *local* drawing error (e.g. one eye drawn out of place) from
    dragging the global pose toward itself — the same logic ``robust_align`` applies to
    the 2D similarity fit — so the residual that survives downstream is the local error,
    not a pose the local error created. No RANSAC randomness, so the estimate is stable.

    The rotation is the rigid model→camera orientation; the sketch's drawing errors
    never enter a 3D fit, they only perturb this rigid estimate (principle #4).
    """
    pts = np.asarray(points_2d, dtype=np.float64)
    if pts.shape != (CANONICAL_FACE_3D.shape[0], 2):
        raise ValueError(
            f"pose needs the full {CANONICAL_FACE_3D.shape[0]}-point 2D set; "
            f"got {pts.shape}"
        )
    K = camera_matrix(image_size)
    # Initial solve with SQPNP: a global, deterministic, init-free PnP that picks the
    # geometrically correct branch. (Plain iterative PnP can land on the planar two-fold
    # ambiguity — flipping a near-frontal face upside down — which would fabricate a
    # bogus pose difference between two front-facing portraits.)
    rvec, tvec = _solve_pnp(CANONICAL_FACE_3D, pts, K, flag=cv2.SOLVEPNP_SQPNP)

    # Trimmed refinement: drop the worst-fitting points and re-solve on the inliers,
    # seeded from the SQPNP solve. One pass is enough to shed a localized outlier.
    n = len(pts)
    keep = max(int(round(n * (1.0 - POSE_TRIM))), 6)
    if keep < n:
        err = _reprojection_error(CANONICAL_FACE_3D, pts, K, rvec, tvec)
        inliers = np.argsort(err)[:keep]
        rvec, tvec = _solve_pnp(
            CANONICAL_FACE_3D[inliers], pts[inliers], K, rvec=rvec, tvec=tvec
        )

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _euler_yaw_pitch_roll(R)
    return Pose(rvec=rvec, tvec=tvec, R=R, yaw=yaw, pitch=pitch, roll=roll)


def _solve_pnp(
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
    flag: int = cv2.SOLVEPNP_ITERATIVE,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ``solvePnP`` wrapper, optionally seeded with a guess.

    Only the iterative solver accepts an extrinsic guess; the seeded path (used by the
    trimmed refinement) therefore always runs iterative refinement from the seed.
    """
    use_guess = rvec is not None and tvec is not None
    ok, rvec, tvec = cv2.solvePnP(
        object_pts.reshape(-1, 1, 3).astype(np.float64),
        image_pts.reshape(-1, 1, 2).astype(np.float64),
        K,
        None,
        rvec=rvec.copy() if use_guess else None,
        tvec=tvec.copy() if use_guess else None,
        useExtrinsicGuess=use_guess,
        flags=cv2.SOLVEPNP_ITERATIVE if use_guess else flag,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to recover a head pose")
    return rvec, tvec


def _reprojection_error(
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Per-point reprojection error (pixels) of ``object_pts`` under ``(rvec, tvec)``."""
    proj, _ = cv2.projectPoints(object_pts.reshape(-1, 1, 3), rvec, tvec, K, None)
    return np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1)


def _euler_yaw_pitch_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Read (yaw about Y, pitch about X, roll about Z) in degrees from a rotation.

    Uses the small-angle-stable Rodrigues read-out (axis·angle components) rather
    than a full Euler decomposition: for the moderate head turns this layer targets
    it is monotone and free of gimbal/wrap ambiguity, and for a pure single-axis
    rotation it returns exactly that axis's angle.
    """
    rvec, _ = cv2.Rodrigues(R)
    pitch, yaw, roll = (float(np.degrees(a)) for a in rvec.ravel())
    return yaw, pitch, roll


def pose_difference(reference: Pose, student: Pose) -> tuple[float, float, float]:
    """Relative (yaw, pitch, roll) of the student's head w.r.t. the reference, in degrees.

    Computed from the relative rotation ``R_student · R_referenceᵀ`` so it is the
    clean "how much further has the head turned" signal, independent of either
    absolute pose. Sign convention (resolved against the synthetic +yaw projection in
    the M1.5 tests): +yaw = turned further toward image-right, +pitch = tipped
    further down, +roll = rotated clockwise on screen.
    """
    R_rel = student.R @ reference.R.T
    return _euler_yaw_pitch_roll(R_rel)


def reproject_reference(
    reference_2d: np.ndarray,
    reference_pose: Pose,
    student_pose: Pose,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Reproject the reference's landmarks as if its head were at the student's pose.

    Identity-preserving (spec §8 M1.5): each reference landmark keeps the camera-frame
    **depth** of its canonical counterpart at the reference pose, is back-projected to
    a 3D point, lifted into the head frame, then re-posed to the student's rotation and
    re-projected. The canonical model supplies depth only — never a 3D reconstruction
    of the sketch (principle #4). At the reference's own pose this returns the input
    landmarks unchanged, so a perfect-but-rotated student reprojects onto the sketch.
    """
    ref2d = np.asarray(reference_2d, dtype=np.float64)
    K = camera_matrix(image_size)
    f = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]

    R_ref, t_ref = reference_pose.R, reference_pose.tvec.reshape(3)
    R_stu, t_stu = student_pose.R, student_pose.tvec.reshape(3)

    # Depth each landmark sits at under the reference pose comes from the canonical
    # model; the reference's *own* (u, v) are back-projected to that depth, so the
    # reference's identity (its departure from canonical) is retained exactly.
    canon_cam = (R_ref @ CANONICAL_FACE_3D.T).T + t_ref      # (N, 3) camera frame
    z = canon_cam[:, 2]
    cam_ref = np.column_stack([
        (ref2d[:, 0] - cx) * z / f,
        (ref2d[:, 1] - cy) * z / f,
        z,
    ])
    # Camera frame → head frame (using the reference pose) → student's camera frame.
    head = (R_ref.T @ (cam_ref - t_ref).T).T
    cam_stu = (R_stu @ head.T).T + t_stu

    zs = np.where(np.abs(cam_stu[:, 2]) < _EPS, _EPS, cam_stu[:, 2])
    u = f * cam_stu[:, 0] / zs + cx
    v = f * cam_stu[:, 1] / zs + cy
    return np.column_stack([u, v])


def pose_finding(reference: Pose, student: Pose) -> Finding | None:
    """One ``Level.GLOBAL`` pose finding if the heads face different ways, else ``None``.

    Yaw and pitch are compared (roll = page tilt is absorbed by alignment, never
    flagged — principle #2). If the larger of the two crosses
    :data:`config.POSE_DIFF_OK_MAX`, a single finding describes that dominant turn
    (spec §8 M1.5: "emit ONE Level.GLOBAL finding"); the other component rides along
    in ``evidence``. Magnitude is in degrees and the severity uses the shared angle
    tiers (§6).
    """
    yaw, pitch, roll = pose_difference(reference, student)

    # Dominant out-of-plane axis decides the single finding.
    if abs(yaw) >= abs(pitch):
        axis_id, magnitude, signed = "pose_yaw", abs(yaw), yaw
        direction = "rotated further right" if signed > 0 else "rotated further left"
    else:
        axis_id, magnitude, signed = "pose_pitch", abs(pitch), pitch
        direction = "rotated further down" if signed > 0 else "rotated further up"

    if magnitude < POSE_DIFF_OK_MAX:
        return None

    severity = _severity_from_tiers(magnitude, ANGLE_TIERS)
    return Finding(
        id=axis_id,
        level=Level.GLOBAL,
        severity=severity,
        feature="head",
        axis="pose",
        direction=direction,
        magnitude=magnitude,
        units="deg",
        score=POSE_WEIGHT * magnitude / POSE_SEVERITY_UNIT,
        evidence={
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "reference_rvec": reference.rvec.ravel().tolist(),
            "student_rvec": student.rvec.ravel().tolist(),
        },
    )


def _severity_from_tiers(magnitude: float, tiers: tuple[float, float, float]) -> Severity:
    """Map a non-negative magnitude to a severity using ``(ok, inacc, mistake)``
    upper bounds (spec §6). At/above the last bound is a BLUNDER."""
    ok_max, inacc_max, mistake_max = tiers
    if magnitude < ok_max:
        return Severity.OK
    if magnitude < inacc_max:
        return Severity.INACCURACY
    if magnitude < mistake_max:
        return Severity.MISTAKE
    return Severity.BLUNDER


@dataclass(frozen=True)
class PoseConditioning:
    """Result of the pose stage handed to the pipeline before local residuals run.

    ``reference`` is what downstream measurement should compare the sketch against:
    the **reprojected** reference when the pose difference was attributed, otherwise
    the original reference untouched. ``finding`` is the single GLOBAL pose finding
    (or ``None``). ``pose`` is the per-image read-out stored on the ``Report``.
    """

    reference: Landmarks
    finding: Finding | None
    pose: dict | None
    applied: bool


def condition_on_pose(reference: Landmarks, sketch: Landmarks) -> PoseConditioning:
    """Pose-attribution stage (spec §7 diagram, inserted before residuals).

    Estimates both heads' poses, and — only when they differ past the threshold —
    emits the single GLOBAL pose finding and swaps in the reference reprojected at the
    student's pose so the downstream residuals are measured "given the angle the
    student drew" (spec §8 M1.5). Below threshold (or if PnP cannot be solved) it is a
    no-op: the original reference passes through unchanged, so the M0 pipeline is
    unaffected. Pose estimates are always reported in ``pose`` for inspection.
    """
    ref_pts = np.asarray(reference.points, dtype=np.float64)
    sketch_pts = np.asarray(sketch.points, dtype=np.float64)

    try:
        ref_pose = estimate_pose(ref_pts, reference.image_size)
        stu_pose = estimate_pose(sketch_pts, sketch.image_size)
    except (cv2.error, RuntimeError, ValueError):
        # Pose is an optional attribution layer; if it cannot be solved, fall back to
        # the plain residual pipeline rather than failing the whole critique.
        return PoseConditioning(reference=reference, finding=None, pose=None, applied=False)

    yaw, pitch, roll = pose_difference(ref_pose, stu_pose)
    pose_dict = {
        "reference": {"yaw": ref_pose.yaw, "pitch": ref_pose.pitch, "roll": ref_pose.roll},
        "student": {"yaw": stu_pose.yaw, "pitch": stu_pose.pitch, "roll": stu_pose.roll},
        "difference": {"yaw": yaw, "pitch": pitch, "roll": roll},
    }

    finding = pose_finding(ref_pose, stu_pose)
    if finding is None:
        return PoseConditioning(
            reference=reference, finding=None, pose=pose_dict, applied=False
        )

    reprojected = reproject_reference(ref_pts, ref_pose, stu_pose, reference.image_size)
    conditioned = Landmarks(
        points=reprojected, names=reference.names, image_size=reference.image_size
    )
    return PoseConditioning(
        reference=conditioned, finding=finding, pose=pose_dict, applied=True
    )
