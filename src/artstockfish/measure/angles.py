"""Feature angle comparisons (spec §9.4).

For each relevant line feature — the **eye line**, the **mouth line**, and the
left/right **jaw tangents** — we fit a least-squares/PCA line through its landmarks
in *both* images and critique the *difference* in orientation, in degrees. The
target is always "match the reference," never a textbook angle (spec §9.4).

All measurement is deterministic geometry (design principle #1). The sketch is
registered to the reference with a robust **similarity** transform only
(principle #2/#3, :func:`artstockfish.align.robust_align`), so a globally tilted
page is absorbed by the alignment and leaves zero angle residual — only a real
relational/contour tilt survives the fit.

Sign convention: angles are measured with :func:`numpy.arctan2` in image
coordinates (x right, y **down**), so increasing angle is a **clockwise** tilt on
screen. A positive ``sketch − reference`` delta therefore reads "tilted
clockwise"; negative reads "tilted counterclockwise".
"""

from __future__ import annotations

import numpy as np

from ..align import apply_similarity, robust_align
from ..config import (
    ANGLE_LINES,
    ANGLE_OK_MAX,
    ANGLE_TIERS,
    DEFAULT_IMPORTANCE_WEIGHT,
    IMPORTANCE_WEIGHTS,
)
from ..schema import Finding, Landmarks, Level, Severity

_EPS = 1e-9


# --- small helpers -----------------------------------------------------------

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


def _line_direction(points: np.ndarray) -> np.ndarray:
    """Unit direction of the best-fit line through ``points`` (PCA principal axis).

    A line is orientation-ambiguous (``v`` and ``-v`` describe the same line), so
    the direction is canonicalized to point toward image-right (``v[0] >= 0``);
    near-vertical lines are oriented downward. This makes the per-image angle
    single-valued so the sketch−reference delta is meaningful.
    """
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v = vt[0]
    if v[0] < -_EPS or (abs(v[0]) <= _EPS and v[1] < 0):
        v = -v
    return v / max(np.linalg.norm(v), _EPS)


def _line_angle_deg(points: np.ndarray) -> float:
    """Orientation of the best-fit line through ``points``, in degrees.

    Image coordinates (y down): ``arctan2(vy, vx)`` so increasing angle is a
    clockwise tilt on screen. Range is ``(-90, 90]`` after canonicalization.
    """
    v = _line_direction(points)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap_pm90(angle_deg: float) -> float:
    """Wrap a line-angle difference into ``(-90, 90]`` (lines repeat every 180°)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def _weight(weight_key: str) -> float:
    """Importance weight for a line's ranking score (spec §9.5)."""
    return IMPORTANCE_WEIGHTS.get(weight_key, DEFAULT_IMPORTANCE_WEIGHT)


# --- public API --------------------------------------------------------------

def measure_angles(
    reference: Landmarks,
    sketch: Landmarks,
    *,
    align: bool = True,
) -> list[Finding]:
    """Feature-angle findings between a reference and a sketch (spec §9.4).

    Args:
        reference: the reference (target) landmarks, full 68-point set.
        sketch: the student's landmarks, same ordering/length as ``reference``.
        align: when ``True`` (default) the sketch is registered to the reference
            with a robust similarity transform first (§9.1) so global page tilt is
            absorbed. Pass ``False`` if the sketch is already aligned (e.g. the
            pose stage reprojected it).

    Returns:
        ``Finding`` objects (``axis="angle"``, ``units="deg"``), one per line
        whose orientation differs from the reference by more than the OK floor.
    """
    ref_pts = np.asarray(reference.points, dtype=np.float64)
    sketch_pts = np.asarray(sketch.points, dtype=np.float64)
    if ref_pts.shape != sketch_pts.shape:
        raise ValueError(
            f"reference and sketch must share shape; got {ref_pts.shape} vs "
            f"{sketch_pts.shape}"
        )

    if align:
        s, R, t = robust_align(ref_pts, sketch_pts)
        aligned = apply_similarity(s, R, t, sketch_pts)
    else:
        aligned = sketch_pts

    return angle_findings(ref_pts, aligned)


def angle_findings(
    ref_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
) -> list[Finding]:
    """Core angle comparison over already-aligned points (pure geometry).

    ``aligned_sketch_points`` must already be registered to ``ref_points`` (same
    coordinate system), so a same-system line-angle difference is the drawing
    error and nothing the similarity transform could have absorbed.
    """
    ref_points = np.asarray(ref_points, dtype=np.float64)
    aligned_sketch_points = np.asarray(aligned_sketch_points, dtype=np.float64)

    findings: list[Finding] = []
    for spec in ANGLE_LINES.values():
        idx = list(spec["indices"])
        ref_g = ref_points[idx]
        sketch_g = aligned_sketch_points[idx]

        ref_angle = _line_angle_deg(ref_g)
        sketch_angle = _line_angle_deg(sketch_g)
        delta = _wrap_pm90(sketch_angle - ref_angle)
        magnitude = abs(delta)

        severity = _severity_from_tiers(magnitude, ANGLE_TIERS)
        if severity is Severity.OK:
            continue

        direction = "tilted clockwise" if delta > 0 else "tilted counterclockwise"
        weight = _weight(spec["weight_key"])
        findings.append(
            Finding(
                id=spec["id"],
                level=Level(spec["level"]),
                severity=severity,
                feature=spec["feature"],
                axis="angle",
                direction=direction,
                magnitude=magnitude,
                units="deg",
                score=weight * magnitude / ANGLE_OK_MAX,
                evidence={
                    "indices": idx,
                    "ref_points": ref_g,
                    "sketch_points": sketch_g,
                    "ref_angle_deg": ref_angle,
                    "sketch_angle_deg": sketch_angle,
                    "delta_deg": delta,
                    "ref_direction": _line_direction(ref_g),
                    "sketch_direction": _line_direction(sketch_g),
                },
            )
        )

    return findings
