"""Per-group landmark residual decomposition (spec §9.3).

For each semantic group (left eye, right eye, nose, mouth, jaw, brows) we take the
**mean residual vector** of the group in the reference's face frame (§9.2) and
split it into vertical/horizontal components. Each component over the noise floor
becomes one ``Level.PLACEMENT`` ``Finding`` ("too high / too low", "too far left /
too far right"). Separately, the group's **internal spread ratio** (sketch vs
reference) yields a scale finding ("too large / too small").

All measurement is deterministic geometry (design principle #1): nothing here is
learned, and every magnitude is expressed in size/tilt-invariant face-frame units
(% of head height for placement, % of area for scale).

The sketch is registered to the reference with a robust **similarity** transform
only (principle #2/#3, :func:`artstockfish.align.robust_align`) — never affine —
so what survives the fit is the drawing error and nothing the transform could
legitimately absorb (global tilt, uniform scale, translation).
"""

from __future__ import annotations

import numpy as np

from ..align import apply_similarity, robust_align
from ..config import (
    AREA_OK_MAX,
    AREA_TIERS,
    DEFAULT_IMPORTANCE_WEIGHT,
    DISPLACEMENT_OK_MAX,
    DISPLACEMENT_TIERS,
    IMPORTANCE_WEIGHTS,
    LANDMARK_GROUP_FEATURE_NAMES,
    LANDMARK_GROUP_WEIGHT_KEYS,
)
from ..frame import SEMANTIC_GROUPS, FaceFrame, build_face_frame
from ..schema import Finding, Landmarks, Level, Severity

# Measurement groups (spec §9.3). These reuse ``frame.SEMANTIC_GROUPS`` but merge
# the nose bridge + nostril base into a single "nose" a teacher would name.
MEASURE_GROUPS: dict[str, tuple[int, ...]] = {
    "jaw": SEMANTIC_GROUPS["jaw"],
    "right_brow": SEMANTIC_GROUPS["right_brow"],
    "left_brow": SEMANTIC_GROUPS["left_brow"],
    "nose": SEMANTIC_GROUPS["nose_bridge"] + SEMANTIC_GROUPS["nose_bottom"],
    "right_eye": SEMANTIC_GROUPS["right_eye"],
    "left_eye": SEMANTIC_GROUPS["left_eye"],
    "mouth": SEMANTIC_GROUPS["mouth"],
}

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


def _group_weight(group: str) -> float:
    """Importance weight for a group's ranking score (spec §9.5)."""
    key = LANDMARK_GROUP_WEIGHT_KEYS.get(group, group)
    return IMPORTANCE_WEIGHTS.get(key, DEFAULT_IMPORTANCE_WEIGHT)


def _feature_name(group: str) -> str:
    """Human feature name for a group ("left_eye" → "left eye")."""
    return LANDMARK_GROUP_FEATURE_NAMES.get(group, group.replace("_", " "))


def _spread(points: np.ndarray) -> float:
    """RMS radius of a point cloud about its centroid (linear size proxy)."""
    centroid = points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))


# --- public API --------------------------------------------------------------

def measure_landmarks(
    reference: Landmarks,
    sketch: Landmarks,
    *,
    frame: FaceFrame | None = None,
    align: bool = True,
) -> list[Finding]:
    """Per-group placement + scale findings between a reference and a sketch.

    Args:
        reference: the reference (target) landmarks, full 68-point set.
        sketch: the student's landmarks, same ordering/length as ``reference``.
        frame: precomputed reference face frame; built from ``reference`` if
            omitted (the residuals are always expressed in the reference frame,
            §9.2).
        align: when ``True`` (default) the sketch is registered to the reference
            with a robust similarity transform first (§9.1). Pass ``False`` if the
            sketch is already aligned (e.g. the pose stage reprojected it).

    Returns:
        ``Finding`` objects (``Level.PLACEMENT``), one per over-threshold
        vertical/horizontal component and one per over-threshold group scale.
    """
    ref_pts = np.asarray(reference.points, dtype=np.float64)
    sketch_pts = np.asarray(sketch.points, dtype=np.float64)
    if ref_pts.shape != sketch_pts.shape:
        raise ValueError(
            f"reference and sketch must share shape; got {ref_pts.shape} vs "
            f"{sketch_pts.shape}"
        )

    if frame is None:
        frame = build_face_frame(ref_pts)

    if align:
        s, R, t = robust_align(ref_pts, sketch_pts)
        aligned = apply_similarity(s, R, t, sketch_pts)
    else:
        aligned = sketch_pts

    return landmark_findings(ref_pts, aligned, frame)


def landmark_findings(
    ref_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
    frame: FaceFrame,
) -> list[Finding]:
    """Core decomposition over already-aligned points (pure geometry).

    ``aligned_sketch_points`` must already be registered to ``ref_points``.
    Residual components come from :meth:`FaceFrame.residual_components`, which
    returns ``(horizontal, vertical)`` in % of head height with +vertical = up
    the face ("too high") and +horizontal = toward image-right.
    """
    ref_points = np.asarray(ref_points, dtype=np.float64)
    aligned_sketch_points = np.asarray(aligned_sketch_points, dtype=np.float64)
    components = frame.residual_components(ref_points, aligned_sketch_points)

    findings: list[Finding] = []
    for group, members in MEASURE_GROUPS.items():
        idx = list(members)
        ref_g = ref_points[idx]
        sketch_g = aligned_sketch_points[idx]
        mean_res = components[idx].mean(axis=0)  # (h, v) in % head height
        h, v = float(mean_res[0]), float(mean_res[1])
        weight = _group_weight(group)
        feature = _feature_name(group)

        findings.extend(
            _placement_findings(group, feature, weight, idx, ref_g, sketch_g, mean_res, h, v)
        )
        scale = _scale_finding(group, feature, weight, idx, ref_g, sketch_g)
        if scale is not None:
            findings.append(scale)

    return findings


# --- per-finding builders ----------------------------------------------------

def _placement_findings(
    group: str,
    feature: str,
    weight: float,
    idx: list[int],
    ref_g: np.ndarray,
    sketch_g: np.ndarray,
    mean_res: np.ndarray,
    h: float,
    v: float,
) -> list[Finding]:
    """Vertical and/or horizontal placement findings for one group."""
    out: list[Finding] = []

    axes = (
        ("vertical", v, ("too high", "too low")),
        ("horizontal", h, ("too far right", "too far left")),
    )
    for axis, component, (pos_dir, neg_dir) in axes:
        magnitude = abs(component)
        severity = _severity_from_tiers(magnitude, DISPLACEMENT_TIERS)
        if severity is Severity.OK:
            continue
        direction = pos_dir if component > 0 else neg_dir
        out.append(
            Finding(
                id=f"{group}_{axis}",
                level=Level.PLACEMENT,
                severity=severity,
                feature=feature,
                axis=axis,
                direction=direction,
                magnitude=magnitude,
                units="%head_height",
                score=weight * magnitude / DISPLACEMENT_OK_MAX,
                evidence={
                    "group": group,
                    "indices": idx,
                    "ref_points": ref_g,
                    "sketch_points": sketch_g,
                    "mean_residual": mean_res,  # (h, v) in % head height
                    "component": component,
                    "axis": axis,
                },
            )
        )
    return out


def _scale_finding(
    group: str,
    feature: str,
    weight: float,
    idx: list[int],
    ref_g: np.ndarray,
    sketch_g: np.ndarray,
) -> Finding | None:
    """Scale finding from the group's internal spread ratio (spec §9.3).

    The linear spread ratio is squared into an area deviation so the magnitude is
    comparable to the area severity tiers (§6). After a *similarity* alignment the
    global scale is already normalized, so a non-unit per-group ratio is a real
    "this feature is drawn too big/small" error, not page scaling.
    """
    spread_ref = _spread(ref_g)
    spread_sketch = _spread(sketch_g)
    ratio = spread_sketch / max(spread_ref, _EPS)
    area_pct = (ratio**2 - 1.0) * 100.0
    magnitude = abs(area_pct)
    severity = _severity_from_tiers(magnitude, AREA_TIERS)
    if severity is Severity.OK:
        return None
    direction = "too large" if area_pct > 0 else "too small"
    return Finding(
        id=f"{group}_scale",
        level=Level.PLACEMENT,
        severity=severity,
        feature=feature,
        axis="area",
        direction=direction,
        magnitude=magnitude,
        units="%area",
        score=weight * magnitude / AREA_OK_MAX,
        evidence={
            "group": group,
            "indices": idx,
            "ref_points": ref_g,
            "sketch_points": sketch_g,
            "spread_ref": spread_ref,
            "spread_sketch": spread_sketch,
            "spread_ratio": ratio,
        },
    )
