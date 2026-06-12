"""Labeled synthetic landmark distortions (spec §8 M1).

Each generator takes a reference :class:`~artstockfish.schema.Landmarks`, applies a
deterministic geometric error, and returns ``(distorted_landmarks,
expected_findings)``. Magnitudes use the same units as the measurement pipeline:
placement shifts are percent of head height, feature scale is percent area, and
line rotations are degrees.

The functions are intentionally small and parameterized so tests can sample honest
known errors without relying on the evaluator to manufacture the labels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from ..config import ANGLE_LINES, ANGLE_TIERS, AREA_TIERS, DISPLACEMENT_TIERS
from ..frame import SEMANTIC_GROUPS, build_face_frame
from ..schema import Landmarks, Severity


@dataclass(frozen=True)
class ExpectedFinding:
    """A synthetic ground-truth finding injected by a distortion generator."""

    id: str
    direction: str
    magnitude: float
    units: str
    severity: Severity


DistortionOp = Callable[[Landmarks], tuple[Landmarks, tuple[ExpectedFinding, ...]]]

_FEATURE_ALIASES: dict[str, tuple[int, ...]] = {
    "nose": SEMANTIC_GROUPS["nose_bridge"] + SEMANTIC_GROUPS["nose_bottom"],
    "face_oval": SEMANTIC_GROUPS["jaw"],
    **SEMANTIC_GROUPS,
}


def _severity(magnitude: float, tiers: tuple[float, float, float]) -> Severity:
    ok, inaccuracy, mistake = tiers
    if magnitude < ok:
        return Severity.OK
    if magnitude < inaccuracy:
        return Severity.INACCURACY
    if magnitude < mistake:
        return Severity.MISTAKE
    return Severity.BLUNDER


def _indices(mapping: dict[str, tuple[int, ...]], name: str) -> tuple[int, ...]:
    try:
        return mapping[name]
    except KeyError as exc:
        choices = ", ".join(sorted(mapping))
        raise ValueError(f"unknown distortion target {name!r}; choose one of: {choices}") from exc


def _copy_landmarks(landmarks: Landmarks, points: np.ndarray) -> Landmarks:
    return Landmarks(
        points=np.asarray(points, dtype=np.float64),
        names=landmarks.names,
        image_size=landmarks.image_size,
    )


def _non_ok(expected: Iterable[ExpectedFinding]) -> tuple[ExpectedFinding, ...]:
    return tuple(f for f in expected if f.severity is not Severity.OK)


def shift_feature(
    landmarks: Landmarks,
    name: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    """Translate a semantic feature in the reference face frame.

    Args:
        landmarks: full 68-point reference landmarks.
        name: semantic group name, e.g. ``"left_eye"`` or ``"mouth"``.
        dx: horizontal shift in percent of head height; positive is image-right
            in the reference face frame.
        dy: vertical shift in percent of head height; positive is up the face.

    Returns:
        Distorted landmarks plus expected placement findings for every shifted
        component above the displacement OK floor.
    """
    idx = list(_indices(_FEATURE_ALIASES, name))
    frame = build_face_frame(landmarks.points)
    delta = (dx / 100.0) * frame.head_height * frame.x_axis
    delta += (dy / 100.0) * frame.head_height * frame.y_axis

    points = np.asarray(landmarks.points, dtype=np.float64).copy()
    points[idx] += delta

    expected = []
    if dx:
        mag = abs(float(dx))
        expected.append(
            ExpectedFinding(
                id=f"{name}_horizontal",
                direction="too far right" if dx > 0 else "too far left",
                magnitude=mag,
                units="%head_height",
                severity=_severity(mag, DISPLACEMENT_TIERS),
            )
        )
    if dy:
        mag = abs(float(dy))
        expected.append(
            ExpectedFinding(
                id=f"{name}_vertical",
                direction="too high" if dy > 0 else "too low",
                magnitude=mag,
                units="%head_height",
                severity=_severity(mag, DISPLACEMENT_TIERS),
            )
        )
    return _copy_landmarks(landmarks, points), _non_ok(expected)


def scale_feature(
    landmarks: Landmarks,
    name: str,
    scale: float,
) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    """Scale a semantic feature about its own centroid.

    The expected magnitude mirrors ``measure.landmarks``: the linear spread ratio
    is squared into a percent area deviation.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")

    idx = list(_indices(_FEATURE_ALIASES, name))
    points = np.asarray(landmarks.points, dtype=np.float64).copy()
    center = points[idx].mean(axis=0)
    points[idx] = center + (points[idx] - center) * float(scale)

    area_pct = (float(scale) ** 2 - 1.0) * 100.0
    mag = abs(area_pct)
    expected = ExpectedFinding(
        id=f"{name}_scale",
        direction="too large" if area_pct > 0 else "too small",
        magnitude=mag,
        units="%area",
        severity=_severity(mag, AREA_TIERS),
    )
    return _copy_landmarks(landmarks, points), _non_ok((expected,))


def rotate_line(
    landmarks: Landmarks,
    name: str,
    deg: float,
) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    """Rotate one configured line feature about its centroid.

    ``name`` is a key from ``config.ANGLE_LINES`` such as ``"mouth_line"`` or
    ``"left_jaw"``. Positive degrees are clockwise in image coordinates, matching
    ``measure.angles``.
    """
    spec = ANGLE_LINES[name] if name in ANGLE_LINES else None
    if spec is None:
        choices = ", ".join(sorted(ANGLE_LINES))
        raise ValueError(f"unknown line {name!r}; choose one of: {choices}")

    idx = list(spec["indices"])
    points = np.asarray(landmarks.points, dtype=np.float64).copy()
    center = points[idx].mean(axis=0)
    theta = np.radians(float(deg))
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    points[idx] = (rot @ (points[idx] - center).T).T + center

    mag = abs(float(deg))
    expected = ExpectedFinding(
        id=str(spec["id"]),
        direction="tilted clockwise" if deg > 0 else "tilted counterclockwise",
        magnitude=mag,
        units="deg",
        severity=_severity(mag, ANGLE_TIERS),
    )
    return _copy_landmarks(landmarks, points), _non_ok((expected,))


def tps_bulge(
    landmarks: Landmarks,
    region: str,
    amount: float,
) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    """Apply a smooth local bulge to a contour-ish region.

    This is the M1 placeholder for the M3 contour-era harness. The displacement is
    a thin-plate-spline-style local radial warp over the named region, with a peak
    displacement of ``amount`` percent of head height. Positive values move points
    away from the face centroid; negative values cave inward.
    """
    idx = list(_indices(_FEATURE_ALIASES, region))
    frame = build_face_frame(landmarks.points)
    points = np.asarray(landmarks.points, dtype=np.float64).copy()

    face_center = points.mean(axis=0)
    region_points = points[idx]
    region_center = region_points.mean(axis=0)
    radial = region_center - face_center
    norm = np.linalg.norm(radial)
    if norm < 1e-9:
        radial = frame.x_axis
    else:
        radial = radial / norm

    offsets = region_points - region_center
    radii = np.linalg.norm(offsets, axis=1)
    bandwidth = max(float(np.percentile(radii, 75)), 1e-9)
    weights = np.exp(-0.5 * (radii / bandwidth) ** 2)
    peak = (float(amount) / 100.0) * frame.head_height
    points[idx] = region_points + weights[:, None] * peak * radial

    mag = abs(float(amount))
    expected = ExpectedFinding(
        id=f"{region}_contour_bulge",
        direction="bulges outward" if amount > 0 else "caves in",
        magnitude=mag,
        units="%head_height",
        severity=_severity(mag, DISPLACEMENT_TIERS),
    )
    return _copy_landmarks(landmarks, points), _non_ok((expected,))


def compose(
    landmarks: Landmarks,
    *operations: DistortionOp,
) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    """Apply distortion operations in order and concatenate their labels."""
    current = landmarks
    expected: list[ExpectedFinding] = []
    for operation in operations:
        current, labels = operation(current)
        expected.extend(labels)
    return current, tuple(expected)
