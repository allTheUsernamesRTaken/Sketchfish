"""Canon-ratio proportions (spec §9.4).

Compute each v1 canon ratio **in both images** and critique the *difference*. The
target is always "match the reference," never "match the textbook": a ratio is a
finding only when the sketch's value departs from the *reference's* value, so a
stylized or non-canonical reference is handled correctly (spec §9.4, principle in
§2 and pitfall §12 "Do not compare ratios against textbook canon").

Why no alignment step here: every ratio is a quotient of two lengths (or two
midline heights), so it is invariant to the similarity transform (translation,
rotation, uniform scale) that aligns the two faces. We therefore measure each
ratio in each image's *own* face frame and compare the dimensionless results —
no Procrustes fit is needed for proportions.

v1 ratio set (spec §9.4):

==========================  ============================================  =======
ratio                       geometry (iBUG 68-point indices)               level
==========================  ============================================  =======
eye-line height/head height (eye-line height above chin) / head height    GLOBAL
face thirds                 midface span / lower-face span                 GLOBAL
interocular/eye width       inner-corner gap / mean eye width             PLACEMENT
nose length/face height     bridge-top→subnasale / head height            PLACEMENT
mouth width/interocular     mouth-corner span / inner-corner gap          PLACEMENT
==========================  ============================================  =======

A finding's ``magnitude`` is the deviation of the sketch ratio from the reference
ratio as a percentage of the reference ratio; ``units`` is ``"%ratio"`` and the
severity tiers live in ``config`` (``# --- proportions ---``). All functions are
pure; inputs are plain ``numpy`` arrays or :class:`~artstockfish.schema.Landmarks`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import config
from ..frame import FaceFrame, build_face_frame
from ..schema import Finding, Level, Severity

# iBUG 68-point landmarks this module reads (see frame.ANCHORS / SEMANTIC_GROUPS).
_CHIN = 8
_NOSE_BRIDGE_TOP = 27
_SUBNASALE = 33                       # nose-base centre (nostril row, middle point)
_RIGHT_EYE = tuple(range(36, 42))     # subject's right eye (image left)
_LEFT_EYE = tuple(range(42, 48))      # subject's left eye (image right)
_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER = 36, 39
_LEFT_EYE_INNER, _LEFT_EYE_OUTER = 42, 45
_BROWS = tuple(range(17, 27))         # both brows
_NOSE_BASE = tuple(range(31, 36))     # nostril base row
_MOUTH_RIGHT_CORNER, _MOUTH_LEFT_CORNER = 48, 54


def _as_points(obj) -> np.ndarray:
    """Accept a :class:`Landmarks` or an ``(N, 2)`` array → ``(N, 2)`` float64."""
    pts = getattr(obj, "points", obj)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) landmark points; got {pts.shape}")
    return pts


def _height(frame: FaceFrame, point: np.ndarray) -> float:
    """Signed height of a point up the face frame (chin → forehead is positive)."""
    return float((np.asarray(point, dtype=np.float64) - frame.origin) @ frame.y_axis)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points (similarity-invariant)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# --- the five canon ratios (each computed identically in ref and sketch) ----------


def _eye_line_height(pts: np.ndarray, frame: FaceFrame) -> float:
    """Height of the eye line above the chin, in units of head height."""
    eye_line = pts[list(_RIGHT_EYE + _LEFT_EYE)].mean(axis=0)
    return (_height(frame, eye_line) - _height(frame, pts[_CHIN])) / frame.head_height


def _face_thirds(pts: np.ndarray, frame: FaceFrame) -> float:
    """Midface span (brow→nose base) over lower-face span (nose base→chin).

    The classic "face in thirds" rule with the two *measurable* thirds from the
    68-point set — there is no hairline/crown landmark, so the upper third is not
    measurable (same limitation noted for head height in DECISIONS.md). The middle
    and lower thirds are canonically equal, so this ratio is ~1 for a canonical
    face; we still critique its departure from the *reference*, not from 1.
    """
    brow = pts[list(_BROWS)].mean(axis=0)
    nose_base = pts[list(_NOSE_BASE)].mean(axis=0)
    mid = _height(frame, brow) - _height(frame, nose_base)
    low = _height(frame, nose_base) - _height(frame, pts[_CHIN])
    return mid / low


def _interocular_eye_width(pts: np.ndarray, frame: FaceFrame) -> float:
    """Inner-corner gap over mean eye width (the "eyes one eye-width apart" rule)."""
    interocular = _dist(pts[_RIGHT_EYE_INNER], pts[_LEFT_EYE_INNER])
    eye_w = 0.5 * (
        _dist(pts[_RIGHT_EYE_OUTER], pts[_RIGHT_EYE_INNER])
        + _dist(pts[_LEFT_EYE_OUTER], pts[_LEFT_EYE_INNER])
    )
    return interocular / eye_w


def _nose_length(pts: np.ndarray, frame: FaceFrame) -> float:
    """Nose length (bridge top → subnasale, along the midline) over head height."""
    nose_len = abs(_height(frame, pts[_NOSE_BRIDGE_TOP]) - _height(frame, pts[_SUBNASALE]))
    return nose_len / frame.head_height


def _mouth_interocular(pts: np.ndarray, frame: FaceFrame) -> float:
    """Mouth width over inner-corner gap (the "mouth ≈ eye-spacing" rule)."""
    mouth_w = _dist(pts[_MOUTH_RIGHT_CORNER], pts[_MOUTH_LEFT_CORNER])
    interocular = _dist(pts[_RIGHT_EYE_INNER], pts[_LEFT_EYE_INNER])
    return mouth_w / interocular


_RATIO_FNS: dict[str, Callable[[np.ndarray, FaceFrame], float]] = {
    "eye_line_height": _eye_line_height,
    "face_thirds": _face_thirds,
    "interocular_eye_width": _interocular_eye_width,
    "nose_length": _nose_length,
    "mouth_interocular": _mouth_interocular,
}

# Landmark indices each ratio reads → echoed into Finding.evidence so annotate.py
# can draw the ratio without re-deriving the geometry.
_RATIO_INDICES: dict[str, tuple[int, ...]] = {
    "eye_line_height": _RIGHT_EYE + _LEFT_EYE + (_CHIN,),
    "face_thirds": _BROWS + _NOSE_BASE + (_CHIN,),
    "interocular_eye_width": (
        _RIGHT_EYE_OUTER,
        _RIGHT_EYE_INNER,
        _LEFT_EYE_INNER,
        _LEFT_EYE_OUTER,
    ),
    "nose_length": (_NOSE_BRIDGE_TOP, _SUBNASALE),
    "mouth_interocular": (
        _MOUTH_RIGHT_CORNER,
        _MOUTH_LEFT_CORNER,
        _RIGHT_EYE_INNER,
        _LEFT_EYE_INNER,
    ),
}


@dataclass(frozen=True)
class _RatioSpec:
    """One canon-ratio rule, assembled from the config metadata + the geometry fn."""

    key: str
    id: str
    level: Level
    feature: str
    weight: float
    higher: str
    lower: str
    compute: Callable[[np.ndarray, FaceFrame], float]
    indices: tuple[int, ...]


def _build_specs() -> tuple[_RatioSpec, ...]:
    specs = []
    for key, meta in config.PROPORTION_RATIOS.items():
        weight = config.PROPORTION_WEIGHTS.get(
            meta["weight_key"], config.DEFAULT_PROPORTION_WEIGHT
        )
        specs.append(
            _RatioSpec(
                key=key,
                id=meta["id"],
                level=Level(meta["level"]),
                feature=meta["feature"],
                weight=weight,
                higher=meta["higher"],
                lower=meta["lower"],
                compute=_RATIO_FNS[key],
                indices=_RATIO_INDICES[key],
            )
        )
    return tuple(specs)


_SPECS = _build_specs()


def _classify(magnitude: float) -> Severity:
    """Map a ratio-deviation magnitude (%) to a severity tier (config thresholds)."""
    ok, inaccuracy, mistake = config.PROPORTION_TIERS
    if magnitude < ok:
        return Severity.OK
    if magnitude < inaccuracy:
        return Severity.INACCURACY
    if magnitude < mistake:
        return Severity.MISTAKE
    return Severity.BLUNDER


def proportion_findings(reference, sketch) -> list[Finding]:
    """Critique the sketch's canon ratios against the reference's (spec §9.4).

    Args:
        reference: the reference face — :class:`~artstockfish.schema.Landmarks`
            or an ``(68, 2)`` array of iBUG landmark coordinates.
        sketch: the student's sketch, same form. Need NOT be pre-aligned: every
            ratio is similarity-invariant, so alignment cannot change it.

    Returns:
        Findings for every canon ratio whose sketch value departs from the
        reference value past the OK noise floor, sorted ``(Level asc, score
        desc)`` to match the schema's ranking contract. A perfectly matched set
        yields an empty list.
    """
    ref_pts = _as_points(reference)
    sketch_pts = _as_points(sketch)
    ref_frame = build_face_frame(ref_pts)
    sketch_frame = build_face_frame(sketch_pts)

    findings: list[Finding] = []
    for spec in _SPECS:
        r_ref = spec.compute(ref_pts, ref_frame)
        r_sketch = spec.compute(sketch_pts, sketch_frame)
        if abs(r_ref) < 1e-12:
            continue  # degenerate reference ratio — nothing to compare against
        deviation = (r_sketch - r_ref) / r_ref * 100.0
        magnitude = abs(deviation)
        severity = _classify(magnitude)
        if severity is Severity.OK:
            continue  # below the noise floor — never shown (spec §6)

        direction = spec.higher if deviation > 0 else spec.lower
        score = spec.weight * magnitude / config.PROPORTION_SEVERITY_UNIT
        findings.append(
            Finding(
                id=spec.id,
                level=spec.level,
                severity=severity,
                feature=spec.feature,
                axis="proportion",
                direction=direction,
                magnitude=magnitude,
                units="%ratio",
                score=score,
                evidence={
                    "ratio_id": spec.id,
                    "reference_ratio": r_ref,
                    "sketch_ratio": r_sketch,
                    "deviation_pct": deviation,
                    "landmark_indices": spec.indices,
                    "reference_points": ref_pts[list(spec.indices)],
                    "sketch_points": sketch_pts[list(spec.indices)],
                },
            )
        )

    findings.sort(key=lambda f: (int(f.level), -f.score))
    return findings
