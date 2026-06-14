"""Corresponded contour-shape measurement (spec §8 M3).

Contours are measured after the same robust **similarity** alignment used by the
landmark pipeline. For each named contour segment (v1: the 68-point jaw / lower
face oval proxy), we sample the reference and aligned sketch along normalized arc
length, compute the sketch's signed perpendicular distance from the reference,
smooth that profile, and surface maximal same-sign runs:

- positive signed distance = the sketch bulges **outward** from the face;
- negative signed distance = the sketch **caves in** toward the face.

Every finding carries drawing-ready geometry in ``evidence``: sampled polylines,
normals, signed distance profiles, run endpoints, anchor labels, and the segment
sub-polylines M4 can render directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .. import config
from ..align import apply_similarity, robust_align
from ..frame import FaceFrame, build_face_frame
from ..schema import Finding, Landmarks, Level, Severity

_EPS = 1e-9


@dataclass(frozen=True)
class _ContourProfile:
    """Sampled signed-distance profile for one corresponded contour segment."""

    contour_id: str
    feature: str
    indices: tuple[int, ...]
    anchor_names: tuple[str, ...]
    arc: np.ndarray
    ref_samples: np.ndarray
    sketch_samples: np.ndarray
    normals: np.ndarray
    signed_distance: np.ndarray
    smoothed_distance: np.ndarray


def _as_points(obj) -> np.ndarray:
    """Accept ``Landmarks`` or an ``(N, 2)`` array and return float64 points."""
    pts = getattr(obj, "points", obj)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points; got {pts.shape}")
    return pts


def _severity_from_tiers(magnitude: float, tiers: tuple[float, float, float]) -> Severity:
    ok, inaccuracy, mistake = tiers
    if magnitude < ok:
        return Severity.OK
    if magnitude < inaccuracy:
        return Severity.INACCURACY
    if magnitude < mistake:
        return Severity.MISTAKE
    return Severity.BLUNDER


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving average with an odd window."""
    vals = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return vals.copy()
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    pad = window // 2
    padded = np.pad(vals, pad_width=pad, mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _polyline_arclength(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Cumulative arc length and total length for a polyline."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("a contour segment needs at least two points")
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    if total <= _EPS:
        raise ValueError("degenerate contour segment: zero arc length")
    return cumulative, total


def _sample_polyline(points: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """Sample a polyline at normalized arc positions in ``[0, 1]``."""
    pts = np.asarray(points, dtype=np.float64)
    cumulative, total = _polyline_arclength(pts)
    target = np.asarray(arc, dtype=np.float64) * total
    x = np.interp(target, cumulative, pts[:, 0])
    y = np.interp(target, cumulative, pts[:, 1])
    return np.stack([x, y], axis=-1)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, _EPS)


def _outward_normals(samples: np.ndarray, frame: FaceFrame) -> np.ndarray:
    """Reference-contour normals oriented away from the face centroid."""
    tangents = _unit_rows(np.gradient(samples, axis=0))
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
    outward_hint = samples - frame.origin
    flip = np.sum(normals * outward_hint, axis=1) < 0.0
    normals[flip] *= -1.0
    return _unit_rows(normals)


def _anchor_name(anchor_names: tuple[str, ...], arc_value: float) -> str:
    """Nearest anatomical anchor name for a normalized arc value."""
    if not anchor_names:
        return f"{arc_value:.2f} arc"
    if len(anchor_names) == 1:
        return anchor_names[0]
    positions = np.linspace(0.0, 1.0, len(anchor_names))
    return anchor_names[int(np.argmin(np.abs(positions - arc_value)))]


def _profile_for_segment(
    contour_id: str,
    ref_points: np.ndarray,
    sketch_points: np.ndarray,
    frame: FaceFrame,
) -> _ContourProfile:
    """Build the signed-distance profile for one configured segment."""
    try:
        spec = config.CONTOUR_SEGMENTS[contour_id]
    except KeyError as exc:
        choices = ", ".join(sorted(config.CONTOUR_SEGMENTS))
        raise ValueError(f"unknown contour segment {contour_id!r}; choose one of {choices}") from exc

    indices = tuple(int(i) for i in spec["indices"])
    arc = np.linspace(0.0, 1.0, int(config.CONTOUR_SAMPLE_COUNT))
    ref_segment = ref_points[list(indices)]
    sketch_segment = sketch_points[list(indices)]
    ref_samples = _sample_polyline(ref_segment, arc)
    sketch_samples = _sample_polyline(sketch_segment, arc)
    normals = _outward_normals(ref_samples, frame)
    signed = np.sum((sketch_samples - ref_samples) * normals, axis=1)
    signed = signed / frame.head_height * 100.0
    smoothed = _moving_average(signed, int(config.CONTOUR_SMOOTH_WINDOW))
    return _ContourProfile(
        contour_id=contour_id,
        feature=str(spec["feature"]),
        indices=indices,
        anchor_names=tuple(str(a) for a in spec["anchor_names"]),
        arc=arc,
        ref_samples=ref_samples,
        sketch_samples=sketch_samples,
        normals=normals,
        signed_distance=signed,
        smoothed_distance=smoothed,
    )


def _same_sign_runs(values: np.ndarray, floor: float) -> list[tuple[int, int, int]]:
    """Maximal contiguous runs whose values share sign and exceed ``floor``."""
    signs = np.zeros(len(values), dtype=np.int8)
    signs[values >= floor] = 1
    signs[values <= -floor] = -1

    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    sign = 0
    for i, current in enumerate(signs):
        if current == 0:
            if start is not None:
                runs.append((start, i - 1, sign))
                start = None
                sign = 0
            continue
        if start is None:
            start = i
            sign = int(current)
            continue
        if int(current) != sign:
            runs.append((start, i - 1, sign))
            start = i
            sign = int(current)
    if start is not None:
        runs.append((start, len(values) - 1, sign))
    return runs


def _contour_run_findings(profile: _ContourProfile) -> list[Finding]:
    """Convert maximal same-sign distance runs into contour bulge findings."""
    findings: list[Finding] = []
    min_span = float(config.CONTOUR_MIN_RUN_ARC_FRAC)
    for run_index, (start, end, sign) in enumerate(
        _same_sign_runs(profile.smoothed_distance, float(config.CONTOUR_RUN_OK_MAX))
    ):
        start_arc = float(profile.arc[start])
        end_arc = float(profile.arc[end])
        if end_arc - start_arc < min_span:
            continue

        run_values = profile.smoothed_distance[start : end + 1]
        peak_rel = int(np.argmax(np.abs(run_values)))
        peak_index = start + peak_rel
        peak_arc = float(profile.arc[peak_index])
        magnitude = float(abs(profile.smoothed_distance[peak_index]))
        severity = _severity_from_tiers(magnitude, config.DISPLACEMENT_TIERS)
        if severity is Severity.OK:
            continue

        direction = "bulges outward" if sign > 0 else "caves in"
        anchor_a = _anchor_name(profile.anchor_names, start_arc)
        anchor_b = _anchor_name(profile.anchor_names, end_arc)
        findings.append(
            Finding(
                id=f"{profile.contour_id}_contour_bulge",
                level=Level.SHAPE,
                severity=severity,
                feature=profile.feature,
                axis="contour",
                direction=direction,
                magnitude=magnitude,
                units="%head_height",
                score=config.CONTOUR_WEIGHT * magnitude / config.CONTOUR_SEVERITY_UNIT,
                evidence={
                    "contour_id": profile.contour_id,
                    "run_index": run_index,
                    "indices": profile.indices,
                    "arc": profile.arc,
                    "ref_samples": profile.ref_samples,
                    "sketch_samples": profile.sketch_samples,
                    "normals": profile.normals,
                    "signed_distance": profile.signed_distance,
                    "smoothed_distance": profile.smoothed_distance,
                    "run_start_index": start,
                    "run_end_index": end,
                    "run_start_arc": start_arc,
                    "run_end_arc": end_arc,
                    "run_mid_arc": (start_arc + end_arc) / 2.0,
                    "peak_index": peak_index,
                    "peak_arc": peak_arc,
                    "peak_distance": float(profile.smoothed_distance[peak_index]),
                    "anchor_a": anchor_a,
                    "anchor_b": anchor_b,
                    "anchor_a_point": profile.ref_samples[start],
                    "anchor_b_point": profile.ref_samples[end],
                    "run_ref_segment": profile.ref_samples[start : end + 1],
                    "run_sketch_segment": profile.sketch_samples[start : end + 1],
                },
            )
        )
    return findings


def _turning_profile(samples: np.ndarray) -> np.ndarray:
    """Discrete unsigned curvature profile, in degrees per sampled segment."""
    tangents = _unit_rows(np.gradient(samples, axis=0))
    dots = np.sum(tangents[:-1] * tangents[1:], axis=1)
    turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return np.concatenate([[turns[0]], 0.5 * (turns[:-1] + turns[1:]), [turns[-1]]])


def _curvature_finding(profile: _ContourProfile) -> Finding | None:
    """One curvature-profile finding for a segment, if angularity differs enough."""
    ref_curv = _moving_average(
        _turning_profile(profile.ref_samples), int(config.CONTOUR_CURVATURE_SMOOTH_WINDOW)
    )
    sketch_curv = _moving_average(
        _turning_profile(profile.sketch_samples), int(config.CONTOUR_CURVATURE_SMOOTH_WINDOW)
    )
    delta = sketch_curv - ref_curv
    peak_index = int(np.argmax(np.abs(delta)))
    magnitude = float(abs(delta[peak_index]))
    severity = _severity_from_tiers(magnitude, config.CONTOUR_CURVATURE_TIERS)
    if severity is Severity.OK:
        return None
    direction = "too angular" if delta[peak_index] > 0.0 else "too rounded"
    return Finding(
        id=f"{profile.contour_id}_curvature",
        level=Level.SHAPE,
        severity=severity,
        feature=profile.feature,
        axis="curvature",
        direction=direction,
        magnitude=magnitude,
        units="deg",
        score=config.CONTOUR_WEIGHT * magnitude / config.CONTOUR_CURVATURE_OK_MAX,
        evidence={
            "contour_id": profile.contour_id,
            "indices": profile.indices,
            "arc": profile.arc,
            "ref_samples": profile.ref_samples,
            "sketch_samples": profile.sketch_samples,
            "reference_curvature": ref_curv,
            "sketch_curvature": sketch_curv,
            "curvature_delta": delta,
            "peak_index": peak_index,
            "peak_arc": float(profile.arc[peak_index]),
            "anchor": _anchor_name(profile.anchor_names, float(profile.arc[peak_index])),
        },
    )


def contour_findings(
    reference_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
    *,
    frame: FaceFrame | None = None,
    segments: Iterable[str] | None = None,
) -> list[Finding]:
    """Measure contour findings over already-aligned landmark points.

    Args:
        reference_points: reference landmark points, shape ``(N, 2)``.
        aligned_sketch_points: sketch points already in the reference coordinate
            system.
        frame: reference face frame. Built from ``reference_points`` if omitted.
        segments: configured contour ids to measure. Defaults to
            ``config.CONTOUR_DEFAULT_SEGMENTS``.
    """
    ref_pts = _as_points(reference_points)
    sketch_pts = _as_points(aligned_sketch_points)
    if ref_pts.shape != sketch_pts.shape:
        raise ValueError(f"point shapes must match; got {ref_pts.shape} vs {sketch_pts.shape}")
    if frame is None:
        frame = build_face_frame(ref_pts)

    findings: list[Finding] = []
    for contour_id in tuple(segments or config.CONTOUR_DEFAULT_SEGMENTS):
        profile = _profile_for_segment(contour_id, ref_pts, sketch_pts, frame)
        findings.extend(_contour_run_findings(profile))
        curvature = _curvature_finding(profile)
        if curvature is not None:
            findings.append(curvature)
    return findings


def measure_contours(
    reference: Landmarks,
    sketch: Landmarks,
    *,
    align: bool = True,
    segments: Iterable[str] | None = None,
) -> list[Finding]:
    """Measure corresponded contour findings between reference and sketch.

    ``align=True`` performs the required robust similarity alignment first
    (spec §8 M3 "after alignment"). Pass ``align=False`` when the caller has
    already applied the shared pipeline transform.
    """
    ref_pts = _as_points(reference)
    sketch_pts = _as_points(sketch)
    if ref_pts.shape != sketch_pts.shape:
        raise ValueError(f"point shapes must match; got {ref_pts.shape} vs {sketch_pts.shape}")

    frame = build_face_frame(ref_pts)
    if align:
        s, R, t = robust_align(ref_pts, sketch_pts)
        aligned = apply_similarity(s, R, t, sketch_pts)
    else:
        aligned = sketch_pts
    return contour_findings(ref_pts, aligned, frame=frame, segments=segments)
