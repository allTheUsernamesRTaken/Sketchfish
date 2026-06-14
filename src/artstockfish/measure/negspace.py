"""Closed negative-space region measurement (spec §8 M3).

This module works on aligned binary ink masks. It finds closed background regions
by flood-filling the outside background from the image border; any remaining
background component is an enclosed negative-space shape. Regions are
corresponded greedily by centroid distance, then area and aspect ratio deviations
become ``Finding`` objects with drawing-ready region geometry in ``evidence``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .. import config
from ..schema import Finding, Level, Severity

_EPS = 1e-9


@dataclass(frozen=True)
class NegativeSpaceRegion:
    """One closed background component extracted from a binary mask."""

    label: int
    area: float
    centroid: np.ndarray      # (x, y)
    bbox: tuple[int, int, int, int]  # x, y, width, height
    aspect: float             # width / height
    pixels: np.ndarray        # (N, 2) x/y pixel coordinates
    contour: np.ndarray       # boundary pixels, (M, 2) x/y coordinates


def _severity_from_tiers(magnitude: float, tiers: tuple[float, float, float]) -> Severity:
    ok, inaccuracy, mistake = tiers
    if magnitude < ok:
        return Severity.OK
    if magnitude < inaccuracy:
        return Severity.INACCURACY
    if magnitude < mistake:
        return Severity.MISTAKE
    return Severity.BLUNDER


def _ink(mask: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary mask to a boolean ink mask."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"negative-space masks must be 2D; got {arr.shape}")
    return arr.astype(bool)


def _outside_background(background: np.ndarray) -> np.ndarray:
    """Flood-fill background connected to the image border."""
    h, w = background.shape
    outside = np.zeros_like(background, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(y: int, x: int) -> None:
        if background[y, x] and not outside[y, x]:
            outside[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w:
                push(yy, xx)
    return outside


def _component_boundary(component: np.ndarray) -> np.ndarray:
    """Return boundary pixels for a component mask as x/y coordinates."""
    h, w = component.shape
    padded = np.pad(component, 1, mode="constant", constant_values=False)
    center = padded[1 : h + 1, 1 : w + 1]
    interior = (
        padded[0:h, 1 : w + 1]
        & padded[2 : h + 2, 1 : w + 1]
        & padded[1 : h + 1, 0:w]
        & padded[1 : h + 1, 2 : w + 2]
    )
    boundary = center & ~interior
    yx = np.argwhere(boundary)
    return yx[:, ::-1].astype(np.float64)


def _closed_components(closed_background: np.ndarray) -> Iterable[np.ndarray]:
    """Yield masks for each connected closed background component."""
    remaining = closed_background.copy()
    h, w = remaining.shape
    while True:
        seeds = np.argwhere(remaining)
        if len(seeds) == 0:
            return
        seed_y, seed_x = (int(seeds[0, 0]), int(seeds[0, 1]))
        comp = np.zeros_like(remaining, dtype=bool)
        q: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
        remaining[seed_y, seed_x] = False
        comp[seed_y, seed_x] = True
        while q:
            y, x = q.popleft()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and remaining[yy, xx]:
                    remaining[yy, xx] = False
                    comp[yy, xx] = True
                    q.append((yy, xx))
        yield comp


def extract_closed_background_regions(
    ink_mask: np.ndarray,
    *,
    min_area: int = config.NEGSPACE_MIN_REGION_AREA,
) -> tuple[NegativeSpaceRegion, ...]:
    """Find closed background regions in a binary ink mask via flood fill.

    Args:
        ink_mask: 2D mask where truthy pixels are drawn strokes/filled shape.
        min_area: components smaller than this many pixels are discarded.
    """
    ink = _ink(ink_mask)
    background = ~ink
    outside = _outside_background(background)
    closed = background & ~outside

    regions: list[NegativeSpaceRegion] = []
    for label, comp in enumerate(_closed_components(closed)):
        yx = np.argwhere(comp)
        area = int(len(yx))
        if area < min_area:
            continue
        y_min, x_min = yx.min(axis=0)
        y_max, x_max = yx.max(axis=0)
        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)
        pixels = yx[:, ::-1].astype(np.float64)
        centroid = pixels.mean(axis=0)
        regions.append(
            NegativeSpaceRegion(
                label=label,
                area=float(area),
                centroid=centroid,
                bbox=(int(x_min), int(y_min), width, height),
                aspect=width / max(float(height), _EPS),
                pixels=pixels,
                contour=_component_boundary(comp),
            )
        )
    return tuple(regions)


def _match_by_centroid(
    reference: tuple[NegativeSpaceRegion, ...],
    sketch: tuple[NegativeSpaceRegion, ...],
) -> list[tuple[NegativeSpaceRegion, NegativeSpaceRegion]]:
    """Greedy one-to-one correspondence by centroid distance."""
    candidates: list[tuple[float, int, int]] = []
    for i, ref in enumerate(reference):
        for j, sk in enumerate(sketch):
            dist = float(np.linalg.norm(ref.centroid - sk.centroid))
            candidates.append((dist, i, j))
    matches: list[tuple[NegativeSpaceRegion, NegativeSpaceRegion]] = []
    used_ref: set[int] = set()
    used_sketch: set[int] = set()
    for _, i, j in sorted(candidates):
        if i in used_ref or j in used_sketch:
            continue
        used_ref.add(i)
        used_sketch.add(j)
        matches.append((reference[i], sketch[j]))
    return matches


def _region_evidence(ref: NegativeSpaceRegion, sketch: NegativeSpaceRegion) -> dict:
    return {
        "reference_region": {
            "label": ref.label,
            "area": ref.area,
            "centroid": ref.centroid,
            "bbox": ref.bbox,
            "aspect": ref.aspect,
            "contour": ref.contour,
        },
        "sketch_region": {
            "label": sketch.label,
            "area": sketch.area,
            "centroid": sketch.centroid,
            "bbox": sketch.bbox,
            "aspect": sketch.aspect,
            "contour": sketch.contour,
        },
        "centroid_delta": sketch.centroid - ref.centroid,
    }


def negative_space_findings(
    reference_regions: tuple[NegativeSpaceRegion, ...],
    sketch_regions: tuple[NegativeSpaceRegion, ...],
) -> list[Finding]:
    """Compare already-extracted negative-space regions by centroid match."""
    findings: list[Finding] = []
    for match_index, (ref, sketch) in enumerate(_match_by_centroid(reference_regions, sketch_regions)):
        area_dev = (sketch.area - ref.area) / max(ref.area, _EPS) * 100.0
        area_mag = abs(float(area_dev))
        area_sev = _severity_from_tiers(area_mag, config.AREA_TIERS)
        if area_sev is not Severity.OK:
            findings.append(
                Finding(
                    id=f"negative_space_{match_index}_area",
                    level=Level.SHAPE,
                    severity=area_sev,
                    feature="negative space",
                    axis="area",
                    direction="too wide" if area_dev > 0.0 else "too narrow",
                    magnitude=area_mag,
                    units="%area",
                    score=config.NEGSPACE_WEIGHT * area_mag / config.NEGSPACE_AREA_OK_MAX,
                    evidence={
                        **_region_evidence(ref, sketch),
                        "region_index": match_index,
                        "area_deviation_pct": float(area_dev),
                    },
                )
            )

        aspect_dev = (sketch.aspect - ref.aspect) / max(ref.aspect, _EPS) * 100.0
        aspect_mag = abs(float(aspect_dev))
        aspect_sev = _severity_from_tiers(aspect_mag, config.NEGSPACE_ASPECT_TIERS)
        if aspect_sev is not Severity.OK:
            findings.append(
                Finding(
                    id=f"negative_space_{match_index}_aspect",
                    level=Level.SHAPE,
                    severity=aspect_sev,
                    feature="negative space",
                    axis="aspect",
                    direction="too wide" if aspect_dev > 0.0 else "too narrow",
                    magnitude=aspect_mag,
                    units="%aspect",
                    score=config.NEGSPACE_WEIGHT * aspect_mag / config.NEGSPACE_ASPECT_OK_MAX,
                    evidence={
                        **_region_evidence(ref, sketch),
                        "region_index": match_index,
                        "aspect_deviation_pct": float(aspect_dev),
                    },
                )
            )
    return findings


def measure_negative_space(
    reference_ink_mask: np.ndarray,
    sketch_ink_mask: np.ndarray,
    *,
    min_area: int = config.NEGSPACE_MIN_REGION_AREA,
) -> list[Finding]:
    """Extract and compare closed negative-space regions from aligned masks.

    The masks should already share a coordinate frame; callers that start with
    images should apply the same similarity alignment used for landmarks before
    calling this function.
    """
    ref_regions = extract_closed_background_regions(reference_ink_mask, min_area=min_area)
    sketch_regions = extract_closed_background_regions(sketch_ink_mask, min_area=min_area)
    return negative_space_findings(ref_regions, sketch_regions)
