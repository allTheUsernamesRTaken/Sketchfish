"""M3 contour acceptance tests: signed bulge localization on corresponded arcs."""

from __future__ import annotations

import numpy as np
import pytest

from artstockfish import config
from artstockfish.frame import SEMANTIC_GROUPS, build_face_frame
from artstockfish.measure.contour import measure_contours
from artstockfish.schema import Landmarks
from artstockfish.synth.distort import tps_bulge

from fixtures import canonical_face_landmarks


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def _jaw_arc(points: np.ndarray) -> np.ndarray:
    jaw = points[list(SEMANTIC_GROUPS["jaw"])]
    steps = np.linalg.norm(np.diff(jaw, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    return cumulative / cumulative[-1]


def _jaw_normals(points: np.ndarray) -> np.ndarray:
    frame = build_face_frame(points)
    jaw = points[list(SEMANTIC_GROUPS["jaw"])]
    tangents = _unit_rows(np.gradient(jaw, axis=0))
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
    outward_hint = jaw - frame.origin
    normals[np.sum(normals * outward_hint, axis=1) < 0.0] *= -1.0
    return _unit_rows(normals)


def _localized_jaw_bulge(
    reference: Landmarks,
    *,
    center_arc: float,
    amount: float,
    width_arc: float = 0.075,
) -> Landmarks:
    """Smooth test-only contour bulge centered at a known jaw arc position."""
    points = reference.points.copy()
    idx = list(SEMANTIC_GROUPS["jaw"])
    frame = build_face_frame(points)
    arc = _jaw_arc(points)
    normals = _jaw_normals(points)
    weights = np.exp(-0.5 * ((arc - center_arc) / width_arc) ** 2)
    displacement = (amount / 100.0) * frame.head_height
    points[idx] += weights[:, None] * displacement * normals
    return Landmarks(points=points, names=reference.names, image_size=reference.image_size)


def _best_bulge_match(findings, expected_direction: str, expected_arc: float):
    bulges = [
        f
        for f in findings
        if f.id == "jaw_contour_bulge" and f.direction == expected_direction
    ]
    if not bulges:
        return None
    return min(bulges, key=lambda f: abs(float(f.evidence["peak_arc"]) - expected_arc))


def test_identical_contours_produce_no_findings():
    reference = canonical_face_landmarks()
    assert measure_contours(reference, reference) == []


def test_existing_tps_bulge_label_is_detected_with_correct_sign():
    reference = canonical_face_landmarks()
    sketch, expected = tps_bulge(reference, "jaw", 7.5)

    findings = measure_contours(reference, sketch)
    match = _best_bulge_match(findings, expected[0].direction, 0.5)

    assert match is not None
    assert match.direction == "bulges outward"
    assert abs(float(match.evidence["peak_arc"]) - 0.5) <= 0.10
    assert match.evidence["anchor_a"]
    assert match.evidence["anchor_b"]
    assert match.evidence["run_ref_segment"].shape[1] == 2
    assert match.evidence["run_sketch_segment"].shape[1] == 2


def test_tps_bulge_distortions_localize_midpoint_and_sign_at_m3_gate():
    reference = canonical_face_landmarks()
    rng = np.random.default_rng(config.SYNTH_RANDOM_SEED + 303)
    cases = 100
    successes = 0
    failures = []

    for case_index in range(cases):
        center_arc = float(rng.uniform(0.15, 0.85))
        sign = float(rng.choice((-1.0, 1.0)))
        amount = sign * float(rng.uniform(7.0, 11.0))
        sketch = _localized_jaw_bulge(reference, center_arc=center_arc, amount=amount)
        expected_direction = "bulges outward" if amount > 0.0 else "caves in"

        findings = measure_contours(reference, sketch)
        match = _best_bulge_match(findings, expected_direction, center_arc)
        if match is not None:
            arc_error = abs(float(match.evidence["peak_arc"]) - center_arc)
            if arc_error <= 0.10:
                successes += 1
                continue
        failures.append(
            {
                "case": case_index,
                "center_arc": center_arc,
                "direction": expected_direction,
                "reported": [
                    (f.id, f.direction, float(f.evidence.get("peak_arc", np.nan)))
                    for f in findings
                ],
            }
        )

    rate = successes / cases
    print(f"\nM3 contour bulge localization: {successes}/{cases} = {rate:.3f}")
    assert rate >= 0.90, failures[:10]
