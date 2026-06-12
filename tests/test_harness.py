"""M1 synthetic distortion harness and headline metrics (spec §8 M1)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from artstockfish import config
from artstockfish.pipeline import critique_pair, demo_reference
from artstockfish.schema import Landmarks
from artstockfish.synth.distort import (
    DistortionOp,
    ExpectedFinding,
    compose,
    rotate_line,
    scale_feature,
    shift_feature,
    tps_bulge,
)


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    median_magnitude_error: float
    false_positives: tuple[tuple[int, str, str], ...]
    misses: tuple[tuple[int, str, str], ...]


def _op_shift_eye(rng: np.random.Generator, used: set[str]) -> DistortionOp | None:
    if "eye_vertical_shift" in used:
        return None
    choices = [name for name in ("left_eye", "right_eye") if f"{name}_vertical" not in used]
    if not choices:
        return None
    name = str(rng.choice(choices))
    dy = float(rng.choice((-1.0, 1.0)) * rng.uniform(4.8, 7.4))
    used.add(f"{name}_vertical")
    used.add("eye_vertical_shift")
    return lambda lm, name=name, dy=dy: shift_feature(lm, name, dy=dy)


def _op_scale_brow(rng: np.random.Generator, used: set[str]) -> DistortionOp | None:
    choices = [name for name in ("left_brow", "right_brow") if f"{name}_scale" not in used]
    if not choices:
        return None
    name = str(rng.choice(choices))
    if rng.random() < 0.5:
        scale = float(rng.uniform(1.10, 1.17))
    else:
        scale = float(rng.uniform(0.84, 0.92))
    used.add(f"{name}_scale")
    return lambda lm, name=name, scale=scale: scale_feature(lm, name, scale)


def _op_rotate_line(rng: np.random.Generator, used: set[str]) -> DistortionOp | None:
    line_names = ("mouth_line", "left_jaw", "right_jaw")
    choices = [
        name for name in line_names if str(config.ANGLE_LINES[name]["id"]) not in used
    ]
    if not choices:
        return None
    name = str(rng.choice(choices))
    deg = float(rng.choice((-1.0, 1.0)) * rng.uniform(5.5, 9.0))
    used.add(str(config.ANGLE_LINES[name]["id"]))
    return lambda lm, name=name, deg=deg: rotate_line(lm, name, deg)


def _random_case(rng: np.random.Generator, n_ops: int) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    reference = demo_reference()
    factories: tuple[Callable[[np.random.Generator, set[str]], DistortionOp | None], ...] = (
        _op_shift_eye,
        _op_scale_brow,
        _op_rotate_line,
    )

    used: set[str] = set()
    operations: list[DistortionOp] = []
    attempts = 0
    while len(operations) < n_ops and attempts < 20:
        attempts += 1
        factory = factories[int(rng.integers(0, len(factories)))]
        op = factory(rng, used)
        if op is not None:
            operations.append(op)
    if len(operations) != n_ops:
        raise AssertionError("could not assemble a unique synthetic case")
    return compose(reference, *operations)


def _key(obj) -> tuple[str, str]:
    return (obj.id, obj.direction)


def _score_case(
    case_index: int,
    expected: tuple[ExpectedFinding, ...],
    reported,
) -> tuple[int, int, int, list[float], list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    expected_by_key = {_key(f): f for f in expected}
    reported_by_key = {_key(f): f for f in reported}

    expected_keys = set(expected_by_key)
    reported_keys = set(reported_by_key)
    matches = expected_keys & reported_keys

    errors = [
        abs(reported_by_key[key].magnitude - expected_by_key[key].magnitude)
        / max(expected_by_key[key].magnitude, 1e-9)
        for key in matches
    ]
    false_positives = [
        (case_index, key[0], key[1]) for key in sorted(reported_keys - expected_keys)
    ]
    misses = [(case_index, key[0], key[1]) for key in sorted(expected_keys - reported_keys)]
    return (
        len(matches),
        len(reported_keys),
        len(expected_keys),
        errors,
        false_positives,
        misses,
    )


def _run_harness() -> Metrics:
    rng = np.random.default_rng(config.SYNTH_RANDOM_SEED)
    reference = demo_reference()
    true_positive = total_reported = total_expected = 0
    magnitude_errors: list[float] = []
    false_positives: list[tuple[int, str, str]] = []
    misses: list[tuple[int, str, str]] = []

    for i in range(config.SYNTH_HARNESS_CASES):
        n_ops = 1 if i < config.SYNTH_HARNESS_CASES // 2 else int(rng.integers(2, 4))
        sketch, expected = _random_case(rng, n_ops)
        reported = critique_pair(reference, sketch).report.findings
        tp, n_reported, n_expected, errors, fp, miss = _score_case(i, expected, reported)
        true_positive += tp
        total_reported += n_reported
        total_expected += n_expected
        magnitude_errors.extend(errors)
        false_positives.extend(fp)
        misses.extend(miss)

    precision = true_positive / total_reported
    recall = true_positive / total_expected
    median_error = float(np.median(magnitude_errors))
    return Metrics(
        precision=precision,
        recall=recall,
        median_magnitude_error=median_error,
        false_positives=tuple(false_positives[:10]),
        misses=tuple(misses[:10]),
    )


def test_distortion_generators_label_known_errors():
    ref = demo_reference()

    shifted, shift_expected = shift_feature(ref, "left_eye", dy=6.0)
    assert shifted.points.shape == ref.points.shape
    assert [(f.id, f.direction, f.magnitude, f.units) for f in shift_expected] == [
        ("left_eye_vertical", "too high", 6.0, "%head_height")
    ]

    scaled, scale_expected = scale_feature(ref, "right_brow", 1.12)
    assert scaled.points.shape == ref.points.shape
    assert scale_expected[0].id == "right_brow_scale"
    assert scale_expected[0].magnitude == pytest.approx(25.44)

    rotated, rotate_expected = rotate_line(ref, "mouth_line", -7.0)
    assert rotated.points.shape == ref.points.shape
    assert [(f.id, f.direction, f.magnitude, f.units) for f in rotate_expected] == [
        ("mouth_line_angle", "tilted counterclockwise", 7.0, "deg")
    ]

    bulged, bulge_expected = tps_bulge(ref, "jaw", 6.0)
    assert bulged.points.shape == ref.points.shape
    assert bulge_expected[0].id == "jaw_contour_bulge"
    assert bulge_expected[0].direction == "bulges outward"


def test_m1_harness_precision_recall_and_magnitude_gates():
    metrics = _run_harness()

    assert metrics.precision >= config.SYNTH_PRECISION_GATE, metrics
    assert metrics.recall >= config.SYNTH_RECALL_GATE, metrics
    assert metrics.median_magnitude_error <= config.SYNTH_MAG_ERROR_GATE, metrics

    # These are the README headline numbers for this deterministic harness run.
    print(
        "\nM1 harness metrics:"
        f"\nprecision={metrics.precision:.3f}"
        f"\nrecall={metrics.recall:.3f}"
        f"\nmedian_magnitude_error={metrics.median_magnitude_error:.3f}"
    )
