"""Deterministic ``(reference, distorted-sketch, ground-truth)`` triples (spec §8 M5).

The benchmark needs labeled error sets it can score *both* systems against. We reuse
the M1 labeled distortion harness (``synth.distort``) so the ground truth is honest
geometry, not something the evaluator manufactured — and we draw from the same op
menu the M1 harness uses (single eye shift / brow scale / line rotation), which is
the menu proven to keep secondary-consequence false positives negligible (the M1
headline precision is ~0.99). Each op writes to a distinct feature/axis within a
case so the injected labels never collide.

No whole-page transform is injected: the sketch carries only the labeled local
errors, so ground truth equals the injected set exactly. (Page tilt/scale is what
the similarity alignment is *supposed* to absorb — §2 principle #2 — and injecting
it would only add un-scored noise to the comparison, not test error detection.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from artstockfish import config
from artstockfish.pipeline import demo_reference
from artstockfish.schema import Landmarks
from artstockfish.synth.distort import (
    DistortionOp,
    ExpectedFinding,
    compose,
    rotate_line,
    scale_feature,
    shift_feature,
)

OpFactory = Callable[[np.random.Generator, set[str]], DistortionOp | None]


@dataclass(frozen=True)
class Triple:
    """One benchmark case: a reference, a labeled distorted sketch, and its truth."""

    case_id: str
    reference: Landmarks
    sketch: Landmarks
    expected: tuple[ExpectedFinding, ...]


def _op_shift_eye(rng: np.random.Generator, used: set[str]) -> DistortionOp | None:
    if "eye_vertical_shift" in used:
        return None
    choices = [n for n in ("left_eye", "right_eye") if f"{n}_vertical" not in used]
    if not choices:
        return None
    name = str(rng.choice(choices))
    dy = float(rng.choice((-1.0, 1.0)) * rng.uniform(4.8, 7.4))
    used.add(f"{name}_vertical")
    used.add("eye_vertical_shift")
    return lambda lm, name=name, dy=dy: shift_feature(lm, name, dy=dy)


def _op_scale_brow(rng: np.random.Generator, used: set[str]) -> DistortionOp | None:
    choices = [n for n in ("left_brow", "right_brow") if f"{n}_scale" not in used]
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
    choices = [n for n in line_names if str(config.ANGLE_LINES[n]["id"]) not in used]
    if not choices:
        return None
    name = str(rng.choice(choices))
    deg = float(rng.choice((-1.0, 1.0)) * rng.uniform(5.5, 9.0))
    used.add(str(config.ANGLE_LINES[name]["id"]))
    return lambda lm, name=name, deg=deg: rotate_line(lm, name, deg)


_FACTORIES: tuple[OpFactory, ...] = (_op_shift_eye, _op_scale_brow, _op_rotate_line)


def _build_case(rng: np.random.Generator, n_ops: int) -> tuple[Landmarks, tuple[ExpectedFinding, ...]]:
    reference = demo_reference()
    used: set[str] = set()
    ops: list[DistortionOp] = []
    attempts = 0
    while len(ops) < n_ops and attempts < 20:
        attempts += 1
        op = _FACTORIES[int(rng.integers(0, len(_FACTORIES)))](rng, used)
        if op is not None:
            ops.append(op)
    if len(ops) != n_ops:
        raise AssertionError("could not assemble a unique benchmark case")
    return compose(reference, *ops)


def build_dataset(
    n_cases: int = config.BENCH_N_CASES,
    seed: int = config.BENCH_SEED,
) -> list[Triple]:
    """Generate the fixed benchmark triples.

    The first half are single-error cases (cleanest possible localization signal);
    the second half compose 2–3 distinct labeled errors to exercise ranking and the
    coarse-to-fine ordering. Identical seed → identical dataset (principle #7).
    """
    rng = np.random.default_rng(seed)
    reference = demo_reference()
    triples: list[Triple] = []
    for i in range(n_cases):
        n_ops = 1 if i < n_cases // 2 else int(rng.integers(2, 4))
        sketch, expected = _build_case(rng, n_ops)
        triples.append(
            Triple(case_id=f"case_{i:03d}", reference=reference, sketch=sketch, expected=expected)
        )
    return triples
