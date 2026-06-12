"""M1-T4 stability under small landmark jitter."""

from __future__ import annotations

from collections import Counter

import numpy as np

from artstockfish import config
from artstockfish.frame import build_face_frame
from artstockfish.pipeline import critique_pair, demo_reference
from artstockfish.schema import Landmarks
from artstockfish.synth.distort import compose, rotate_line, scale_feature, shift_feature


def _jitter_landmarks(landmarks: Landmarks, jitter: np.ndarray) -> Landmarks:
    return Landmarks(
        points=landmarks.points + jitter,
        names=landmarks.names,
        image_size=landmarks.image_size,
    )


def _finding_signature(result) -> tuple[tuple[str, str], ...]:
    return tuple((f.id, f.severity.value) for f in result.report.findings)


def test_m1_t4_jittered_findings_are_stable():
    reference = demo_reference()
    sketch, _ = compose(
        reference,
        lambda lm: shift_feature(lm, "left_eye", dy=6.0),
        lambda lm: scale_feature(lm, "right_brow", 1.12),
        lambda lm: rotate_line(lm, "mouth_line", 7.0),
    )

    frame = build_face_frame(reference.points)
    sigma_px = config.SYNTH_JITTER_SIGMA_HEAD_FRAC * frame.head_height
    rng = np.random.default_rng(config.SYNTH_RANDOM_SEED + 1)

    signatures = []
    for _ in range(config.SYNTH_STABILITY_RUNS):
        jitter = rng.normal(0.0, sigma_px, size=reference.points.shape)
        jittered_ref = _jitter_landmarks(reference, jitter)
        jittered_sketch = _jitter_landmarks(sketch, jitter)
        signatures.append(_finding_signature(critique_pair(jittered_ref, jittered_sketch)))

    modal_count = Counter(signatures).most_common(1)[0][1]
    stability = modal_count / config.SYNTH_STABILITY_RUNS
    assert stability >= config.SYNTH_STABILITY_GATE, {
        "stability": stability,
        "signatures": signatures,
    }
