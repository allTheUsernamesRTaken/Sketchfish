"""Wave 0 acceptance tests — M0-T3 (robustness) and M0-T4 (rotation absorb).

Run: ``pytest tests/test_align.py -q``

These exercise the alignment spine on the hardcoded canonical 68-point face
(``tests/fixtures.py``). They encode two design principles from spec §2:

- **#3 Robust alignment.** One huge drawing error must not drag the fit and smear
  blame across correct features (M0-T3). The test also shows the *naive*
  least-squares fit fails this — that's the point of using robust Procrustes.
- **#2 Similarity-only semantics.** A globally tilted page is not an error; the
  similarity transform absorbs it, leaving zero residual (M0-T4).
"""

from __future__ import annotations

import numpy as np
import pytest

from artstockfish.align import (
    apply_similarity,
    robust_align,
    rotation_angle_deg,
    similarity_procrustes,
)
from artstockfish.config import DISPLACEMENT_OK_MAX, DISPLACEMENT_MISTAKE_MAX
from artstockfish.frame import SEMANTIC_GROUPS, build_face_frame

from fixtures import canonical_face_points


def _other_indices(group: str) -> list[int]:
    members = set(SEMANTIC_GROUPS[group])
    return [i for i in range(68) if i not in members]


# --- M0-T3: robust alignment is not dragged by one large group error ----------

def test_t3_robust_alignment_not_dragged():
    """Displace one landmark group by 25% of head height → every OTHER feature's
    residual stays below the OK threshold (the alignment was not dragged)."""
    ref = canonical_face_points()
    frame = build_face_frame(ref)
    group = "left_eye"
    others = _other_indices(group)

    sketch = ref.copy()
    sketch[list(SEMANTIC_GROUPS[group]), 0] += 0.25 * frame.head_height  # 25% sideways

    s, R, t = robust_align(ref, sketch)
    aligned = apply_similarity(s, R, t, sketch)
    residual_pct = frame.residual_magnitude(ref, aligned)

    # Robustness: untouched features are not smeared with blame.
    assert residual_pct[others].max() < DISPLACEMENT_OK_MAX, (
        f"robust alignment was dragged: max other residual "
        f"{residual_pct[others].max():.3f}% ≥ OK {DISPLACEMENT_OK_MAX}%"
    )

    # Sanity: the real error is still right there on the displaced group.
    displaced = list(SEMANTIC_GROUPS[group])
    assert residual_pct[displaced].max() > DISPLACEMENT_MISTAKE_MAX


def test_t3_naive_least_squares_is_dragged():
    """Same input, naive (uniform-weight) Procrustes: the big error DOES smear
    onto correct features — confirming why robust alignment is required."""
    ref = canonical_face_points()
    frame = build_face_frame(ref)
    group = "left_eye"
    others = _other_indices(group)

    sketch = ref.copy()
    sketch[list(SEMANTIC_GROUPS[group]), 0] += 0.25 * frame.head_height

    s, R, t = similarity_procrustes(ref, sketch, np.ones(len(ref)))
    aligned = apply_similarity(s, R, t, sketch)
    residual_pct = frame.residual_magnitude(ref, aligned)

    assert residual_pct[others].max() >= DISPLACEMENT_OK_MAX, (
        "naive least-squares unexpectedly stayed below OK — fixture no longer "
        "demonstrates the robustness contrast"
    )


# --- M0-T4: a globally tilted page is absorbed, not flagged --------------------

@pytest.mark.parametrize("center_label", ["centroid", "offset"])
def test_t4_global_rotation_absorbed(center_label):
    """Rotate the whole sketch by 7° → the similarity transform absorbs it and
    every residual is ~zero (page tilt is not a drawing error)."""
    ref = canonical_face_points()
    frame = build_face_frame(ref)

    theta = np.radians(7.0)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    center = ref.mean(axis=0) if center_label == "centroid" else np.array([0.0, 0.0])
    sketch = (rot @ (ref - center).T).T + center

    s, R, t = robust_align(ref, sketch)
    aligned = apply_similarity(s, R, t, sketch)
    residual_pct = frame.residual_magnitude(ref, aligned)

    # Zero residual everywhere — well below the OK noise floor.
    assert residual_pct.max() < 1e-6
    assert residual_pct.max() < DISPLACEMENT_OK_MAX

    # The transform actually recovered the 7° page tilt (mapping sketch back).
    assert abs(rotation_angle_deg(R)) == pytest.approx(7.0, abs=1e-3)
