"""Face coordinate frame (spec §9.2).

After alignment, residuals are expressed in the **reference's** face frame so that
every magnitude is size- and tilt-invariant and reads as "% of head height":

- origin    = face centroid (mean of the reference landmarks)
- y-axis    = midline direction (fit through nose-bridge + chin landmarks),
              oriented to point *up the face* (chin → forehead)
- x-axis    = perpendicular to the midline
- unit length = head height (chin → top of the face cloud along the midline)

This module also owns the v1 **68-point** landmark vocabulary (300-W / iBUG
convention, spec §5): the semantic groups and the anatomical anchors the frame
needs. Downstream measurement modules may reuse these groupings.

Note on head height: the 68-point set has no crown/forehead-top landmark, so the
true chin→crown head height is not measurable. We approximate it by the span of
the reference landmark cloud projected onto the midline (chin → brow line). This
is internally consistent: every residual and threshold is scaled by the same
unit. (Logged in DECISIONS.md.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Landmarks

# v1 68-point semantic groups (iBUG / 300-W ordering, spec §5).
SEMANTIC_GROUPS: dict[str, tuple[int, ...]] = {
    "jaw": tuple(range(0, 17)),          # 0–16, chin contour; index 8 = chin
    "right_brow": tuple(range(17, 22)),  # subject's right (image left)
    "left_brow": tuple(range(22, 27)),
    "nose_bridge": tuple(range(27, 31)),  # 27 top of bridge → 30 nose tip
    "nose_bottom": tuple(range(31, 36)),  # nostril base
    "right_eye": tuple(range(36, 42)),
    "left_eye": tuple(range(42, 48)),
    "mouth": tuple(range(48, 68)),       # 48–59 outer, 60–67 inner
}

# Anatomical anchors by index (iBUG 68-point convention).
ANCHORS: dict[str, int] = {
    "chin": 8,
    "nose_bridge_top": 27,
    "nose_tip": 30,
    "right_eye_outer": 36,
    "right_eye_inner": 39,
    "left_eye_inner": 42,
    "left_eye_outer": 45,
}

# Landmarks that define the facial midline (spec §9.2): the nose bridge plus the
# chin. A line fit through these is the vertical axis of the face frame.
MIDLINE_INDICES: tuple[int, ...] = SEMANTIC_GROUPS["nose_bridge"] + (ANCHORS["chin"],)

N_LANDMARKS_68 = 68

# Canonical names for the 68-point set: "<group>_<k>" with the chin called out.
LANDMARK_NAMES_68: tuple[str, ...] = tuple(
    "chin"
    if idx == ANCHORS["chin"]
    else next(
        f"{group}_{members.index(idx)}"
        for group, members in SEMANTIC_GROUPS.items()
        if idx in members
    )
    for idx in range(N_LANDMARKS_68)
)


@dataclass(frozen=True)
class FaceFrame:
    """An orthonormal face frame plus a head-height normalization length.

    ``x_axis``/``y_axis`` are unit vectors in image coordinates (y points down).
    ``y_axis`` points up the face (chin → forehead). All residual helpers return
    values in **percent of head height**.
    """

    origin: np.ndarray      # (2,) face centroid, image coords
    x_axis: np.ndarray      # (2,) unit, perpendicular to midline
    y_axis: np.ndarray      # (2,) unit, up the face (chin → forehead)
    head_height: float      # image units; the normalization length

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Express absolute points in frame coords, in units of head height.

        Returns ``(N, 2)`` with columns ``(along-x, along-y)``; multiply by 100
        for percent of head height.
        """
        p = np.asarray(points, dtype=np.float64) - self.origin
        x = p @ self.x_axis
        y = p @ self.y_axis
        return np.stack([x, y], axis=-1) / self.head_height

    def residual_components(
        self, ref_points: np.ndarray, sketch_points: np.ndarray
    ) -> np.ndarray:
        """Per-point ``(horizontal, vertical)`` residual in % of head height.

        ``sketch_points`` must already be aligned to ``ref_points`` (e.g. via
        :func:`artstockfish.align.robust_align`). The residual is a *displacement*
        vector, so only the axes — not the origin — are used. Positive vertical
        means the sketch point sits further up the face ("too high").
        """
        d = np.asarray(sketch_points, dtype=np.float64) - np.asarray(
            ref_points, dtype=np.float64
        )
        h = d @ self.x_axis
        v = d @ self.y_axis
        return np.stack([h, v], axis=-1) / self.head_height * 100.0

    def residual_magnitude(
        self, ref_points: np.ndarray, sketch_points: np.ndarray
    ) -> np.ndarray:
        """Per-point residual magnitude in % of head height."""
        comp = self.residual_components(ref_points, sketch_points)
        return np.linalg.norm(comp, axis=-1)


def build_face_frame(points: np.ndarray) -> FaceFrame:
    """Build a :class:`FaceFrame` from 68-point landmark coordinates.

    Args:
        points: reference landmarks, shape ``(68, 2)`` in image coordinates,
            following the iBUG 68-point ordering.

    Returns:
        The reference face frame (§9.2).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {pts.shape}")
    if pts.shape[0] < N_LANDMARKS_68:
        raise ValueError(
            f"face frame needs the full 68-point set; got {pts.shape[0]} points"
        )

    origin = pts.mean(axis=0)

    # Midline direction: principal axis of the nose-bridge + chin points.
    midline_pts = pts[list(MIDLINE_INDICES)]
    centered = midline_pts - midline_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    y_axis = vt[0]
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Orient "up the face": from chin toward the top of the nose bridge.
    chin = pts[ANCHORS["chin"]]
    bridge_top = pts[ANCHORS["nose_bridge_top"]]
    if np.dot(bridge_top - chin, y_axis) < 0:
        y_axis = -y_axis

    # x-axis ⊥ midline (CCW 90° of y_axis in image coords).
    x_axis = np.array([-y_axis[1], y_axis[0]])

    # Head height: span of the landmark cloud projected onto the midline.
    proj = (pts - origin) @ y_axis
    head_height = float(proj.max() - proj.min())
    if head_height <= 0:
        raise ValueError("degenerate landmarks: head height is non-positive")

    return FaceFrame(origin=origin, x_axis=x_axis, y_axis=y_axis, head_height=head_height)


def face_frame(landmarks: Landmarks) -> FaceFrame:
    """Convenience wrapper: build the face frame from a :class:`Landmarks`."""
    return build_face_frame(landmarks.points)
