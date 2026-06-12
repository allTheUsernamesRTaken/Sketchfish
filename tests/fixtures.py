"""Test fixtures — a hardcoded canonical 68-point face (iBUG / 300-W ordering).

A tiny, deterministic, front-facing reference face used by the Wave 0 alignment
tests. Coordinates are in image space (x right, y down), centred near (250, 290)
with a head height of ~270 px. Groups follow the v1 68-point convention
(spec §5; see ``artstockfish.frame.SEMANTIC_GROUPS``).
"""

from __future__ import annotations

import numpy as np

from artstockfish.frame import LANDMARK_NAMES_68
from artstockfish.schema import Landmarks

# 68 points, grouped by the iBUG convention. Jaw is a smooth lower arc; eyes,
# brows, nose and mouth are placed to look like a plausible frontal face.
_CANONICAL_FACE_68 = np.array(
    [
        # --- jaw (0–16), left temple → chin (8) → right temple ---
        [102.18, 278.90], [108.91, 307.72], [119.89, 334.59], [134.87, 358.96],
        [153.37, 380.02], [174.83, 397.12], [198.57, 409.69], [223.88, 417.40],
        [250.00, 420.00], [276.12, 417.40], [301.43, 409.69], [325.17, 397.12],
        [346.63, 380.02], [365.13, 358.96], [380.11, 334.59], [391.09, 307.72],
        [397.82, 278.90],
        # --- right brow (17–21), subject's right / image left ---
        [120.00, 152.00], [142.00, 142.00], [166.00, 139.00], [190.00, 142.00],
        [212.00, 150.00],
        # --- left brow (22–26) ---
        [288.00, 150.00], [310.00, 142.00], [334.00, 139.00], [358.00, 142.00],
        [380.00, 152.00],
        # --- nose bridge (27–30), top → tip ---
        [250.00, 165.00], [250.00, 200.00], [250.00, 235.00], [250.00, 270.00],
        # --- nose bottom (31–35), nostril base ---
        [228.00, 285.00], [239.00, 290.00], [250.00, 293.00], [261.00, 290.00],
        [272.00, 285.00],
        # --- right eye (36–41), outer → around ---
        [135.00, 200.00], [150.00, 190.00], [180.00, 190.00], [195.00, 200.00],
        [180.00, 210.00], [150.00, 210.00],
        # --- left eye (42–47), inner → around ---
        [305.00, 200.00], [320.00, 190.00], [350.00, 190.00], [365.00, 200.00],
        [350.00, 210.00], [320.00, 210.00],
        # --- mouth outer (48–59) ---
        [213.00, 345.00], [225.00, 335.00], [238.00, 330.00], [250.00, 332.00],
        [262.00, 330.00], [275.00, 335.00], [287.00, 345.00], [275.00, 357.00],
        [262.00, 362.00], [250.00, 364.00], [238.00, 362.00], [225.00, 357.00],
        # --- mouth inner (60–67) ---
        [220.00, 345.00], [235.00, 340.00], [250.00, 341.00], [265.00, 340.00],
        [280.00, 345.00], [265.00, 350.00], [250.00, 351.00], [235.00, 350.00],
    ],
    dtype=np.float64,
)

assert _CANONICAL_FACE_68.shape == (68, 2), "canonical face must be 68 points"


def canonical_face_points() -> np.ndarray:
    """Return a fresh ``(68, 2)`` copy of the canonical reference face."""
    return _CANONICAL_FACE_68.copy()


def canonical_face_landmarks() -> Landmarks:
    """Return the canonical face as a :class:`Landmarks` with 68-point names."""
    return Landmarks(
        points=canonical_face_points(),
        names=LANDMARK_NAMES_68,
        image_size=(500, 500),
    )
