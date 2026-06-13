"""MediaPipe FaceLandmarker → 68-point Landmarks (spec §8 M2).

Role per the de-risk report (``data/detection_report.md``): MediaPipe is the
**reference-side** detector — it is reliable on photos (80–100% control hit rate) —
and only an *opportunistic, sanity-gated* fast-path on sketches, where it fails on
exactly the clean line art v1 targets (33% on real drawings, 20% on XDoG). The
default sketch path is CPD (:mod:`artstockfish.detect.cpd_register`).

The 478-point Tasks-API mesh is downsampled to the project's frozen 68-point iBUG /
300-W convention (spec §5) via a fixed index map, so everything downstream
(``frame``, ``measure/*``, ``pose``) consumes detections unchanged.

ML is used here for *correspondence only* — finding where landmarks are. No learned
model produces a number that appears in a critique (principle #1).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..frame import LANDMARK_NAMES_68
from ..schema import Landmarks


class DetectionError(RuntimeError):
    """Raised when an image yields no usable landmarks."""


# MediaPipe face-mesh index for each iBUG 68-point landmark (spec §5 convention).
# This is the widely used mesh→68 correspondence: the 17 jaw points follow the mesh's
# face-oval ring; brows, eye rings, nose bridge/base, and lip rings map to the mesh
# vertices at the same anatomical positions. Verified by eye on data/ photos (the
# 68-point overlay sits on the same features the 478-point mesh marks).
MP478_TO_IBUG68: tuple[int, ...] = (
    # jaw 0–16 (face oval, subject's right ear → chin → left ear; 152 = chin)
    127, 234, 93, 58, 172, 136, 150, 176, 152, 400, 379, 365, 288, 361, 323, 454, 356,
    # right brow 17–21
    70, 63, 105, 66, 107,
    # left brow 22–26
    336, 296, 334, 293, 300,
    # nose bridge 27–30 (nasion → tip)
    168, 197, 5, 4,
    # nose base row 31–35 (right flare → subnasale → left flare)
    98, 97, 2, 326, 327,
    # right eye 36–41 (outer corner, upper ×2, inner corner, lower ×2)
    33, 160, 158, 133, 153, 144,
    # left eye 42–47 (inner corner, upper ×2, outer corner, lower ×2)
    362, 385, 387, 263, 373, 380,
    # mouth outer 48–59 (right corner → upper lip → left corner → lower lip)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # mouth inner 60–67
    78, 82, 13, 312, 308, 317, 14, 87,
)
assert len(MP478_TO_IBUG68) == 68


def resolve_model_path(model_path: str | os.PathLike | None = None) -> Path:
    """Locate the FaceLandmarker model bundle.

    Order: explicit argument → ``ARTSTOCKFISH_FACE_MODEL`` env var →
    ``config.DETECT_MODEL_PATH`` resolved against the repo root (the package lives at
    ``<root>/src/artstockfish``, an editable install).
    """
    candidate = model_path or os.environ.get("ARTSTOCKFISH_FACE_MODEL")
    if candidate is None:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / config.DETECT_MODEL_PATH
    candidate = Path(candidate)
    if not candidate.is_file():
        raise DetectionError(
            f"FaceLandmarker model not found at {candidate}. Download it with:\n"
            "  curl -sL -o data/_scripts/face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )
    return candidate


@lru_cache(maxsize=4)
def _landmarker(model_path: str, confidence: float):
    """One cached FaceLandmarker per (model, confidence) — they are expensive to build."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_mesh(
    image: np.ndarray,
    *,
    confidence: float = config.DETECT_MIN_CONFIDENCE,
    model_path: str | os.PathLike | None = None,
) -> np.ndarray | None:
    """Run FaceLandmarker on one image → ``(478, 2)`` pixel coords, or ``None``."""
    import mediapipe as mp

    rgb = np.ascontiguousarray(_to_rgb(image))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker(str(resolve_model_path(model_path)), confidence).detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    mesh = result.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in mesh], dtype=np.float64)


def mesh_to_68(mesh: np.ndarray) -> np.ndarray:
    """Downsample a ``(478, 2)`` mesh to the iBUG 68-point set."""
    return np.asarray(mesh, dtype=np.float64)[list(MP478_TO_IBUG68)]


def detect_landmarks_68(
    image: np.ndarray,
    *,
    confidence: float = config.DETECT_MIN_CONFIDENCE,
    model_path: str | os.PathLike | None = None,
) -> Landmarks | None:
    """Detect one face and return 68-point :class:`Landmarks`, or ``None``."""
    mesh = detect_mesh(image, confidence=confidence, model_path=model_path)
    if mesh is None:
        return None
    h, w = np.asarray(image).shape[:2]
    return Landmarks(points=mesh_to_68(mesh), names=LANDMARK_NAMES_68, image_size=(w, h))


def mesh_sanity_gate(points68: np.ndarray, ink_mask: np.ndarray) -> bool:
    """Is a sketch-side MediaPipe mesh believable, given the drawing's ink?

    The de-risk report's la14 case shows the regressor can return a confidently
    *misplaced* mesh (collapsed onto a small shadowed patch); silently accepting one
    would poison the alignment (principle #3). Two deterministic geometry checks,
    both relative to the ink (the drawing's actual marks):

    1. **Span** — the mesh bbox diagonal must be a sensible fraction of the ink bbox
       diagonal (a collapsed mesh is far smaller than the drawing).
    2. **On-ink** — most landmarks must lie near *some* ink (a mesh on blank paper
       or on an isolated noise patch fails).
    """
    pts = np.asarray(points68, dtype=np.float64)
    ink_yx = np.argwhere(ink_mask)
    if len(ink_yx) < 10:
        return False
    ink_xy = ink_yx[:, ::-1].astype(np.float64)

    ink_diag = float(np.linalg.norm(ink_xy.max(0) - ink_xy.min(0)))
    mesh_diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    if ink_diag <= 0 or mesh_diag < config.DETECT_GATE_MIN_SPAN * ink_diag:
        return False

    radius = config.DETECT_GATE_INK_RADIUS_FRAC * ink_diag
    # Distance-to-ink via a distance transform on the non-ink region.
    not_ink = (~ink_mask).astype(np.uint8)
    dist = cv2.distanceTransform(not_ink, cv2.DIST_L2, 3)
    h, w = ink_mask.shape
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    near_ink = float(np.mean(dist[ys, xs] <= radius))
    return near_ink >= config.DETECT_GATE_MIN_NEAR_INK
