"""M2 detection orchestration: images → Landmarks → noise-floored critique.

Implements the de-risk report's chosen architecture (``data/detection_report.md``,
PATH 2):

- **Reference** (a photo / clean image): MediaPipe FaceLandmarker, with one relaxed
  retry — the report's control shows photos detect reliably.
- **Sketch**: MediaPipe is tried first as an *opportunistic fast-path*, but every
  sketch detection must pass the ink-based sanity gate (the report's la14 case
  returned a confidently-misplaced mesh). On miss or gate failure, the default
  path runs: **CPD transfers the reference's landmarks onto the sketch's strokes**
  (:mod:`artstockfish.detect.cpd_register`).
- **Noise floor**: findings measured from detected landmarks are surfaced only above
  the raised detection floors (spec §8 M2-T2; pitfall §12 "do not show findings
  below the detector-noise floor"). The floors live in ``config`` (``# --- detect
  ---``) and were calibrated on detector-jitter pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from ..critique import critique_report
from ..evaluate import build_report
from ..frame import LANDMARK_NAMES_68
from ..schema import Finding, Landmarks
from .cpd_register import cpd_transfer_landmarks, extract_ink_points, ink_mask
from .mediapipe_face import (
    DetectionError,
    detect_landmarks_68,
    mesh_sanity_gate,
)

__all__ = [
    "DetectionError",
    "DetectedPair",
    "load_image",
    "detect_reference",
    "detect_sketch",
    "detect_pair",
    "apply_detection_noise_floor",
    "critique_images",
]


@dataclass(frozen=True)
class DetectedPair:
    """Landmarks for one (reference, sketch) image pair, plus provenance."""

    reference: Landmarks
    sketch: Landmarks
    sketch_detector: str        # "mediapipe" (gated fast-path) or "cpd" (default path)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file (BGR uint8); raise :class:`DetectionError` if unreadable."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise DetectionError(f"could not read image: {path}")
    return img


def detect_reference(image: np.ndarray) -> Landmarks:
    """Detect the reference's 68 landmarks (MediaPipe; relaxed retry per the report)."""
    lm = detect_landmarks_68(image, confidence=config.DETECT_MIN_CONFIDENCE)
    if lm is None:
        lm = detect_landmarks_68(image, confidence=config.DETECT_RELAXED_CONFIDENCE)
    if lm is None:
        raise DetectionError(
            "no face detected in the reference image — the reference must be a "
            "photo or clean image with a visible, roughly frontal face (spec §3)"
        )
    return lm


def _reference_ink(reference_image: np.ndarray, reference_landmarks: Landmarks) -> np.ndarray:
    """Reference-side CPD cloud: XDoG edges of the photo, cropped to the head.

    The sketch input is, per scope (§3), a drawing *of the head* — cropping the
    reference to the expanded landmark bbox makes both clouds describe the same
    subject instead of registering the drawing against the photo's background.
    """
    from ..synth.sketchify import xdog

    pts = np.asarray(reference_landmarks.points, dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = config.DETECT_FACE_CROP_MARGIN * (hi - lo)
    h, w = np.asarray(reference_image).shape[:2]
    x0, y0 = np.maximum(np.floor(lo - margin).astype(int), 0)
    x1 = min(int(np.ceil(hi[0] + margin[0])), w)
    y1 = min(int(np.ceil(hi[1] + margin[1])), h)

    # XDoG on the crop itself (not the full frame) so its z-scored line density is
    # set by the head, exactly as it is in a sketch of the head.
    edges = ink_mask(xdog(np.asarray(reference_image)[y0:y1, x0:x1]))
    return extract_ink_points(edges) + np.array([x0, y0], dtype=np.float64)


def detect_sketch(
    sketch_image: np.ndarray,
    reference_image: np.ndarray,
    reference_landmarks: Landmarks,
) -> tuple[Landmarks, str]:
    """Turn a sketch image into 68-point Landmarks.

    The CPD transfer (the trusted default path per the de-risk report) always runs.
    If MediaPipe also fires on the sketch, its mesh is preferred — it reads feature
    positions directly rather than off edge correspondence — but only when it passes
    the ink-plausibility check AND agrees with the CPD answer
    (``config.DETECT_GATE_AGREEMENT_MAX``): the report's la14 case shows MediaPipe
    can return a confidently-misplaced mesh, and a junk mesh cannot agree with an
    independent classical reading of the same drawing.

    Returns the landmarks plus which detector produced them (``"mediapipe"`` /
    ``"cpd"``).
    """
    from ..frame import build_face_frame

    sketch = np.asarray(sketch_image)
    h, w = sketch.shape[:2]
    strokes = ink_mask(sketch)

    sketch_pts = extract_ink_points(strokes)
    ref_ink = _reference_ink(reference_image, reference_landmarks)
    cpd_points = cpd_transfer_landmarks(ref_ink, reference_landmarks.points, sketch_pts)

    fast = detect_landmarks_68(sketch, confidence=config.DETECT_MIN_CONFIDENCE)
    if fast is not None and mesh_sanity_gate(fast.points, strokes):
        from .cpd_register import _LOCAL_GROUPS

        # Agreement is judged per semantic group (worst group-mean distance): the
        # measurement layer consumes group means, so a median over all 68 points
        # would hide exactly the disagreement that changes the critique (measured:
        # a mesh within 4% median still differed 6% on the jaw → 20° of pitch).
        head_height = build_face_frame(cpd_points).head_height
        disagreement = max(
            float(np.linalg.norm((fast.points[list(idx)] - cpd_points[list(idx)]).mean(axis=0)))
            for idx in _LOCAL_GROUPS.values()
        ) / head_height * 100.0
        if disagreement <= config.DETECT_GATE_AGREEMENT_MAX:
            return fast, "mediapipe"

    return (
        Landmarks(points=cpd_points, names=LANDMARK_NAMES_68, image_size=(w, h)),
        "cpd",
    )


def detect_pair(reference_image: np.ndarray, sketch_image: np.ndarray) -> DetectedPair:
    """Detect landmarks for a (reference, sketch) image pair."""
    reference = detect_reference(reference_image)
    sketch, detector = detect_sketch(sketch_image, reference_image, reference)
    return DetectedPair(reference=reference, sketch=sketch, sketch_detector=detector)


# --- detection noise floor (M2-T2) -------------------------------------------------

def _detection_floor(finding: Finding) -> float:
    """The raised OK floor for one finding, chosen by its units/axis."""
    if finding.units == "%head_height":
        return config.DETECT_OK_DISPLACEMENT
    if finding.units == "deg":
        return config.DETECT_OK_POSE if finding.axis == "pose" else config.DETECT_OK_ANGLE
    if finding.units == "%area":
        return config.DETECT_OK_AREA
    if finding.units == "%ratio":
        return config.DETECT_OK_RATIO
    return 0.0  # unknown units: never silently drop


def apply_detection_noise_floor(findings) -> list[Finding]:
    """Drop findings whose magnitude sits below the detection noise floor.

    The synthetic OK floors (§6) assume exact landmarks; detected landmarks carry
    correspondence jitter, so anything below the calibrated per-axis detection floor
    is indistinguishable from detector noise and must not be shown (M2-T2). The rule
    is uniform over every finding — no special-casing of inputs (Ground Rule 5).
    """
    return [f for f in findings if f.magnitude >= _detection_floor(f)]


def critique_images(
    reference_path: str | Path,
    sketch_path: str | Path,
    *,
    overlay_path: str | None = None,
):
    """End-to-end: two image files → noise-floored CritiqueResult (the M2 demo).

    Detection (above) feeds the unchanged M0/M1.5 pipeline; the detection noise
    floor then filters the findings and the report/sentences are rebuilt so the
    accuracy score and ranking reflect exactly the surfaced findings.

    Returns ``(result, pair)`` — the critique plus the detected landmarks/provenance.
    """
    from ..pipeline import CritiqueResult, critique_pair

    reference_image = load_image(reference_path)
    sketch_image = load_image(sketch_path)
    pair = detect_pair(reference_image, sketch_image)

    raw = critique_pair(pair.reference, pair.sketch)
    surfaced = apply_detection_noise_floor(raw.report.findings)
    report = build_report(surfaced, transform=raw.report.transform, pose=raw.report.pose)
    sentences = critique_report(report)

    overlay = None
    if overlay_path is not None:
        from ..annotate import render_overlay  # lazy: only rendering needs matplotlib

        overlay = render_overlay(
            report, raw.reference_points, raw.aligned_sketch_points, overlay_path
        )

    result = CritiqueResult(
        report=report,
        sentences=sentences,
        reference_points=raw.reference_points,
        aligned_sketch_points=raw.aligned_sketch_points,
        overlay_path=overlay,
    )
    return result, pair
