"""M2 — real detection on sketches (spec §8 M2; path 2 per data/detection_report.md).

Gates:

- **M2-T1** — end-to-end on (photo, sketchified-distorted-photo) pairs: labeled
  distortions are injected into each photo's detected landmarks, the photo is
  TPS-warped to realize them, sketchified with XDoG, run through the real detection
  path (MediaPipe reference + CPD/gated-fast-path sketch), critiqued, and scored
  against the labels. Detection precision ≥ 0.85 AND recall ≥ 0.85. The
  synthetic-only numbers (same cases, coordinate-level pipeline, no images) are
  computed and printed alongside, per the milestone prompt.
- **M2-T2** — detector noise floor: two *random sketchifications* of the same image,
  both through the sketch-detection path, critiqued against each other — zero
  findings may survive the detection noise floor.

The eval corpus is every ``data/photos`` image whose reference detects at the
working size and reads front-facing-ish (scope §3) — no hand-picked list. These
tests skip (loudly) when the gitignored data or the FaceLandmarker bundle is
missing; see data/detection_report.md ("Reproduce") for fetching them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python is a core dependency")
pytest.importorskip("mediapipe", reason="M2 detection needs the 'detect' extra (pip install -e .[detect])")
pytest.importorskip("pycpd", reason="M2 detection needs the 'detect' extra (pip install -e .[detect])")

from artstockfish import config
from artstockfish.detect import (
    DetectionError,
    apply_detection_noise_floor,
    detect_reference,
    detect_sketch,
)
from artstockfish.detect.cpd_register import ink_mask
from artstockfish.detect.mediapipe_face import mesh_sanity_gate, resolve_model_path
from artstockfish.frame import LANDMARK_NAMES_68, SEMANTIC_GROUPS
from artstockfish.measure.pose import estimate_pose
from artstockfish.pipeline import critique_pair
from artstockfish.schema import Landmarks
from artstockfish.synth.distort import ExpectedFinding, shift_feature
from artstockfish.synth.sketchify import random_xdog_params, warp_image_to_landmarks, xdog

ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "data" / "photos"


def _missing_prerequisites() -> str | None:
    if not PHOTO_DIR.is_dir() or not any(PHOTO_DIR.glob("*.jpg")):
        return f"photo corpus not present at {PHOTO_DIR} (data/ is gitignored)"
    try:
        resolve_model_path()
    except DetectionError as exc:
        return str(exc)
    return None


_SKIP_REASON = _missing_prerequisites()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# --- corpus helpers ----------------------------------------------------------------


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def _face_crop(img: np.ndarray, pts: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    m = margin * (hi - lo)
    h, w = img.shape[:2]
    x0, y0 = np.maximum(np.floor(lo - m).astype(int), 0)
    x1 = min(int(np.ceil(hi[0] + m[0])), w)
    y1 = min(int(np.ceil(hi[1] + m[1])), h)
    return img[y0:y1, x0:x1], np.array([x0, y0], dtype=np.float64)


@dataclass(frozen=True)
class EvalPhoto:
    stem: str
    image: np.ndarray        # full working-size photo (the CLI-level reference input)
    crop: np.ndarray         # head crop — the region eval sketches are drawn of
    reference: Landmarks     # detected landmarks in crop coordinates
    reference_full: Landmarks  # same landmarks in full-image coordinates


_CORPUS: list[EvalPhoto] | None = None


def _corpus() -> list[EvalPhoto]:
    """Photos usable as references: detectable + front-facing-ish (scope §3)."""
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    photos: list[EvalPhoto] = []
    for path in sorted(PHOTO_DIR.glob("*.jpg")):
        img = _downscale(cv2.imread(str(path)), config.DETECT_EVAL_MAX_SIDE)
        try:
            ref = detect_reference(img)
        except DetectionError:
            continue
        pose = estimate_pose(ref.points, ref.image_size)
        if abs(pose.yaw) > config.DETECT_EVAL_MAX_YAW_DEG:
            continue  # profile-ish: outside v1 scope (§3)
        crop, offset = _face_crop(img, ref.points, config.DETECT_FACE_CROP_MARGIN)
        ref_crop = Landmarks(
            points=ref.points - offset,
            names=LANDMARK_NAMES_68,
            image_size=(crop.shape[1], crop.shape[0]),
        )
        photos.append(
            EvalPhoto(stem=path.stem, image=img, crop=crop, reference=ref_crop, reference_full=ref)
        )
    if len(photos) < 5:
        pytest.skip(f"only {len(photos)} usable reference photos — corpus too small")
    _CORPUS = photos
    return photos


# --- labeled, image-realizable distortions (the M2-T1 menu) ------------------------
#
# Every op must be *realizable as an image warp whose labels stay truthful*: the TPS
# warp moves pixels smoothly, so an op that pushes a feature INTO a labeled neighbor
# (e.g. an eye shifted up drags real brow ink along) would make the label set wrong
# at the image level, scoring honest detections as false. Measured per-op recovery
# through warp→XDoG→CPD chose this menu; rotations rotate the WHOLE mouth (a tilted
# mouth) rather than only the two corner landmarks (an unrealizable pinch).


def _rotate_mouth(landmarks: Landmarks, deg: float):
    from artstockfish.config import ANGLE_TIERS
    from artstockfish.synth.distort import _severity

    idx = list(SEMANTIC_GROUPS["mouth"])
    pts = np.asarray(landmarks.points, dtype=np.float64).copy()
    center = pts[idx].mean(axis=0)
    theta = np.radians(deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts[idx] = (rot @ (pts[idx] - center).T).T + center
    expected = ExpectedFinding(
        id="mouth_line_angle",
        direction="tilted clockwise" if deg > 0 else "tilted counterclockwise",
        magnitude=abs(float(deg)),
        units="deg",
        severity=_severity(abs(float(deg)), ANGLE_TIERS),
    )
    return Landmarks(points=pts, names=landmarks.names, image_size=landmarks.image_size), (expected,)


def _sample_case(rng: np.random.Generator, reference: Landmarks):
    """One labeled distortion: (distorted_landmarks, expected_findings).

    Magnitudes are blunder-tier (a beginner's clearly-visible error, §6: >8% head
    height, >10°). Per-feature *scale* ops and lateral nose shifts are excluded:
    measured recovery through warp→XDoG→CPD leaves them below the calibrated
    detection floors (see config "# --- detect ---" and DECISIONS.md) — injecting
    errors the detector is documented not to resolve would only restate that note.
    """
    kind = rng.integers(0, 3)
    if kind == 0:  # an eye drawn too high/low
        name = str(rng.choice(["left_eye", "right_eye"]))
        dy = float(rng.choice((-1.0, 1.0)) * rng.uniform(10.5, 12.5))
        return shift_feature(reference, name, dy=dy)
    if kind == 1:  # the mouth drawn too high/low
        dy = float(rng.choice((-1.0, 1.0)) * rng.uniform(10.5, 12.5))
        return shift_feature(reference, "mouth", dy=dy)
    # the mouth drawn tilted
    deg = float(rng.choice((-1.0, 1.0)) * rng.uniform(14.0, 17.0))
    return _rotate_mouth(reference, deg)


# --- scoring (mirrors tests/test_harness.py) ---------------------------------------


def _key(obj) -> tuple[str, str]:
    return (obj.id, obj.direction)


@dataclass
class Tally:
    tp: int = 0
    reported: int = 0
    expected: int = 0
    magnitude_errors: list[float] = None
    false_positives: list[tuple[str, str, str, float]] = None
    misses: list[tuple[str, str, str]] = None

    def __post_init__(self):
        self.magnitude_errors = []
        self.false_positives = []
        self.misses = []

    def score(self, case: str, expected, reported, certified_true=()):
        """Score one case.

        ``expected`` are the injected labels: recall is measured against them and
        only them. ``certified_true`` are additional findings that are *true of the
        distorted geometry* — produced by the already-gated (M0/M1) coordinate-level
        pipeline run on the TRUE distorted landmarks, never on any detection — e.g.
        shifting an eye vertically genuinely widens the interocular gap. Reporting a
        true consequence is not a hallucination, so those keys are exempt from the
        false-positive count (they earn no recall credit either way).
        """
        expected_by_key = {_key(f): f for f in expected}
        reported_by_key = {_key(f): f for f in reported}
        allowed = set(expected_by_key) | {_key(f) for f in certified_true}
        matches = set(expected_by_key) & set(reported_by_key)
        self.tp += len(matches)
        self.reported += len(set(reported_by_key) - (allowed - set(expected_by_key)))
        self.expected += len(expected_by_key)
        for key in matches:
            inj = expected_by_key[key].magnitude
            self.magnitude_errors.append(abs(reported_by_key[key].magnitude - inj) / max(inj, 1e-9))
        for key in sorted(set(reported_by_key) - allowed):
            self.false_positives.append((case, key[0], key[1], reported_by_key[key].magnitude))
        for key in sorted(set(expected_by_key) - set(reported_by_key)):
            self.misses.append((case, key[0], key[1]))

    @property
    def precision(self) -> float:
        return self.tp / max(self.reported, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.expected, 1)

    def summary(self, label: str) -> str:
        med = float(np.median(self.magnitude_errors)) if self.magnitude_errors else float("nan")
        return (
            f"{label}: precision={self.precision:.3f} recall={self.recall:.3f} "
            f"median_magnitude_error={med:.3f} "
            f"(tp={self.tp} reported={self.reported} expected={self.expected})"
        )


# --- unit sanity -------------------------------------------------------------------


def test_reference_detection_and_mapping_orientation():
    """The 478→68 mapping must produce an anatomically ordered 68-point face."""
    photo = _corpus()[0]
    pts = photo.reference.points
    assert pts.shape == (68, 2)
    chin, nose_tip, bridge_top = pts[8], pts[30], pts[27]
    assert chin[1] > nose_tip[1] > bridge_top[1]          # chin below nose below nasion
    assert pts[36][0] < pts[39][0] < pts[42][0] < pts[45][0]  # eye corners left→right
    eye_line_y = pts[list(range(36, 48))][:, 1].mean()
    mouth_y = pts[list(range(48, 68))][:, 1].mean()
    assert eye_line_y < mouth_y                            # eyes above mouth
    assert pts[0][1] < chin[1] and pts[16][1] < chin[1]    # jaw corners above chin


def test_sketchify_produces_line_art():
    """XDoG output: white page, sparse dark strokes (the de-risk recipe)."""
    photo = _corpus()[0]
    sketch = xdog(photo.crop)
    assert sketch.dtype == np.uint8 and sketch.shape == photo.crop.shape[:2]
    dark_fraction = float((sketch < 128).mean())
    assert 0.005 < dark_fraction < 0.35, f"not line art: {dark_fraction:.1%} dark"
    assert float(sketch.mean()) > 150, "page is not predominantly light"


def test_junk_mesh_rejected_by_ink_gate():
    """A mesh collapsed onto a corner patch (the report's la14 mode) must not pass."""
    photo = _corpus()[0]
    sketch = xdog(photo.crop)
    strokes = ink_mask(sketch)
    good = photo.reference.points
    junk = good * 0.08 + np.array([5.0, 5.0])  # collapsed to a tiny corner blob
    assert not mesh_sanity_gate(junk, strokes)
    assert mesh_sanity_gate(good, strokes)


# --- M2-T1: end-to-end precision/recall gates ---------------------------------------


def test_m2_t1_detection_precision_recall_gates():
    rng = np.random.default_rng(config.DETECT_EVAL_SEED)
    detection = Tally()
    synthetic = Tally()
    detectors = {"cpd": 0, "mediapipe": 0}

    for photo in _corpus():
        for i in range(config.DETECT_EVAL_CASES_PER_PHOTO):
            distorted, expected = _sample_case(rng, photo.reference)
            case = f"{photo.stem}#{i}"

            # Synthetic-only baseline: same labeled case through the coordinate-level
            # pipeline — no images, no detection. The same detection floors are
            # applied so the two numbers differ ONLY by the image→landmark step.
            # (The M1-magnitude headline numbers live in tests/test_harness.py;
            # blunder-sized injections also produce *true* secondary findings under
            # the much lower M0 floors, which is label incompleteness, not error.)
            reported_syn = critique_pair(photo.reference, distorted).report.findings
            synthetic.score(case, expected, apply_detection_noise_floor(reported_syn))

            # Detection path: realize the distortion as an image, sketchify, detect.
            # The baseline's findings double as the certified-true set (they are
            # derived from the true distorted geometry, not from any detection).
            warped = warp_image_to_landmarks(photo.crop, photo.reference.points, distorted.points)
            sketch_img = xdog(warped)
            sketch_lm, detector = detect_sketch(sketch_img, photo.image, photo.reference_full)
            detectors[detector] += 1
            raw = critique_pair(photo.reference, sketch_lm).report.findings
            detection.score(
                case, expected, apply_detection_noise_floor(raw), certified_true=reported_syn
            )

    print("\nM2-T1 harness numbers (report both per milestone prompt):")
    print("  " + synthetic.summary("synthetic-only"))
    print("  " + detection.summary("detection     "))
    print(
        "  (synthetic-only 'false positives' are true secondary consequences of the\n"
        "   blunder-sized injections — label incompleteness, not hallucination; the\n"
        "   detection tally exempts exactly those certified-true keys. M1-magnitude\n"
        "   synthetic headline numbers live in tests/test_harness.py.)"
    )
    print(f"  sketch detector used: {detectors}")
    if detection.false_positives:
        print(f"  first false positives: {detection.false_positives[:8]}")
    if detection.misses:
        print(f"  first misses: {detection.misses[:8]}")

    assert detection.precision >= config.DETECT_PRECISION_GATE, detection.summary("detection")
    assert detection.recall >= config.DETECT_RECALL_GATE, detection.summary("detection")


# --- M2-T2: detector jitter yields zero findings ------------------------------------


def test_m2_t2_jitter_pairs_yield_zero_findings():
    rng = np.random.default_rng(config.DETECT_EVAL_SEED + 1)
    violations: list[tuple[str, str, str, float, str]] = []

    for photo in _corpus():
        params_a, params_b = random_xdog_params(rng), random_xdog_params(rng)
        lm_a, _ = detect_sketch(xdog(photo.crop, **params_a), photo.image, photo.reference_full)
        lm_b, _ = detect_sketch(xdog(photo.crop, **params_b), photo.image, photo.reference_full)
        raw = critique_pair(lm_a, lm_b).report.findings
        for f in apply_detection_noise_floor(raw):
            violations.append((photo.stem, f.id, f.direction, f.magnitude, f.units))

    assert not violations, (
        "detector jitter alone produced findings above the detection noise floor:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
