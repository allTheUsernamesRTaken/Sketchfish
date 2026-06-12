"""End-to-end M0 orchestration: landmarks → measure → evaluate → critique → annotate.

Inputs are coordinate lists (:class:`~artstockfish.schema.Landmarks`), not images —
real detection is M2. The pipeline:

1. Registers the sketch to the reference once with the robust **similarity** transform
   (§9.1) and reuses that single alignment for every measurement, so the report's
   ``transform`` and the overlay are consistent.
2. Runs the three measurement modules (placement residuals, feature angles, canon
   ratios) on the shared alignment.
3. **Explains away** derived angle findings (:func:`suppress_explained_angles`): a
   line tilt that is fully accounted for by feature placements already reported is a
   symptom, not a second mistake (spec §2 principle #5, pitfall §12). See DECISIONS.md.
4. Ranks + scores into a :class:`~artstockfish.schema.Report` (``evaluate``), generates
   teacher-voiced sentences (``critique``), and optionally renders the overlay PNG.

All geometry is deterministic (principle #1); the same inputs give the same report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import apply_similarity, robust_align, rotation_angle_deg
from .critique import critique_report
from .evaluate import build_report
from .frame import LANDMARK_NAMES_68
from .measure.angles import angle_findings, measure_angles
from .measure.landmarks import measure_landmarks
from .measure.proportions import proportion_findings
from .schema import Finding, Landmarks, Report


@dataclass(frozen=True)
class CritiqueResult:
    """Everything the M0 pipeline produces for one (reference, sketch) pair."""

    report: Report
    sentences: tuple[str, ...]          # parallel to report.findings (ranked order)
    reference_points: np.ndarray        # (68, 2) reference landmarks
    aligned_sketch_points: np.ndarray   # (68, 2) sketch registered to the reference
    overlay_path: str | None            # set iff an overlay was rendered


def _group_mean_displacement(aligned: np.ndarray, ref: np.ndarray, idx: list[int]) -> np.ndarray:
    """Mean image-space displacement of a group (aligned − reference)."""
    return (aligned[idx] - ref[idx]).mean(axis=0)


def suppress_explained_angles(
    angle_results: list[Finding],
    placement_results: list[Finding],
    reference_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
) -> list[Finding]:
    """Drop angle findings that are fully explained by reported placements.

    Counterfactual ("if the student fixed the feature *positions*, would this line
    still be tilted?"): translate every group that has a placement finding back onto
    the reference — removing its mean displacement while keeping its internal shape —
    then re-measure the feature angles. An original angle finding is **kept** only if
    its id still fires after that correction (an independent tilt, e.g. a feature
    rotated in place or a jaw tangent with no group displacement). One whose tilt
    collapses below the OK floor was a symptom of the placement error and is suppressed,
    so a single mistake is reported once (spec §2 principle #5, pitfall §12).

    This is a general rule over all angle findings, not a special case (Ground Rule 5).
    """
    if not angle_results or not placement_results:
        return list(angle_results)

    ref = np.asarray(reference_points, dtype=np.float64)
    corrected = np.asarray(aligned_sketch_points, dtype=np.float64).copy()

    # Correct each flagged group once (a group may carry several placement findings).
    corrected_groups: dict[str, list[int]] = {}
    for pf in placement_results:
        group = pf.evidence.get("group")
        idx = pf.evidence.get("indices")
        if group is None or idx is None or group in corrected_groups:
            continue
        idx = list(idx)
        corrected_groups[group] = idx
        corrected[idx] -= _group_mean_displacement(corrected, ref, idx)

    surviving_ids = {f.id for f in angle_findings(ref, corrected)}
    return [f for f in angle_results if f.id in surviving_ids]


def critique_pair(
    reference: Landmarks,
    sketch: Landmarks,
    *,
    overlay_path: str | None = None,
) -> CritiqueResult:
    """Run the full M0 critique on a (reference, sketch) landmark pair.

    Args:
        reference: the reference (target) landmarks, full 68-point set.
        sketch: the student's landmarks, same ordering/length.
        overlay_path: if given, render the annotated overlay PNG there.

    Returns:
        A :class:`CritiqueResult` with the ranked report, parallel critique
        sentences, the shared alignment's aligned sketch points, and the overlay path.
    """
    ref_pts = np.asarray(reference.points, dtype=np.float64)
    sketch_pts = np.asarray(sketch.points, dtype=np.float64)

    # One shared robust similarity alignment for the whole report (§9.1).
    s, R, t = robust_align(ref_pts, sketch_pts)
    aligned = apply_similarity(s, R, t, sketch_pts)
    aligned_lm = Landmarks(
        points=aligned, names=reference.names, image_size=reference.image_size
    )

    # Measure on the shared alignment (align=False ⇒ use the pre-aligned points).
    placement = measure_landmarks(reference, aligned_lm, align=False)
    angles = measure_angles(reference, aligned_lm, align=False)
    # Proportions are similarity-invariant; measure them on the raw sketch.
    proportions = proportion_findings(reference, sketch)

    angles = suppress_explained_angles(angles, placement, ref_pts, aligned)

    findings = placement + angles + proportions
    transform = {
        "scale": float(s),
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "rotation_deg": rotation_angle_deg(R),
    }
    report = build_report(findings, transform=transform, pose=None)
    sentences = critique_report(report)

    overlay = None
    if overlay_path is not None:
        from .annotate import render_overlay  # lazy: only rendering needs matplotlib

        overlay = render_overlay(report, ref_pts, aligned, overlay_path)

    return CritiqueResult(
        report=report,
        sentences=sentences,
        reference_points=ref_pts,
        aligned_sketch_points=aligned,
        overlay_path=overlay,
    )


# --- synthetic demo data (spec §8 M0: "hardcode one canonical face") --------------
#
# A plausible front-facing 68-point face (iBUG / 300-W ordering), mirroring the Wave 0
# test fixture so the CLI demo and the M0 acceptance tests describe the same face. Demo
# input only — not a measurement constant.
_CANONICAL_FACE_68 = np.array(
    [
        # jaw (0–16)
        [102.18, 278.90], [108.91, 307.72], [119.89, 334.59], [134.87, 358.96],
        [153.37, 380.02], [174.83, 397.12], [198.57, 409.69], [223.88, 417.40],
        [250.00, 420.00], [276.12, 417.40], [301.43, 409.69], [325.17, 397.12],
        [346.63, 380.02], [365.13, 358.96], [380.11, 334.59], [391.09, 307.72],
        [397.82, 278.90],
        # right brow (17–21)
        [120.00, 152.00], [142.00, 142.00], [166.00, 139.00], [190.00, 142.00],
        [212.00, 150.00],
        # left brow (22–26)
        [288.00, 150.00], [310.00, 142.00], [334.00, 139.00], [358.00, 142.00],
        [380.00, 152.00],
        # nose bridge (27–30)
        [250.00, 165.00], [250.00, 200.00], [250.00, 235.00], [250.00, 270.00],
        # nose bottom (31–35)
        [228.00, 285.00], [239.00, 290.00], [250.00, 293.00], [261.00, 290.00],
        [272.00, 285.00],
        # right eye (36–41)
        [135.00, 200.00], [150.00, 190.00], [180.00, 190.00], [195.00, 200.00],
        [180.00, 210.00], [150.00, 210.00],
        # left eye (42–47)
        [305.00, 200.00], [320.00, 190.00], [350.00, 190.00], [365.00, 200.00],
        [350.00, 210.00], [320.00, 210.00],
        # mouth outer (48–59)
        [213.00, 345.00], [225.00, 335.00], [238.00, 330.00], [250.00, 332.00],
        [262.00, 330.00], [275.00, 335.00], [287.00, 345.00], [275.00, 357.00],
        [262.00, 362.00], [250.00, 364.00], [238.00, 362.00], [225.00, 357.00],
        # mouth inner (60–67)
        [220.00, 345.00], [235.00, 340.00], [250.00, 341.00], [265.00, 340.00],
        [280.00, 345.00], [265.00, 350.00], [250.00, 351.00], [235.00, 350.00],
    ],
    dtype=np.float64,
)


def _landmarks(points: np.ndarray) -> Landmarks:
    return Landmarks(points=points, names=LANDMARK_NAMES_68, image_size=(500, 500))


def demo_reference() -> Landmarks:
    """The canonical demo reference face."""
    return _landmarks(_CANONICAL_FACE_68.copy())


def demo_synthetic_pair() -> tuple[Landmarks, Landmarks]:
    """A (reference, realistically-perturbed sketch) pair for the M0 demo / eyeball.

    The sketch carries several *independent* beginner errors so the demo exercises the
    coarse-to-fine ranking and the redundancy suppression, not a single isolated shift:

    - the whole **nose is set too low** → a GLOBAL "midface too tall" proportion error
      (the headline structural note) plus a nose placement error;
    - the **left eye is too high** → a placement error whose induced eye-line tilt is
      *explained away* (no redundant angle finding surfaces);
    - the **right eye is drawn too large** → a scale error;
    - the whole drawing sits on a slightly **tilted page**, which the similarity
      alignment must absorb rather than flag (principle #2).
    """
    from .frame import SEMANTIC_GROUPS, build_face_frame

    ref = _CANONICAL_FACE_68.copy()
    frame = build_face_frame(ref)
    up = frame.y_axis
    hh = frame.head_height

    sk = ref.copy()
    # Whole nose ~5% of head height too low (rigid translation → shifts the midface
    # third without changing the nose's own spread): GLOBAL midface + nose placement.
    nose = list(SEMANTIC_GROUPS["nose_bridge"]) + list(SEMANTIC_GROUPS["nose_bottom"])
    sk[nose] -= 0.05 * hh * up
    # Left eye ~6% too high: a placement mistake; its induced eye-line tilt is a
    # symptom that suppress_explained_angles removes.
    sk[list(SEMANTIC_GROUPS["left_eye"])] += 0.06 * hh * up
    # Subject's right eye drawn ~10% larger (scale about its own centre).
    re = list(SEMANTIC_GROUPS["right_eye"])
    c = sk[re].mean(axis=0)
    sk[re] = c + (sk[re] - c) * 1.10

    # A 5° page tilt about the centroid — not a drawing error; alignment absorbs it.
    theta = np.radians(5.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = sk.mean(axis=0)
    sk = (rot @ (sk - centre).T).T + centre

    return _landmarks(ref), _landmarks(sk)
