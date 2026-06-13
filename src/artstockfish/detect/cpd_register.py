"""CPD landmark transfer — the default sketch detector (spec §8 M2, path 2).

The de-risk report (``data/detection_report.md``) showed MediaPipe's face *detector*
simply does not fire on flat line art, while the reference side (a photo) detects
reliably. This module exploits that asymmetry: it never tries to detect on the
sketch at all. Instead it

1. extracts **edge/ink points** from both sides — XDoG of the reference photo
   (cropped to the head) and the sketch's own strokes;
2. registers the reference cloud onto the sketch cloud with **Coherent Point
   Drift** — a similarity-mode CPD first (global position/scale/rotation), then a
   deformable CPD for the residual shape differences;
3. **reads the 68 landmark positions off where they land**: the reference's known
   landmarks are carried through both fitted transforms (for the deformable stage,
   via the Gaussian kernel between the landmarks and the registered cloud).

Fully classical, no training (spec §8 path 2). CPD is used for *correspondence
only* — the critique's alignment remains similarity-only Procrustes downstream
(glossary §13, principle #2). Everything here is deterministic: thresholding is
Otsu, subsampling is grid-based, and pycpd's EM has no random initialization
(principle #7).
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..frame import SEMANTIC_GROUPS

# Groups refined locally (mirrors measure/landmarks.py's MEASURE_GROUPS: the nose is
# one feature to a teacher, and to the local registration).
_LOCAL_GROUPS: dict[str, tuple[int, ...]] = {
    "jaw": SEMANTIC_GROUPS["jaw"],
    "right_brow": SEMANTIC_GROUPS["right_brow"],
    "left_brow": SEMANTIC_GROUPS["left_brow"],
    "nose": SEMANTIC_GROUPS["nose_bridge"] + SEMANTIC_GROUPS["nose_bottom"],
    "right_eye": SEMANTIC_GROUPS["right_eye"],
    "left_eye": SEMANTIC_GROUPS["left_eye"],
    "mouth": SEMANTIC_GROUPS["mouth"],
}

# Open-curve groups suffer the aperture problem: an arc registered onto a similar arc
# can slide along itself, so only the displacement component *normal* to the curve is
# observable. Their local correction is restricted to translation along that normal;
# rotation/scale/tangential motion from an arc window is noise, not measurement.
_ARC_GROUPS = frozenset({"jaw", "right_brow", "left_brow"})


def ink_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask of a drawing's strokes ("ink").

    Otsu-threshold the grayscale image and take the *minority* side as ink: line art
    is sparse strokes on an empty page, whichever polarity (dark-on-light or the
    inverse) the file uses. Tiny connected components are then dropped — real strokes
    are extended curves, while XDoG of skin/paper texture yields isolated specks that
    would pollute the registration cloud with structureless clutter.
    """
    import cv2

    img = np.asarray(image)
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    _, dark = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = dark.astype(bool)
    mask = dark if dark.mean() <= 0.5 else ~dark

    diag = float(np.hypot(*mask.shape))
    min_area = max(int((config.DETECT_INK_MIN_COMPONENT_FRAC * diag) ** 2), 1)
    if min_area > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False  # background label
        mask = keep[labels]
    return mask


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton: reduce strokes to ~1-px centerlines.

    Registration clouds must describe stroke *geometry*, not stroke *weight*: two
    renderings of the same drawing with different pen pressure (or two XDoG
    parameterizations of the same photo) lay down the same centerlines under
    different thicknesses. Without this, thickness differences read as shape
    differences and detector jitter explodes (measured: per-group scale jitter up
    to ~140% area between two random sketchifications; skeletonizing removes the
    thickness component). Classical and deterministic.
    """
    import cv2

    img = mask.astype(np.uint8)
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while img.any():
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel |= img & ~opened
        img = eroded
    return skel.astype(bool)


def extract_ink_points(
    image_or_mask: np.ndarray,
    *,
    max_points: int = config.CPD_MAX_POINTS,
) -> np.ndarray:
    """Subsample a drawing's stroke centerlines to a registration point cloud.

    The ink mask is skeletonized (stroke-width invariance, see :func:`_skeletonize`)
    and then grid-subsampled deterministically (no RNG — principle #7): skeleton
    pixels are binned into square cells sized so roughly ``max_points`` cells are
    occupied, and each occupied cell contributes its centroid. This keeps the cloud
    spatially uniform — random subsampling would over-represent dense hatching.

    Args:
        image_or_mask: a grayscale/BGR drawing, or a precomputed boolean ink mask.
        max_points: approximate point budget (bounds CPD's O(M·N) EM cost).

    Returns:
        ``(N, 2)`` float64 (x, y) pixel coordinates, ``N ≲ max_points``.
    """
    mask = (
        image_or_mask
        if image_or_mask.dtype == bool
        else ink_mask(image_or_mask)
    )
    yx = np.argwhere(_skeletonize(mask))
    if len(yx) == 0:
        return np.empty((0, 2), dtype=np.float64)
    xy = yx[:, ::-1].astype(np.float64)
    if len(xy) <= max_points:
        return xy

    # Skeleton pixels lie on ~1-px curves: a curve crosses a g-sized cell in ~g
    # pixels, so g = n/max_points yields ≈ max_points occupied cells.
    cell = max(len(xy) / max_points, 2.0)
    keys = np.floor((xy - xy.min(0)) / cell).astype(np.int64)
    flat = keys[:, 0] * (keys[:, 1].max() + 1) + keys[:, 1]
    order = np.argsort(flat, kind="stable")
    flat_sorted, xy_sorted = flat[order], xy[order]
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    groups = np.split(xy_sorted, boundaries)
    return np.array([g.mean(axis=0) for g in groups], dtype=np.float64)


def _normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center on the centroid and scale RMS radius to 1 (CPD works best normalized)."""
    mu = points.mean(axis=0)
    centered = points - mu
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    scale = max(scale, 1e-9)
    return centered / scale, mu, scale


def _make_stable_deformable(**kwargs):
    """pycpd ``DeformableRegistration`` with two numerical guards.

    Vanilla pycpd diverges on *well-matched* clouds: as the match improves σ²
    anneals toward zero, the ``α·σ²·I`` term stops regularizing, and solving
    against the near-singular Gaussian kernel matrix produces wild ``W``
    (measured: 0.8 normalized-units of phantom displacement on two *identical*
    1.6k-point clouds — the better the data, the worse the failure). Guards:

    1. **σ² floor** — annealing stops at a scale already below the measurement
       noise floor; finer "precision" would be fitting nothing real.
    2. **min-norm solve** — ``lstsq`` instead of ``solve``, so near-singular
       directions of the kernel get zero weight instead of exploding.
    """
    from pycpd import DeformableRegistration

    class _StableDeformable(DeformableRegistration):
        def update_transform(self):
            A = np.dot(np.diag(self.P1), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
            self.W = np.linalg.lstsq(A, B, rcond=1e-9)[0]

        def update_variance(self):
            super().update_variance()
            if self.sigma2 < config.CPD_SIGMA2_FLOOR:
                self.sigma2 = config.CPD_SIGMA2_FLOOR
                self.diff = 0.0  # at the floor: converged, stop annealing

    return _StableDeformable(**kwargs)


def _mutual_trim(
    moving: np.ndarray, target: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop points of each cloud with no counterpart within ``radius`` in the other.

    After the global (similarity) registration, content present on only one side —
    shoulders the student drew, background texture only the photo has — sits far
    from everything in the other cloud. Removing it symmetrically keeps the
    deformable stage focused on structure the two drawings share, instead of letting
    unmatched clutter drag the deformation field (the same robustness instinct as
    the trimmed Procrustes, principle #3).
    """
    from scipy.spatial import cKDTree

    d_m = cKDTree(target).query(moving, k=1)[0]
    d_t = cKDTree(moving).query(target, k=1)[0]
    return moving[d_m <= radius], target[d_t <= radius]


def _carry_through_displacements(
    points: np.ndarray,
    cloud_before: np.ndarray,
    cloud_after: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Move ``points`` by the Gaussian-weighted average of nearby cloud displacements.

    The principled-looking alternative — evaluating the fitted CPD kernel
    ``G(points, Y) @ W`` at the landmarks — is numerically unusable: when the clouds
    match well CPD's σ² collapses, its ``α·σ²`` regularization vanishes, and ``W``
    explodes against the ill-conditioned Gaussian kernel matrix (observed |W| ≈ 500
    on *identical* clouds, i.e. 20%-of-head-height landmark error for a perfect
    registration). Interpolating the *registered points' own displacements* is
    bounded by construction (a convex combination of real displacements), smooth,
    and deterministic.
    """
    disp = cloud_after - cloud_before
    d2 = np.sum((points[:, None, :] - cloud_before[None, :, :]) ** 2, axis=2)
    weights = np.exp(-d2 / (2.0 * beta**2))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return points + weights @ disp


def _local_refine(
    landmarks: np.ndarray,
    ref_cloud: np.ndarray,
    sketch_cloud: np.ndarray,
) -> np.ndarray:
    """Per-feature similarity re-registration around each semantic group.

    The global stages park every group near its strokes but under-recover
    feature-level errors: on a dense edge cloud a displaced feature's points are
    pulled by its unmoved neighbors (an eye drawn 10% too high sits closer to the
    brow's ink than to its own), so a single global field splits the difference.
    Locally the ambiguity disappears — the window around one group contains mostly
    that feature's strokes on both sides.

    Each group gets one **similarity-mode** local CPD (rotation/scale/translation):
    that measures exactly what the landmark layer reads — where the feature went,
    how big it is, how it is tilted — and by construction cannot invent
    within-feature shape (local contour form is M3's problem and stays untouched).

    All inputs are in the normalized sketch frame; ``ref_cloud`` must already be
    carried through the global stages.
    """
    from pycpd import RigidRegistration
    from scipy.spatial import cKDTree

    refined = landmarks.copy()

    for name, idx in _LOCAL_GROUPS.items():
        group = landmarks[list(idx)]
        # Window = ink within reach of the group's landmarks (distance to the
        # landmark *set*, so the window hugs elongated features like the jaw).
        lm_tree = cKDTree(group)
        d_ref = lm_tree.query(ref_cloud, k=1)[0]
        local_ref = ref_cloud[d_ref <= config.CPD_LOCAL_WINDOW_MARGIN]
        d_sk = lm_tree.query(sketch_cloud, k=1)[0]
        slack = config.CPD_LOCAL_WINDOW_MARGIN + config.CPD_LOCAL_SEARCH_SLACK
        local_sk = sketch_cloud[d_sk <= slack]
        if len(local_ref) < config.CPD_LOCAL_MIN_POINTS or len(local_sk) < config.CPD_LOCAL_MIN_POINTS:
            continue

        reg = RigidRegistration(
            X=local_sk,
            Y=local_ref,
            w=config.CPD_LOCAL_W,
            max_iterations=config.CPD_LOCAL_MAX_ITER,
            tolerance=config.CPD_TOLERANCE,
        )
        reg.register()
        moved = reg.transform_point_cloud(Y=group)

        if name in _ARC_GROUPS:
            # Aperture problem (see _ARC_GROUPS): keep only the normal-direction
            # translation the local fit found.
            delta = moved.mean(axis=0) - group.mean(axis=0)
            centered = group - group.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            normal = vt[-1] / np.linalg.norm(vt[-1])
            moved = group + normal * float(delta @ normal)

        # A fit that teleports the group found the wrong structure — keep the
        # global estimate rather than trust a degenerate window (principle #3).
        shift = float(np.linalg.norm(moved.mean(axis=0) - group.mean(axis=0)))
        if shift <= config.CPD_LOCAL_MAX_SHIFT:
            refined[list(idx)] = moved

    return refined


def cpd_transfer_landmarks(
    reference_ink: np.ndarray,
    reference_landmarks: np.ndarray,
    sketch_ink: np.ndarray,
) -> np.ndarray:
    """Transfer the reference's 68 landmarks onto the sketch via CPD registration.

    Args:
        reference_ink: ``(M, 2)`` reference edge/ink points (e.g. XDoG of the photo,
            cropped to the head), pixel coords.
        reference_landmarks: ``(68, 2)`` landmarks detected on the reference photo,
            same coordinate system as ``reference_ink``.
        sketch_ink: ``(N, 2)`` sketch stroke points, sketch pixel coords.

    Returns:
        ``(68, 2)`` estimated landmark positions in sketch pixel coordinates.
    """
    from pycpd import RigidRegistration

    ref = np.asarray(reference_ink, dtype=np.float64)
    lms = np.asarray(reference_landmarks, dtype=np.float64)
    sk = np.asarray(sketch_ink, dtype=np.float64)
    if len(ref) < 20 or len(sk) < 20:
        raise ValueError(
            f"too few ink points to register (reference {len(ref)}, sketch {len(sk)})"
        )

    ref_n, _, _ = _normalize(ref)
    sk_n, sk_mu, sk_scale = _normalize(sk)
    # Landmarks ride along in the reference cloud's normalized coordinates.
    lm_n = (lms - ref.mean(axis=0)) / max(
        float(np.sqrt(np.mean(np.sum((ref - ref.mean(axis=0)) ** 2, axis=1)))), 1e-9
    )

    # Stage 1 — similarity-mode CPD: global rotation/scale/translation between the
    # clouds (a tilted or differently-sized drawing must not be mistaken for shape).
    rigid = RigidRegistration(
        X=sk_n,
        Y=ref_n,
        w=config.CPD_RIGID_W,
        max_iterations=config.CPD_RIGID_MAX_ITER,
        tolerance=config.CPD_TOLERANCE,
    )
    rigid.register()
    ref_r = rigid.transform_point_cloud(Y=ref_n)
    lm_r = rigid.transform_point_cloud(Y=lm_n)

    # Inter-stage trim: shed content that exists on only one side (see _mutual_trim)
    # so the deformable stage registers shared structure, not clutter.
    ref_t, sk_t = _mutual_trim(ref_r, sk_n, config.CPD_TRIM_RADIUS)
    if len(ref_t) < 20 or len(sk_t) < 20:  # degenerate overlap — keep everything
        ref_t, sk_t = ref_r, sk_n

    # Stage 2 — deformable CPD: the residual non-rigid shape difference (the
    # student's drawing errors), regularized so the field is smooth. Runs on a
    # strided subset (config.CPD_DEFORM_MAX_POINTS): it solves an O(M³) system per
    # EM iteration and only has to produce a smooth carry field — feature-level
    # precision is stage 3's job on the dense clouds.
    ref_s = ref_t[:: max(len(ref_t) // config.CPD_DEFORM_MAX_POINTS, 1)]
    sk_s = sk_t[:: max(len(sk_t) // config.CPD_DEFORM_MAX_POINTS, 1)]
    deform = _make_stable_deformable(
        X=sk_s,
        Y=ref_s,
        alpha=config.CPD_DEFORM_ALPHA,
        beta=config.CPD_DEFORM_BETA,
        w=config.CPD_DEFORM_W,
        sigma2=config.CPD_SIGMA2_INIT,
        max_iterations=config.CPD_DEFORM_MAX_ITER,
        tolerance=config.CPD_TOLERANCE,
    )
    deform.register()
    deform.transform_point_cloud()
    lm_d = _carry_through_displacements(lm_r, ref_s, deform.TY, config.CPD_CARRY_BETA)

    # Stage 3 — per-feature local similarity refinement (see _local_refine). The
    # dense reference cloud rides the same global field so both windows live in the
    # registered frame.
    ref_d = _carry_through_displacements(ref_r, ref_s, deform.TY, config.CPD_CARRY_BETA)
    lm_final = _local_refine(lm_d, ref_d, sk_n)

    return lm_final * sk_scale + sk_mu
