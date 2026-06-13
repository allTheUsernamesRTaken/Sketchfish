"""Photo → sketch conversion and eval-pair image warping (spec §8 M2, §10).

Two jobs, both classical image processing (no learning — principle #1):

1. **XDoG sketchify**: turn a photo into a clean line-drawing-styled image. This is
   the z-scored difference-of-Gaussians recipe the de-risk sidecar validated by eye
   on all five test photos (``data/detection_report.md``: "the z-scored-DoG recipe in
   ``_scripts/run_facemesh.py::xdog`` works well"). Plain DoG is ~0 in flat regions
   regardless of brightness, so soft-thresholding its z-score makes line density
   robust across exposures.

2. **Landmark-driven image warp**: move a photo's pixels so that its landmarks land
   on given target positions (thin-plate-spline interpolation of the landmark
   displacements). Combined with :mod:`artstockfish.synth.distort` this turns any
   photo with detected landmarks into a *sketchified, distorted* eval image whose
   ground-truth findings are known — the M2-T1 pair generator (spec §10:
   "annotations carry over for free").

Both are deterministic given their inputs; the only randomness is the explicitly
seeded parameter sampling for M2-T2's "two random sketchifications".
"""

from __future__ import annotations

import cv2
import numpy as np

# XDoG defaults — the de-risk sidecar's tuned variant "A" (data/detection_report.md,
# Method): sigma/k define the two Gaussians, phi the soft-threshold steepness, z_thr
# the z-score at which a pixel starts reading as line.
XDOG_SIGMA = 1.2
XDOG_K = 1.6
XDOG_PHI = 1.4
XDOG_Z_THR = -0.9

# M2-T2 "random sketchification" parameter ranges: wide enough that two draws give
# visibly different stroke weight/density (real re-sketching variance), narrow enough
# that every draw still reads as clean line art (eyeballed across the data/ photos).
XDOG_SIGMA_RANGE = (1.0, 1.5)
XDOG_PHI_RANGE = (1.2, 1.6)
XDOG_Z_THR_RANGE = (-1.1, -0.7)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Accept gray/BGR/BGRA uint8 → float32 gray in [0, 1]."""
    img = np.asarray(image)
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img.astype(np.float32) / 255.0


def xdog(
    image: np.ndarray,
    *,
    sigma: float = XDOG_SIGMA,
    k: float = XDOG_K,
    phi: float = XDOG_PHI,
    z_thr: float = XDOG_Z_THR,
) -> np.ndarray:
    """eXtended Difference-of-Gaussians: photo → line-drawing image.

    Args:
        image: uint8 gray or BGR(A) photo.
        sigma: inner Gaussian sigma; ``sigma * k`` is the outer one.
        k: outer/inner sigma ratio.
        phi: soft-threshold steepness.
        z_thr: z-score threshold; lower values produce denser/darker lines.

    Returns:
        uint8 grayscale image, white paper with dark lines (same size as input).
    """
    g = _to_gray(image)
    g1 = cv2.GaussianBlur(g, (0, 0), sigma)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma * k)
    dog = g1 - g2
    z = (dog - dog.mean()) / (dog.std() + 1e-6)
    out = np.clip(1.0 + np.tanh(phi * (z - z_thr)), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def random_xdog_params(rng: np.random.Generator) -> dict[str, float]:
    """Sample one random-but-plausible XDoG parameterization (M2-T2 jitter source)."""
    return {
        "sigma": float(rng.uniform(*XDOG_SIGMA_RANGE)),
        "phi": float(rng.uniform(*XDOG_PHI_RANGE)),
        "z_thr": float(rng.uniform(*XDOG_Z_THR_RANGE)),
    }


def warp_image_to_landmarks(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """Warp ``image`` so pixels at ``src_points`` move to ``dst_points``.

    Thin-plate-spline interpolation of the control-point displacements, evaluated as
    a *backward* map (for each output pixel, where in the input to sample) so the
    result has no holes. The image border (corners + edge midpoints) is pinned so
    the page itself stays put and only the face geometry moves — matching how a
    student mis-draws a feature on an otherwise fine page.

    Args:
        image: uint8 gray or BGR image.
        src_points: ``(N, 2)`` original landmark positions, image (x, y) pixels.
        dst_points: ``(N, 2)`` where those landmarks should land.

    Returns:
        The warped image, same dtype/shape as the input.
    """
    from scipy.interpolate import RBFInterpolator  # lazy: scipy only needed here

    img = np.asarray(image)
    h, w = img.shape[:2]
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"control points must be matching (N, 2); got {src.shape} vs {dst.shape}")

    # Pin the page: corners + edge midpoints get zero displacement.
    anchors = np.array(
        [
            [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
            [w / 2, 0], [w / 2, h - 1], [0, h / 2], [w - 1, h / 2],
        ],
        dtype=np.float64,
    )
    # Backward map: interpolate (dst → src) displacement at every output pixel.
    control_dst = np.vstack([dst, anchors])
    control_disp = np.vstack([src - dst, np.zeros_like(anchors)])
    interp = RBFInterpolator(control_dst, control_disp, kernel="thin_plate_spline")

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)
    disp = np.empty_like(pixels)
    chunk = 65536  # bound the (pixels × controls) kernel matrix memory
    for start in range(0, len(pixels), chunk):
        disp[start : start + chunk] = interp(pixels[start : start + chunk])
    sample = pixels + disp

    map_x = sample[:, 0].reshape(h, w).astype(np.float32)
    map_y = sample[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
