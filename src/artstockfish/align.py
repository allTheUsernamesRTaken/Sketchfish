"""Robust similarity Procrustes alignment (spec §9.1).

The alignment transform class *defines critique semantics* (design principle #2):
we fit a **similarity transform only** — translation, rotation, uniform scale.
Anything the transform absorbs is something the critique becomes blind to, so we
never fit affine/projective/non-rigid warps here. The residual left after a
similarity fit is, by definition, the drawing error.

Alignment must also be **robust** (principle #3): one huge drawing error must not
drag the fit and smear blame across correct features. ``robust_align`` re-fits
while down-weighting the worst residuals (IRLS / trimmed Procrustes).

All functions are pure (no hidden state); inputs are plain ``numpy`` arrays.
"""

from __future__ import annotations

import numpy as np

from .config import ROBUST_ITERS, ROBUST_TRIM


def similarity_procrustes(
    A: np.ndarray, B: np.ndarray, w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted similarity transform mapping ``B`` → ``A``.

    Args:
        A: target points, shape ``(N, 2)``.
        B: source points, shape ``(N, 2)``.
        w: per-point weights, shape ``(N,)``, non-negative.

    Returns:
        ``(s, R, t)`` — scalar scale ``s``, ``(2, 2)`` rotation ``R``, ``(2,)``
        translation ``t`` such that ``s * (R @ B.T).T + t`` best matches ``A`` in
        the weighted least-squares sense. ``R`` is a proper rotation (``det = +1``);
        reflections are disallowed so a similarity stays orientation-preserving.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    wa, wb = (w[:, None] * A), (w[:, None] * B)
    muA, muB = wa.sum(0) / w.sum(), wb.sum(0) / w.sum()
    A0, B0 = A - muA, B - muB
    H = (w[:, None] * B0).T @ A0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    s = (S * np.array([1.0, d])).sum() / (w * (B0 ** 2).sum(1)).sum()
    t = muA - s * (R @ muB)
    return float(s), R, t


def robust_align(
    A: np.ndarray,
    B: np.ndarray,
    iters: int = ROBUST_ITERS,
    trim: float = ROBUST_TRIM,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Trimmed/IRLS similarity fit mapping ``B`` → ``A`` (spec §9.1).

    Re-fits ``iters`` times, each pass down-weighting the worst ``trim`` fraction
    of residuals so large drawing errors don't drag the alignment toward
    themselves (design principle #3). Points inside the cutoff keep weight 1;
    points beyond it get weight ``cutoff / residual`` (a soft Huber-like falloff).

    Args:
        A: target points, shape ``(N, 2)``.
        B: source points, shape ``(N, 2)``.
        iters: number of re-weighting passes.
        trim: fraction of points treated as outliers each pass (0–1).

    Returns:
        ``(s, R, t)`` as in :func:`similarity_procrustes`.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    w = np.ones(len(A))
    s, R, t = similarity_procrustes(A, B, w)
    for _ in range(iters):
        s, R, t = similarity_procrustes(A, B, w)
        r = np.linalg.norm(A - (s * (R @ B.T).T + t), axis=1)
        cutoff = np.quantile(r, 1 - trim)
        w = np.where(r <= cutoff, 1.0, cutoff / np.maximum(r, 1e-9))
    return s, R, t


def apply_similarity(
    s: float, R: np.ndarray, t: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Apply a similarity transform to points: ``s * (R @ B.T).T + t``.

    Args:
        s: scale, ``R``: ``(2, 2)`` rotation, ``t``: ``(2,)`` translation.
        B: points to transform, shape ``(N, 2)``.

    Returns:
        Transformed points, shape ``(N, 2)``.
    """
    B = np.asarray(B, dtype=np.float64)
    return s * (R @ B.T).T + t


def rotation_angle_deg(R: np.ndarray) -> float:
    """Signed rotation angle of a ``(2, 2)`` rotation matrix, in degrees.

    Positive is counter-clockwise in standard math axes. Useful for inspecting
    how much page tilt the similarity transform absorbed (e.g. M0-T4).
    """
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
