"""Render a 68-point landmark set as a clean line-drawing face (PNG bytes).

The VLM baseline works on *images*; our pipeline works on the coordinates directly.
To make the comparison fair the same geometry drives both: each face is drawn as the
standard iBUG 68-point wireframe (jaw arc, brows, nose, eyes, mouth) in plain black
on white, with no shading or texture cues a teacher couldn't also see.

The reference and the sketch are always rendered in the **same fixed frame** (axis
limits computed once from the reference), so a local error — an eye drawn too high —
is visibly higher, while a uniform translation/scale would not move anything relative
to the frame. That mirrors the similarity-invariance our own measurement enforces
(spec §2 principle #2): we compare the two systems on the error class both are meant
to judge.
"""

from __future__ import annotations

import io

import numpy as np

from artstockfish import config

# iBUG 68-point polyline topology. Each entry is (indices, closed?).
_CONNECTIONS: tuple[tuple[tuple[int, ...], bool], ...] = (
    (tuple(range(0, 17)), False),   # jaw
    (tuple(range(17, 22)), False),  # right brow
    (tuple(range(22, 27)), False),  # left brow
    (tuple(range(27, 31)), False),  # nose bridge
    (tuple(range(31, 36)), False),  # lower nose
    (tuple(range(36, 42)), True),   # right eye
    (tuple(range(42, 48)), True),   # left eye
    (tuple(range(48, 60)), True),   # outer mouth
    (tuple(range(60, 68)), True),   # inner mouth
)


def frame_limits(reference_points: np.ndarray) -> tuple[float, float, float, float]:
    """Square ``(xmin, xmax, ymin, ymax)`` framing the reference with margin.

    A square frame keeps aspect neutral so the VLM can't read distortion off an
    anisotropic canvas; the margin leaves room for a sketch whose features drift
    a little outside the reference's own bounding box.
    """
    pts = np.asarray(reference_points, dtype=np.float64)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0
    half *= 1.0 + 2.0 * config.BENCH_RENDER_MARGIN
    return (
        float(center[0] - half), float(center[0] + half),
        float(center[1] - half), float(center[1] + half),
    )


def render_face(
    points: np.ndarray,
    limits: tuple[float, float, float, float],
    *,
    px: int = config.BENCH_RENDER_PX,
) -> bytes:
    """Render one 68-point face to PNG bytes within the given fixed frame."""
    import matplotlib

    matplotlib.use("Agg")  # headless, deterministic raster
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=np.float64)
    xmin, xmax, ymin, ymax = limits
    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for idx, closed in _CONNECTIONS:
        seq = list(idx) + ([idx[0]] if closed else [])
        ax.plot(
            pts[seq, 0], pts[seq, 1],
            color="black", linewidth=config.BENCH_RENDER_LINEWIDTH,
            solid_capstyle="round", solid_joinstyle="round", antialiased=True,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # invert y: image coords have y pointing down
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def render_pair(reference_points: np.ndarray, sketch_points: np.ndarray) -> tuple[bytes, bytes]:
    """Render (reference, sketch) into a shared frame; returns ``(ref_png, sketch_png)``."""
    limits = frame_limits(reference_points)
    return render_face(reference_points, limits), render_face(sketch_points, limits)
