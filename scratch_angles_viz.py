"""Throwaway visual sanity check for measure/angles.py (NOT part of the module).

Renders the reference vs. the aligned sketch, draws each fitted feature line in
both, and labels the angle Findings. Run: python scratch_angles_viz.py
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from artstockfish.align import apply_similarity, robust_align
from artstockfish.config import ANGLE_LINES
from artstockfish.frame import LANDMARK_NAMES_68
from artstockfish.measure.angles import _line_direction, measure_angles
from artstockfish.schema import Landmarks, Severity
from fixtures import canonical_face_points

SEV_COLOR = {
    Severity.INACCURACY: "#e1ad01",
    Severity.MISTAKE: "#e8590c",
    Severity.BLUNDER: "#c92a2a",
}


def lm(p):
    return Landmarks(points=p, names=LANDMARK_NAMES_68, image_size=(500, 500))


def rotate(points, idx, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out = points.copy()
    sub = out[idx]
    c = sub.mean(0)
    out[idx] = (R @ (sub - c).T).T + c
    return out


def draw_fit_line(ax, pts, color, ls):
    c = pts.mean(0)
    v = _line_direction(pts)
    half = 1.15 * np.max(np.linalg.norm(pts - c, axis=1))
    a, b = c - v * half, c + v * half
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=2, zorder=3)


def main():
    ref = canonical_face_points()
    # Inject: eye line +6 deg (clockwise), right jaw -7 deg, mouth +4 deg.
    sketch = ref.copy()
    sketch = rotate(sketch, list(ANGLE_LINES["eye_line"]["indices"]), 6.0)
    sketch = rotate(sketch, list(ANGLE_LINES["right_jaw"]["indices"]), -7.0)
    sketch = rotate(sketch, list(ANGLE_LINES["mouth_line"]["indices"]), 4.0)

    findings = measure_angles(lm(ref), lm(sketch))
    by_line = {f.id: f for f in findings}

    s, R, t = robust_align(ref, sketch)
    aligned = apply_similarity(s, R, t, sketch)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(ref[:, 0], ref[:, 1], s=14, c="#1c7ed6", label="reference", zorder=2)
    ax.scatter(aligned[:, 0], aligned[:, 1], s=14, c="#868e96",
               label="sketch (aligned)", zorder=2)

    for key, spec in ANGLE_LINES.items():
        idx = list(spec["indices"])
        draw_fit_line(ax, ref[idx], "#1c7ed6", "-")
        f = by_line.get(spec["id"])
        sk_color = SEV_COLOR.get(f.severity, "#868e96") if f else "#2f9e44"
        draw_fit_line(ax, aligned[idx], sk_color, "--")
        c = aligned[idx].mean(0)
        if f:
            ax.annotate(f"{f.feature}: {f.magnitude:.1f}° {f.direction.split()[-1]}\n"
                        f"[{f.severity.value}]",
                        xy=(c[0], c[1]), xytext=(c[0] + 12, c[1] - 18),
                        fontsize=8, color=sk_color, weight="bold",
                        bbox=dict(boxstyle="round", fc="white", ec=sk_color, alpha=0.9))

    ax.set_title("angles.py sanity check — solid blue = reference fit, "
                 "dashed = aligned sketch fit\n(colored by severity)", fontsize=9)
    ax.invert_yaxis()  # image coords: y down
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = "scratch_angles_viz.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    for f in findings:
        print(f"  {f.id}: {f.magnitude:.2f} deg {f.direction} ({f.severity.value})")


if __name__ == "__main__":
    main()
