"""Annotated overlay — matplotlib debug render (spec §8 M0; SVG comes in M4).

Draws both landmark sets *after alignment* (reference vs. the registered sketch),
then overlays each surfaced :class:`~artstockfish.schema.Finding` using the raw
geometry the measurement modules stashed in ``Finding.evidence`` (never the critique
text — spec §6). Placement/scale findings become correction **arrows/circles**, angle
findings become a reference-vs-sketch **line pair**, proportion findings mark their
landmarks — all coloured by severity, with the chess-style severity **badge**
(``!?`` / ``?`` / ``??``).

matplotlib is imported lazily (Agg backend) so importing the measurement /
evaluation / critique path — and running its tests — never requires it; only
rendering does (spec §4: matplotlib is the M0/M1 overlay tool).
"""

from __future__ import annotations

import numpy as np

from .schema import Report, Severity

# Severity → colour (chess.com-ish: yellow inaccuracy, orange mistake, red blunder)
# and badge glyph (spec §6: !? / ? / ??). OK is never surfaced, so it has no entry.
_SEV_COLOR = {
    Severity.INACCURACY: "#e1ad01",
    Severity.MISTAKE: "#e8590c",
    Severity.BLUNDER: "#c92a2a",
}
_SEV_BADGE = {
    Severity.INACCURACY: "!?",
    Severity.MISTAKE: "?",
    Severity.BLUNDER: "??",
}
_REF_COLOR = "#1c7ed6"      # reference landmarks / fitted lines
_SKETCH_COLOR = "#868e96"   # aligned sketch landmarks


def _centroid(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).mean(axis=0)


def _short_direction(direction: str) -> str:
    """Trim the leading "tilted "/"too " so labels stay compact."""
    return direction.replace("tilted ", "").replace("too far ", "").replace("too ", "")


def _label(finding) -> str:
    unit = "°" if finding.units == "deg" else "%"
    badge = _SEV_BADGE.get(finding.severity, "")
    return (
        f"{finding.feature}: {finding.magnitude:.0f}{unit} "
        f"{_short_direction(finding.direction)} {badge}"
    )


def render_overlay(
    report: Report,
    reference_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
    out_path: str,
    *,
    title: str | None = None,
    dpi: int = 120,
) -> str:
    """Render the annotated overlay PNG and return its path.

    Args:
        report: the ranked report; every surfaced finding gets one annotation.
        reference_points: ``(68, 2)`` reference landmarks (image coords).
        aligned_sketch_points: ``(68, 2)`` sketch landmarks already registered to
            the reference (same frame as ``reference_points``).
        out_path: where to write the PNG.
        title: optional figure title; a default is used if omitted.
        dpi: output resolution.

    Returns:
        ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; we only ever save to file
    import matplotlib.pyplot as plt

    ref = np.asarray(reference_points, dtype=np.float64)
    sk = np.asarray(aligned_sketch_points, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(ref[:, 0], ref[:, 1], s=14, c=_REF_COLOR, label="reference", zorder=2)
    ax.scatter(
        sk[:, 0], sk[:, 1], s=14, c=_SKETCH_COLOR, label="sketch (aligned)", zorder=2
    )

    for i, finding in enumerate(report.findings):
        color = _SEV_COLOR.get(finding.severity, _SKETCH_COLOR)
        _draw_finding(ax, finding, color, i)

    ax.set_title(
        title
        or "Art Stockfish — reference vs. aligned sketch\n"
        "(arrows/lines = corrections, coloured by severity)",
        fontsize=9,
    )
    ax.invert_yaxis()  # image coordinates: y points down
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _annotate_label(ax, xy: np.ndarray, color: str, text: str, row: int) -> None:
    """Place a finding label with a small per-finding vertical offset to de-clutter."""
    ax.annotate(
        text,
        xy=(float(xy[0]), float(xy[1])),
        xytext=(float(xy[0]) + 14, float(xy[1]) - 14 - 14 * (row % 3)),
        fontsize=7,
        color=color,
        weight="bold",
        bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9),
        zorder=5,
    )


def _draw_finding(ax, finding, color: str, row: int) -> None:
    """Draw one finding's geometry from its evidence (axis-specific)."""
    ev = finding.evidence

    if finding.axis in ("vertical", "horizontal"):
        ref_g = np.asarray(ev["ref_points"], dtype=np.float64)
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        tail, head = _centroid(sk_g), _centroid(ref_g)  # drawn → correct
        ax.annotate(
            "",
            xy=(head[0], head[1]),
            xytext=(tail[0], tail[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2),
            zorder=4,
        )
        _annotate_label(ax, tail, color, _label(finding), row)
        return

    if finding.axis == "area":
        from matplotlib.patches import Circle

        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        c = _centroid(sk_g)
        r_ref = float(ev.get("spread_ref", 0.0))
        r_sk = float(ev.get("spread_sketch", 0.0))
        ax.add_patch(Circle((c[0], c[1]), r_ref, fill=False, ec=_REF_COLOR,
                            ls="-", lw=1.5, zorder=3))
        ax.add_patch(Circle((c[0], c[1]), r_sk, fill=False, ec=color,
                            ls="--", lw=1.8, zorder=3))
        _annotate_label(ax, c, color, _label(finding), row)
        return

    if finding.axis == "angle":
        ref_g = np.asarray(ev["ref_points"], dtype=np.float64)
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        _draw_line(ax, ref_g, np.asarray(ev["ref_direction"]), _REF_COLOR, "-")
        _draw_line(ax, sk_g, np.asarray(ev["sketch_direction"]), color, "--")
        _annotate_label(ax, _centroid(sk_g), color, _label(finding), row)
        return

    if finding.axis == "proportion":
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        ax.scatter(sk_g[:, 0], sk_g[:, 1], s=40, facecolors="none", edgecolors=color,
                   lw=2, zorder=4)
        _annotate_label(ax, _centroid(sk_g), color, _label(finding), row)
        return

    # Unknown axis: drop a labelled marker at whatever points the evidence carries.
    pts = ev.get("sketch_points") or ev.get("ref_points")
    if pts is not None:
        _annotate_label(ax, _centroid(np.asarray(pts, dtype=np.float64)), color,
                        _label(finding), row)


def _draw_line(ax, points: np.ndarray, direction: np.ndarray, color: str, ls: str) -> None:
    """Draw the fitted line for an angle finding through the landmark centroid."""
    c = _centroid(points)
    v = np.asarray(direction, dtype=np.float64)
    half = 1.2 * float(np.max(np.linalg.norm(points - c, axis=1)))
    a, b = c - v * half, c + v * half
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=2, zorder=3)
