"""Annotated overlay — matplotlib debug render (M0) and SVG product render (M4).

Two renderers share this module; both read raw geometry from
:attr:`~artstockfish.schema.Finding.evidence` (never the critique text — spec §6):

- :func:`render_overlay` — the M0/M1 matplotlib PNG debug overlay (spec §8 M0). Still
  used by the CLI ``demo-synthetic`` / ``critique`` paths and the detection pipeline.
- :func:`render_svg` — the **M4 product overlay** (spec §8 M4): the student's sketch as
  a base layer, per-finding displacement **arrows** (drawn→correct), a **ghost outline**
  of the corrected feature, a signed-deviation **contour heatmap**, and severity
  **badges** (``!?`` / ``?`` / ``??``) pinned to each region. Every surfaced finding
  becomes exactly **one** annotation group (``<g class="as-annotation" data-finding-id=…>``),
  so the web list can highlight a finding's annotation by id (the FastAPI app + static
  page in ``web/`` drive the interaction).

Both heavy renderers import their backend lazily — matplotlib (Agg) for the PNG,
``svgwrite`` for the SVG — so importing the measurement / evaluation / critique path,
and running its tests, never requires either; only rendering does (spec §4).
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


# =============================================================================
# M4 — SVG product overlay (spec §8 M4)
# =============================================================================
#
# Image coordinates (y points DOWN) match the SVG coordinate system, so — unlike the
# matplotlib path above — no y-inversion is needed: reference/sketch points map
# straight to SVG user space.

_GHOST_COLOR = "#2f9e44"        # corrected/target geometry reads "green = where it goes"
# Diverging heatmap endpoints for signed contour deviation (caves in ↔ bulges out).
_HEAT_NEG = (28, 126, 214)      # signed distance < 0: contour caves inward (cool/blue)
_HEAT_ZERO = (208, 208, 208)    # on the reference contour (neutral grey)
_HEAT_POS = (201, 42, 42)       # signed distance > 0: contour bulges outward (warm/red)

# Standalone interactivity: the web page toggles ``as-has-selection`` on the <svg> and
# ``as-selected`` on the clicked finding's group; everything else dims. Kept in the SVG
# so a saved overlay is self-contained (spec §8 M4 "click a finding → highlight it").
_SVG_CSS = (
    ".as-annotation{cursor:pointer}"
    "svg.as-has-selection .as-annotation{opacity:0.12}"
    "svg.as-has-selection .as-annotation.as-selected{opacity:1}"
)

# Face connectivity for the 68-point set (iBUG ordering) → polylines for the base /
# reference outlines. ``True`` = closed loop (eyes, mouth rings).
_FACE_POLYLINES: tuple[tuple[tuple[int, ...], bool], ...] = (
    (tuple(range(0, 17)), False),    # jaw
    (tuple(range(17, 22)), False),   # right brow
    (tuple(range(22, 27)), False),   # left brow
    (tuple(range(27, 31)), False),   # nose bridge
    (tuple(range(31, 36)), False),   # nose bottom
    (tuple(range(36, 42)), True),    # right eye
    (tuple(range(42, 48)), True),    # left eye
    (tuple(range(48, 60)), True),    # mouth outer
    (tuple(range(60, 68)), True),    # mouth inner
)


def _group_local_polylines(group: str, n: int) -> list[tuple[list[int], bool]]:
    """Within-group connectivity (local indices into a group's own point list).

    The measurement modules store a group's ``ref_points``/``sketch_points`` in global
    iBUG order, so the ghost outline of a feature is drawn from those points alone.
    """
    if group in ("left_eye", "right_eye"):
        return [(list(range(n)), True)]
    if group == "mouth":  # group = 48..67 → local 0..11 outer ring, 12..19 inner ring
        return [(list(range(0, 12)), True), (list(range(12, n)), True)]
    if group == "nose":  # group = bridge 27..30 + bottom 31..35 → local 0..3, 4..8
        return [(list(range(0, 4)), False), (list(range(4, n)), False)]
    return [(list(range(n)), False)]  # jaw, brows: a single open polyline


def _lerp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], f: float) -> str:
    return "rgb({},{},{})".format(*(int(round(a + (b - a) * f)) for a, b in zip(c0, c1)))


def _heat_color(t: float) -> str:
    """Diverging colour for a signed deviation ``t`` normalised to ``[-1, 1]``."""
    t = max(-1.0, min(1.0, float(t)))
    return _lerp_rgb(_HEAT_ZERO, _HEAT_POS, t) if t >= 0 else _lerp_rgb(_HEAT_ZERO, _HEAT_NEG, -t)


def _xy(point) -> tuple[float, float]:
    p = np.asarray(point, dtype=np.float64).ravel()
    return float(p[0]), float(p[1])


def _evidence_points(finding) -> np.ndarray | None:
    """Best available (N, 2) point cloud for a finding (for bounds / fallback anchor)."""
    ev = finding.evidence
    for key in ("ref_samples", "sketch_samples", "sketch_points", "ref_points",
                "reference_points", "run_ref_segment"):
        v = ev.get(key)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64).reshape(-1, 2)
            if arr.size:
                return arr
    return None


def _poly(dwg, points, closed: bool, **attrs):
    pts = [(_xy(p)) for p in np.asarray(points, dtype=np.float64).reshape(-1, 2)]
    attrs.setdefault("fill", "none")
    return (dwg.polygon if closed else dwg.polyline)(points=pts, **attrs)


def _arrow_markers(dwg, u: float) -> dict[Severity, str]:
    """One arrowhead marker per severity colour; returns severity → funciri."""
    s = 0.026 * u
    iris: dict[Severity, str] = {}
    for sev, color in _SEV_COLOR.items():
        marker = dwg.marker(
            insert=(s, s / 2.0), size=(s, s), orient="auto", markerUnits="userSpaceOnUse",
            id=f"as-arrow-{sev.value}",
        )
        marker.add(dwg.path(d=f"M0,0 L{s:.2f},{s/2:.2f} L0,{s:.2f} Z", fill=color))
        dwg.defs.add(marker)
        iris[sev] = marker.get_funciri()
    return iris


def _badge(dwg, xy, color: str, glyph: str, r: float, font: float):
    x, y = _xy(xy)
    grp = dwg.g(class_="as-badge")
    grp.add(dwg.circle(center=(x, y), r=r, fill=color, stroke="white", stroke_width=max(r * 0.16, 0.6)))
    grp.add(dwg.text(
        glyph, insert=(x, y + font * 0.34), text_anchor="middle",
        font_size=f"{font:.2f}px", font_family="sans-serif", font_weight="bold", fill="white",
    ))
    return grp


def _draw_finding_svg(dwg, group, finding, sizes: dict, arrow_iris: dict) -> tuple[float, float]:
    """Draw one finding's geometry into ``group``; return the badge anchor point."""
    ev = finding.evidence
    color = _SEV_COLOR.get(finding.severity, _SKETCH_COLOR)
    ghost_w, line_w, arrow_w = sizes["ghost"], sizes["line"], sizes["arrow"]

    # --- placement (vertical / horizontal): ghost feature + drawn→correct arrow ----
    if finding.axis in ("vertical", "horizontal"):
        ref_g = np.asarray(ev["ref_points"], dtype=np.float64)
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        for local, closed in _group_local_polylines(ev.get("group", ""), len(ref_g)):
            group.add(_poly(dwg, ref_g[local], closed, stroke=_GHOST_COLOR,
                            stroke_width=ghost_w, stroke_dasharray="4,3", opacity=0.9))
        tail, head = _centroid(sk_g), _centroid(ref_g)  # drawn → correct
        group.add(dwg.line(start=_xy(tail), end=_xy(head), stroke=color,
                           stroke_width=arrow_w, marker_end=arrow_iris[finding.severity]))
        return _xy(tail)

    # --- scale (area): ghost outline of the correctly-sized feature ----------------
    if finding.axis == "area":
        ref_g = np.asarray(ev["ref_points"], dtype=np.float64)
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        for local, closed in _group_local_polylines(ev.get("group", ""), len(ref_g)):
            group.add(_poly(dwg, ref_g[local], closed, stroke=_GHOST_COLOR,
                            stroke_width=ghost_w, stroke_dasharray="4,3", opacity=0.9))
            group.add(_poly(dwg, sk_g[local], closed, stroke=color, stroke_width=line_w))
        return _xy(_centroid(sk_g))

    # --- angle: reference (ghost) line vs. the drawn line --------------------------
    if finding.axis == "angle":
        ref_g = np.asarray(ev["ref_points"], dtype=np.float64)
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        _svg_line_through(dwg, group, ref_g, ev["ref_direction"], _GHOST_COLOR, ghost_w, "4,3")
        _svg_line_through(dwg, group, sk_g, ev["sketch_direction"], color, line_w, None)
        return _xy(_centroid(sk_g))

    # --- proportion: highlight the landmarks the ratio reads -----------------------
    if finding.axis == "proportion":
        sk_g = np.asarray(ev["sketch_points"], dtype=np.float64)
        for p in sk_g:
            group.add(dwg.circle(center=_xy(p), r=sizes["dot"], fill="none",
                                 stroke=color, stroke_width=line_w))
        return _xy(_centroid(sk_g))

    # --- contour / curvature: signed-deviation heatmap along the reference arc -----
    if finding.axis in ("contour", "curvature") and ev.get("ref_samples") is not None:
        samples = np.asarray(ev["ref_samples"], dtype=np.float64)
        signed = ev.get("signed_distance")
        scalar = (np.asarray(signed, dtype=np.float64) if signed is not None
                  else np.zeros(len(samples)))
        scale = float(np.max(np.abs(scalar))) or 1.0
        for a, b, sa in zip(samples[:-1], samples[1:], scalar[:-1]):
            group.add(dwg.line(start=_xy(a), end=_xy(b), stroke=_heat_color(sa / scale),
                               stroke_width=sizes["heat"], stroke_linecap="round"))
        run = ev.get("run_ref_segment")
        if run is not None:
            group.add(_poly(dwg, run, False, stroke=color, stroke_width=line_w, opacity=0.8))
        anchor = ev.get("anchor_a_point")
        return _xy(anchor if anchor is not None else _centroid(samples))

    # --- pose / unknown axis: a badge over the face, geometry permitting -----------
    pts = _evidence_points(finding)
    return _xy(_centroid(pts)) if pts is not None else (sizes["cx"], sizes["cy"])


def _svg_line_through(dwg, group, points, direction, color, width, dash) -> None:
    c = _centroid(points)
    v = np.asarray(direction, dtype=np.float64)
    half = 1.25 * float(np.max(np.linalg.norm(np.asarray(points) - c, axis=1)))
    a, b = c - v * half, c + v * half
    kw = {"stroke": color, "stroke_width": width}
    if dash:
        kw["stroke_dasharray"] = dash
    group.add(dwg.line(start=_xy(a), end=_xy(b), **kw))


def render_svg(
    report: Report,
    reference_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
    *,
    title: str | None = None,
) -> str:
    """Render the M4 annotated overlay as an SVG string (spec §8 M4).

    The student's sketch is the base layer; each surfaced finding becomes exactly one
    ``<g class="as-annotation" data-finding-id=…>`` carrying its correction geometry —
    a displacement arrow (drawn→correct), a ghost outline of the corrected feature, a
    signed-deviation contour heatmap, and an angle line-pair as the finding's axis
    dictates — plus a severity badge. The one-group-per-finding contract lets the web
    page highlight a finding's annotation by id.

    Args:
        report: the ranked report; every finding gets one annotation group.
        reference_points: ``(68, 2)`` reference landmarks (image coords).
        aligned_sketch_points: ``(68, 2)`` sketch landmarks registered to the reference.
        title: optional ``<title>`` for the document.

    Returns:
        The SVG markup as a string.
    """
    import svgwrite

    ref = np.asarray(reference_points, dtype=np.float64)
    sk = np.asarray(aligned_sketch_points, dtype=np.float64)

    # View box spans every point we might draw (landmarks + any evidence geometry).
    clouds = [ref, sk]
    for f in report.findings:
        pts = _evidence_points(f)
        if pts is not None:
            clouds.append(pts)
    allpts = np.vstack(clouds)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    extent = float(np.max(hi - lo)) or 1.0
    pad = 0.10 * extent
    minx, miny = lo[0] - pad, lo[1] - pad
    w, h = (hi[0] - lo[0]) + 2 * pad, (hi[1] - lo[1]) + 2 * pad
    u = max(w, h)

    sizes = {
        "base": 0.0045 * u, "ghost": 0.006 * u, "line": 0.006 * u, "arrow": 0.007 * u,
        "heat": 0.012 * u, "dot": 0.014 * u, "badge": 0.030 * u, "font": 0.050 * u,
        "cx": minx + w / 2.0, "cy": miny + h / 2.0,
    }

    dwg = svgwrite.Drawing(
        size=(f"{w:.1f}", f"{h:.1f}"),
        viewBox=f"{minx:.2f} {miny:.2f} {w:.2f} {h:.2f}",
        profile="full", debug=False,
    )
    dwg.add(dwg.style(_SVG_CSS))
    dwg.add(dwg.rect(insert=(minx, miny), size=(w, h), fill="white"))
    if title:
        dwg.set_desc(title=title)
    arrow_iris = _arrow_markers(dwg, u)

    # Base layer: the student's sketch (the drawing being corrected).
    base = dwg.g(class_="as-base", fill="none", stroke=_SKETCH_COLOR,
                 stroke_width=sizes["base"], stroke_linejoin="round")
    for idx, closed in _FACE_POLYLINES:
        base.add(_poly(dwg, sk[list(idx)], closed))
    dwg.add(base)

    # Faint reference guide so corrections read against the target face.
    guide = dwg.g(class_="as-reference", fill="none", stroke=_REF_COLOR,
                  stroke_width=sizes["base"], opacity=0.28, stroke_linejoin="round")
    for idx, closed in _FACE_POLYLINES:
        guide.add(_poly(dwg, ref[list(idx)], closed))
    dwg.add(guide)

    # One annotation group per surfaced finding (the SVG↔list contract).
    for finding in report.findings:
        color = _SEV_COLOR.get(finding.severity, _SKETCH_COLOR)
        g = dwg.g(**{"class": "as-annotation", "data-finding-id": finding.id,
                     "data-severity": finding.severity.value})
        g.set_desc(title=_label(finding))
        anchor = _draw_finding_svg(dwg, g, finding, sizes, arrow_iris)
        g.add(_badge(dwg, anchor, color, _SEV_BADGE.get(finding.severity, ""),
                     sizes["badge"], sizes["font"]))
        dwg.add(g)

    return dwg.tostring()


def save_svg(
    report: Report,
    reference_points: np.ndarray,
    aligned_sketch_points: np.ndarray,
    out_path: str,
    *,
    title: str | None = None,
) -> str:
    """Render the SVG overlay (:func:`render_svg`) and write it to ``out_path``."""
    svg = render_svg(report, reference_points, aligned_sketch_points, title=title)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return out_path
