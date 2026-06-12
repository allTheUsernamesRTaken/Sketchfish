"""Template critique sentences (spec §11).

One template per finding *axis* (with per-id action verbs where the axis alone is
ambiguous), filled **only** from the :class:`~artstockfish.schema.Finding` — never from
the images or any learned model (design principle #1: an LLM may later *paraphrase* a
sentence but the geometry, feature, direction and number all come from the finding).

Tone rules (spec §11): direct, specific, exactly one actionable instruction per
sentence, no hedging words ("maybe", "slightly off") — the geometry earns the
confidence. When a finding is a MISTAKE or BLUNDER we append the coarse-to-fine nudge
"Fix this before refining details." (spec §11, the ``left_eye_vertical`` example).

Magnitudes are rendered with ``{:.0f}`` and the axis's unit symbol (``%`` for the
head-height / area / ratio percentages, ``°`` for angles), matching the §11 examples
("5%", "9°"). All functions are pure.
"""

from __future__ import annotations

from collections.abc import Iterable

from .schema import Finding, Report, Severity

# Severities at or above which we append the coarse-to-fine nudge (spec §11).
_FIX_FIRST_SEVERITIES = frozenset({Severity.MISTAKE, Severity.BLUNDER})
_FIX_FIRST_SUFFIX = " Fix this before refining details."

# Unit symbol per schema ``units`` string (spec §6). Everything percent-like prints
# "%"; angles print "°".
_UNIT_SYMBOL = {
    "%head_height": "%",
    "%area": "%",
    "%ratio": "%",
    "deg": "°",
}

# Action clauses keyed by the direction string the measurement module emits. Each is
# the single actionable instruction the sentence ends on.
_PLACEMENT_ACTIONS = {
    # vertical (landmarks.py)
    "too high": "bring it down to meet the reference",
    "too low": "bring it up to meet the reference",
    # horizontal (landmarks.py): direction names where the feature *is*, so the
    # instruction moves it the other way.
    "too far right": "slide it left to meet the reference",
    "too far left": "slide it right to meet the reference",
    # scale / area (landmarks.py)
    "too large": "draw it smaller to match the reference",
    "too small": "draw it larger to match the reference",
}

# Per-id action clauses for the canon-ratio proportions (axis="proportion"); the same
# direction word ("too wide") means different fixes for different ratios.
_PROPORTION_ACTIONS = {
    ("eye_line_height", "too high"): "lower the eye line on the head",
    ("eye_line_height", "too low"): "raise the eye line on the head",
    ("face_thirds", "too tall"): "shorten the midface",
    ("face_thirds", "too short"): "lengthen the midface",
    ("interocular_eye_width", "too wide"): "bring the eyes closer together",
    ("interocular_eye_width", "too narrow"): "spread the eyes farther apart",
    ("nose_length", "too long"): "shorten the nose",
    ("nose_length", "too short"): "lengthen the nose",
    ("mouth_interocular", "too wide"): "narrow the mouth",
    ("mouth_interocular", "too narrow"): "widen the mouth",
}


def _magnitude_str(finding: Finding) -> str:
    """Render the finding magnitude as ``"5%"`` / ``"9°"`` (spec §11 formatting)."""
    return f"{finding.magnitude:.0f}{_UNIT_SYMBOL.get(finding.units, '')}"


def _core_sentence(f: Finding) -> str:
    """The instruction sentence for a finding, before the coarse-to-fine suffix."""
    mag = _magnitude_str(f)

    if f.axis == "vertical":
        action = _PLACEMENT_ACTIONS.get(f.direction, "move it to meet the reference")
        return f"The {f.feature} sits {mag} of head height {f.direction} — {action}."

    if f.axis == "horizontal":
        action = _PLACEMENT_ACTIONS.get(f.direction, "move it to meet the reference")
        return f"The {f.feature} is {mag} of head height {f.direction} — {action}."

    if f.axis == "area":
        action = _PLACEMENT_ACTIONS.get(f.direction, "resize it to match the reference")
        return f"The {f.feature} is drawn {mag} {f.direction} — {action}."

    if f.axis == "angle":
        # direction is "tilted clockwise" / "tilted counterclockwise"; keep the
        # orientation word, give a level-it instruction.
        orientation = f.direction.replace("tilted ", "")
        return (
            f"The {f.feature} is tilted {mag} {orientation} relative to the "
            f"reference — rotate it back to level."
        )

    if f.axis == "proportion":
        action = _PROPORTION_ACTIONS.get(
            (f.id, f.direction), "adjust it to match the reference"
        )
        return (
            f"The {f.feature} is {mag} {f.direction} relative to the reference "
            f"— {action}."
        )

    # Defensive fallback for any future axis: still grounded only in the finding.
    return (
        f"The {f.feature} is {f.direction} by {mag} relative to the reference."
    )


def critique_finding(f: Finding) -> str:
    """One teacher-voiced sentence for a single finding (spec §11).

    The sentence is built entirely from the finding's own fields; severities of
    MISTAKE or above append the coarse-to-fine nudge.
    """
    sentence = _core_sentence(f)
    if f.severity in _FIX_FIRST_SEVERITIES:
        sentence += _FIX_FIRST_SUFFIX
    return sentence


def critique_findings(findings: Iterable[Finding]) -> tuple[str, ...]:
    """Critique sentences for an ordered iterable of findings, order preserved."""
    return tuple(critique_finding(f) for f in findings)


def critique_report(report: Report) -> tuple[str, ...]:
    """Critique sentences for a report, parallel to ``report.findings`` (ranked)."""
    return critique_findings(report.findings)
