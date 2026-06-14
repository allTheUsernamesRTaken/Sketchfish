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

import re
from collections.abc import Callable, Iterable

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


# ---------------------------------------------------------------------------
# Optional LLM verbalizer (spec §8 M5) with a hard, code-enforced guard.
#
# An LLM may rewrite the template sentences into warmer teacher voice, but it is a
# *paraphraser only* (spec §2 principle #1): it never measures, adds findings, or
# changes magnitudes. The LLM receives ONLY the findings (the Report JSON, no
# images). Every rewritten sentence is then validated against its own finding and,
# on ANY violation, the deterministic template is used instead — so a hallucination
# can never reach the user. The guarantee is enforced here in code, not trusted to
# the model.
#
# What the guard enforces, per sentence, against its finding:
#   • numbers     — no number the finding's magnitude doesn't support.
#   • features    — no anatomical region the finding isn't about.
#   • direction   — no error *kind* (vertical / size / width / extent / rotation)
#                   outside the finding's own axis. (The corrective phrasing names
#                   the opposite pole of the same axis — "too high → lower it" — so
#                   within-axis poles are both legitimate and not gated; left↔right
#                   side words are intentionally uncontrolled for the same reason.)
# The magnitude + feature + axis checks are what make the paraphrase faithful;
# anything the guard can't certify falls back to the template. ---------------------

# Anatomical region words → canonical feature token. A token here in a verbalized
# sentence must be covered by its finding's feature. "head"/"face" are deliberately
# omitted (they appear in many non-feature phrasings, e.g. "% of head height").
_FACE_CANON: dict[str, str] = {
    "jaw": "jaw", "jawline": "jaw",
    "brow": "brow", "brows": "brow", "eyebrow": "brow", "eyebrows": "brow",
    "nose": "nose", "nostril": "nose", "nostrils": "nose",
    "eye": "eye", "eyes": "eye",
    "mouth": "mouth", "lip": "mouth", "lips": "mouth",
    "chin": "chin",
    "ear": "ear", "ears": "ear",
    "cheek": "cheek", "cheeks": "cheek",
    "forehead": "forehead", "hairline": "hairline",
    "temple": "temple", "temples": "temple",
    "midface": "midface",
}

# Direction words → the error-axis family they belong to. A finding constrains the
# candidate to its own axis; both poles of that axis are allowed (the fix names the
# opposite pole). Left/right are omitted — corrective phrasing uses the other side.
_DIRECTION_FAMILY: dict[str, str] = {}
for _fam, _words in {
    "vertical": (
        "high higher highest low lower lowest up upward upwards down downward "
        "downwards raise raised lift lifted drop dropped sink sunk sunken"
    ),
    "size": (
        "large larger largest big bigger biggest small smaller smallest enlarge "
        "enlarged shrink shrunk shrunken oversized undersized"
    ),
    "width": "wide wider widen widened narrow narrower narrowed",
    "extent": (
        "long longer lengthen lengthened short shorter shorten shortened "
        "tall taller tallest"
    ),
    "rotation": "clockwise counterclockwise anticlockwise",
}.items():
    for _w in _words.split():
        _DIRECTION_FAMILY.setdefault(_w, set()).add(_fam)  # type: ignore[attr-defined]
# Freeze the family sets.
_DIRECTION_FAMILY = {w: frozenset(f) for w, f in _DIRECTION_FAMILY.items()}  # type: ignore[assignment]

_WORD_RE = re.compile(r"[a-z]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _finding_face_tokens(finding: Finding) -> set[str]:
    """Canonical feature tokens this finding is allowed to mention."""
    return {
        _FACE_CANON[w]
        for w in _WORD_RE.findall(finding.feature.lower())
        if w in _FACE_CANON
    }


def _finding_direction_families(finding: Finding) -> frozenset[str]:
    """Error-axis families this finding's direction belongs to."""
    fams: set[str] = set()
    for w in _WORD_RE.findall(finding.direction.lower()):
        fams |= _DIRECTION_FAMILY.get(w, frozenset())
    return frozenset(fams)


def verbalizer_violation(sentence: str, finding: Finding) -> str | None:
    """Return a reason string if ``sentence`` is not a faithful paraphrase, else None.

    A faithful paraphrase mentions no anatomical feature, no error axis, and no
    number that the finding doesn't support (spec §2 principle #1). This is the hard
    rule; callers fall back to the deterministic template on any non-None result.
    """
    if not sentence or not sentence.strip():
        return "empty sentence"

    text = sentence.lower()
    allowed_faces = _finding_face_tokens(finding)
    allowed_families = _finding_direction_families(finding)

    for word in _WORD_RE.findall(text):
        canon = _FACE_CANON.get(word)
        if canon is not None and canon not in allowed_faces:
            return f"mentions feature {word!r} absent from finding {finding.id!r}"
        fams = _DIRECTION_FAMILY.get(word)
        if fams is not None and fams.isdisjoint(allowed_families):
            return f"mentions direction {word!r} outside finding {finding.id!r}'s axis"

    magnitude = float(finding.magnitude)
    for token in _NUMBER_RE.findall(text):
        value = float(token)
        if abs(value - magnitude) > 1.0 and round(value) != round(magnitude):
            return f"mentions number {token!r} unsupported by magnitude {magnitude:.2f}"
    return None


def report_verbalizer_payload(report: Report, sentences: Iterable[str]) -> list[dict]:
    """The findings-only JSON the verbalizer LLM is allowed to see (no images).

    One entry per finding, in the report's ranked order, carrying just the fields a
    paraphraser needs plus the template sentence to rewrite.
    """
    return [
        {
            "id": f.id,
            "feature": f.feature,
            "axis": f.axis,
            "direction": f.direction,
            "magnitude": round(float(f.magnitude)),
            "units": f.units,
            "level": f.level.name,
            "template_sentence": s,
        }
        for f, s in zip(report.findings, sentences)
    ]


def verbalize_report(
    report: Report,
    sentences: tuple[str, ...] | None = None,
    *,
    llm: Callable[[list[dict]], list[str]],
) -> tuple[str, ...]:
    """Rewrite the template sentences in teacher voice via ``llm``, guarded.

    ``llm`` is given only :func:`report_verbalizer_payload` (the findings; never an
    image) and must return one rewritten sentence per finding in order. Each result
    is validated by :func:`verbalizer_violation`; any that fails — or the whole batch
    if the LLM errors or returns the wrong count — falls back to the deterministic
    template (spec §2 principle #1). The output is always parallel to
    ``report.findings``.
    """
    if sentences is None:
        sentences = critique_report(report)
    templates = tuple(sentences)

    try:
        rewritten = list(llm(report_verbalizer_payload(report, templates)))
    except Exception:
        return templates  # LLM unavailable / errored → templates, never a failure

    if len(rewritten) != len(templates):
        return templates  # shape mismatch → don't trust any of it

    out: list[str] = []
    for finding, template, candidate in zip(report.findings, templates, rewritten):
        if isinstance(candidate, str) and verbalizer_violation(candidate, finding) is None:
            out.append(candidate.strip())
        else:
            out.append(template)
    return tuple(out)


def make_anthropic_verbalizer(
    model: str = "claude-opus-4-8",
    *,
    client=None,
) -> Callable[[list[dict]], list[str]]:
    """Build a real LLM verbalizer backed by the Anthropic SDK (paraphrase only).

    The returned callable takes the findings payload (no images) and returns one
    teacher-voiced sentence per finding via structured output. It is still validated
    by :func:`verbalize_report`'s guard — the model is never trusted, only checked
    (principle #1). The ``anthropic`` SDK is imported lazily so this module's core
    template path never requires it.
    """
    import json

    system = (
        "You are a warm, encouraging atelier drawing teacher. You are given a JSON "
        "list of geometric critique findings, each already measured, with a plain "
        "template sentence. Rewrite each into one natural, encouraging, actionable "
        "sentence in a teacher's voice.\n"
        "HARD CONSTRAINTS (the geometry is already decided; you only rephrase):\n"
        "  - Mention only the feature, direction, and number present in that finding.\n"
        "  - Never introduce a feature, measurement, or claim not in the finding.\n"
        "  - Keep the finding's number if you cite one; never invent or change a number.\n"
        "  - One sentence per finding, same order, one actionable instruction each."
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentences": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["sentences"],
    }

    def _verbalize(payload: list[dict]) -> list[str]:
        nonlocal client
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        user = (
            "Rewrite each finding's template_sentence in teacher voice. Return JSON "
            '{"sentences": [...]} with exactly one sentence per finding, in order.\n\n'
            + json.dumps(payload, indent=2)
        )
        with client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        text = next((b.text for b in message.content if b.type == "text"), "{}")
        return list(json.loads(text).get("sentences", []))

    return _verbalize
