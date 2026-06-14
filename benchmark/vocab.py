"""The closed finding vocabulary, derived from the frozen contract.

Both systems are scored on the same target: the exact ``(id, direction)`` keys our
deterministic pipeline can emit. We hand that vocabulary to the VLM verbatim in its
prompt so the baseline is judged on *measurement ability*, not on guessing our
naming convention — a fluent "the left eye looks a touch high" that we couldn't map
to ``left_eye_vertical / too high`` would unfairly tank the VLM's recall.

Everything here is derived from ``config`` and ``measure`` module conventions (the
frozen §6 schema and §9 ids), never hand-duplicated, so it cannot drift from what
the pipeline actually produces.
"""

from __future__ import annotations

from dataclasses import dataclass

from artstockfish import config
from artstockfish.measure.landmarks import MEASURE_GROUPS
from artstockfish.schema import Level


@dataclass(frozen=True)
class VocabEntry:
    """One critique kind the system can report."""

    id: str
    feature: str
    axis: str
    units: str
    directions: tuple[str, ...]
    level: Level


def _feature_name(group: str) -> str:
    return config.LANDMARK_GROUP_FEATURE_NAMES.get(group, group.replace("_", " "))


def _build_vocab() -> tuple[VocabEntry, ...]:
    entries: list[VocabEntry] = []

    # Placement + scale, per measurement group (measure/landmarks.py).
    for group in MEASURE_GROUPS:
        feature = _feature_name(group)
        entries.append(VocabEntry(
            f"{group}_vertical", feature, "vertical", "%head_height",
            ("too high", "too low"), Level.PLACEMENT,
        ))
        entries.append(VocabEntry(
            f"{group}_horizontal", feature, "horizontal", "%head_height",
            ("too far right", "too far left"), Level.PLACEMENT,
        ))
        entries.append(VocabEntry(
            f"{group}_scale", feature, "area", "%area",
            ("too large", "too small"), Level.PLACEMENT,
        ))

    # Feature line angles (measure/angles.py via config.ANGLE_LINES).
    for spec in config.ANGLE_LINES.values():
        entries.append(VocabEntry(
            str(spec["id"]), str(spec["feature"]), "angle", "deg",
            ("tilted clockwise", "tilted counterclockwise"), Level(int(spec["level"])),
        ))

    # Canon proportions (measure/proportions.py via config.PROPORTION_RATIOS).
    for spec in config.PROPORTION_RATIOS.values():
        entries.append(VocabEntry(
            str(spec["id"]), str(spec["feature"]), "proportion", "%ratio",
            (str(spec["higher"]), str(spec["lower"])), Level(int(spec["level"])),
        ))

    # Head pose (measure/pose.py) — one GLOBAL finding on the dominant out-of-plane axis.
    entries.append(VocabEntry(
        "pose_yaw", "head", "pose", "deg",
        ("rotated further right", "rotated further left"), Level.GLOBAL,
    ))
    entries.append(VocabEntry(
        "pose_pitch", "head", "pose", "deg",
        ("rotated further down", "rotated further up"), Level.GLOBAL,
    ))
    return tuple(entries)


FINDING_VOCAB: tuple[VocabEntry, ...] = _build_vocab()

# All legal (id, direction) keys — used to validate VLM output before scoring.
VALID_KEYS: frozenset[tuple[str, str]] = frozenset(
    (e.id, d) for e in FINDING_VOCAB for d in e.directions
)
VALID_IDS: frozenset[str] = frozenset(e.id for e in FINDING_VOCAB)
_UNITS_BY_ID: dict[str, str] = {e.id: e.units for e in FINDING_VOCAB}


def units_for(finding_id: str) -> str | None:
    """The units string a given finding id reports in, or ``None`` if unknown."""
    return _UNITS_BY_ID.get(finding_id)


def vocab_prompt_block() -> str:
    """A human-readable enumeration of the vocabulary for the VLM prompt.

    Grouped by coarse-to-fine level so the model is nudged toward the same
    structural-first priority our ranking enforces (spec §2 principle #5).
    """
    by_level: dict[Level, list[VocabEntry]] = {}
    for e in FINDING_VOCAB:
        by_level.setdefault(e.level, []).append(e)

    lines: list[str] = []
    for level in sorted(by_level, key=int):
        lines.append(f"{level.name} findings (report these first):")
        for e in by_level[level]:
            dirs = " | ".join(e.directions)
            lines.append(
                f'  - id "{e.id}" (feature: {e.feature}) — direction one of [{dirs}]; '
                f"magnitude in {e.units}"
            )
    return "\n".join(lines)
