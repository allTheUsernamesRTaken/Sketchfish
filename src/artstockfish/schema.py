"""Core data schema — the frozen contract (spec §6).

Everything downstream of measurement consumes ``Finding`` objects. These
dataclasses are defined EXACTLY as in spec §6 and are frozen: after Wave 0 this
module is a read-only synchronization barrier. If it seems wrong, stop and
report — do not edit it while other agents depend on it (AGENTS.md, Ground
Rule 3).

Note on hashing: ``Finding`` and ``Report`` carry mutable ``dict``/``tuple``
fields (``evidence``, ``transform``, ``pose``), so although the dataclasses are
``frozen=True`` they are not reliably hashable. Treat them as immutable value
records, not as dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


@dataclass(frozen=True)
class Landmarks:
    points: np.ndarray          # (N, 2) float64, image coords
    names: tuple[str, ...]      # semantic names, e.g. "left_eye_outer"
    image_size: tuple[int, int]


class Severity(Enum):
    OK = "ok"                   # below noise floor — never shown
    INACCURACY = "inaccuracy"   # !?  small but real
    MISTAKE = "mistake"         # ?   clearly visible
    BLUNDER = "blunder"         # ??  structural


class Level(IntEnum):           # coarse-to-fine ranking tiers
    GLOBAL = 0                  # pose, tilt, overall proportion
    PLACEMENT = 1               # feature position/size
    SHAPE = 2                   # local contour form


@dataclass(frozen=True)
class Finding:
    id: str                     # stable, e.g. "left_eye_vertical"
    level: Level
    severity: Severity
    feature: str                # "left eye"
    axis: str                   # "vertical" | "horizontal" | "angle" | "area" | ...
    direction: str              # "too high" | "tilted clockwise" | "too narrow" | ...
    magnitude: float            # normalized (fraction of head height, or degrees)
    units: str                  # "%head_height" | "deg" | "%area"
    score: float                # weight * normalized magnitude (for ranking)
    evidence: dict              # raw geometry: points, vectors, segment indices
    #                             → consumed by annotate.py, never by critique text


@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]   # sorted: Level asc, then score desc
    accuracy_score: float           # 0–100 aggregate ("eval bar")
    transform: dict                 # the fitted similarity transform params
    pose: dict | None               # per-image pose estimates (M1.5+)
