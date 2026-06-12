"""Tunable constants — thresholds, weights, severity tiers.

This file is **append-only and section-partitioned** (AGENTS.md / Ground Rule 4).
Every module adds its constants under a ``# --- <module> ---`` header and never
edits another module's block, so parallel merges stay trivial.

Conventions for Wave 0 foundational constants:
- Linear displacements and the face-frame residuals are expressed in **percent of
  head height** (so ``2.0`` means 2 % of head height), matching the ``%head_height``
  units in the schema and the ``{magnitude:.0f}%`` critique templates (spec §11).
- Angles are in **degrees**; areas in **percent of the reference region area**.

Every magic number cites its source per spec §4.
"""

from __future__ import annotations

# --- severity ---
# Severity tiers from spec §6. Each tuple is the *upper bound* (exclusive) of a
# tier; anything at or above the last bound is a BLUNDER.
#
# Displacement (% of head height):  <2 OK | 2–4 inaccuracy | 4–8 mistake | >8 blunder
DISPLACEMENT_OK_MAX = 2.0          # spec §6: "<2% OK"
DISPLACEMENT_INACCURACY_MAX = 4.0  # spec §6: "2–4% inaccuracy"
DISPLACEMENT_MISTAKE_MAX = 8.0     # spec §6: "4–8% mistake" (>8% blunder)

# Angles (degrees):  <2 OK | 2–5 inaccuracy | 5–10 mistake | >10 blunder
ANGLE_OK_MAX = 2.0                 # spec §6: "<2° OK"
ANGLE_INACCURACY_MAX = 5.0         # spec §6: "2–5° inaccuracy"
ANGLE_MISTAKE_MAX = 10.0           # spec §6: "5–10° mistake" (>10° blunder)

# Areas (% of reference region area):  <8 OK | 8–15 | 15–30 | >30 blunder
AREA_OK_MAX = 8.0                  # spec §6: "<8% OK"
AREA_INACCURACY_MAX = 15.0         # spec §6: "8–15% inaccuracy"
AREA_MISTAKE_MAX = 30.0            # spec §6: "15–30% mistake" (>30% blunder)

# Ordered (inaccuracy_max, mistake_max, blunder_floor) bundles for downstream
# classification helpers (evaluate.py, Wave 2). Kept here so the tiers live in
# exactly one place.
DISPLACEMENT_TIERS = (
    DISPLACEMENT_OK_MAX,
    DISPLACEMENT_INACCURACY_MAX,
    DISPLACEMENT_MISTAKE_MAX,
)
ANGLE_TIERS = (ANGLE_OK_MAX, ANGLE_INACCURACY_MAX, ANGLE_MISTAKE_MAX)
AREA_TIERS = (AREA_OK_MAX, AREA_INACCURACY_MAX, AREA_MISTAKE_MAX)

# --- align ---
# Robust Procrustes defaults (spec §9.1). ``robust_align`` down-weights the worst
# ``ROBUST_TRIM`` fraction of residuals so one big drawing error can't drag the
# alignment (design principle #3). Mirrored as the function's keyword defaults.
ROBUST_ITERS = 5                   # spec §9.1: robust_align(iters=5)
ROBUST_TRIM = 0.25                 # spec §9.1: robust_align(trim=0.25)

# --- evaluate ---
# Importance weights per feature (spec §9.5). Used by evaluate.py (Wave 2) as the
# ``importance_weight`` factor in ``score = weight × (magnitude / severity_unit)``.
# Tuned against human redlines in M5; these are the spec's initial values.
IMPORTANCE_WEIGHTS = {
    "left_eye": 1.0,
    "right_eye": 1.0,
    "mouth": 0.8,
    "nose": 0.7,
    "face_oval": 0.7,
    "left_brow": 0.5,
    "right_brow": 0.5,
    "ears": 0.3,
    "hairline": 0.3,
}
DEFAULT_IMPORTANCE_WEIGHT = 0.5    # spec §9.5: fallback for unlisted features

# Accuracy ("eval bar") decay: accuracy_score = 100 · exp(-ACCURACY_K · Σ scores),
# with ACCURACY_K chosen so a typical first attempt lands ~55–70 (spec §9.5).
# Placeholder until calibrated against real sketches in M5.
ACCURACY_K = 0.04                  # spec §9.5: tune so first attempts score ~55–70
