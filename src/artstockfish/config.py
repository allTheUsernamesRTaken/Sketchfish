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

# --- landmarks ---
# Per-group residual decomposition (spec §9.3). A semantic group's mean residual
# vector in the face frame is split into vertical/horizontal components; any
# component at or above this floor (in % head height) becomes one PLACEMENT
# Finding. We reuse the shared displacement OK floor (§6) so the landmark layer
# shares the schema's noise floor instead of inventing a second one.
LANDMARK_COMPONENT_OK_MAX = DISPLACEMENT_OK_MAX     # spec §6: <2% head height is OK
# Severity for vertical/horizontal components uses the shared DISPLACEMENT_TIERS;
# the per-group scale residual uses the shared AREA_TIERS (both in the severity
# section above) — kept there so every tier lives in exactly one place.

# Display names for the measurement groups (spec §9.3 "left eye, right eye,
# nose, mouth, jaw…"). ``nose`` merges the bridge + nostril sub-groups that
# ``frame.SEMANTIC_GROUPS`` splits, since a teacher critiques "the nose" as one.
LANDMARK_GROUP_FEATURE_NAMES = {
    "jaw": "jaw",
    "left_eye": "left eye",
    "right_eye": "right eye",
    "nose": "nose",
    "mouth": "mouth",
    "left_brow": "left brow",
    "right_brow": "right brow",
}
# Importance-weight key per group for the §9.5 ranking score. Only overrides where
# the group name differs from the weight table key; everything else falls back to
# the group name, then DEFAULT_IMPORTANCE_WEIGHT. The jaw contour IS the face oval,
# so it borrows the "face_oval" weight (spec §9.5 lists "face oval", not "jaw").
LANDMARK_GROUP_WEIGHT_KEYS = {
    "jaw": "face_oval",
}

# --- angles ---
# Feature angle comparisons (spec §9.4): fit a least-squares/PCA line through the
# relevant landmarks in BOTH images and critique the *difference* in degrees. A
# global page tilt is already absorbed by the similarity alignment (§9.1), so any
# residual angle is a real relational/contour error. Severity uses the shared
# ANGLE_TIERS (2/5/10°, spec §6) in the severity section above — no second floor;
# the score normalizes by ANGLE_OK_MAX (the 2° noise floor) from that section.
#
# ``level`` codes mirror schema.Level (0=GLOBAL, 1=PLACEMENT, 2=SHAPE). Stored as
# ints so config stays dependency-free; angles.py maps them back to Level. Eye and
# mouth lines describe how features are *arranged* → PLACEMENT. Jaw tangents
# describe contour *form* → SHAPE (spec §2 coarse-to-fine ranking; the GLOBAL tilt
# tier is reserved for whole-head tilt, which alignment already absorbs).
#
# ``indices`` are iBUG 68-point landmarks (see frame.ANCHORS / SEMANTIC_GROUPS):
# eye line = the four eye corners (36/39 right, 42/45 left); mouth line = the two
# outer mouth corners (48/54); jaw tangents = the descending jaw segment on each
# side between the jaw corner and the chin (index 8). ``weight_key`` indexes the
# §9.5 IMPORTANCE_WEIGHTS table for the ranking score (jaw borrows "face_oval").
ANGLE_LINES = {
    "eye_line": {
        "id": "eye_line_angle",
        "feature": "eye line",
        "indices": (36, 39, 42, 45),  # right+left eye outer/inner corners
        "level": 1,                   # PLACEMENT
        "weight_key": "left_eye",     # eyes weight 1.0 (spec §9.5)
    },
    "mouth_line": {
        "id": "mouth_line_angle",
        "feature": "mouth line",
        "indices": (48, 54),          # outer mouth corners
        "level": 1,                   # PLACEMENT
        "weight_key": "mouth",        # 0.8
    },
    "right_jaw": {                     # subject's right = image left (lower indices)
        "id": "right_jaw_angle",
        "feature": "right jaw line",
        "indices": (4, 5, 6, 7),      # jaw corner → chin, image-left side
        "level": 2,                   # SHAPE (contour form)
        "weight_key": "face_oval",    # 0.7
    },
    "left_jaw": {                      # subject's left = image right (higher indices)
        "id": "left_jaw_angle",
        "feature": "left jaw line",
        "indices": (9, 10, 11, 12),   # chin → jaw corner, image-right side
        "level": 2,                   # SHAPE (contour form)
        "weight_key": "face_oval",    # 0.7
    },
}

# --- proportions ---
# Canon-ratio comparison (spec §9.4): each v1 ratio is computed in BOTH images and
# the *difference* is critiqued — the target is always "match the reference," never
# "match the textbook" (so non-canonical references are handled correctly).
#
# A proportion magnitude is the deviation of the sketch ratio from the reference
# ratio, as a percentage of the reference ratio:
#     magnitude = |ratio_sketch - ratio_ref| / ratio_ref * 100      (units: "%ratio")
#
# Severity tiers for that deviation. Spec §6 only tabulates tiers for displacement
# (%head height), angles, and areas — not for dimensionless ratios — so these are a
# design choice: the ratio-deviation analogue of the displacement tiers, deliberately
# a touch coarser because a ratio couples two measured lengths and so accumulates the
# noise of both. Tune against human redlines in M5. (Logged in DECISIONS.md.)
PROPORTION_OK_MAX = 5.0            # <5% ratio deviation: matched, below noise floor
PROPORTION_INACCURACY_MAX = 10.0  # 5–10%: inaccuracy
PROPORTION_MISTAKE_MAX = 20.0     # 10–20%: mistake (>20% blunder)
PROPORTION_TIERS = (
    PROPORTION_OK_MAX,
    PROPORTION_INACCURACY_MAX,
    PROPORTION_MISTAKE_MAX,
)

# Score normalization unit (spec §9.5: score = weight × magnitude / severity_unit).
# One OK-band of ratio deviation counts as one unit of score, so a finding that just
# crosses the noise floor contributes ~weight to the ranking.
PROPORTION_SEVERITY_UNIT = PROPORTION_OK_MAX

# Per-ratio importance weights (spec §9.5 initial values: eyes 1.0, mouth 0.8,
# nose 0.7; the overall vertical-proportion cues are treated as high-leverage
# structural signals just under the eyes).
PROPORTION_WEIGHTS = {
    "eye_line_height": 1.0,        # where the eyes sit on the head — structural
    "face_thirds": 0.9,            # overall vertical proportion
    "interocular_eye_width": 1.0,  # eye spacing (spec §9.5: eyes 1.0)
    "nose_length": 0.7,            # spec §9.5: nose 0.7
    "mouth_interocular": 0.8,      # spec §9.5: mouth 0.8
}
DEFAULT_PROPORTION_WEIGHT = 0.5    # fallback, mirrors DEFAULT_IMPORTANCE_WEIGHT

# The v1 canon ratio set (spec §9.4). Each entry is one rule:
#   id           — stable Finding id.
#   feature      — human feature name for the critique sentence.
#   level        — schema.Level code (0=GLOBAL overall proportion, 1=PLACEMENT).
#                  Eye-line height and the face-thirds balance describe the head's
#                  overall proportion → GLOBAL; the feature-relative ratios are
#                  PLACEMENT (spec §2 coarse-to-fine).
#   weight_key   — key into PROPORTION_WEIGHTS for the §9.5 ranking score.
#   higher/lower — direction string when the sketch ratio is above / below the
#                  reference ratio.
# The actual geometry for each ratio lives in measure/proportions.py (the landmark
# indices it reads are documented there); config only holds the tunables + metadata.
PROPORTION_RATIOS = {
    "eye_line_height": {
        "id": "eye_line_height",
        "feature": "eye line",
        "level": 0,                # GLOBAL — overall vertical proportion
        "weight_key": "eye_line_height",
        "higher": "too high",
        "lower": "too low",
    },
    "face_thirds": {
        "id": "face_thirds",
        "feature": "midface",
        "level": 0,                # GLOBAL — overall vertical proportion
        "weight_key": "face_thirds",
        "higher": "too tall",
        "lower": "too short",
    },
    "interocular_eye_width": {
        "id": "interocular_eye_width",
        "feature": "eye spacing",
        "level": 1,                # PLACEMENT
        "weight_key": "interocular_eye_width",
        "higher": "too wide",
        "lower": "too narrow",
    },
    "nose_length": {
        "id": "nose_length",
        "feature": "nose",
        "level": 1,                # PLACEMENT
        "weight_key": "nose_length",
        "higher": "too long",
        "lower": "too short",
    },
    "mouth_interocular": {
        "id": "mouth_interocular",
        "feature": "mouth",
        "level": 1,                # PLACEMENT
        "weight_key": "mouth_interocular",
        "higher": "too wide",
        "lower": "too narrow",
    },
}
