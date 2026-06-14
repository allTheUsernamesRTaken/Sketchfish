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

# --- synth ---
# Synthetic harness constants (spec §8 M1). The randomized M1 gate uses a fixed
# seed and N=200 so the headline precision/recall/magnitude numbers are
# deterministic and reproducible.
SYNTH_RANDOM_SEED = 20260612       # spec §2 principle #7: deterministic stability
SYNTH_HARNESS_CASES = 200          # spec §8 M1: N=200 randomized cases
SYNTH_PRECISION_GATE = 0.95        # spec §8 M1 gate
SYNTH_RECALL_GATE = 0.95           # spec §8 M1 gate
SYNTH_MAG_ERROR_GATE = 0.20        # spec §8 M1: median error <= 20% injected mag
SYNTH_STABILITY_RUNS = 20          # spec §8 M1-T4: 20 jittered runs
SYNTH_STABILITY_GATE = 0.95        # spec §8 M1-T4: identical ids+severities >=95%
SYNTH_JITTER_SIGMA_HEAD_FRAC = 0.005  # spec §8 M1-T4: Gaussian σ=0.5% head height

# --- pose ---
# Head-pose attribution layer (spec §8 M1.5, principle #4). We estimate head pose
# for the reference and the sketch INDEPENDENTLY via cv2.solvePnP on a canonical 3D
# 68-point model, compare the rotations, and — if they differ past the threshold
# below — emit ONE Level.GLOBAL pose finding and reproject the reference at the
# student's pose before the local residuals run. 3D is used for attribution only:
# we never fit 3D geometry to the sketch's drawing errors (principle #4).
#
# Threshold: the in-image angle floor is 2° (ANGLE_OK_MAX, §6), but a solvePnP pose
# estimate off noisy/perturbed landmarks is coarser than a single line fit, so the
# pose floor is set higher so detector/perturbation noise alone does not trip a
# spurious GLOBAL pose finding (mirrors the "don't show below the noise floor" rule,
# pitfall §12). A real turn of the head (the M1.5 tests use ±10° yaw) is well above
# it. Yaw/pitch only — in-plane roll is page tilt, already absorbed by the
# similarity alignment (principle #2), so it never becomes a pose finding. Tune
# against labelled 300W-LP poses in a later wave. (Logged in DECISIONS.md.)
POSE_DIFF_OK_MAX = 4.0             # deg; below this the two heads face the same way

# Robust pose: after the full solvePnP, drop the worst POSE_TRIM fraction of points
# by reprojection error and re-solve on the inliers, so a localized drawing error
# can't drag the global pose toward itself (principle #3; mirrors align.ROBUST_TRIM).
POSE_TRIM = 0.25                   # spec §9.1 robustness analogue (ROBUST_TRIM = 0.25)

# Severity for the pose finding reuses the shared ANGLE_TIERS (2/5/10°, §6): a head
# turned 5–10° further than the reference is a MISTAKE, >10° a BLUNDER. The score
# normalizes by the angle floor so a pose finding that just crosses POSE_DIFF_OK_MAX
# still ranks above placement noise.
POSE_SEVERITY_UNIT = ANGLE_OK_MAX  # spec §9.5: score = weight × magnitude / unit

# Pose is the most structural cue (the coarsest level, spec §2 #5), so it carries a
# full importance weight — it should sort to the very top of the GLOBAL tier.
POSE_WEIGHT = 1.0                  # spec §9.5 scale (eyes 1.0); pose is top-level

# Camera model for solvePnP / reprojection. A long focal length (a multiple of the
# image's long side) keeps perspective mild so the synthetic projection stays near
# the weak-perspective regime faces are usually shot in; the exact value only has to
# be used consistently for both solvePnP and the reprojection (it is, in pose.py).
POSE_FOCAL_LENGTH_FACTOR = 2.0     # focal = factor × max(image_w, image_h)

# --- detect ---
# M2 real detection (spec §8 M2). Path choice per data/detection_report.md: PATH 2 —
# CPD is the default sketch detector; MediaPipe detects the *reference* (a photo,
# where it is reliable) and serves only as a sanity-gated opportunistic fast-path on
# the sketch (the report's la14 case shows it can return a confidently-misplaced mesh).

# MediaPipe FaceLandmarker model bundle (BlazeFace short-range + 478-pt mesh), the
# same file the de-risk sidecar used; path is relative to the repo root. Re-fetch:
#   curl -sL -o data/_scripts/face_landmarker.task https://storage.googleapis.com/
#     mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
DETECT_MODEL_PATH = "data/_scripts/face_landmarker.task"
DETECT_MIN_CONFIDENCE = 0.5        # MediaPipe default (de-risk report: method)
DETECT_RELAXED_CONFIDENCE = 0.2    # retry for tight close-ups (report: ph11 control)

# Sketch-side MediaPipe gating (de-risk report: "gate every sketch detection with
# the overlay/geometry sanity-check"). Two layers:
# 1. A cheap ink-plausibility check — the mesh must span a sensible fraction of the
#    drawing's ink extent and sit on ink. Measured on data/line_art: this alone does
#    NOT reject the report's junk case (la14 spans 0.34 of a charcoal drawing whose
#    ink is everywhere), so it only filters gross failures (mesh on blank paper).
DETECT_GATE_MIN_SPAN = 0.25        # mesh bbox diagonal ≥ 25% of ink bbox diagonal
DETECT_GATE_MIN_NEAR_INK = 0.50    # ≥ 50% of the 68 landmarks within the ink radius
DETECT_GATE_INK_RADIUS_FRAC = 0.025  # "near ink" radius, fraction of ink bbox diagonal
# 2. The decisive gate: agreement with the CPD transfer (the trusted default path,
#    which runs regardless). MediaPipe's mesh is used only when its worst per-group
#    mean distance to the CPD answer is below this fraction of head height — a junk
#    mesh cannot agree with an independent classical reading of the same drawing.
#    Per-GROUP because the measurement layer consumes group means: a whole-face
#    median hides exactly the disagreement that changes the critique (measured: a
#    mesh within 4% median still differed 6% on the jaw → 20° of implied pitch).
#    Tight ≈ the displacement floor: which internal path answered must not change
#    the critique (principle #7 stability).
DETECT_GATE_AGREEMENT_MAX = 4.0    # %head_height, worst group-mean over the 68 pts

# CPD registration (the default sketch path). CPD is *correspondence only* — it finds
# where the landmarks are; alignment stays similarity-only Procrustes (glossary §13,
# principle #2). Point budget bounds the O(M·N) EM cost; values below are in
# normalized cloud units (centroid 0, RMS radius 1).
CPD_MAX_POINTS = 1200              # per-cloud subsample budget (grid, deterministic);
#                                    dense enough that one feature (an eye) holds ~20+
#                                    points — the local refinement needs that support
CPD_TOLERANCE = 1e-5               # EM convergence tolerance (pycpd default 1e-3 is too loose)
CPD_RIGID_W = 0.10                 # rigid outlier weight (background/hair clutter)
CPD_RIGID_MAX_ITER = 60
CPD_DEFORM_ALPHA = 2.0             # pycpd default; smooth global drift only — feature-level
#                                    precision comes from the local refinement stage below
CPD_DEFORM_BETA = 0.30             # kernel width: ~feature scale
CPD_DEFORM_W = 0.15                # deformable outlier weight
CPD_DEFORM_MAX_ITER = 80
# The deformable stage solves an O(M³) system per EM iteration, but it only has to
# produce a SMOOTH carry field (precision is the local stage's job on the dense
# clouds), so it runs on a strided subset. Striding the grid-ordered points keeps
# spatial coverage uniform and deterministic.
CPD_DEFORM_MAX_POINTS = 600
# σ² annealing floor for the deformable EM (normalized units²). σ = 0.03 ≈ 1–2% of
# head height — already below the detection noise floor; annealing past it stops
# regularizing the kernel solve and the registration explodes numerically (see
# cpd_register._make_stable_deformable).
CPD_SIGMA2_FLOOR = 1e-3
# Initial σ² for the deformable EM. pycpd's default (mean pairwise distance² of the
# whole cloud) starts with a cloud-wide "blur" phase that walks self-similar texture
# (hatching, engraving curls) into wrong basins. The rigid stage has already aligned
# the clouds, so correspondence search only needs to span the largest *drawing error*
# (~0.2 units, a blunder), not the whole face.
CPD_SIGMA2_INIT = 0.04
# After the rigid stage, points with no counterpart within this radius (normalized
# units) in the other cloud are trimmed symmetrically — content only one side has
# (shoulders, background texture) must not drag the deformable field (principle #3).
# The radius must comfortably exceed the largest drawing error the system measures
# (a blunder displacement ≈ 0.2–0.3 units) or the moved feature itself gets trimmed
# and becomes invisible; true cross-content sits much farther (≳ 0.5 units).
CPD_TRIM_RADIUS = 0.35
# Landmarks are carried through the registration as the Gaussian-weighted average of
# the nearby registered points' displacements (see cpd_register: evaluating the CPD
# kernel at off-cloud points is numerically unstable when σ² collapses). Wide-ish =
# stable: this stage only needs to park each group near its strokes; precision is the
# local stage's job.
CPD_CARRY_BETA = 0.25

# Local per-feature refinement. Point-based registration on a dense edge cloud cannot
# disambiguate a displaced feature from its unmoved neighbors (an eye drawn 10% too
# high sits closer to the brow's strokes than to its own), so after the global stages
# each semantic group is re-registered LOCALLY: a similarity-mode CPD between the ink
# points around the group on both sides. A similarity fit measures exactly what the
# landmark layer reads — where the feature went, its size, its rotation — and cannot
# fabricate within-feature shape (that's M3's contour problem, not faked here).
CPD_LOCAL_WINDOW_MARGIN = 0.18     # window: ink within this distance (normalized units)
#                                    of the group's carried landmarks. Must exceed the
#                                    residual displacement the global stage leaves on a
#                                    blunder-sized error, or the window misses the
#                                    feature's true strokes and nothing is recovered.
CPD_LOCAL_SEARCH_SLACK = 0.0       # extra sketch-side margin. MUST stay ≈ 0: asymmetric
#                                    windows force the local GMM to explain the extra
#                                    target ring and bias the fit outward (measured: it
#                                    alone pushed identity-pair error from ~0 to 3–10%).
#                                    The deformable stage already parks each group near
#                                    its strokes, so no slack is needed.
CPD_LOCAL_MIN_POINTS = 12          # skip refinement when a window has fewer ink points
CPD_LOCAL_W = 0.25                 # local outlier weight (neighboring features' strokes)
CPD_LOCAL_MAX_ITER = 60
CPD_LOCAL_MAX_SHIFT = 0.30         # a local fit that moves a group's centroid farther
#                                    than this is a degenerate window — keep the global
#                                    estimate instead (robustness, principle #3)
# Ink-mask despeckling: connected components smaller than (frac × image diagonal)²
# pixels are texture specks, not strokes, and are dropped from registration clouds.
DETECT_INK_MIN_COMPONENT_FRAC = 0.005
# Reference ink comes from XDoG of the photo cropped to the landmark bbox expanded by
# this margin (the sketch input is, per scope §3, a drawing of the head — cropping the
# reference to the head makes the two clouds describe the same subject).
DETECT_FACE_CROP_MARGIN = 0.40

# Detection noise floors (M2-T2). Real detection adds correspondence jitter far above
# the synthetic floors, so findings measured from *detected* landmarks are surfaced
# only above these raised OK floors (spec §8 M2-T2, pitfall §12 "do not show findings
# below the detector-noise floor"). Calibrated 2026-06-12 on the usable data/photos
# corpus: per photo, one bias pair (MediaPipe reference vs CPD-detected undistorted
# sketchification) and one jitter pair (two random sketchifications, both CPD path);
# each floor sits above the largest spurious magnitude observed for its units.
# tests/test_detect.py M2-T2 re-runs the jitter experiment as the gate.
# NOTE the honest consequence of the large area/ratio floors: per-feature *scale*
# and canon-*proportion* findings are nearly muted in detect mode — XDoG re-renders
# change stroke support enough that feature spread/ratios jitter wildly (measured
# max 70.6 %area, 32.6 %ratio). Placement, line-angle, and pose findings carry the
# critique. Improving scale/ratio observability is M3-adjacent future work.
DETECT_OK_DISPLACEMENT = 4.0       # %head_height; max observed jitter 3.34
DETECT_OK_ANGLE = 5.0              # deg; max observed 4.13 (jaw tangents)
DETECT_OK_POSE = 10.0              # deg; max observed 8.98 (pitch, CPD pairs)
DETECT_OK_AREA = 75.0              # %area; max observed 70.6 (eye scale, re-render)
DETECT_OK_RATIO = 35.0             # %ratio; max observed 32.6 (mouth/interocular)

# M2 evaluation harness (tests/test_detect.py). The eval set is every data/photos
# image whose reference detects at the working size and reads front-facing-ish
# (scope §3) — no hand-picked list.
DETECT_PRECISION_GATE = 0.85       # spec §8 M2-T1
DETECT_RECALL_GATE = 0.85          # spec §8 M2-T1
DETECT_EVAL_MAX_SIDE = 640         # working size: photos downscale; detection stays reliable
DETECT_EVAL_MAX_YAW_DEG = 20.0     # scope §3 "front-facing-ish": excludes profile photos
DETECT_EVAL_CASES_PER_PHOTO = 4    # distorted eval pairs sampled per usable photo
DETECT_EVAL_SEED = 20260612        # deterministic harness (principle #7)

# --- contour ---
# M3 contour measurement (spec §8 M3): after similarity alignment, sample
# corresponded contour segments by arc length, measure signed perpendicular
# distance in % head height, smooth, then surface maximal same-sign runs.
CONTOUR_SAMPLE_COUNT = 96          # dense enough to localize a 68-pt jaw bulge within 10% arc
CONTOUR_SMOOTH_WINDOW = 7          # odd moving-average window for stable same-sign runs
CONTOUR_RUN_OK_MAX = DISPLACEMENT_OK_MAX  # spec §6 displacement floor for local contour offsets
CONTOUR_MIN_RUN_ARC_FRAC = 0.06    # suppress tiny point-noise runs; M3 wants visible segments
CONTOUR_CURVATURE_OK_MAX = 12.0    # deg/segment; curvature profile noise floor for angularity
CONTOUR_CURVATURE_TIERS = (12.0, 20.0, 35.0)  # angular/rounded visibility tiers
CONTOUR_CURVATURE_SMOOTH_WINDOW = 5
CONTOUR_SEVERITY_UNIT = DISPLACEMENT_OK_MAX  # spec §9.5 score normalization analogue
CONTOUR_WEIGHT = IMPORTANCE_WEIGHTS["face_oval"]  # spec §9.5: face oval importance 0.7
CONTOUR_SEGMENTS = {
    "jaw": {
        "indices": tuple(range(0, 17)),
        "feature": "jaw contour",
        "anchor_names": (
            "right temple",
            "right jaw",
            "right lower jaw",
            "chin",
            "left lower jaw",
            "left jaw",
            "left temple",
        ),
    },
    "face_oval": {
        # The 68-point set has only the visible lower oval/jaw contour; this is
        # the v1 face-oval proxy until a detector supplies crown/cheek contours.
        "indices": tuple(range(0, 17)),
        "feature": "face oval",
        "anchor_names": (
            "right temple",
            "right jaw",
            "right lower jaw",
            "chin",
            "left lower jaw",
            "left jaw",
            "left temple",
        ),
    },
}
CONTOUR_DEFAULT_SEGMENTS = ("jaw",)

# --- negspace ---
# M3 negative-space measurement (spec §8 M3): find closed background regions via
# flood fill, correspond regions by centroid after alignment, then compare area
# and aspect ratio. Magnitudes are percent deviation from the reference region.
NEGSPACE_MIN_REGION_AREA = 24      # px; drop specks from binary flood-fill regions
NEGSPACE_AREA_OK_MAX = AREA_OK_MAX # spec §6 area floor: <8% OK
NEGSPACE_ASPECT_OK_MAX = 8.0       # %aspect deviation floor, area-tier analogue
NEGSPACE_ASPECT_TIERS = (8.0, 15.0, 30.0)
NEGSPACE_WEIGHT = 0.5              # background-shape cue, below face features
