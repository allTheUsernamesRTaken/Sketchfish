# Decisions log

Dated, one-line departures from `ART_STOCKFISH_SPEC.md` with the reason
(AGENTS.md / Ground Rule 6).

- 2026-06-11 (Wave 0) — Head-height unit: spec §9.2 defines the frame's unit length as
  "chin → top of forehead/oval", but the v1 68-point convention (§5) has no crown/forehead-top
  landmark. `frame.build_face_frame` approximates head height as the span of the reference
  landmark cloud projected onto the midline axis (chin → brow line). This is internally
  consistent — every residual and every threshold is scaled by the same unit — and can be
  refined in M2 when a forehead/oval point becomes available from detection.

- 2026-06-12 (Wave 1B, proportions) — Severity tiers for canon ratios: spec §6 tabulates
  severity tiers for displacement (%head height), angles, and areas but not for dimensionless
  ratios. `config` (`# --- proportions ---`) defines ratio-deviation tiers 5/10/20% as the
  analogue of the displacement tiers, deliberately a touch coarser because a ratio couples two
  measured lengths and accumulates both lengths' noise. Magnitude = ratio deviation as a
  percent of the reference ratio. To be retuned against human redlines in M5.

- 2026-06-12 (Wave 2, integration) — Redundant-angle suppression: through the full pipeline a
  single feature displacement also tilts any line that spans it (e.g. shifting the left eye up 5 %
  tilts the eye line, since the eye-line fit reads the eye corners), so a *pure* placement error
  surfaced both a placement finding and a derived `*_angle` finding. That double-counts one
  mistake and reads like a coordinate diff, not a teacher (spec §2 principle #5 coarse-to-fine,
  pitfall §12 "don't report correlated locals when a more fundamental finding explains them").
  `pipeline.suppress_explained_angles` runs a counterfactual: translate every group that has a
  placement finding back onto the reference (mean displacement removed, internal shape kept) and
  re-measure angles; an angle finding that then drops below the OK floor was fully explained by
  placements and is suppressed, while a genuinely independent tilt (e.g. a feature rotated in
  place, or a jaw tangent with no group displacement) survives. This is a general
  explain-away rule, not a special case, and it is the mechanism M0-T1 ("one 5 % eye shift →
  exactly one finding") relies on.

- 2026-06-12 (Wave 2, integration) — matplotlib dependency: Wave 0 deliberately left matplotlib
  out of `pyproject.toml` ("add … later"); Wave 2 is the wave that first needs it, since
  `annotate.py` renders the M0 debug overlay (spec §4 lists matplotlib as the M0/M1 overlay
  tool). Added `matplotlib` to core `dependencies`. `annotate.py` imports it lazily so the
  measurement/evaluation/critique path and its tests never require it — only rendering does.
  This touches `pyproject.toml` (a Wave 0 file) but the build plan schedules the matplotlib dep
  for this milestone and no parallel agent owns it now; flagged here and in the handoff rather
  than done silently (AGENTS.md / Ground Rule 2).

- 2026-06-12 (Wave 3B, M1.5 pose) — Canonical 3D model source: spec §8 M1.5 suggests "a
  published canonical set or MediaPipe's metric face model" for the 68-point 3D landmarks, but
  neither asset is downloaded in this repo (and `data/` is gitignored). `measure/pose.py`
  synthesizes the canonical 3D model from the project's canonical 2D face by adding an
  ellipsoidal-head depth profile (face centre bulges toward the camera, rim/temples/jaw recede)
  with the nose protruded. This is *attribution scaffolding* — a fixed rigid model used to solve
  both images' poses — never a reconstruction of the sketch (principle #4); only a consistent,
  non-planar shape is needed for a well-posed PnP. Swap in a published 3DMM mean face when one is
  available; the pose API is unaffected.

- 2026-06-12 (Wave 3B, M1.5 pose) — SQPNP for branch selection: a near-frontal face is a
  near-planar PnP problem with a two-fold ambiguity, and plain iterative `solvePnP` was landing on
  the flipped (upside-down, roll ≈ ±180°) branch for one image but not the other — fabricating a
  bogus ~8° pose difference between two genuinely front-facing portraits (it broke M0-T1 once the
  pose stage was wired in). `estimate_pose` uses `cv2.SOLVEPNP_SQPNP` (global, deterministic,
  init-free) for the initial solve, which picks the geometrically correct upright branch, then
  refines with seeded iterative PnP. Portraits are upright/front-facing by scope (§3), so this is
  the right prior, not a special case.

- 2026-06-12 (Wave 3B, M1.5 pose) — Robust (trimmed) PnP: mirroring `align.robust_align`
  (principle #3), `estimate_pose` drops the worst `POSE_TRIM` (=0.25) fraction of points by
  reprojection error and re-solves on the inliers, so a *localized* drawing error (e.g. one eye
  drawn out of place) cannot drag the global pose toward itself. Without this the perturbed eye in
  M1.5-T2 biased the recovered yaw by ~0.8° and pulled the conditioned eye magnitude down to
  ~4.75%; with it the pose stays at 10.0° and the eye measures 5.01% (well inside the ±1% gate).
  Deterministic (inlier selection by sorted residual; no RANSAC randomness — principle #7).

- 2026-06-12 (Wave 3B, M1.5 pose) — Pose threshold & finding shape: the in-image angle floor is
  2° (§6), but a solvePnP pose off noisy/perturbed landmarks is coarser than a single line fit, so
  `POSE_DIFF_OK_MAX = 4°` is the pose noise floor (don't surface below the noise floor, pitfall
  §12). Only yaw/pitch can trip it — in-plane roll is page tilt, already absorbed by the similarity
  alignment (principle #2). Per spec "emit ONE Level.GLOBAL finding", the layer emits a single
  finding for the dominant out-of-plane axis (`pose_yaw` / `pose_pitch`, the other component in
  `evidence`), `axis="pose"`, severity from the shared `ANGLE_TIERS`. critique.py is owned by
  Wave 2 and not edited here; its defensive fallback template renders the pose finding cleanly
  ("The head is rotated further right by 10° relative to the reference."), matching the §11
  `pose_yaw` intent, so no template change was required.

- 2026-06-12 (Wave 1B, proportions) — Face-thirds ratio: spec §9.4 lists "face thirds" among
  the v1 ratios, but the 68-point set has no hairline/crown landmark, so the *upper* third is
  not measurable (same limitation as the head-height note above). `measure/proportions.py`
  implements face-thirds as the ratio of the two measurable thirds (midface brow→nose-base
  over lower-face nose-base→chin), which are canonically equal; it is still critiqued against
  the reference, never against the textbook value of 1.
