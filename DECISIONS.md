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

- 2026-06-12 (Wave 1B, proportions) — Face-thirds ratio: spec §9.4 lists "face thirds" among
  the v1 ratios, but the 68-point set has no hairline/crown landmark, so the *upper* third is
  not measurable (same limitation as the head-height note above). `measure/proportions.py`
  implements face-thirds as the ratio of the two measurable thirds (midface brow→nose-base
  over lower-face nose-base→chin), which are canonically equal; it is still critiqued against
  the reference, never against the textbook value of 1.
