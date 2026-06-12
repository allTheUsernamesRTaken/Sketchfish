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

- 2026-06-12 (Wave 1B, proportions) — Face-thirds ratio: spec §9.4 lists "face thirds" among
  the v1 ratios, but the 68-point set has no hairline/crown landmark, so the *upper* third is
  not measurable (same limitation as the head-height note above). `measure/proportions.py`
  implements face-thirds as the ratio of the two measurable thirds (midface brow→nose-base
  over lower-face nose-base→chin), which are canonically equal; it is still critiqued against
  the reference, never against the textbook value of 1.
