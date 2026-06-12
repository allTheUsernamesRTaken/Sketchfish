# Decisions log

Dated, one-line departures from `ART_STOCKFISH_SPEC.md` with the reason
(AGENTS.md / Ground Rule 6).

- 2026-06-11 (Wave 0) — Head-height unit: spec §9.2 defines the frame's unit length as
  "chin → top of forehead/oval", but the v1 68-point convention (§5) has no crown/forehead-top
  landmark. `frame.build_face_frame` approximates head height as the span of the reference
  landmark cloud projected onto the midline axis (chin → brow line). This is internally
  consistent — every residual and every threshold is scaled by the same unit — and can be
  refined in M2 when a forehead/oval point becomes available from detection.
