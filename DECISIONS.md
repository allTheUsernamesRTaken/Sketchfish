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

- 2026-06-12 (Wave 4, M2) — Path 2 per `data/detection_report.md`: MediaPipe detects the
  **reference** (photo) only; the **default sketch detector is CPD landmark transfer**
  (`detect/cpd_register.py`), a three-stage chain — similarity-mode CPD (global), deformable
  CPD (smooth drift), then **per-feature local similarity CPD** around each semantic group.
  The local stage exists because point-based registration on a dense edge cloud cannot
  disambiguate a displaced feature from its unmoved neighbors (measured: a 10 % eye shift
  recovered only ~25 % globally, ~60–80 % with the local stage). A local *similarity* fit
  measures exactly what `measure/landmarks.py` reads (position/scale/rotation) and cannot
  fabricate within-feature shape. Open-curve groups (jaw, brows) get a normal-direction
  translation only — the aperture problem makes tangential motion/rotation/scale from an arc
  window unobservable noise (measured: full similarity on the jaw *worsened* it 2.2→6.7 %).

- 2026-06-12 (Wave 4, M2) — pycpd numerical guards (`_make_stable_deformable`): vanilla pycpd
  diverges on *well-matched* clouds — σ² anneals toward zero, the α·σ²·I term stops
  regularizing, and solving against the near-singular Gaussian kernel produces wild weights
  (measured: 0.8 normalized units of phantom displacement on two *identical* 1.6 k-point
  clouds). Guards: a σ² annealing floor (`CPD_SIGMA2_FLOOR`), a min-norm `lstsq` solve, and a
  blunder-scale σ² *init* (`CPD_SIGMA2_INIT` — pycpd's cloud-wide default walks self-similar
  texture like engraving curls into wrong basins; the rigid stage has already aligned the
  clouds, so the EM only needs to search drawing-error range). Landmarks are carried through
  the fitted field as the Gaussian-weighted average of the registered points' *displacements*
  — evaluating the CPD kernel at off-cloud points amplifies the same ill-conditioning (20 %
  head-height error on a perfect registration). Also: pycpd's
  `DeformableRegistration.transform_point_cloud(Y=...)` is simply wrong for new points (it
  reuses the training kernel), so it is never used.

- 2026-06-12 (Wave 4, M2) — Sketch-side MediaPipe gate is **agreement with CPD**, not the ink
  heuristic the de-risk report sketched: measured on `data/line_art`, the report's junk case
  la14 *passes* any reasonable span/on-ink check (a charcoal drawing has ink everywhere), so
  ink checks only filter gross failures. MediaPipe's sketch mesh is used only when its worst
  per-**group** mean distance to the CPD answer is ≤ 4 % head height — per-group because the
  measurement layer consumes group means (measured: a mesh within 4 % *median* still differed
  6 % on the jaw, which solvePnP turned into 20° of phantom pitch). Consequence: the fast path
  fires only when both paths nearly agree, so which path answered cannot change the critique
  (principle #7).

- 2026-06-12 (Wave 4, M2) — Input contract + eval corpus: the sketch is a drawing **of the
  head** (scope §3), so the CPD reference cloud is XDoG of the photo cropped to the expanded
  landmark bbox, and M2 eval sketches are generated head-cropped (full-frame XDoG against a
  face-cropped reference cloud mis-registers catastrophically — measured ~150 % head-height
  errors). The eval corpus is *every* `data/photos` image that detects at the working size
  and reads front-facing-ish (|yaw| ≤ 20°, scope §3) — no hand-picked list; this excludes the
  genuinely 3/4-view photos (ph03/ph06/ph16) and the profile (ph13).

- 2026-06-12 (Wave 4, M2) — Detection noise floors (M2-T2) calibrated on bias pairs
  (MediaPipe-reference vs CPD-detected undistorted sketchification) and jitter pairs (two
  random XDoG re-renders, both CPD): displacement 4.0 %hh, line angle 5°, pose 10°, area
  75 %, ratio 35 % — each above the worst observed spurious magnitude. Honest consequence:
  per-feature **scale** and canon-**proportion** findings are nearly muted in detect mode
  (XDoG re-renders change stroke support enough that spread/ratio readings jitter wildly,
  max 70.6 %area / 32.6 %ratio), and the M2-T1 menu therefore injects placements and mouth
  tilts, not scale or lateral-nose ops (their measured recovery lands *below* the calibrated
  floors — injecting errors the detector documentedly cannot resolve would only restate
  this note). Placement, line-angle and pose findings carry the v1 detect-mode critique.

- 2026-06-12 (Wave 4, M2) — M2-T1 scoring: recall is measured against the injected labels
  only; for precision, findings whose (id, direction) the **coordinate-level pipeline**
  (M0/M1-certified) also reports on the TRUE distorted landmarks are exempt from the
  false-positive count. Blunder-sized injections have real secondary consequences (shifting
  an eye vertically genuinely widens the interocular gap) — counting a true statement as a
  hallucination would measure the labels, not the detector. The exemption set is computed
  the same way for every case, never from any detection output (no self-judging).

- 2026-06-12 (Wave 4, M2) — Cross-boundary touches for the mandated demo
  (`artstockfish critique ref.jpg sketch.png`): added the `critique` subcommand to `cli.py`
  (Wave 2 file; additive), a `[project.scripts]` console entry to `pyproject.toml`, and
  `pycpd` to the `detect` extra. The M2 task prompt defines this demo command, so the wiring
  is in-scope; flagged here rather than done silently (Ground Rule 2).

- 2026-06-13 (Wave 5B, M4) — `annotate.py` SVG overlay added **alongside** the matplotlib
  `render_overlay`, not as a replacement, even though the M4 prompt says "upgrade annotate.py
  to SVG". `pipeline.py` and `detect/__init__.py` (both read-only for this wave) still call
  `render_overlay` for the M0/M2 debug PNG; removing it would break them. New `render_svg`/
  `save_svg` are the M4 product renderer; the PNG path is untouched. (Ground Rule 2.)

- 2026-06-13 (Wave 5B, M4) — `tests/test_api.py` overrides the server's `get_detector`
  dependency with a deterministic stub returning known landmarks, rather than running the M2
  detection stack. The M4 surface under test is multipart handling + `Report` JSON
  serialization + the SVG overlay; detection accuracy has its own gates (`tests/test_detect.py`)
  and heavy optional deps. The request still POSTs two real PNG fixtures, so upload/decode/
  encode/SVG render run end to end — only the ML step is swapped (standard FastAPI
  `app.dependency_overrides` pattern). Not a weakened gate: it isolates the unit it owns.

- 2026-06-13 (Wave 5B, M4) — Web deps beyond the prompt's "fastapi+svgwrite": added
  `python-multipart` (FastAPI requires it to parse file uploads), `uvicorn` (the mandated
  `uvicorn ...` / `artstockfish web` run command), and `httpx` (FastAPI `TestClient`) to a
  new `web` optional-dependencies group (+ `httpx` in `dev`). `pyproject.toml` is a Wave 0
  file; the edit is an additive new section. A `web` subcommand was added to `cli.py`
  (additive, per the prompt's "cli web command" ownership).

- 2026-06-14 (Wave 6, M5) — `benchmark/` lives at the **repo root**, not under
  `src/artstockfish/`: it is evaluation tooling, not part of the shipped library, and the spec
  layout (§5) has no benchmark package. It depends only on the light core plus the optional
  `anthropic` SDK (lazy-imported), so the library install is unaffected. Run via
  `python -m benchmark.run` (repo root is on `sys.path` for `-m`). New `tests/conftest.py`
  inserts the repo root on `sys.path` so `tests/test_benchmark.py` can `import benchmark`
  (the installed editable package only exposes `src/`). Both are additive new files; no
  parallel agent owns them (Wave 6 is serial). A `# --- benchmark ---` section was appended to
  `config.py` per the standard append-only convention (Ground Rule 4).

- 2026-06-14 (Wave 6, M5) — Benchmark dataset reuses the **M1 op menu** (single eye shift /
  brow scale / line rotation) with **no whole-page transform** injected, so ground truth equals
  the injected labels exactly. Page tilt/scale is precisely what the similarity alignment is
  meant to absorb (principle #2); injecting it would add un-scored noise to the comparison
  rather than test error detection. This menu is the one the M1 harness proves keeps
  secondary-consequence false positives negligible (~0.99 precision), so our column is honest,
  not tuned.

- 2026-06-14 (Wave 6, M5) — Scoring is computed **raw** (no false-positive exemption like the
  M2 `tests/test_detect.py` "true secondary consequence" carve-out). The headline claim is
  strongest when computed the simplest, most defensible way; a real secondary finding our
  pipeline surfaces is counted against our precision (conservative *against* us, the honest
  direction — Ground Rule 5). Both systems are scored identically: exact `(id, direction)`
  match for precision/recall, id-match (any direction) for localization, median fractional
  magnitude error, and mean pairwise Jaccard of per-case key sets across the 3 repeats for
  consistency.

- 2026-06-14 (Wave 6, M5) — The VLM baseline is given the **closed finding vocabulary** (every
  legal `id`/`direction`/units, derived from the frozen config in `benchmark/vocab.py`) and its
  output is constrained to it via structured-output **enums**. The comparison is therefore about
  *which findings the model picks and how well it measures them*, never about whether it guessed
  our id strings or emitted valid JSON — a fluent-but-unmappable critique would otherwise tank
  the baseline's recall unfairly. The prompt also tells it to ignore global position/scale/page
  rotation (the same similarity-invariant error class our pipeline measures) and to report
  structural errors first, so both systems answer the same question.

- 2026-06-14 (Wave 6, M5) — Verbalizer guard (principle #1, enforced in `critique.py`, not
  trusted to the model): the LLM receives ONLY the findings JSON (no images) and rewrites each
  template sentence; every result is validated per-finding and, on any violation, the
  deterministic template is used (granular per-sentence fallback; whole-batch fallback on LLM
  error or count mismatch). The guard forbids any sentence from naming a **feature**, **number**,
  or **error axis** (vertical / size / width / extent / rotation) the finding doesn't support.
  Two things are intentionally *not* lexically gated: the within-axis pole (the corrective verb
  legitimately names the opposite pole — "too high → lower it"), and left↔right side words (same
  corrective-phrasing entanglement). The faithful-paraphrase contract that matters — no invented
  feature, no invented/changed magnitude, no off-axis claim — is fully enforced; a regression
  test asserts every shipped template passes its own guard.

- 2026-06-14 (Wave 6, M5) — The published comparison table's VLM column is left **pending a real
  API run**: this environment has no API key and no SDK installed, and 50×3 frontier-VLM calls
  send rendered data to an external service at real cost. Per Ground Rule 5, VLM numbers are NOT
  fabricated; our column is real and deterministic. The benchmark runs end-to-end against an
  offline `StubVLM` (proving render→client→parse→score→table and the consistency metric in
  `tests/test_benchmark.py`); the identical code path hits the real API once a provider SDK + key
  are present (`python -m benchmark.run`, responses cached on disk).

- 2026-06-14 (Wave 6, M5) — Added an **OpenAI** frontier-VLM baseline alongside Anthropic (at the
  user's request — they have an OpenAI key). `benchmark/vlm.py` was refactored so both providers
  share a `_CachedVLM` base (prompt, closed vocabulary, schema, parse, on-disk cache); only the
  API call differs. Model id was **researched live** (2026-06-14) rather than recalled: `gpt-4o`
  is superseded; the current vision family is GPT-5.x, so the default is `gpt-5.5` (`--model`
  overrides, e.g. `gpt-5.4-mini` for a cheaper run). GPT-5.x are reasoning models, so the OpenAI
  client uses Chat Completions with `max_completion_tokens` (not `max_tokens`; generous headroom
  for reasoning tokens — `BENCH_OPENAI_MAX_TOKENS = 16000`) and leaves `temperature` at default
  (non-default is rejected, and the default gives the run-to-run variance the consistency metric
  measures). Output is constrained with strict `json_schema` — the same closed vocabulary. The
  default `--provider` is now `openai`. `openai`+`anthropic` added as a new `bench` extra in
  `pyproject.toml` (additive section, mirroring the M4 `web` extra precedent).

- 2026-06-14 (Wave 6, M5) — API keys: the runner loads a gitignored repo-root `.env`
  (`benchmark/_env.py`, a tiny dependency-free `KEY=value` parser; real exported env vars win via
  `setdefault`). `.env` / `.env.*` are gitignored (with `!.env.example` kept), and a committed
  `.env.example` documents `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. This keeps the "easy to set,
  safe to push to GitHub" property the user asked for — no secret ever reaches the repo, and no
  third-party dotenv dependency is added. A missing key fails fast with a message pointing at
  `.env.example` and the `--provider none` escape hatch.
