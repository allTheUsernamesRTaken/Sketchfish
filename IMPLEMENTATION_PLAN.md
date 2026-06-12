# Art Stockfish — Implementation & Handoff Plan

> Companion to `ART_STOCKFISH_SPEC.md`. The spec says **what** to build and **why**.
> This doc says **in what order**, **what can run in parallel**, and gives **copy-paste
> prompts** you can hand to separate Claude instances.
>
> **How to use this doc:** pick a wave, open the prompt block(s) for that wave, copy each
> one into a fresh Claude Code instance pointed at this repo. Within a wave, the prompts
> marked *parallel* can run at the same time; waves themselves are serial — don't start a
> wave until the previous one's "Definition of done" is met.

---

## Ground rules (every agent reads this)

1. **The spec is law.** `ART_STOCKFISH_SPEC.md` §2 principles, §6 schema, and each
   milestone's acceptance tests are fixed. Do not weaken them to make progress.
2. **File ownership is law too.** Each prompt lists the files you may create/edit. Do **not**
   touch files owned by another agent. If you think you need to, **stop and report** — that's
   a coordination decision for the human, not a thing to silently do.
3. **`schema.py` is the synchronization barrier.** It is frozen in Wave 0 and treated as a
   read-only contract after that. If you believe the schema is wrong, stop and report; do not
   edit it mid-flight while others depend on it.
4. **`config.py` is append-only and section-partitioned.** Add constants under a header
   `# --- <your module> ---`. Never edit another module's block. (This keeps parallel merges
   trivial — every edit is a disjoint append.)
5. **Don't game gates.** Never special-case synthetic inputs, weaken a test, or tune to the
   harness's own distribution to turn a number green. A gamed gate is worse than a red one —
   if a gate is unreachable, report the gap with failing cases.
6. **Log deviations.** Anything you do that departs from the spec → one dated line in
   `DECISIONS.md` with the reason.

---

## The Loop Contract (every build prompt enforces this)

You are not done when the code is written. You are done when **you have run it and watched it
work**. Specifically:

1. Build the thing.
2. **Write the acceptance tests listed in your prompt, then run them yourself.**
3. **Loop:** read failures, fix, re-run — until every listed test is green. Where the prompt
   asks for an "eyeball" check, actually look at the output and judge it.
4. Do **not** report success until tests pass. If you get stuck after a genuine effort,
   report the blocker and the failing output — don't hand back something broken and call it done.
5. **Your final handoff message must contain, in this order:**
   - **Files created/changed** (list).
   - **How to reproduce:** the exact command(s) to re-run your tests (copy-pasteable).
   - **Pasted passing output** of those commands.
   - **Demo command** (if your milestone defines one) + what it prints/saves.
   - **One-paragraph README note** describing what now works (append it to `README.md` under a
     `## Progress` section).
   - **Deviations:** anything you added to `DECISIONS.md`, or "none."

---

## Dependency graph — what's parallel and why

Most of this project is a **serial spine** with a few genuinely independent bursts. I am
*not* forcing parallelism where serial is safer — the bursts below are parallel only because
the work is mathematically independent and writes to disjoint files.

```
        ┌─────────────────────── DE-RISK SIDECAR (independent, run anytime) ──────────────────┐
        │  MediaPipe on 15 real sketches → data/detection_report.md  (gates M2's path choice)  │
        └──────────────────────────────────────────────────────────────────────────────────────┘

WAVE 0  [serial, 1 agent]   schema.py · config.py · align.py · frame.py        ← the contract
   │                         (M0-T3 robustness, M0-T4 rotation-absorb)
   ▼
WAVE 1  [PARALLEL ×3]        1A measure/landmarks.py                            ← independent
   │                         1B measure/proportions.py                            math, disjoint
   │                         1C measure/angles.py                                  files
   ▼
WAVE 2  [serial, 1 agent]   evaluate.py · critique.py · annotate.py(mpl) ·     ← wires M0,
   │                         pipeline.py · cli.py  (M0-T1, M0-T2, eyeball)         M0 DONE
   ▼
WAVE 3  [PARALLEL ×2]        3A M1  synth/distort.py · test_harness · stability ← disjoint
   │                         3B M1.5 measure/pose.py · pipeline pose-stage         domains
   ▼
WAVE 4  [serial, 1 agent]   M2 detect/* · synth/sketchify.py                   ← decision tree,
   │   (may overlap W3)      (de-risk report picks the path)                       inherently serial
   ▼
WAVE 5  [PARALLEL ×2]        5A M3 measure/contour.py · negspace.py            ← M3 produces data,
   │                         5B M4 annotate.py(SVG) · web/ · FastAPI              M4 renders it
   ▼
WAVE 6  [serial, 1 agent]   M5 benchmark vs VLM (+ optional verbalizer)
```

**Why these and not more:** Wave 0 is the contract — parallelizing it would mean agents
guessing each other's interfaces. Waves 2, 4, 6 are integration/decision steps that need the
previous work present. The three parallel bursts (1, 3, 5) are the only places where the work
is truly independent — take the speed there, don't manufacture it elsewhere.

### Running parallel agents safely
Give each parallel agent its **own git worktree**, then merge:
```bash
git worktree add ../as-1A -b wave1-landmarks
git worktree add ../as-1B -b wave1-proportions
git worktree add ../as-1C -b wave1-angles
# ...each agent works in its worktree; when all green:
git checkout main && git merge wave1-landmarks wave1-proportions wave1-angles
```
Because file ownership is disjoint and `config.py` edits are append-only sections, these
merges are trivial. If two parallel agents ever need the same file, that's a signal the split
was wrong — fall back to serial for that pair.

---

## Copy-paste prompts

Each prompt is self-contained; the agent will read the spec and this doc itself. Replace
`<repo path>` only if the instance isn't already opened in the repo.

---

### ▶ De-risk sidecar — run this first / in parallel with Wave 0

```
Read ART_STOCKFISH_SPEC.md (the "Before M0 — one-hour de-risk" note and §2, §10) and the
Loop Contract + Ground Rules in IMPLEMENTATION_PLAN.md.

TASK: Validate the core architectural bet before we build on it. The system assumes face
landmark detection works on line drawings.
- Collect ~15 real portrait line drawings of varied skill (internet sources are fine; record
  sources). Save under data/ (gitignored).
- Run MediaPipe Face Mesh on each, and on 5 XDoG-converted photos.
- Record per-image: detected (y/n), and a 1-line quality note.
- Write data/detection_report.md with the hit rate and a recommendation: does detection work
  on clean line art (→ M2 path 1), is it spotty (→ CPD, path 2), or does it broadly fail
  (→ synthetic fine-tune, path 3)?

OWNERSHIP: data/** and data/detection_report.md only. Do not write src/ code.

HANDOFF: follow the Loop Contract handoff format. The key deliverable is the hit rate and the
path recommendation — that decision gates Wave 4. If the answer is "broadly fails," say so
loudly; it changes the architecture.
```

---

### ▶ Wave 0 — Foundation (serial, 1 agent)

```
Read ART_STOCKFISH_SPEC.md fully — especially §2 (principles), §4 (stack/conventions),
§6 (schema), §9.1 (Procrustes), §9.2 (face frame) — and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md.

TASK: Build the frozen contract and the alignment spine.
- pyproject.toml (Python 3.11+, deps: numpy scipy opencv-python pytest; add mediapipe/shapely
  later). src/artstockfish/__init__.py.
- config.py: severity thresholds and weights from §6/§9.5 as named constants, each with a
  comment citing its source. Use the "# --- <module> ---" section convention.
- schema.py: Landmarks, Severity, Level, Finding, Report EXACTLY as in §6. Frozen dataclasses.
  This is the contract every other agent depends on — get it right and stable.
- align.py: similarity_procrustes + robust_align from §9.1. Pure functions.
- frame.py: face coordinate frame from §9.2 — residuals expressed in % of head height,
  size/tilt-invariant.
- tests/test_align.py implementing M0-T3 (one landmark group displaced 25% → all OTHER
  residuals stay below OK; must FAIL with naive least-squares and PASS with robust_align) and
  M0-T4 (global 7° rotation → zero residual, transform absorbs page tilt). Use a hardcoded
  canonical 68-point face as a fixture (note 68-point convention per §5).

OWNERSHIP: pyproject.toml, src/artstockfish/__init__.py, config.py, schema.py, align.py,
frame.py, tests/test_align.py, a fixtures helper if needed (tests/fixtures.py).

LOOP until test_align.py is green. Per Loop Contract, hand back with reproduce command, pasted
output, and a README progress note. After this, schema.py is treated as read-only by everyone.
```

---

### ▶ Wave 1 — Measurement (PARALLEL ×3)

Hand all three out at once, each in its own worktree.

**1A — landmark residuals**
```
Read ART_STOCKFISH_SPEC.md §2, §6, §9.2, §9.3, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md. schema.py / align.py / frame.py already exist and are READ-ONLY.

TASK: measure/landmarks.py — per semantic group (left eye, right eye, nose, mouth, jaw…),
mean residual vector in the face frame, split into vertical/horizontal components → one
Finding per component over threshold; per-group scale residual → "too large/small". Emit
proper Finding objects (level=PLACEMENT, correct severity from config thresholds, evidence
dict with the raw points/vectors).

TESTS: tests/test_measure_landmarks.py — shift left-eye group up 5% head height → exactly one
Finding left_eye_vertical / "too high", magnitude 5% ± 0.5%; identical inputs → zero findings.

OWNERSHIP: measure/landmarks.py, tests/test_measure_landmarks.py, and a "# --- landmarks ---"
section in config.py. Do not edit schema/align/frame or other measure/* files.

LOOP until green; hand back per Loop Contract.
```

**1B — proportion ratios**
```
Read ART_STOCKFISH_SPEC.md §2, §6, §9.2, §9.4, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md. schema.py / align.py / frame.py exist and are READ-ONLY.

TASK: measure/proportions.py — compute each v1 canon ratio (eye-line height/head height; face
thirds; interocular/eye width; nose length/face height; mouth width/interocular) IN BOTH
images and critique the DIFFERENCE (match the reference, never the textbook — principle in
§9.4). Emit Findings (level=PLACEMENT or GLOBAL for overall proportion, correct severity).

TESTS: tests/test_measure_proportions.py — widen interocular distance by a known amount in the
"sketch" → one proportion Finding with correct sign/magnitude; matched ratios → zero findings.

OWNERSHIP: measure/proportions.py, tests/test_measure_proportions.py, a "# --- proportions ---"
section in config.py. Touch nothing else.

LOOP until green; hand back per Loop Contract.
```

**1C — feature angles**
```
Read ART_STOCKFISH_SPEC.md §2, §6, §9.2, §9.4, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md. schema.py / align.py / frame.py exist and are READ-ONLY.

TASK: measure/angles.py — PCA/least-squares line fits through the relevant landmark pairs
(eye line, mouth line, jaw tangents); report Δ in degrees between sketch and reference. Emit
Findings (axis="angle", correct severity from the angle thresholds in §6).

TESTS: tests/test_measure_angles.py — rotate the eye-line landmarks by a known angle in the
sketch → one eye_line_angle Finding, magnitude within ±0.5°, correct direction; level/global
rotation already handled by alignment so a pure page-tilt yields zero.

OWNERSHIP: measure/angles.py, tests/test_measure_angles.py, a "# --- angles ---" section in
config.py. Touch nothing else.

LOOP until green; hand back per Loop Contract.
```

---

### ▶ Wave 2 — Wire it up + M0 acceptance (serial, 1 agent)

```
Read ART_STOCKFISH_SPEC.md §6, §8 (M0), §9.5, §11, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md. schema/align/frame and all measure/* now exist and are READ-ONLY.

TASK: integrate the measurement modules into a working M0.
- evaluate.py: score = importance_weight × (magnitude/severity_unit); sort by (Level asc,
  score desc); accuracy_score = 100·exp(-k·Σscores), k so a typical first attempt lands 55–70.
- critique.py: template sentences per §11 (one per finding id × axis), filled from the Finding
  only. Tone rules per §11.
- annotate.py: matplotlib overlay — both point sets after alignment, displacement arrows
  colored by severity, finding labels. (SVG comes in M4; matplotlib only here.)
- pipeline.py: orchestrate landmarks→measure→evaluate→critique→annotate over coordinate-list
  inputs. cli.py: `python -m artstockfish.cli demo-synthetic`.
- tests/test_evaluate.py + the M0 acceptance tests M0-T1 (single 5% eye shift → exactly one
  PLACEMENT finding, nothing else above OK) and M0-T2 (identical sets → zero findings,
  accuracy 100).

THEN the usefulness eyeball (spec M0): print the top 3 findings for one realistically-perturbed
face and judge — does it read like a teacher or a coordinate diff? If it's a useless firehose,
fix ranking/thresholds before declaring done, and note what you changed in DECISIONS.md.

OWNERSHIP: evaluate.py, critique.py, annotate.py, pipeline.py, cli.py, tests/test_evaluate.py,
tests/test_m0_acceptance.py, "# --- evaluate ---" / "# --- critique ---" config sections.

LOOP until all M0 tests are green AND the eyeball reads like a teacher. Hand back per Loop
Contract, including the demo command output and the eyeball verdict. **M0 is DONE here.**
```

---

### ▶ Wave 3 — Harness + Pose (PARALLEL ×2)

**3A — M1 synthetic harness & metrics**
```
Read ART_STOCKFISH_SPEC.md §8 (M1), §1, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md. The full M0 pipeline exists and is READ-ONLY for you.

TASK:
- synth/distort.py: parameterized LABELED distortion generators shift_feature / scale_feature
  / rotate_line / tps_bulge / compose; each returns (distorted_landmarks, expected_findings).
- tests/test_harness.py: N=200 randomized single+multi-error cases; compute precision, recall,
  median magnitude error. GATES: precision ≥ 0.95, recall ≥ 0.95, median mag err ≤ 20%.
- tests/test_stability.py (M1-T4): Gaussian jitter σ=0.5% head height; finding ids+severities
  identical across 20 runs ≥ 95% of the time.

Do NOT game the gates (Ground Rule 5). If a gate is unreachable, report the failing cases.
Put the three headline numbers in the README progress note.

OWNERSHIP: synth/distort.py, synth/__init__.py, tests/test_harness.py, tests/test_stability.py,
"# --- synth ---" config section. Do NOT edit pipeline.py (that's 3B's).

LOOP until gates pass; hand back per Loop Contract with the precision/recall/mag-error table.
```

**3B — M1.5 pose attribution**
```
Read ART_STOCKFISH_SPEC.md §2 (principle #4!), §8 (M1.5), §7, and the Loop Contract + Ground
Rules in IMPLEMENTATION_PLAN.md. M0 pipeline exists.

TASK: measure/pose.py — canonical 3D positions for the 68 landmarks; cv2.solvePnP per image;
compare rotations. If yaw/pitch difference exceeds threshold: emit ONE Level.GLOBAL pose
finding, reproject the reference's 3D landmarks at the STUDENT's pose, and run downstream
residuals against the reprojection. Respect principle #4: pose is attribution only — never
reconstruct the sketch.
You own the single pose-stage hook in pipeline.py (insert before residuals, per §7 diagram).

TESTS: tests/test_pose.py — M1.5-T1 (frontal vs +10° yaw → exactly one GLOBAL pose finding,
PLACEMENT findings ≈ zero) and M1.5-T2 (+10° yaw AND eye up 5% → both findings present, eye
magnitude still 5% ±1% after pose conditioning). Use synthetic 3D→2D projection if 300W-LP
isn't downloaded.

OWNERSHIP: measure/pose.py, tests/test_pose.py, the pose-stage hook in pipeline.py,
"# --- pose ---" config section. Coordinate the pipeline edit is yours alone this wave.

LOOP until green; hand back per Loop Contract.
```

---

### ▶ Wave 4 — Real detection (serial, 1 agent; may overlap Wave 3)

```
Read ART_STOCKFISH_SPEC.md §8 (M2), §10, data/detection_report.md (from the de-risk sidecar),
and the Loop Contract + Ground Rules in IMPLEMENTATION_PLAN.md.

TASK: turn images into Landmarks. FOLLOW THE PATH the de-risk report chose:
- Path 1 (detection works): detect/mediapipe_face.py, constrain input, done.
- Path 2 (spotty): add detect/cpd_register.py — CPD warps the reference contour point set onto
  the sketch's extracted edges; read landmark positions off where they land. Classical only.
- Path 3 (broadly fails): synth/sketchify.py (XDoG, direct OpenCV DoG+threshold; optional HED),
  convert an annotated dataset to sketch style, fine-tune a small landmark model. Last resort.
Also build synth/sketchify.py regardless (needed for M2 eval pairs).

TESTS (tests/test_detect.py): M2-T1 end-to-end on (photo, sketchified-distorted-photo) pairs,
harness precision ≥ 0.85 AND recall ≥ 0.85 (report BOTH the synthetic-only and detection
numbers). M2-T2: raise the OK-noise floor so detector jitter on two random sketchifications of
the same image yields ZERO findings.

OWNERSHIP: detect/**, synth/sketchify.py, tests/test_detect.py, "# --- detect ---" config
section. This milestone is a decision tree — work it serially; don't parallelize internally.

Demo: `artstockfish critique ref.jpg sketch.png` works on real files. LOOP until the chosen
path's gates pass; hand back per Loop Contract with both number sets.
```

---

### ▶ Wave 5 — Contours & Product (PARALLEL ×2)

> Split chosen so files don't collide: **M3 produces contour/negspace Findings with geometry
> in `evidence`; M4 owns `annotate.py` and renders that evidence.** M3 must NOT edit annotate.py.

**5A — M3 contours & negative space**
```
Read ART_STOCKFISH_SPEC.md §8 (M3), §9.2, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md.

TASK: measure/contour.py — corresponded contour segments (face oval, jaw); after alignment,
signed perpendicular distance along arc length; smooth; maximal same-sign runs → "bulges
outward / caves in between {anchor A} and {anchor B}" with the segment in evidence; curvature
profile → "too angular/rounded". measure/negspace.py — closed background regions via flood
fill; correspond by centroid after alignment; compare area+aspect → "too narrow/wide".
Put all geometry needed for rendering into Finding.evidence (M4 will draw it).

TESTS: extend the harness with tps_bulge distortions; detect bulge midpoint within 10% arc
length and correct sign in ≥ 90% of cases (tests/test_contour.py).

OWNERSHIP: measure/contour.py, measure/negspace.py, tests/test_contour.py, tps_bulge support
already in synth/distort.py (read-only — extend via your own test helpers if needed),
"# --- contour ---"/"# --- negspace ---" config sections. Do NOT edit annotate.py.

LOOP until green; hand back per Loop Contract.
```

**5B — M4 product surface (SVG + web)**
```
Read ART_STOCKFISH_SPEC.md §8 (M4), §6, and the Loop Contract + Ground Rules in
IMPLEMENTATION_PLAN.md.

TASK: upgrade annotate.py to SVG — sketch base layer; displacement arrows (drawn→correct);
ghost outline of corrected feature; contour heatmap colored by signed deviation (read from
Finding.evidence); severity badges (!?, ?, ??) pinned to regions; clicking a finding in the
list highlights its annotation. FastAPI: POST /critique (two images) → JSON Report + SVG.
One static HTML page in web/ (no framework).

TESTS (tests/test_api.py): POST two fixture images → 200 with a valid Report JSON and
well-formed SVG; the SVG contains one annotation element per surfaced finding.

OWNERSHIP: annotate.py (you own the SVG rewrite), web/**, the FastAPI app (e.g. server.py),
cli web command, tests/test_api.py. Add fastapi+svgwrite to pyproject. Do NOT edit measure/*.

LOOP until green; hand back per Loop Contract with the run command (e.g. `uvicorn ...`) and a
screenshot or saved SVG path.
```

---

### ▶ Wave 6 — Benchmark vs VLM (serial, 1 agent)

```
Read ART_STOCKFISH_SPEC.md §8 (M5), §1, §2 (principle #1), and the Loop Contract + Ground
Rules in IMPLEMENTATION_PLAN.md. For any Anthropic API usage, also load the claude-api skill.

TASK: 50 harness-generated (reference, distorted-sketch, ground-truth-findings) triples. Send
each pair to a frontier VLM with a fixed prompt requesting critiques in OUR JSON schema. Score
both systems on finding precision/recall vs ground truth, localization, magnitude accuracy, and
run-to-run consistency (3 repeats, Jaccard of finding sets). Publish the comparison table in
the README — this is the headline claim.

Optional verbalizer: an LLM receives ONLY the Report JSON (no images) and rewrites template
sentences in teacher voice. HARD RULE enforced in code: validate the output mentions no
feature/direction/number absent from the findings; on violation, fall back to templates
(principle #1).

OWNERSHIP: benchmark/** (new), tests for the verbalizer guard, README table, optional
verbalizer in critique.py (extend, don't break existing templates).

LOOP until the benchmark runs end-to-end and the verbalizer guard test passes (feed it a
hallucinated sentence → asserts fallback). Hand back per Loop Contract with the published table.
```

---

## Definition of done per wave (gate before advancing)

| Wave | Done when |
|---|---|
| De-risk | `data/detection_report.md` exists with hit rate + path recommendation |
| 0 | `pytest tests/test_align.py` green (M0-T3, M0-T4); schema frozen |
| 1 | all three `test_measure_*` green; merged with no conflicts |
| 2 | M0-T1, M0-T2 green **and** eyeball reads like a teacher → **M0 complete** |
| 3 | harness gates (≥0.95/≥0.95/≤20%) + stability green; pose M1.5-T1/T2 green |
| 4 | M2-T1 (≥0.85/≥0.85) + M2-T2 green; `critique ref.jpg sketch.png` runs on real files |
| 5 | contour bulge ≥90% green; `POST /critique` returns Report JSON + SVG |
| 6 | benchmark table published; verbalizer guard test green |

When you (the human) want to re-verify any wave yourself, the agent's handoff message gives you
the exact `pytest`/demo command — run it and watch it go green.
```
