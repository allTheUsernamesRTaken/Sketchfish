# Art Stockfish — Build Specification

> **Audience:** This document is written as project context for Claude (Claude Code or chat).
> Read it fully before writing any code. It defines the architecture, the build order,
> the design principles that must not be violated, and the acceptance test for every milestone.
> Work through milestones strictly in order — each one is independently runnable and tested.
>
> **What is fixed vs. free.** The §2 principles, the §6 schema, and each milestone's
> acceptance tests are *fixed* — do not weaken them. The §5 repo layout and the reference
> code in §9 are a *suggested end state*, not a build order: reach each test with as few
> files as you like and refactor toward §5 only once tests are green. When you deviate from
> anything in this spec, append a dated line to `DECISIONS.md` with the reason — principled
> deviation is expected; silent drift is not.
>
> **Before M0 — one-hour de-risk (do this first).** The whole landmark-first architecture
> bets on detection working on line drawings. Run MediaPipe Face Mesh on ~15 real portrait
> sketches *today* and note the hit rate in `data/detection_report.md`. If it broadly fails,
> stop and raise it — the architecture changes before any of M0 is built.

---

## 1. What this project is

A computer-vision drawing coach. Given a **reference image** and a **student's sketch** of it,
the system produces teacher-like critiques grounded in *measurable geometry*, not aesthetics:

- "The left eye is 7% of head height too high."
- "The cheek contour bulges outward between the eye line and the mouth line."
- "The shoulder line is tilted 9° too steeply."
- "The negative space between the arm and torso is 30% too narrow."

…plus a chess.com-style annotated overlay (arrows, ghost corrections, severity badges)
and a ranked list of corrections ("best move first").

**The Stockfish analogy is the design document:**

| Chess engine | Art Stockfish |
|---|---|
| Board representation | Landmark graph + vectorized contours |
| Evaluation function | Weighted sum of interpretable geometric error terms |
| Best move | Highest-leverage correction, ranked coarse-to-fine |
| Blunder / mistake / inaccuracy | Error-magnitude severity tiers |
| "Show the line" | Warp the sketch to fix only the top error |
| Eval bar | Aggregate accuracy score |

**Why this beats a VLM at its job:** vision-language models give fluent but unmeasured,
inconsistent, poorly-localized feedback. This system's edge is *measured, localized,
reproducible, annotatable* critique. Protect that edge in every design decision.

---

## 2. Non-negotiable design principles

Claude: do not violate these, even when a violation would be locally convenient.

1. **Measurement is deterministic geometry.** No learned model ever produces a number that
   appears in a critique. ML may be used only for *correspondence* (finding landmarks/contours).
   An LLM may be used only to *paraphrase* an already-computed findings struct into warmer
   language — it never measures, never adds findings, never changes magnitudes.

2. **The alignment transform class defines critique semantics.** Sketch and reference are
   registered with a **similarity transform only** (translation, rotation, uniform scale —
   Procrustes). Never fit affine, homography, or non-rigid warps during alignment: anything the
   transform can absorb is something the system becomes blind to, and affine absorbs real
   proportion errors (e.g., systematically widened faces) that teachers critique.

3. **Alignment must be robust.** A drawing with one huge error must not have its alignment
   dragged toward that error (which smears blame across correct features). Use trimmed/IRLS
   Procrustes or RANSAC. There is an acceptance test for this (M0-T3).

4. **3D is for attribution only — never reconstruct the sketch.** A beginner's drawing is not
   a valid projection of any 3D scene; its inconsistencies ARE the signal. Fitting 3D geometry
   to the sketch regularizes the errors away. The only permitted 3D operation: estimate head
   pose for reference and sketch independently via `solvePnP` on canonical 3D landmarks,
   report pose difference as a *single* finding, and optionally reproject the reference at the
   student's pose before computing local residuals. No depth estimation, no NeRF/splatting,
   no 3DMM fitting to the sketch.

5. **Critique ranking is coarse-to-fine** (atelier pedagogy): global pose/tilt/proportion
   errors outrank feature placement, which outranks local contour shape. Never surface a
   detail-level finding above an unresolved structural one.

6. **Every milestone is gated by automated tests on synthetic ground truth.** The synthetic
   distortion harness (M1) is the project's conscience: inject known errors, assert the system
   reports exactly those errors and nothing else. No milestone is "done" without its tests green.

7. **Determinism and stability.** Same inputs → same report, bit-for-bit where possible.
   Slightly jittered inputs → same findings (tested in M1-T4). If the critique flips between
   runs, users lose trust instantly.

---

## 3. Scope

**v1 (this spec): front-facing-ish portraits, clean line art or digital input, landmark-driven.**

In scope: face landmarks, proportion ratios, feature angles, head-pose attribution,
signed contour deviation, negative space (v1.5), SVG annotated overlay, CLI + minimal web UI,
synthetic evaluation harness, VLM baseline benchmark.

Out of scope for v1 (do not build, do not scaffold "for later"): figure/pose subjects,
arbitrary still life, photographed-pencil-sketch cleanup (page dewarping, adaptive
thresholding), shading/value critique, stroke-quality critique, real-time feedback,
user accounts, mobile.

---

## 4. Tech stack

- Python 3.11+, managed with `uv` (or plain venv + pip)
- `numpy`, `scipy` — Procrustes, geometry, statistics
- `opencv-python` — contours, solvePnP, image I/O
- `mediapipe` — Face Mesh landmark detection (M2)
- `pycpd` — Coherent Point Drift fallback registration (M2, only if needed)
- `shapely` — negative-space polygons (M3)
- `matplotlib` — M0/M1 debug overlays only
- `svgwrite` (or hand-rolled f-strings) — production annotation overlays (M4)
- `fastapi` + a single static HTML page — web UI (M4); no frontend framework
- `pytest` — all tests
- Optional, M5 only: `anthropic` / `openai` SDKs for the verbalizer and VLM baseline

Code conventions: type hints everywhere; dataclasses (or pydantic) for all schema objects;
pure functions for all geometry (no hidden state); every magic number is a named constant in
`config.py` with a comment citing its source (e.g., the proportion canon it comes from).

---

## 5. Repository layout

*Suggested end state, not a build order. Get the earliest acceptance test green in as few
files as you like, then refactor toward this. v1 uses the **68-point** landmark convention
throughout (300-W); WFLW's 98 points are only a fine-tune source in M2, downsampled to 68.*

```
art-stockfish/
  README.md
  ART_STOCKFISH_SPEC.md        # this file
  pyproject.toml
  src/artstockfish/
    config.py                  # thresholds, weights, severity tiers
    schema.py                  # Landmarks, Finding, Report dataclasses
    align.py                   # Procrustes + robust trimming      (M0)
    frame.py                   # face coordinate frame, axes        (M0)
    measure/
      landmarks.py             # residual decomposition             (M0)
      proportions.py           # ratio rules                        (M0)
      angles.py                # line-fit angle comparisons         (M0)
      pose.py                  # solvePnP attribution layer         (M1.5)
      contour.py               # signed deviation, curvature        (M3)
      negspace.py              # negative-space regions             (M3)
    evaluate.py                # weighting, ranking, severity       (M0)
    critique.py                # template sentence generation       (M0)
    annotate.py                # SVG overlay rendering              (M0 mpl → M4 svg)
    detect/
      mediapipe_face.py        # sketch+photo landmark detection    (M2)
      cpd_register.py          # CPD fallback                       (M2, conditional)
    synth/
      distort.py               # known-error generators             (M1)
      sketchify.py             # XDoG/HED photo→sketch              (M2)
    pipeline.py                # end-to-end orchestration
    cli.py                     # `artstockfish critique ref.jpg sketch.jpg`
  tests/
    test_align.py  test_measure.py  test_evaluate.py
    test_harness.py            # synthetic precision/recall gates
    test_pose.py  test_stability.py
  data/                        # gitignored; datasets + fixtures
  web/                         # M4 static page
```

---

## 6. Core data schema (`schema.py`)

Everything downstream of measurement consumes `Finding` objects. Define these first.

```python
@dataclass(frozen=True)
class Landmarks:
    points: np.ndarray          # (N, 2) float64, image coords
    names: tuple[str, ...]      # semantic names, e.g. "left_eye_outer"
    image_size: tuple[int, int]

class Severity(Enum):
    OK = "ok"               # below noise floor — never shown
    INACCURACY = "inaccuracy"   # !?  small but real
    MISTAKE = "mistake"         # ?   clearly visible
    BLUNDER = "blunder"         # ??  structural

class Level(IntEnum):       # coarse-to-fine ranking tiers
    GLOBAL = 0              # pose, tilt, overall proportion
    PLACEMENT = 1           # feature position/size
    SHAPE = 2               # local contour form

@dataclass(frozen=True)
class Finding:
    id: str                 # stable, e.g. "left_eye_vertical"
    level: Level
    severity: Severity
    feature: str            # "left eye"
    axis: str               # "vertical" | "horizontal" | "angle" | "area" | ...
    direction: str          # "too high" | "tilted clockwise" | "too narrow" | ...
    magnitude: float        # normalized (fraction of head height, or degrees)
    units: str              # "%head_height" | "deg" | "%area"
    score: float            # weight * normalized magnitude (for ranking)
    evidence: dict          # raw geometry: points, vectors, segment indices
                            #   → consumed by annotate.py, never by critique text

@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]   # sorted: Level asc, then score desc
    accuracy_score: float           # 0–100 aggregate ("eval bar")
    transform: dict                 # the fitted similarity transform params
    pose: dict | None               # per-image pose estimates (M1.5+)
```

Severity thresholds (in `config.py`, tune later against human redlines):
landmark displacement as % of head height — `<2%` OK, `2–4%` inaccuracy, `4–8%` mistake,
`>8%` blunder. Angles: `<2°` OK, `2–5°`, `5–10°`, `>10°`. Areas: `<8%`, `8–15%`, `15–30%`, `>30%`.

---

## 7. Pipeline stages (reference)

```
reference photo ──► landmarks + contours ─┐
                                          ├─► correspondence ─► robust similarity
student sketch ──► clean/vectorize ───────┘        alignment (Procrustes)
                                                        │
                                          [M1.5] pose attribution (solvePnP both sides;
                                           if poses differ: 1 GLOBAL finding + reproject
                                           reference at student pose before residuals)
                                                        │
                              geometric measurement (residuals, ratios, angles,
                              signed contour deviation, negative space)
                                                        │
                              evaluation (weights × magnitudes → ranked findings)
                                                        │
                          ┌─────────────────────────────┴───────────────┐
                  critique text (templates,                    annotated overlay
                  optional LLM paraphrase)                     (SVG: arrows/ghost/badges)
```

---

## 8. Milestones

Work strictly in order. Each milestone ends with: tests green, a runnable demo command,
and a one-paragraph note in README on what now works.

### M0 — Synthetic core (no computer vision). Target: one weekend.

Inputs are *coordinate lists*, not images. Use ground-truth 68-point annotations from the
300-W dataset (or hardcode one canonical face's landmarks as a fixture) as the "reference";
hand-perturbed copies are the "sketch."

Build: `schema.py`, `align.py` (Procrustes, §9.1), `frame.py` (§9.2),
`measure/landmarks.py` + `proportions.py` + `angles.py` (§9.3–9.4), `evaluate.py`,
`critique.py` (template sentences), `annotate.py` (matplotlib: both point sets after
alignment, displacement arrows colored by severity, finding labels).

Acceptance tests:
- **M0-T1:** shift left-eye landmarks up by 5% of head height → report contains exactly one
  PLACEMENT finding `left_eye_vertical / too high`, magnitude 5% ± 0.5%, and zero other
  findings above OK.
- **M0-T2:** identical landmark sets → zero findings, accuracy_score = 100.
- **M0-T3 (robustness):** one landmark group displaced by 25% of head height → all *other*
  features' residuals stay below the OK threshold (alignment was not dragged). This test
  fails with naive least-squares Procrustes; that's the point.
- **M0-T4:** global 7° rotation of the whole sketch → zero findings (similarity transform
  absorbs page tilt; tilt of the page is not an error).

Get M0-T1 green in as few files as you like before splitting into the §5 module tree.

Demo: `python -m artstockfish.cli demo-synthetic` → prints critique, saves overlay PNG.

**Usefulness smoke test (do not skip):** every acceptance test here is synthetic, and
synthetic-green does *not* mean the critique is useful. Print the top 3 findings for one
real perturbed face and eyeball them: do they read like a teacher, or like a coordinate
diff? If it's a firehose of true-but-useless nitpicks, the ranking/threshold design is
wrong — fix that before M1, not after M2.

### M1 — Synthetic distortion harness + metrics. Target: 1–2 evenings.

`synth/distort.py`: a library of parameterized, *labeled* distortion generators —
`shift_feature(name, dx, dy)`, `scale_feature(name, s)`, `rotate_line(name, deg)`,
`tps_bulge(region, amount)` (thin-plate-spline local warp for contour-era tests),
`compose(...)`. Each returns (distorted_landmarks, expected_findings).

`tests/test_harness.py`: run N=200 randomized single- and multi-error cases; compute
**precision** (no hallucinated findings), **recall** (injected errors detected), and
**magnitude error** (|reported − injected|). Gates: precision ≥ 0.95, recall ≥ 0.95,
median magnitude error ≤ 20% of injected magnitude. These numbers go in the README —
they are the project's headline claim.

**These gates are hard, but they measure the system — not the test.** Never weaken a test,
special-case the synthetic inputs, or tune thresholds to the harness's own distributions to
make a number go green. If a gate isn't reachable, stop and report the gap (with the failing
cases) rather than engineering around it — a gamed gate is worse than a red one.

- **M1-T4 (stability):** add Gaussian jitter σ = 0.5% head height to all points; the set of
  finding ids and severities must be identical across 20 jittered runs ≥ 95% of the time.

### M1.5 — Pose attribution layer. Target: 1–2 evenings.

`measure/pose.py`: canonical 3D positions for the 68 landmarks (use a published canonical
set or MediaPipe's metric face model); `cv2.solvePnP` per image; compare rotations.
If yaw/pitch difference exceeds threshold: emit ONE `Level.GLOBAL` finding
("head rotated N° further right than reference"), reproject the reference's 3D landmarks
at the *student's* pose, and run all downstream residuals against the reprojection.

Acceptance (use 300W-LP, which provides the same faces at labeled yaw angles):
- **M1.5-T1:** frontal reference vs +10° yaw "sketch" → exactly one GLOBAL pose finding;
  the count of PLACEMENT findings is ~zero (not a storm of correlated local errors).
- **M1.5-T2:** +10° yaw AND left eye shifted up 5% → the pose finding AND the eye finding
  both present; eye magnitude still 5% ± 1% *after* pose conditioning.

### M2 — Real detection on sketches. Target: the hard one; budget 2–4 weekends.

**Run the de-risking experiment before building:** collect 15–20 portrait line drawings of
varied skill; run MediaPipe Face Mesh on them and on XDoG-converted photos; record the
failure rate in `data/detection_report.md`. Then choose the path:

- Detection works on clean line art → ship `detect/mediapipe_face.py`, constrain input,
  done.
- Spotty → `detect/cpd_register.py`: Coherent Point Drift warps the reference's contour
  point set onto the sketch's extracted edge points; read landmark positions off where
  they land. Fully classical, no training.
- Broadly fails → synthetic fine-tune: convert an annotated dataset (300-W / WFLW) to
  sketch style with XDoG/HED (`synth/sketchify.py` — annotations carry over for free),
  fine-tune a small landmark model. Only go here if the first two paths fail.

Acceptance:
- **M2-T1:** end-to-end on (photo, sketchified-distorted-photo) pairs: harness precision ≥
  0.85, recall ≥ 0.85 (detection noise lowers the synthetic-only gates; that's expected —
  report both numbers).
- **M2-T2:** raise the OK-noise floor so that detector jitter alone (run on two random
  sketchifications of the same image) produces zero findings.

Demo: `artstockfish critique ref.jpg sketch.png` works on real files.

### M3 — Contours and negative space. Target: 2–3 weekends.

`measure/contour.py`: extract corresponded contour segments (face oval, jaw); after
alignment, signed perpendicular distance sampled along arc length; smooth; find maximal
same-sign runs → "bulges outward / caves in between {anchor A} and {anchor B}" with the
segment in `evidence`. Curvature profile comparison → "too angular / too rounded."
`measure/negspace.py`: closed background regions via flood fill between contours;
correspond by centroid after alignment; compare area + aspect → "too narrow/wide."

Acceptance: extend the harness with `tps_bulge` distortions; detect the bulge's location
(midpoint within 10% arc length of injected) and sign correctly in ≥ 90% of cases.

### M4 — Product surface. Target: 1–2 weekends.

`annotate.py` → SVG: sketch as base layer; displacement arrows (drawn→correct);
ghost outline of corrected feature; contour heatmap colored by signed deviation; severity
badges (!?, ?, ??) pinned to regions; click a finding in the list → highlight its
annotation. FastAPI: `POST /critique` (two images) → JSON Report + SVG. One static page.

Stretch ("show the line"): thin-plate-spline warp of the student's own strokes fixing
*only* the top-ranked finding, rendered as a before/after slider.

### M5 — Benchmark vs VLM baseline + optional verbalizer. Target: 1 weekend.

Protocol: 50 harness-generated (reference, distorted-sketch, ground-truth-findings)
triples. Send each pair to a frontier VLM with a strong, fixed prompt requesting critiques
in this system's JSON schema. Score both systems on: finding precision/recall vs ground
truth, localization (does the named feature match the injected one), magnitude accuracy,
and run-to-run consistency (3 repeats, Jaccard of finding sets). Publish the table.

Optional verbalizer: LLM receives ONLY the `Report` JSON (no images) and rewrites template
sentences in teacher voice. Hard rule enforced in code: the verbalizer output is validated
to mention no feature, direction, or number absent from the findings; on violation, fall
back to templates.

---

## 9. Key algorithms

### 9.1 Robust similarity Procrustes (`align.py`)

```python
def similarity_procrustes(A: np.ndarray, B: np.ndarray, w: np.ndarray):
    """Weighted similarity transform mapping B → A. A, B: (N,2). w: (N,) weights."""
    wa, wb = (w[:, None] * A), (w[:, None] * B)
    muA, muB = wa.sum(0) / w.sum(), wb.sum(0) / w.sum()
    A0, B0 = A - muA, B - muB
    H = (w[:, None] * B0).T @ A0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    s = (S * np.array([1.0, d])).sum() / (w * (B0 ** 2).sum(1)).sum()
    t = muA - s * (R @ muB)
    return s, R, t

def robust_align(A, B, iters=5, trim=0.25):
    """IRLS/trimmed: refit while down-weighting the worst residuals so big drawing
    errors don't drag the alignment (design principle #3)."""
    w = np.ones(len(A))
    for _ in range(iters):
        s, R, t = similarity_procrustes(A, B, w)
        r = np.linalg.norm(A - (s * (R @ B.T).T + t), axis=1)
        cutoff = np.quantile(r, 1 - trim)
        w = np.where(r <= cutoff, 1.0, cutoff / np.maximum(r, 1e-9))
    return s, R, t
```

### 9.2 Face coordinate frame (`frame.py`)

After alignment, express residuals in the *reference's* face frame: origin = face centroid;
y-axis = midline direction (line through nose-bridge and chin landmarks); x-axis ⊥;
unit length = head height (chin → top of forehead/oval). All magnitudes are reported in
this frame → "% of head height," size- and tilt-invariant.

### 9.3 Landmark residual decomposition (`measure/landmarks.py`)

Per semantic group (left eye, right eye, nose, mouth, jaw…): mean residual vector of the
group in the face frame → split into vertical/horizontal components → one Finding per
component over threshold. Per-group scale residual (group's internal spread ratio sketch
vs reference) → "too large / too small."

### 9.4 Proportion ratios and angles (`measure/proportions.py`, `angles.py`)

Compute each canon ratio **in both images** and critique the *difference* — the target is
"match the reference," never "match the textbook" (handles non-canonical references).
v1 ratio set: eye-line height / head height; face thirds; interocular distance / eye width;
nose length / face height; mouth width / interocular. Angles: PCA/least-squares line through
the relevant landmark pairs (eye line, mouth line, jaw tangents); report Δ in degrees.

### 9.5 Evaluation (`evaluate.py`)

`score = importance_weight[feature] × (magnitude / severity_unit)`. Initial weights
(tune in M5 against human redlines): eyes 1.0, mouth 0.8, nose 0.7, face oval 0.7,
brows 0.5, ears/hairline 0.3. Sort findings by (Level asc, score desc). Accuracy score:
`100 × exp(-k × Σ scores)`, k chosen so a typical first-attempt sketch lands ~55–70.

---

## 10. Data

- **300-W / WFLW** — 68/98-point annotated faces (M0 fixtures, M2 fine-tune source).
- **300W-LP** — same faces across labeled yaw angles (M1.5 pose tests).
- **`synth/sketchify.py`** — XDoG (implement directly in OpenCV: difference-of-Gaussians
  with thresholding) and optionally HED edge detection, to turn any annotated photo into a
  sketch-styled training/eval image with free landmark labels.
- A small hand-collected set (~20) of real internet line drawings for the M2 detection
  report. Keep everything in `data/`, gitignored, with a `data/README.md` listing sources.

---

## 11. Critique templates (`critique.py`)

One template per (finding id × axis), filled from the Finding only. Examples:

- `left_eye_vertical`: "The left eye sits {magnitude:.0f}% of head height {direction} —
  drop it toward the eye line." (severity ≥ mistake appends: "Fix this before refining details.")
- `eye_line_angle`: "The eye line is tilted {magnitude:.0f}° {direction} relative to the reference."
- `pose_yaw` (GLOBAL): "The whole head is rotated about {magnitude:.0f}° further
  {direction} than the reference — re-check your big shapes before adjusting features.
  The remaining notes below assume the angle you drew."
- `jaw_contour_bulge`: "The jaw contour {direction} between {evidence.anchor_a} and
  {evidence.anchor_b}."

Tone rules: direct, specific, never insulting, one actionable instruction per sentence,
no hedging words ("maybe", "slightly off"). The geometry earns the confidence.

---

## 12. Pitfalls — do NOT do these

- Do not fit affine/projective/non-rigid transforms during alignment (principle #2).
- Do not let a VLM or LLM produce or modify any measurement (principle #1).
- Do not reconstruct 3D from the sketch in any form (principle #4).
- Do not report many correlated local findings when one GLOBAL finding explains them —
  pose first, then residuals.
- Do not compare ratios against textbook canon instead of the reference.
- Do not show findings below the detector-noise floor (M2-T2 calibrates it).
- Do not skip a milestone's acceptance tests to "come back later."
- Do not add scope (figures, shading, photographed pencil) before M5 ships.

---

## 13. Glossary

- **Procrustes / similarity transform** — best-fit translation+rotation+uniform-scale
  between point sets; the residual after it is, by definition, the drawing error.
- **Face frame** — coordinate system anchored to the reference face (§9.2); makes all
  magnitudes size/tilt-invariant and human-readable.
- **CPD** — Coherent Point Drift, probabilistic non-rigid point-set registration; used only
  for *correspondence* (finding where landmarks are), never for alignment.
- **XDoG** — eXtended Difference-of-Gaussians; cheap photo→line-drawing conversion.
- **300W-LP** — face landmark dataset with large-pose (yaw-labeled) variants.
- **Finding** — one atomic, evidenced, severity-tiered critique (schema §6).
