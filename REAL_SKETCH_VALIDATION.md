# Next phase — validating Art Stockfish on *real sketches*

> Context brief for a fresh agent (Claude Code/Codex) picking up Art Stockfish after M0–M5.
> **Read `AGENTS.md`, then `ART_STOCKFISH_SPEC.md` (§1–§3, §8) and `IMPLEMENTATION_PLAN.md`
> (Ground Rules + Loop Contract) before writing code.** This doc says *where the project
> actually stands*, *what is and isn't proven*, and *what to build next and why*. It does not
> override the spec's §2 principles, the frozen `schema.py`, or the file-ownership rules.

---

## 1. Where the project stands (M0–M5 are built and green)

Art Stockfish is a **face-drawing coach**: given a reference image and a student's sketch, it
produces measured, ranked, teacher-like critiques ("the left eye is 6% of head height too
high") plus an annotated overlay — see `README.md` (Progress) for the per-wave detail. The
pipeline, all tested on synthetic ground truth:

`landmarks → robust similarity alignment (Procrustes) → pose attribution (solvePnP) →
geometric measurement (residuals / ratios / angles / contours / negative space) → weighted
coarse-to-fine ranking → template critique (+ optional guarded LLM verbalizer) → SVG overlay`

The headline M5 benchmark (`benchmark/`, table in `README.md`) scored our pipeline vs a
frontier VLM (gpt-5.5) over 50 triples × 3 repeats:

| Metric | Ours | gpt-5.5 |
|---|---|---|
| Finding precision | 98.9% | 63.5% |
| Finding recall | 100% | 70.5% |
| Localization | 100% | 76.1% |
| Median magnitude error | ~0% | 4.7% |
| Run-to-run consistency (Jaccard) | 1.000 | 0.696 |

**This validated the measurement *engine*, not the product on its real input.** Read §2.

---

## 2. The two honest gaps (this is the whole reason for this doc)

### Gap A — it has never been validated on a real human sketch
Every validation so far is **synthetic on the input side**:
- M1/M5 perturb *ground-truth landmark coordinates* and measure them (the "sketch" is
  perturbed numbers, or a wireframe render of them — see `benchmark/render.py`).
- M2 turns a *photo* into a "sketch" with an XDoG/Canny edge filter + a TPS warp
  (`synth/sketchify.py`). That is a filtered photograph, **not a drawing**.

A real student sketch differs in the ways that matter: wobbly/incomplete lines, construction
marks, hatching/shading, stylization and deliberate exaggeration, omitted or implied features,
non-photometric proportions, varied media. **Critique quality on a real sketch is unmeasured.**
The benchmark measured the *ceiling*, on the system's home turf (our distortion vocabulary,
scored against our own schema), not the real task.

### Gap B — faces only
The whole stack is face-specific by design (spec §3 scopes v1 to front-facing portraits): the
68-point iBUG model, the face frame (midline/head-height in `frame.py`), the proportion canons
(`measure/proportions.py`), the 3D pose model (`measure/pose.py`). "Other objects" is an
**architecture change, not a feature** — there is no universal landmark graph. The only
class-agnostic machinery already here is `measure/contour.py` + `measure/negspace.py` (outlines
and gaps), which is the realistic seed if the project ever generalizes.

---

## 3. Detection reality check (run 2026-06-15, current code)

Real line art is the production input, and the fast-path detector mostly fails on it.
MediaPipe (`detect.detect_landmarks_68`) on all **32 `data/line_art` drawings**:

- **Hit rate: 8/32 = 25% @ conf 0.5, 10/32 = 31% @ conf 0.2.** (Confirms the de-risk's
  ~33% on 15 images, at larger N — see `data/detection_report.md`.)
- **When it fires, the mesh is correct** (overlays saved to `data/_realcheck/`; e.g. la06 lands
  cleanly). The bottleneck is the BlazeFace *detector* not firing on flat/stylized line art,
  not the landmark regressor.
- The other ~70% rely on the **CPD path** (`detect/cpd_register.py`), which warps a *reference's*
  landmarks onto the sketch's edges. CPD has M2 gates on *sketchified photos*, but has **never
  been evaluated on real drawings for critique quality** — and it needs a paired reference,
  which the `data/line_art` corpus does not have.

Reproduce: see the snippet at the bottom of this file.

---

## 4. The task: real-sketch validation (do this before any breadth)

**Goal:** turn "the engine is precise on synthetic data" into "the *product* gives correct,
useful critiques on a real drawing of a real reference." Priority order:

### Task A (highest leverage) — a small real-sketch eval set with *human* ground truth
This is the "tune against human redlines" the spec keeps deferring (§6, §9.5). It's the only
test that escapes the synthetic / home-field framing.
1. Collect ~20–50 **(reference image, real sketch of it, human critique)** triples. Sources:
   draw them, commission a few, or pair real drawings to their references. The corpus must
   include *beginner* sketches (the de-risk corpus is skilled published art — an optimistic
   ceiling; spec note §10).
2. Have a person annotate the **real** errors / write the teacher critique as ground truth
   (and ideally mark the true 68 landmarks, so detection and critique can be scored separately).
3. Run the full pipeline (`artstockfish critique ref.jpg sketch.png`) and measure two things
   *separately*: (a) does detection survive real line art, and (b) do our findings match what a
   human flags (precision/recall/localization vs human ground truth)?
   - Reuse `benchmark/scoring.py` (it's input-agnostic: feed it human-labeled `GroundTruthFinding`s).
   - This is where the §9.5 importance weights and severity thresholds finally get *calibrated*
     against humans rather than carried from the spec's initial values. **Do not tune them to
     make a number green (Ground Rule 5)** — tune to agree with human redlines, report honestly.

### Task B — harden detection on real line art (the bottleneck for shipping)
With ~70% fast-path miss rate, this is the real blocker. Options, cheapest first:
- Better-gated CPD on real sketches (validate `detect/cpd_register.py` end-to-end on Task A's
  real pairs; tune the sanity gates in `# --- detect ---` of `config.py`).
- The deferred **Path 3** (spec §8 M2): fine-tune a small sketch-domain landmark model on
  XDoG/HED-converted annotated data (`synth/sketchify.py` already produces the training images).
- Constrain input or let the user place a few anchor points as a fallback.
Detection that's wrong poisons everything downstream (principle #3) — no point critiquing
geometry you mislocated.

### Task C (only after A & B) — depth vs breadth
Recommendation: **go deep on faces** (3/4 views, expression, the "show the line" TPS correction
in spec §8 M4 stretch, calibrated weights) and ship a genuinely good face coach before
attempting other object classes. If breadth is pursued, build on `contour.py`/`negspace.py`
(class-agnostic), not the face landmark graph. The Stockfish thesis (spec §1) is *be the best
at one game first*.

---

## 5. Starting points (files, commands, data)

- **Pipeline entry:** `src/artstockfish/pipeline.py::critique_pair` (landmarks in), and
  `detect/__init__.py::critique_images` / `detect_pair` (image files in, M2 stack).
- **Detection:** `detect/mediapipe_face.py` (reference + fast-path), `detect/cpd_register.py`
  (default sketch path). `detect_landmarks_68`, `detect_reference`, `detect_sketch` are the
  public entry points. Full-res photos often miss — detect at ≤ `config.DETECT_EVAL_MAX_SIDE`
  (640) px (see `benchmark/demo_realface.py` for the downscale-then-detect pattern).
- **Scoring (reuse for the human eval):** `benchmark/scoring.py` (`GroundTruthFinding`,
  `ReportedFinding`, `score_system`).
- **Synthetic harness / sketchify:** `synth/distort.py`, `synth/sketchify.py`.
- **Data (gitignored):** `data/line_art/` (32 real drawings + `sources.csv`),
  `data/photos/` (18; ~7 front-facing usable: ph08–12, 15, 18), `data/detection_report.md`
  (the de-risk study), the MediaPipe model at `data/_scripts/face_landmarker.task`.
- **Benchmark:** `python -m benchmark.run --provider openai|anthropic|none`
  (keys via a gitignored `.env`, copy `.env.example`; `--workers` parallelizes,
  `--reasoning-effort low` is cheaper). One real-face demo:
  `python -m benchmark.demo_realface --photo data/photos/ph08.jpg`.
- **Tests:** `pytest -q` (heavy `test_detect.py` needs the `detect` extra + corpus, ~10 min).

## 6. Non-negotiables to preserve (do not relearn the hard way)
- **Spec §2 principles are law:** similarity-only alignment; **no ML/LLM ever produces a number
  in a critique** (the verbalizer is paraphrase-only, guarded in `critique.py`); 3D is
  attribution-only; coarse-to-fine ranking; deterministic output.
- **`schema.py` is frozen.** `config.py` is append-only, section-partitioned. Stay in your
  lane on file ownership. Log deviations in `DECISIONS.md`. **Don't game gates** — a real-sketch
  eval that's red is far more valuable than a synthetic one gamed green.
- Follow the **Loop Contract** (build → write tests → run → loop → hand back with pasted output).

## 7. Open decisions for the human
- Where do real sketches + human ground truth come from (draw / commission / source-and-pair)?
  This gates Task A and is a data-collection decision, not a coding one.
- Spec §3 vs reality: the de-risk found clean *flat* line art is the *hardest* case for the
  detector (opposite of the spec's assumption). Decide whether v1 targets tonal sketches, or
  commits to the non-MediaPipe detector (Task B).

---

### Reproduce the detection reality check
```python
# .venv/Scripts/python.exe - , from repo root (needs the `detect` extra + model bundle)
import glob, os
from artstockfish.detect import detect_landmarks_68, load_image
hits = sum(detect_landmarks_68(load_image(p)) is not None
           for p in glob.glob("data/line_art/*"))
print(hits, "/", len(glob.glob("data/line_art/*")))   # ~8/32 @ conf 0.5
```
