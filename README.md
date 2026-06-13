### Sketchfish

big fan of fish

<pre>

 _________         .    .
(..       \_    ,  |\  /|
 \       O  \  /|  \ \/ /
  \______    \/ |   \  / 
     vvvv\    \ |   /  |
     \^^^^  ==   \_/   |
      `\_   ===    \.  |
      / /\_   \ /      |
      |/   \_  \|      /
             \________/

</pre>

## Progress

**Wave 0 — the frozen contract + alignment spine (done).** The project's data
contract (`schema.py`: `Landmarks`, `Severity`, `Level`, `Finding`, `Report`, all frozen
dataclasses exactly per spec §6) is in place and is now read-only for every other agent. The
alignment spine is built and tested: `align.py` implements robust similarity Procrustes (§9.1)
— translation, rotation, uniform scale only, with IRLS trimming so one large drawing error
can't drag the fit — and `frame.py` builds the reference face frame (§9.2) that expresses every
residual as a size- and tilt-invariant "% of head height". Thresholds and importance weights
from §6/§9.5 live in `config.py` as cited, section-partitioned constants. `tests/test_align.py`
passes M0-T3 (a group displaced 25% of head height leaves all other features below the OK floor
under robust alignment, and demonstrates that naive least-squares smears the blame) and M0-T4 (a
7° page rotation is fully absorbed by the similarity transform, leaving zero residual). Run it
with `pytest tests/test_align.py -q`.

**Wave 1A — landmark residual decomposition (done).** `measure/landmarks.py` turns a
(reference, sketch) landmark pair into per-group `Finding` objects (spec §9.3). It robustly
aligns the sketch to the reference (similarity only, §9.1), then for each semantic group
(left/right eye, nose, mouth, jaw, brows) takes the group's mean residual vector in the
reference face frame and splits it into vertical/horizontal components — each component over
the §6 noise floor becomes one `Level.PLACEMENT` finding ("too high/low", "too far left/right")
with severity read from the shared displacement tiers. A separate per-group internal spread
ratio yields a scale finding ("too large/too small"), tiered against the area thresholds. Every
finding carries the raw points and residual vectors in `evidence` for `annotate.py`. The gates
in `tests/test_measure_landmarks.py` pass: shifting the left-eye group up 5% of head height
yields exactly one `left_eye_vertical` / "too high" finding at 5.0% ± 0.5% (measured 4.996%,
MISTAKE tier) with no other group flagged, and identical inputs yield zero findings. Run it with
`pytest tests/test_measure_landmarks.py -q`.

**Wave 1C — feature angle comparisons (done).** `measure/angles.py` fits a least-squares/PCA
line through each relevant feature in both images — the eye line (the four eye corners), the
mouth line (the two outer corners), and the left/right jaw tangents — and critiques the
*difference* in orientation in degrees (spec §9.4, "match the reference, never the textbook").
The sketch is first registered to the reference with the robust similarity transform (§9.1), so
a globally tilted page is absorbed and only a real relational/contour tilt survives. Each line
over the §6 angle floor becomes one `axis="angle"` `Finding` ("tilted clockwise / counter-
clockwise", severity from the shared 2/5/10° tiers), with the eye/mouth lines ranked `PLACEMENT`
and the jaw tangents `SHAPE`; `evidence` carries both fitted line directions and angles for
`annotate.py`. The gates in `tests/test_measure_angles.py` pass: rotating the eye-line landmarks
6° produces exactly one `eye_line_angle` finding at 6.00° (±0.5°) with the correct direction,
while a 7° whole-face page tilt yields zero findings. Run it with
`pytest tests/test_measure_angles.py -q`.

**Wave 1B — canon-ratio proportions (done).** `measure/proportions.py` computes each v1 canon
ratio (spec §9.4) — eye-line height / head height, face thirds, interocular / eye width, nose
length / face height, mouth width / interocular — in **both** images and critiques the
*difference*, so the target is always "match the reference," never "match the textbook" (handles
non-canonical references; pitfall §12). Because every ratio is a quotient of lengths it is
invariant to the similarity alignment, so each is measured in its own face frame and compared
directly — no Procrustes step needed. A ratio whose sketch value departs from the reference's
past the noise floor becomes one `axis="proportion"` `Finding` (magnitude = ratio deviation as a
percent of the reference ratio, `units="%ratio"`); the two overall-proportion cues (eye-line
height, face-thirds balance) rank `GLOBAL` and the feature-relative ratios rank `PLACEMENT`, so
findings sort coarse-to-fine. Severity uses ratio-deviation tiers (5/10/20%) added to `config.py`
under `# --- proportions ---`; `evidence` carries the two ratios, the signed deviation and the
landmark points for `annotate.py`. The gates in `tests/test_measure_proportions.py` pass: a
known 15% interocular widening (mouth widened in proportion to isolate the rule) yields exactly
one `interocular_eye_width` / "too wide" finding at 15.0% ± 0.5% (MISTAKE tier), the same
narrowing flips the sign, matched ratios yield zero findings, and a sanity test confirms the
widening genuinely couples the two interocular ratios when left uncompensated. Run it with
`pytest tests/test_measure_proportions.py -q`.

**Wave 2 — wired-up M0 critique (done; M0 COMPLETE).** The measurement modules are now an
end-to-end coach. `pipeline.py` runs one shared robust similarity alignment and feeds it to all
three measurement modules, then `evaluate.py` ranks the findings coarse-to-fine (`Level` asc,
then `score` desc — global proportion/pose before feature placement before local shape, spec
§9.5/principle #5) and rolls their scores into the aggregate accuracy "eval bar"
(`100·exp(-k·Σscore)`, `k=0.04`, which lands the realistic demo at **60/100**, inside the spec's
55–70 target). `critique.py` turns each finding into one teacher-voiced sentence from the
finding alone (template per axis with per-id action verbs; MISTAKE/BLUNDER append "Fix this
before refining details." — spec §11; no hedging, one instruction per sentence), and
`annotate.py` renders a matplotlib overlay (reference vs. aligned sketch, correction
arrows/circles/line-pairs coloured by severity with `!?`/`?`/`??` badges; matplotlib is
lazy-imported so the measure/evaluate/critique path never requires it). A key integration step,
`pipeline.suppress_explained_angles`, removes angle findings that are merely *symptoms* of a
placement already reported (e.g. a single eye shifted up tilts the eye line; reporting both
double-counts one mistake) via a counterfactual — correct the flagged feature positions and
re-measure; a tilt that collapses was explained, one that survives (a feature rotated in place,
a jaw tangent) is kept (spec §2 principle #5, pitfall §12; see `DECISIONS.md`). The M0 gates in
`tests/test_m0_acceptance.py` pass: a 5%-of-head-height left-eye shift yields **exactly one**
`left_eye_vertical` / "too high" finding at 5.0% (MISTAKE) with the induced eye-line tilt
explained away, and identical inputs yield zero findings with accuracy 100; `tests/test_evaluate.py`
pins the ranking, the accuracy formula/calibration, and report assembly. The usefulness eyeball
reads like a teacher, not a coordinate diff — the demo's top three are *"the midface is 23% too
tall — shorten the midface"* (GLOBAL), *"the left eye sits 6% too high"*, *"the right eye is
drawn 21% too large"*, surfacing the structural error before the feature fixes. Demo:
`python -m artstockfish.cli demo-synthetic` prints the ranked critique + accuracy and saves the
annotated overlay PNG. Run the gates with `pytest tests/test_m0_acceptance.py tests/test_evaluate.py -q`.

**Wave 3A — M1 synthetic distortion harness (done).** `synth/distort.py` now provides labeled,
parameterized landmark distortions for the M1 conscience test: `shift_feature` translates a
semantic group in the reference face frame and labels the expected placement finding,
`scale_feature` scales a feature about its centroid and labels the expected `%area` scale
finding, `rotate_line` rotates configured line features and labels angle findings, `tps_bulge`
adds a smooth local contour-era bulge label for later M3 tests, and `compose` chains operations
while preserving their ground-truth labels. `tests/test_harness.py` runs 200 deterministic
single- and multi-error cases through the read-only M0 pipeline and scores every surfaced
finding against those labels: **precision 0.989**, **recall 1.000**, **median magnitude error
0.000** (fractional error; gate is <=0.200). `tests/test_stability.py` covers M1-T4 with 20
Gaussian-jittered runs at sigma = 0.5% head height and requires the modal `(finding id,
severity)` signature to appear at least 95% of the time. Run the gates with
`pytest tests/test_harness.py tests/test_stability.py -q -s`.

**Wave 4 — M2 real detection on sketches (done).** The coach now runs on real image files:
`artstockfish critique ref.jpg sketch.png`. Following the de-risk report's path choice
(`data/detection_report.md`, PATH 2), MediaPipe FaceLandmarker detects the **reference photo**
(478-pt mesh downsampled to the frozen 68-pt convention), while the **sketch** is read by
classical **CPD landmark transfer** (`detect/cpd_register.py`): XDoG edges of the head-cropped
reference are registered onto the sketch's skeletonized strokes — similarity CPD, then a
numerically-guarded deformable CPD (σ² floor/init + min-norm solve; vanilla pycpd diverges on
well-matched clouds), then a per-feature local similarity refinement (with normal-only
corrections for aperture-limited arcs like the jaw) — and the reference's landmarks are read
off where they land. MediaPipe is only an opportunistic sketch fast-path, gated on per-group
agreement with the CPD answer (the report's la14 junk-mesh case defeats ink-based checks —
measured). Findings from detected landmarks pass a **calibrated detection noise floor**
(M2-T2: two random XDoG re-renders of the same image → **zero** findings; floors 4 %hh /
5° / 10° pose / 75 %area / 35 %ratio sit above the worst observed jitter, which mutes
scale/proportion sensitivity in detect mode — see DECISIONS.md). The M2-T1 gate runs
end-to-end on (photo, TPS-warped + XDoG-sketchified) pairs with labeled blunder-tier errors
across every usable front-facing photo in `data/photos`: **detection precision 0.862,
recall 0.893** (gates ≥ 0.85), vs **synthetic-only precision 0.750, recall 0.964, median
magnitude error 0.010** on the same cases with the same floors (the synthetic "false
positives" are true secondary consequences of blunder-sized injections; M1-magnitude
headline numbers remain in `tests/test_harness.py`). `synth/sketchify.py` provides the
validated XDoG recipe plus the landmark-driven TPS image warp that turns any annotated photo
into a distorted eval sketch with free labels. Run the gates with
`pytest tests/test_detect.py -q -s` (needs the gitignored `data/` corpus + model bundle;
~10 min). Demo: `artstockfish critique data/demo_ref.jpg data/demo_sketch.png` reports
exactly the injected error — "The mouth sits 6% of head height too low" — and saves the
annotated overlay.

**Wave 3B — M1.5 head-pose attribution (done).** `measure/pose.py` adds the pose layer that
keeps a turned head from masquerading as a storm of local errors (spec §8 M1.5, principle #4).
It carries a fixed canonical 3D 68-point model (an ellipsoidal-head depth profile over the
project's canonical face — attribution scaffolding, never a fit to the sketch) and estimates
head pose for the reference and the sketch *independently* with `cv2.solvePnP`. The solve is
robust and deterministic: a global `SQPNP` initial solve picks the upright, camera-facing branch
(plain iterative PnP flips a near-frontal face on the planar two-fold ambiguity), then a *trimmed*
re-solve drops the worst-fitting 25% of points so a single mislocated feature can't drag the
global pose — the same robustness logic `robust_align` applies in 2D (principle #3). When the
heads' yaw/pitch differ past the pose noise floor (`POSE_DIFF_OK_MAX = 4°`; in-plane roll is page
tilt and stays absorbed by the similarity alignment, principle #2), the pipeline's single
pose-stage hook emits **one** `Level.GLOBAL` pose finding ("the head is rotated N° further right
than the reference") and **reprojects the reference at the student's pose** — identity-preserving,
so a perfect-but-rotated student reprojects exactly onto the sketch — and runs every downstream
residual against that reprojection. Below threshold the stage is a no-op and the M0 path is byte
-for-byte unchanged. The gates in `tests/test_pose.py` pass: a frontal reference vs a +10°-yaw
sketch yields **exactly one** GLOBAL pose finding and **zero** placement findings (M1.5-T1, with
the reprojection landing on the sketch to <1px), and a +10° yaw *plus* a left eye shifted up 5%
surfaces **both** the pose finding (10.0°) **and** `left_eye_vertical` at **5.0% ± 1%** after pose
conditioning (M1.5-T2, measured 5.01%), ranked coarse-to-fine with pose first. Run the gates with
`pytest tests/test_pose.py -q`.
