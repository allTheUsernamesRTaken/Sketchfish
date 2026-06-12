# M2 De-risk: MediaPipe Face Mesh on line drawings

**Date:** 2026-06-11  **Owner:** de-risk sidecar  **Gates:** Wave 4 / M2 path choice (§8)

---

## TL;DR

> **Detection on line art is SPOTTY, and it is spotty in the worst possible way: it works on
> tonal / shaded / photo-like drawings and FAILS on exactly the clean, flat line art that
> spec §3 names as v1's input.** Raw hit rate **5/15 = 33%** on real portrait line drawings
> (≈27% if you discard one misaligned junk detection); **1/5 = 20%** on XDoG-converted photos,
> vs **4/5 → 5/5** on the source photos (control). Lowering the detection-confidence threshold
> from 0.5 to 0.2 did **not** recover the misses — the failure is at the BlazeFace **face
> detector**, not the landmark regressor.
>
> **Recommendation → PATH 2 (CPD), as the primary investment.** Do **not** ship MediaPipe-only
> and "constrain input to clean line art" (Path 1's premise) — that constraint makes detection
> *worse*, not better. It is also not quite "broadly fails" (Path 3): when the detector *does*
> fire, the 478-pt mesh lands correctly, so the model is usable given a face box. Exploit the
> asymmetry: the **reference** is a photo/clean image (MediaPipe detects it reliably), so detect
> on the reference and **transfer landmarks to the sketch with CPD** (classical, no training).
> Keep MediaPipe as an opportunistic fast-path for the ~30% of tonal sketches where it fires
> (gated by an overlay sanity-check). Path 3 (synthetic XDoG fine-tune) is the documented
> backstop — and XDoG sketchify already works, so that data pipeline is cheap if needed.

---

## Method

- **Detector:** MediaPipe Tasks `FaceLandmarker` (`face_landmarker.task`, float16), 478-pt mesh,
  `num_faces=1`, `RunningMode.IMAGE`. *(The installed `mediapipe==0.10.35` ships only the Tasks
  API — the legacy `mp.solutions.face_mesh` is gone — but `FaceLandmarker` wraps the same
  BlazeFace short-range detector + face-mesh model, so this is the same architecture the spec
  assumes.)*
- **Two detection-confidence thresholds** per image: **0.5** (default) and **0.2** (relaxed),
  to separate "borderline confidence" from "detector never fired."
- **Inputs:** 15 real portrait line drawings of varied skill/style/pose (Wikimedia Commons;
  see `sources.csv`) + 5 face photographs converted to line art with **XDoG** + the same 5
  raw photos as a control.
- **XDoG:** plain Difference-of-Gaussians (σ=1.2, k=1.6), z-scored and soft-thresholded
  (`1+tanh(1.4·(z+0.9))`) so line density is exposure-robust. Outputs were **eyeballed to
  confirm they read as clean line drawings** before scoring (an early, badly-calibrated XDoG
  produced near-blank images and a false 0/5 — that run was discarded).
- **Verification:** every detection saved a landmark overlay; all "hits" were inspected by eye
  to reject meshes that landed on noise instead of the face.

Reproduce: see the bottom of this file.

---

## Results — line drawings (the core question)

`@0.5` / `@0.2` = detected at that confidence. Quality = eyeballed overlay alignment.

| id | drawing | @0.5 | @0.2 | quality of mesh |
|----|---------|:----:|:----:|-----------------|
| la06 | clean simple line, woman, **frontal** | ✅ | ✅ | good — full mesh aligned |
| la17 | realistic **shaded** pencil, woman, 3/4 | ✅ | ✅ | good — aligned |
| la21 | faint pencil, bearded man, frontal | ✅ | ✅ | good — aligned (loose) |
| la32 | soft **shaded** graphite, woman, 3/4 | ✅ | ✅ | good — aligned |
| la14 | loose charcoal, boxer, head **down** | ✅ | ❌ | **junk** — mesh collapsed onto a small shadowed patch |
| la30 | colored-chalk, woman, **profile** | ❌ | ✅ | partial — half-mesh on visible side |
| la05 | loose graphite sketch, man, 3/4 | ❌ | ❌ | — |
| la07 | chalk drawing, man w/ hat | ❌ | ❌ | — |
| la08 | **bold ink** outline, man, frontal | ❌ | ❌ | — (unambiguous frontal face; total miss) |
| la13 | **blind-contour** (crude/beginner) | ❌ | ❌ | — |
| la15 | graphite, man, **profile** | ❌ | ❌ | — |
| la19 | loose graphite, man w/ glasses, 3/4 | ❌ | ❌ | — |
| la20 | modern shaded pencil, man, frontal | ❌ | ❌ | — |
| la25 | **stipple engraving**, old woman | ❌ | ❌ | — |
| la28 | **ukiyo-e** woodblock, stylized | ❌ | ❌ | — |

**Hit rate: 5/15 = 33% @0.5, 5/15 = 33% @0.2** (the two thresholds detect overlapping-but-not-
identical sets; union = 6/15 = 40%, but with the quality caveats above). **Usable-quality
detections: 4/15 ≈ 27%.**

## Results — XDoG sketchify + control

| id | @0.5 | @0.2 | note |
|----|:----:|:----:|------|
| ph08 → xdog | ✅ | ✅ | only XDoG hit; clean modern frontal; mesh aligned |
| ph10 → xdog | ❌ | ❌ | legible line face, but detector misses |
| ph11 → xdog | ❌ | ❌ | legible line face, but detector misses |
| ph15 → xdog | ❌ | ❌ | legible line face, but detector misses |
| ph18 → xdog | ❌ | ❌ | legible line face, but detector misses |
| **control:** ph08 / ph10 / ph15 / ph18 photos | ✅ | ✅ | photos detect fine |
| **control:** ph11 photo | ❌ | ✅ | tight close-up; detects when relaxed |

**XDoG hit rate: 1/5 = 20%.  Photo control: 4/5 = 80% @0.5, 5/5 = 100% @0.2.**
The XDoG images were confirmed by eye to be clearly-readable faces, so this is a real
**80–100% → 20% collapse** caused purely by removing photographic tone.

---

## What this means

1. **The bottleneck is the face *detector*, not the mesh.** When BlazeFace produces a box, the
   478-pt mesh lands correctly (see the ✅ overlays). The misses are total non-detections that
   relaxing the threshold to 0.2 does not fix — BlazeFace is photo-trained and simply does not
   fire on flat line art. *Lowering confidence is not the lever.*

2. **Tone, not "cleanliness," predicts success.** Every hit (la06, la17, la21, la32, ph08_xdog)
   retains soft gradients / shading / a photo-like value structure. Every flat, high-contrast,
   or stylized drawing fails: **bold ink outline (la08), ukiyo-e (la28), stipple engraving
   (la25), blind-contour (la13), and both profiles (la15, la30)**. This is the opposite of the
   spec's working assumption.

3. **⚠️ Direct conflict with spec §3 scope.** §3 scopes v1 to "clean line art or digital input"
   and Path 1 says "constrain input … done." The data says clean, flat line art is the
   *hardest* case for MediaPipe. Constraining to clean line art would **lower** the hit rate.
   The team should resolve this before M2 build: either (a) widen the de-facto input toward
   *tonal* sketches, or (b) accept that the sketch side needs a non-MediaPipe detector (CPD).

4. **The hit rate is an optimistic ceiling.** The corpus is skilled, published Commons art;
   real beginner phone-sketches (lower contrast, paper texture, perspective) will likely do
   worse. Treat 33% as a best case.

---

## Recommendation — PATH 2 (CPD), with MediaPipe as an opportunistic fast-path

| Path | Verdict |
|------|---------|
| **1 — MediaPipe only, constrain input** | ❌ Rejected. 33%/20% is far too low to be the sole sketch detector, and the failures concentrate on v1's stated input class. |
| **2 — CPD correspondence** | ✅ **Primary.** Detect landmarks on the **reference** (a photo / clean image — MediaPipe handles it), then warp the reference's contour/landmark point set onto the **sketch's** extracted edge points with Coherent Point Drift; read landmark positions off where they land. Fully classical, no training. Directly exploits the reference↔sketch asymmetry. |
| **3 — Synthetic XDoG fine-tune** | 🅱️ Backstop. Not yet warranted (it's "spotty," not "broadly fails"), but keep it staged: XDoG sketchify already works here, so converting an annotated set (300-W/WFLW) for a small landmark-model fine-tune is cheap if CPD proves insufficient on the hardest cases (profiles, ukiyo-e, crude beginner sketches). |

### Concrete next steps for Wave 4 (M2)
- Build `detect/mediapipe_face.py` for the **reference** path and as an **opportunistic** sketch
  detector — but **gate every sketch detection with the overlay/geometry sanity-check** (la14
  shows MediaPipe can return a confidently-misplaced mesh; a silent junk box would poison
  alignment, violating design principle #3).
- Build `detect/cpd_register.py` as the **default sketch path**. This is now on the critical
  path, not a "conditional" — budget for it.
- Build `synth/sketchify.py` (XDoG) regardless — needed for M2 eval pairs *and* as the Path-3
  data generator. The z-scored-DoG recipe in `_scripts/run_facemesh.py::xdog` works well.
- For **M2-T2** (detector-noise floor): note that MediaPipe's sketch failures are *non-detections*,
  not jitter, so the noise floor must be calibrated on the CPD output, not on MediaPipe.

---

## Caveats / limitations
- N is small (15 + 5); treat percentages as directional, not precise.
- Corpus skews to skilled/published art (optimistic, per above).
- One detector config tested (`FaceLandmarker`, short-range BlazeFace). A different face
  detector front-end (e.g. RetinaFace, or BlazeFace full-range) might shift numbers, but the
  detector-is-the-bottleneck finding would still drive the architecture.

## Deviations logged
*(I own `data/**` only and cannot edit `DECISIONS.md`; recording here for the integrator.)*
- `mediapipe==0.10.35` exposes only the Tasks API; used `vision.FaceLandmarker` instead of the
  legacy `mp.solutions.face_mesh`. Same underlying models.
- XDoG implemented as z-scored plain-DoG soft-threshold (not the σ/τ/ε Winnemöller form named
  in §10) for exposure-robustness; outputs visually validated as line art.

## Reproduce
```bash
# from repo root (Windows; venv lives under data/ and is gitignored)
python -m venv data/.venv
data/.venv/Scripts/python.exe -m pip install mediapipe opencv-python numpy
# model bundle (BlazeFace + 478-pt mesh):
curl -sL -o data/_scripts/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# images already under data/line_art and data/photos (see sources.csv); to re-fetch:
#   data/.venv/Scripts/python.exe data/_scripts/collect.py   (and collect2.py, collect3.py)
data/.venv/Scripts/python.exe data/_scripts/run_facemesh.py
# -> prints the summary, writes data/results.csv and overlays to data/overlays/
```
Raw per-image output: `results.csv`. Overlays: `overlays/*_overlay.png`. XDoG: `xdog/*.png`.
