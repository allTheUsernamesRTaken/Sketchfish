"""One-sample demo: benchmark on a REAL face instead of the canonical synthetic one.

This is the seed of a multi-face benchmark (and a cheap sanity demo). It takes a real
photo, detects its 68 landmarks with MediaPipe, injects ONE labeled distortion with the
same harness the main benchmark uses, renders the (reference, sketch) wireframe pair,
and runs both systems on it — our deterministic pipeline and one VLM call.

Using a detected real face as the reference breaks the single-face limitation of
``benchmark.dataset`` (every triple there distorts one hardcoded canonical face). The
real-face geometry was never used to fit the synthetic measurement core (it has no fit
parameters), so for that layer this is effectively held-out.

Run::

    python -m benchmark.demo_realface --photo data/photos/ph01.jpg

Needs the ``detect`` extra (MediaPipe) + the model bundle, an OpenAI key in ``.env``,
and the ``bench`` extra. Makes exactly one VLM call (cheap, low reasoning effort).
"""

from __future__ import annotations

import argparse
import sys
import time

from artstockfish.pipeline import critique_pair
from artstockfish.synth.distort import shift_feature

from ._env import load_dotenv
from .render import render_pair
from .scoring import GroundTruthFinding, ReportedFinding, score_system
from .vlm import OpenAIVLM


def _our_findings(ref, sketch) -> list[ReportedFinding]:
    report = critique_pair(ref, sketch).report
    return [ReportedFinding(f.id, f.direction, float(f.magnitude)) for f in report.findings]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One real-face benchmark sample (M5 demo)")
    parser.add_argument("--photo", default="data/photos/ph01.jpg")
    parser.add_argument("--feature", default="left_eye", help="semantic group to displace")
    parser.add_argument("--dy", type=float, default=6.0, help="vertical shift, %% head height")
    parser.add_argument("--model", default=None, help="OpenAI model (default: config)")
    parser.add_argument("--reasoning-effort", default="low",
                        choices=("minimal", "low", "medium", "high"))
    parser.add_argument("--save", default="artstockfish_bench_realface")
    args = parser.parse_args(argv)

    load_dotenv()
    import cv2

    from artstockfish import config
    from artstockfish.detect import detect_reference, load_image  # lazy: heavy deps

    print(f"Detecting landmarks on {args.photo} ...", file=sys.stderr)
    img = load_image(args.photo)
    # Detect at the M2 working size — full-res photos often miss (config note).
    h, w = img.shape[:2]
    scale = config.DETECT_EVAL_MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    reference = detect_reference(img)

    sketch, expected = shift_feature(reference, args.feature, dy=args.dy)
    gt = [GroundTruthFinding(e.id, e.direction, float(e.magnitude)) for e in expected]

    ref_png, sketch_png = render_pair(reference.points, sketch.points)
    open(f"{args.save}_ref.png", "wb").write(ref_png)
    open(f"{args.save}_sketch.png", "wb").write(sketch_png)

    ours = _our_findings(reference, sketch)

    from artstockfish import config

    vlm = OpenAIVLM(
        model=args.model or config.BENCH_OPENAI_MODEL,
        reasoning_effort=args.reasoning_effort,
    )
    t0 = time.time()
    vlm_found = vlm.critique(ref_png, sketch_png, case_id="demo_realface", repeat=0)
    dt = time.time() - t0

    def fmt(fs):
        return [f"{f.id}/{f.direction} ({f.magnitude:.1f})" for f in fs] or ["(none)"]

    print("\n=== One real-face sample ===")
    print(f"photo: {args.photo}   injected: {args.feature} dy={args.dy}%")
    print(f"renders saved: {args.save}_ref.png, {args.save}_sketch.png\n")
    print("ground truth :", fmt(gt))
    print("ours         :", fmt(ours))
    print(f"VLM ({vlm.model}, reff={args.reasoning_effort}, {dt:.1f}s):", fmt(vlm_found))

    # One-sample scores (recall/precision against the injected label).
    o = score_system([gt], [[ours]])
    v = score_system([gt], [[vlm_found]])
    print(f"\nscores (this 1 case)  ours: P={o.precision:.2f} R={o.recall:.2f} "
          f"loc={o.localization:.2f}   VLM: P={v.precision:.2f} R={v.recall:.2f} loc={v.localization:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
