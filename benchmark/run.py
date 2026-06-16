"""Run the M5 benchmark and publish the comparison table (spec §8 M5).

Usage::

    python -m benchmark.run --provider anthropic     # real frontier-VLM baseline
    python -m benchmark.run --provider none          # our system only (no API needed)

The runner builds the fixed triples, runs our deterministic pipeline and (optionally)
the VLM over 3 repeats, scores both identically, prints the table, writes the raw
results JSON, and splices the table into README.md between the BENCHMARK markers.

The VLM path needs the ``anthropic`` SDK and ``ANTHROPIC_API_KEY``; responses are
cached on disk so re-runs are free. With ``--provider none`` only our (real,
deterministic) column is produced and the VLM column is left pending.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from artstockfish import config
from artstockfish.pipeline import critique_pair

from . import render
from ._env import load_dotenv
from .dataset import Triple, build_dataset
from .scoring import GroundTruthFinding, ReportedFinding, SystemScore, score_system
from .vlm import AnthropicVLM, OpenAIVLM, VLMClient

# VLM backend → (client factory, default model, env var the SDK needs).
_PROVIDERS = {
    "anthropic": (AnthropicVLM, config.BENCH_VLM_MODEL, "ANTHROPIC_API_KEY"),
    "openai": (OpenAIVLM, config.BENCH_OPENAI_MODEL, "OPENAI_API_KEY"),
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_BENCH_START = "<!-- BENCHMARK:START -->"
_BENCH_END = "<!-- BENCHMARK:END -->"

# Per-run-by-case findings: results[repeat][case] -> list[ReportedFinding].
RunMatrix = list[list[list[ReportedFinding]]]


def ground_truth(triple: Triple) -> list[GroundTruthFinding]:
    return [
        GroundTruthFinding(id=e.id, direction=e.direction, magnitude=float(e.magnitude))
        for e in triple.expected
    ]


def our_system_findings(triple: Triple) -> list[ReportedFinding]:
    """Our deterministic pipeline's findings for one triple."""
    report = critique_pair(triple.reference, triple.sketch).report
    return [
        ReportedFinding(id=f.id, direction=f.direction, magnitude=float(f.magnitude))
        for f in report.findings
    ]


def run_our_system(triples: list[Triple], repeats: int) -> RunMatrix:
    # Deterministic: every repeat is identical (and we measure that as consistency=1.0).
    one = [our_system_findings(t) for t in triples]
    return [[list(case) for case in one] for _ in range(repeats)]


def run_vlm(
    triples: list[Triple],
    vlm: VLMClient,
    repeats: int,
    *,
    workers: int = config.BENCH_VLM_WORKERS,
) -> RunMatrix:
    """Sample the VLM `repeats` times per case. Calls are independent, so they run
    concurrently (cache hits short-circuit; cache writes go to distinct files)."""
    rendered = [render.render_pair(t.reference.points, t.sketch.points) for t in triples]
    matrix: RunMatrix = [[[] for _ in triples] for _ in range(repeats)]
    tasks = [(r, c) for r in range(repeats) for c in range(len(triples))]

    def work(rc: tuple[int, int]):
        r, c = rc
        ref_png, sketch_png = rendered[c]
        return rc, vlm.critique(ref_png, sketch_png, case_id=triples[c].case_id, repeat=r)

    if workers <= 1:
        results = (work(rc) for rc in tasks)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(work, tasks))
    for (r, c), found in results:
        matrix[r][c] = found
    return matrix


def run_benchmark(
    triples: list[Triple],
    vlm: VLMClient | None,
    repeats: int,
    *,
    workers: int = config.BENCH_VLM_WORKERS,
) -> tuple[SystemScore, SystemScore | None]:
    expected = [ground_truth(t) for t in triples]
    ours = score_system(expected, run_our_system(triples, repeats))
    vlm_score = (
        score_system(expected, run_vlm(triples, vlm, repeats, workers=workers))
        if vlm is not None else None
    )
    return ours, vlm_score


# --- presentation ------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_table(ours: SystemScore, vlm: SystemScore | None, vlm_label: str) -> str:
    """The headline comparison table as GitHub-flavored markdown."""
    pending = "_pending — run `python -m benchmark.run --provider openai` (or `anthropic`)_"

    def col(score: SystemScore | None, key: str) -> str:
        if score is None:
            return pending
        if key == "precision":
            return _fmt_pct(score.precision)
        if key == "recall":
            return _fmt_pct(score.recall)
        if key == "localization":
            return _fmt_pct(score.localization)
        if key == "magnitude":
            return f"{score.median_magnitude_error * 100:.1f}% (median abs error)"
        if key == "consistency":
            return f"{score.consistency:.3f}"
        return ""

    rows = [
        ("Finding precision (id+direction)", "precision", "higher is better"),
        ("Finding recall", "recall", "higher is better"),
        ("Localization (right feature)", "localization", "higher is better"),
        ("Magnitude error", "magnitude", "lower is better"),
        ("Run-to-run consistency (Jaccard, 3×)", "consistency", "1.0 = identical every run"),
    ]
    n = ours.n_cases
    lines = [
        f"Protocol: **{n} triples** (reference, distorted sketch, ground-truth findings) "
        f"× **{ours.n_repeats} repeats**. Same labeled errors, same scoring for both systems.",
        "",
        f"| Metric | Art Stockfish (ours) | Frontier VLM (`{vlm_label}`) | |",
        "|---|---|---|---|",
    ]
    for label, key, note in rows:
        lines.append(f"| {label} | {col(ours, key)} | {col(vlm, key)} | {note} |")
    return "\n".join(lines)


def splice_into_readme(table: str, readme: Path = _REPO_ROOT / "README.md") -> bool:
    """Replace the BENCHMARK-marked block in README with the table. Returns True if written."""
    text = readme.read_text(encoding="utf-8")
    if _BENCH_START not in text or _BENCH_END not in text:
        return False
    pre = text.split(_BENCH_START)[0]
    post = text.split(_BENCH_END)[1]
    block = f"{_BENCH_START}\n{table}\n{_BENCH_END}"
    readme.write_text(pre + block + post, encoding="utf-8")
    return True


def write_results(ours: SystemScore, vlm: SystemScore | None, vlm_label: str, seed: int) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "vlm_model": vlm_label,
        "our_system": dataclasses.asdict(ours),
        "vlm": dataclasses.asdict(vlm) if vlm is not None else None,
    }
    out = _RESULTS_DIR / "benchmark_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Art Stockfish vs frontier VLM benchmark (M5)")
    parser.add_argument(
        "--provider", choices=("openai", "anthropic", "none"), default="openai",
        help="VLM baseline backend; 'none' runs our system only (no API key needed)",
    )
    parser.add_argument("--cases", type=int, default=config.BENCH_N_CASES)
    parser.add_argument("--repeats", type=int, default=config.BENCH_REPEATS)
    parser.add_argument("--seed", type=int, default=config.BENCH_SEED)
    parser.add_argument("--model", default=None, help="override the provider's default model")
    parser.add_argument(
        "--reasoning-effort", default=config.BENCH_OPENAI_REASONING_EFFORT,
        choices=("minimal", "low", "medium", "high"),
        help="OpenAI reasoning effort (lower = faster/cheaper); ignored for anthropic",
    )
    parser.add_argument(
        "--workers", type=int, default=config.BENCH_VLM_WORKERS,
        help="concurrent VLM requests (wall-clock ~1/workers; cost unchanged)",
    )
    parser.add_argument("--no-readme", action="store_true", help="don't write README")
    args = parser.parse_args(argv)

    load_dotenv()  # pick up keys from a gitignored repo-root .env, if present

    vlm: VLMClient | None = None
    model_label = "—"
    if args.provider != "none":
        factory, default_model, env_var = _PROVIDERS[args.provider]
        if not os.environ.get(env_var):
            print(
                f"error: {env_var} is not set. Put it in a repo-root .env file (copy "
                f".env.example) or export it, then re-run.\n"
                f"To skip the VLM and produce only our column: "
                f"python -m benchmark.run --provider none",
                file=sys.stderr,
            )
            return 2
        model_label = args.model or default_model
        kwargs = {"reasoning_effort": args.reasoning_effort} if args.provider == "openai" else {}
        vlm = factory(model=model_label, **kwargs)

    triples = build_dataset(n_cases=args.cases, seed=args.seed)
    print(f"Running benchmark: {len(triples)} triples × {args.repeats} repeats "
          f"(provider={args.provider}, model={model_label}, workers={args.workers})...",
          file=sys.stderr)
    ours, vlm_score = run_benchmark(triples, vlm, args.repeats, workers=args.workers)

    table = format_table(ours, vlm_score, model_label)
    print("\n" + table + "\n")

    results_path = write_results(ours, vlm_score, model_label, args.seed)
    print(f"Results written to {results_path}", file=sys.stderr)
    if not args.no_readme:
        wrote = splice_into_readme(table)
        print(f"README {'updated' if wrote else 'NOT updated (markers missing)'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
