"""Command-line entry point.

M0 ships one command::

    python -m artstockfish.cli demo-synthetic

It runs the full pipeline on a hardcoded canonical face vs. a realistically-perturbed
copy, prints the ranked critique and the accuracy score ("eval bar"), and saves the
annotated overlay PNG (spec §8 M0 demo). Real-image input arrives in M2.
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import CritiqueResult, critique_pair, demo_synthetic_pair

_SEV_BADGE = {"inaccuracy": "!?", "mistake": "?", "blunder": "??"}
_DEFAULT_OVERLAY = "artstockfish_demo_overlay.png"


def _print_result(result: CritiqueResult) -> None:
    report = result.report
    print(f"Accuracy score: {report.accuracy_score:.1f} / 100")
    print(f"Findings: {len(report.findings)} (ranked best move first)\n")
    if not report.findings:
        print("  No findings above the noise floor — this sketch matches the reference.")
        return
    for i, (finding, sentence) in enumerate(zip(report.findings, result.sentences), 1):
        badge = _SEV_BADGE.get(finding.severity.value, "")
        print(f"  {i}. [{finding.level.name:<9} {finding.severity.value:<10} {badge:>2}] {sentence}")


def cmd_demo_synthetic(args: argparse.Namespace) -> int:
    reference, sketch = demo_synthetic_pair()
    result = critique_pair(reference, sketch, overlay_path=args.out)
    print("Art Stockfish — synthetic M0 demo")
    print("=" * 60)
    _print_result(result)
    print()
    print(f"Annotated overlay saved to: {result.overlay_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="artstockfish", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo-synthetic",
        help="run the M0 critique on a synthetic perturbed face and save the overlay",
    )
    demo.add_argument(
        "--out", default=_DEFAULT_OVERLAY, help="overlay PNG output path"
    )
    demo.set_defaults(func=cmd_demo_synthetic)
    return parser


def _force_utf8_stdout() -> None:
    """Print UTF-8 regardless of console code page (the critiques use "—"/"°").

    Windows consoles default to cp1252, which can't encode the em dash / degree sign
    in the critique sentences. Reconfigure stdout/stderr to UTF-8 when possible.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
