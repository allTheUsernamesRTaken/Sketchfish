# Agent guide — Art Stockfish

Canonical context for any agent (Claude Code, Codex, etc.) working in this repo.
**Read this, then the two docs below, before writing code.**

## Required reading
1. `ART_STOCKFISH_SPEC.md` — what to build and the non-negotiable design principles (§2).
2. `IMPLEMENTATION_PLAN.md` — build order, what's parallel, and the copy-paste task prompts.

Your specific task, file ownership, and acceptance tests come from a prompt in
`IMPLEMENTATION_PLAN.md`. If you were handed a prompt, it overrides general guidance here only
where it is more specific — never where this file says "do not."

## Non-negotiables (full detail in the two docs)
- **Principles in spec §2 are law.** Similarity-only alignment; no ML/LLM ever produces a
  number in a critique; 3D is attribution-only; coarse-to-fine ranking; deterministic output.
- **`schema.py` is a frozen contract.** After Wave 0 it is read-only. If it seems wrong, stop
  and report — do not edit it while others depend on it.
- **Stay in your lane.** Only create/edit the files your task prompt assigns. Need another
  file? Stop and report; don't silently reach across boundaries.
- **`config.py` is append-only, section-partitioned** (`# --- <module> ---`). Never edit
  another module's block.
- **Don't game gates.** Never weaken a test or special-case synthetic inputs to turn a number
  green. Unreachable gate → report the gap with failing cases.
- **Log deviations.** Any departure from the spec → one dated line in `DECISIONS.md` with why.

## The Loop Contract (how "done" is defined)
You are done when you have **run the code and watched it work**, not when it's written:
build → write the listed tests → run them → loop until green (and pass any "eyeball" check) →
only then hand back. Your handoff message must include: files changed, the exact command to
re-run the tests, the pasted passing output, the demo command (if any), and a one-paragraph
`README.md` progress note. Full text: see "The Loop Contract" in `IMPLEMENTATION_PLAN.md`.

## Dev setup
- Python 3.11+. Install: `pip install -e .` (or `uv pip install -e .`).
- Tests live in `tests/`. Run all: `pytest`. Run one file: `pytest tests/test_align.py -q`.
- Pure functions for geometry; type hints everywhere; magic numbers → `config.py` with a
  source comment (spec §4).
- `data/` is gitignored; never commit datasets or fixtures larger than a tiny canonical face.
