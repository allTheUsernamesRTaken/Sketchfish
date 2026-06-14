"""End-to-end smoke test for the M5 benchmark machinery (spec §8 M5).

Runs the whole pipeline — dataset → render → VLM client → scoring → table — with a
deterministic offline :class:`StubVLM` so no API key or network is needed. The real
frontier-VLM run uses the identical code path (only the client differs), so this
proves the benchmark "runs end-to-end". It also pins the scoring metrics on hand-built
cases and the determinism of our own pipeline (consistency = 1.0).

These StubVLM numbers are a test fixture, NOT a frontier-VLM result.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from benchmark.dataset import build_dataset
from benchmark.render import render_pair
from benchmark.run import format_table, our_system_findings, run_benchmark, splice_into_readme
from benchmark.scoring import (
    GroundTruthFinding,
    ReportedFinding,
    score_system,
)
from benchmark.vlm import StubVLM
from benchmark.vocab import VALID_KEYS


# --- dataset -----------------------------------------------------------------------

def test_dataset_is_deterministic_and_labeled():
    a = build_dataset(n_cases=20, seed=123)
    b = build_dataset(n_cases=20, seed=123)
    assert len(a) == 20
    # Same seed → identical sketches and labels (principle #7).
    for ta, tb in zip(a, b):
        assert np.array_equal(ta.sketch.points, tb.sketch.points)
        assert [e.id for e in ta.expected] == [e.id for e in tb.expected]
    # Every case carries at least one ground-truth finding, all in the vocabulary.
    for t in a:
        assert t.expected
        for e in t.expected:
            assert (e.id, e.direction) in VALID_KEYS


# --- rendering ---------------------------------------------------------------------

def test_render_pair_emits_two_pngs():
    t = build_dataset(n_cases=1, seed=1)[0]
    ref_png, sketch_png = render_pair(t.reference.points, t.sketch.points)
    png_magic = b"\x89PNG\r\n\x1a\n"
    assert ref_png.startswith(png_magic) and sketch_png.startswith(png_magic)
    assert len(ref_png) > 500 and len(sketch_png) > 500


# --- scoring -----------------------------------------------------------------------

def test_scoring_perfect_and_degraded():
    expected = [
        [GroundTruthFinding("left_eye_vertical", "too high", 5.0)],
        [GroundTruthFinding("mouth_line_angle", "tilted clockwise", 7.0)],
    ]
    # A perfect predictor: exact keys + magnitudes, identical across 3 repeats.
    perfect_repeat = [
        [ReportedFinding("left_eye_vertical", "too high", 5.0)],
        [ReportedFinding("mouth_line_angle", "tilted clockwise", 7.0)],
    ]
    perfect = score_system(expected, [perfect_repeat, perfect_repeat, perfect_repeat])
    assert perfect.precision == 1.0
    assert perfect.recall == 1.0
    assert perfect.localization == 1.0
    assert perfect.median_magnitude_error == 0.0
    assert perfect.consistency == 1.0

    # An inconsistent predictor: each repeat reports a different subset → recall and
    # consistency both drop.
    r0 = [[ReportedFinding("left_eye_vertical", "too high", 5.0)], []]
    r1 = [[], [ReportedFinding("mouth_line_angle", "tilted clockwise", 7.0)]]
    r2 = [[ReportedFinding("left_eye_vertical", "too high", 5.0)], []]
    noisy = score_system(expected, [r0, r1, r2])
    assert noisy.recall < 1.0
    assert noisy.consistency < 1.0


# --- end-to-end --------------------------------------------------------------------

def _perfect_responder(triples):
    by_case = {
        t.case_id: [ReportedFinding(e.id, e.direction, float(e.magnitude)) for e in t.expected]
        for t in triples
    }
    return lambda case_id, repeat: by_case[case_id]


def _degraded_responder(triples):
    """A noisy, repeat-varying VLM: drops findings, perturbs magnitudes, hallucinates."""
    gt = {t.case_id: list(t.expected) for t in triples}
    all_keys = sorted(VALID_KEYS)

    def respond(case_id: str, repeat: int):
        seed = int(hashlib.sha256(f"{case_id}:{repeat}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        out: list[ReportedFinding] = []
        for e in gt[case_id]:
            if rng.random() < 0.35:
                continue  # missed finding
            mag = float(e.magnitude) * rng.uniform(0.6, 1.5)  # poorly calibrated magnitude
            out.append(ReportedFinding(e.id, e.direction, mag))
        if rng.random() < 0.3:  # occasional hallucination
            fid, direction = rng.choice(all_keys)
            out.append(ReportedFinding(fid, direction, rng.uniform(3, 9)))
        return out

    return respond


def test_end_to_end_pipeline_runs_and_distinguishes_systems():
    triples = build_dataset(n_cases=6, seed=7)
    repeats = 3

    perfect_vlm = StubVLM(_perfect_responder(triples))
    ours, vlm = run_benchmark(triples, perfect_vlm, repeats)

    # Our deterministic pipeline: identical every run, recovers the injected errors.
    assert ours.consistency == 1.0
    assert ours.recall >= 0.9
    assert ours.precision >= 0.9
    # A perfect VLM stub scores a clean sweep — the scoring plumbing is sound.
    assert vlm.recall == 1.0 and vlm.consistency == 1.0

    # A realistic (noisy) VLM scores strictly worse on consistency — the axis the
    # spec says deterministic measurement wins (§1).
    noisy_vlm = StubVLM(_degraded_responder(triples))
    _, noisy = run_benchmark(triples, noisy_vlm, repeats)
    assert noisy.consistency < ours.consistency
    assert noisy.recall < 1.0


def test_format_table_and_readme_splice(tmp_path):
    triples = build_dataset(n_cases=4, seed=2)
    ours, vlm = run_benchmark(triples, StubVLM(_perfect_responder(triples)), 3)

    table = format_table(ours, vlm, "claude-opus-4-8")
    assert "Art Stockfish" in table and "Frontier VLM" in table
    assert "Run-to-run consistency" in table

    # Pending VLM column when no VLM was run.
    pending = format_table(ours, None, "claude-opus-4-8")
    assert "pending" in pending

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- BENCHMARK:START -->\nold\n<!-- BENCHMARK:END -->\noutro\n", encoding="utf-8")
    assert splice_into_readme(table, readme) is True
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "Art Stockfish" in text and text.startswith("intro")
