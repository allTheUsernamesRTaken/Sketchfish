"""The verbalizer's hard guard (spec §8 M5, §2 principle #1).

The optional LLM verbalizer may only *paraphrase* already-measured findings. These
tests pin the contract that is enforced in code: a rewritten sentence that mentions a
feature, direction, or number the finding doesn't support is rejected and the
deterministic template is used instead. The headline case the spec calls out — feed
it a hallucinated sentence and assert it falls back — is :func:`test_hallucinated_*`.
"""

from __future__ import annotations

from artstockfish.critique import (
    critique_finding,
    critique_report,
    report_verbalizer_payload,
    verbalize_report,
    verbalizer_violation,
)
from artstockfish.evaluate import build_report
from artstockfish.pipeline import critique_pair, demo_synthetic_pair
from artstockfish.schema import Finding, Level, Severity


def _finding(
    fid: str,
    feature: str,
    axis: str,
    direction: str,
    magnitude: float,
    units: str,
    level: Level = Level.PLACEMENT,
    severity: Severity = Severity.MISTAKE,
) -> Finding:
    return Finding(
        id=fid, level=level, severity=severity, feature=feature, axis=axis,
        direction=direction, magnitude=magnitude, units=units, score=1.0, evidence={},
    )


def _report(*findings: Finding):
    return build_report(list(findings), transform={})


_LEFT_EYE = _finding("left_eye_vertical", "left eye", "vertical", "too high", 5.93, "%head_height")


def test_hallucinated_sentence_falls_back_to_template():
    """A sentence inventing features and a number is rejected; the template is used."""
    report = _report(_LEFT_EYE)
    template = critique_report(report)[0]
    hallucination = "Your ears are drawn 12% too small and the nose looks short."

    # The guard flags it on its own...
    assert verbalizer_violation(hallucination, _LEFT_EYE) is not None
    # ...and verbalize_report falls back to the template, never surfacing the lie.
    out = verbalize_report(report, llm=lambda payload: [hallucination])
    assert out == (template,)
    assert hallucination not in out


def test_clean_paraphrase_passes_through():
    report = _report(_LEFT_EYE)
    clean = "Your left eye sits about 6% of head height too high — gently bring it down to the eye line."
    assert verbalizer_violation(clean, _LEFT_EYE) is None
    out = verbalize_report(report, llm=lambda payload: [clean])
    assert out == (clean,)


def test_invented_number_falls_back():
    report = _report(_LEFT_EYE)
    template = critique_report(report)[0]
    bad = "The left eye is 40% too high — bring it down."  # 40 ≠ 6
    assert verbalizer_violation(bad, _LEFT_EYE) is not None
    assert verbalize_report(report, llm=lambda p: [bad]) == (template,)


def test_off_axis_direction_falls_back():
    report = _report(_LEFT_EYE)
    template = critique_report(report)[0]
    bad = "The left eye is too wide — narrow it."  # width axis ≠ vertical finding
    assert verbalizer_violation(bad, _LEFT_EYE) is not None
    assert verbalize_report(report, llm=lambda p: [bad]) == (template,)


def test_granular_fallback_only_bad_sentence():
    """A clean rewrite is kept; a sibling hallucination falls back to its own template."""
    mouth = _finding("mouth_line_angle", "mouth line", "angle", "tilted clockwise", 7.0, "deg")
    report = _report(_LEFT_EYE, mouth)
    templates = critique_report(report)
    clean_eye = "Your left eye is a touch high — about 6% of head height; ease it down."
    bad_mouth = "The jaw is 9% too wide."  # wrong feature + wrong axis + wrong number

    # Map each finding (in ranked order) to a candidate.
    by_id = {"left_eye_vertical": clean_eye, "mouth_line_angle": bad_mouth}
    out = verbalize_report(report, templates, llm=lambda payload: [by_id[f["id"]] for f in payload])

    eye_idx = [f.id for f in report.findings].index("left_eye_vertical")
    mouth_idx = [f.id for f in report.findings].index("mouth_line_angle")
    assert out[eye_idx] == clean_eye
    assert out[mouth_idx] == templates[mouth_idx]  # fell back


def test_llm_error_and_shape_mismatch_fall_back_to_all_templates():
    report = _report(_LEFT_EYE)
    templates = critique_report(report)

    def boom(payload):
        raise RuntimeError("LLM down")

    assert verbalize_report(report, templates, llm=boom) == templates
    # Wrong count → don't trust any of it.
    assert verbalize_report(report, templates, llm=lambda p: ["a", "b"]) == templates


def test_payload_carries_only_findings_no_images():
    report = _report(_LEFT_EYE)
    payload = report_verbalizer_payload(report, critique_report(report))
    assert payload and set(payload[0]) == {
        "id", "feature", "axis", "direction", "magnitude", "units", "level", "template_sentence"
    }
    # No image/geometry leakage into what the verbalizer LLM sees.
    blob = repr(payload).lower()
    assert "evidence" not in blob and "points" not in blob and "image" not in blob


def test_every_real_template_passes_its_own_guard():
    """Regression: the deterministic templates must never trip the guard themselves.

    Otherwise a "clean" paraphrase identical to the template could be wrongly rejected.
    Runs on a realistic multi-finding report (pose-free synthetic demo).
    """
    ref, sketch = demo_synthetic_pair()
    result = critique_pair(ref, sketch)
    assert result.report.findings  # the demo produces several findings
    for finding, sentence in zip(result.report.findings, result.sentences):
        assert verbalizer_violation(sentence, finding) is None, (finding.id, sentence)
