"""M4 product-surface acceptance tests (spec §8 M4).

Gate: ``POST /critique`` with two images returns **200** with a valid ``Report`` JSON
and a well-formed SVG overlay, and the SVG carries **exactly one annotation element per
surfaced finding** (the contract the web list relies on to highlight a finding by id).

The image→landmarks step is the M2 detection stack (``artstockfish.detect``), which has
its own gates in ``tests/test_detect.py`` and heavy optional deps (MediaPipe, pycpd). The
M4 surface under test here is multipart handling + ``Report`` serialization + the SVG
overlay, so the detector dependency is overridden with a deterministic stub
(``app.dependency_overrides`` — the standard FastAPI pattern). The request still posts two
real PNG files, so multipart upload, image decoding by the stub, JSON encoding and SVG
rendering are all exercised end to end; only the ML detection is swapped for known
landmarks. Run: ``pytest tests/test_api.py -q``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="M4 web UI needs the 'web' extra (pip install -e .[web])")
pytest.importorskip("svgwrite", reason="M4 SVG overlay needs the 'web' extra")
pytest.importorskip("httpx", reason="FastAPI TestClient needs httpx (the 'dev' extra)")
cv2 = pytest.importorskip("cv2", reason="opencv-python is a core dependency")

from fastapi.testclient import TestClient

from artstockfish.annotate import _FACE_POLYLINES
from artstockfish.frame import SEMANTIC_GROUPS, build_face_frame
from artstockfish.schema import Landmarks
from artstockfish.server import app, get_detector

from fixtures import canonical_face_landmarks, canonical_face_points

_SVG_NS = "{http://www.w3.org/2000/svg}"


@dataclass
class _StubPair:
    """Stand-in for ``detect.DetectedPair`` — the server only reads these attributes."""

    reference: Landmarks
    sketch: Landmarks
    sketch_detector: str = "synthetic"


def _perturbed_sketch() -> Landmarks:
    """A canonical face with several independent beginner errors (≥3 findings)."""
    ref = canonical_face_points()
    frame = build_face_frame(ref)
    sk = ref.copy()
    sk[list(SEMANTIC_GROUPS["left_eye"])] += 0.06 * frame.head_height * frame.y_axis   # eye too high
    sk[list(SEMANTIC_GROUPS["nose_bridge"])] -= 0.05 * frame.head_height * frame.y_axis  # nose low
    sk[list(SEMANTIC_GROUPS["nose_bottom"])] -= 0.05 * frame.head_height * frame.y_axis
    base = canonical_face_landmarks()
    return Landmarks(points=sk, names=base.names, image_size=base.image_size)


def _render_face_png(landmarks: Landmarks) -> bytes:
    """A small white-on-black line drawing of a landmark set (a real fixture image)."""
    w, h = landmarks.image_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    pts = np.asarray(landmarks.points, dtype=np.int32)
    for idx, closed in _FACE_POLYLINES:
        cv2.polylines(canvas, [pts[list(idx)]], closed, (40, 40, 40), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok, "failed to encode fixture PNG"
    return buf.tobytes()


@pytest.fixture
def reference_sketch() -> tuple[Landmarks, Landmarks]:
    return canonical_face_landmarks(), _perturbed_sketch()


@pytest.fixture
def client(reference_sketch):
    """A TestClient whose detector returns the fixture (reference, sketch) landmarks."""
    reference, sketch = reference_sketch

    def _stub_detector():
        def _detect(reference_bytes: bytes, sketch_bytes: bytes) -> _StubPair:
            assert reference_bytes and sketch_bytes  # the uploads really arrived
            return _StubPair(reference=reference, sketch=sketch)

        return _detect

    app.dependency_overrides[get_detector] = _stub_detector
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _post_two_images(client, reference_sketch):
    reference, sketch = reference_sketch
    files = {
        "reference": ("reference.png", _render_face_png(reference), "image/png"),
        "sketch": ("sketch.png", _render_face_png(sketch), "image/png"),
    }
    return client.post("/critique", files=files)


# --- the acceptance gate ------------------------------------------------------------

def test_critique_returns_report_and_svg(client, reference_sketch):
    resp = _post_two_images(client, reference_sketch)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # A valid Report JSON: required keys, sane types, ranked findings.
    assert set(body) == {"report", "svg", "detector"}
    report = body["report"]
    assert set(report) >= {"accuracy_score", "findings", "transform", "pose"}
    assert isinstance(report["accuracy_score"], (int, float))
    assert 0.0 <= report["accuracy_score"] <= 100.0
    findings = report["findings"]
    assert isinstance(findings, list) and findings, "expected a non-empty critique"
    for f in findings:
        assert set(f) >= {
            "id", "level", "severity", "feature", "axis", "direction",
            "magnitude", "units", "score", "sentence", "evidence",
        }
        assert f["severity"] in {"inaccuracy", "mistake", "blunder"}  # OK is never shown
        assert isinstance(f["sentence"], str) and f["sentence"]
    # Ranked coarse-to-fine (Level asc) per the schema contract.
    assert [f["level"] for f in findings] == sorted(f["level"] for f in findings)

    # Well-formed SVG.
    svg = body["svg"]
    assert isinstance(svg, str) and svg.lstrip().startswith("<")
    root = ET.fromstring(svg)               # raises on malformed XML
    assert root.tag == f"{_SVG_NS}svg"

    # Exactly one annotation element per surfaced finding, ids matching the report.
    annotations = [el for el in root.iter() if el.get("data-finding-id") is not None]
    assert len(annotations) == len(findings)
    assert {el.get("data-finding-id") for el in annotations} == {f["id"] for f in findings}
    # Each annotation carries its severity badge for the pinned chess-style marker.
    for el in annotations:
        assert el.get("data-severity") in {"inaccuracy", "mistake", "blunder"}


def test_missing_image_is_rejected(client):
    """The endpoint requires both files (FastAPI validation → 422)."""
    resp = client.post("/critique", files={"reference": ("r.png", b"x", "image/png")})
    assert resp.status_code == 422


def test_index_page_served():
    with TestClient(app) as c:
        resp = c.get("/")
        assert resp.status_code == 200
        assert "Art Stockfish" in resp.text
        assert resp.headers["content-type"].startswith("text/html")
