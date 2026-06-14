"""FastAPI product surface (spec §8 M4).

One endpoint and one static page:

- ``POST /critique`` — multipart upload of two images (``reference`` + ``sketch``);
  returns JSON ``{"report": <Report>, "svg": <overlay>, "detector": …}``. The images
  are turned into 68-point :class:`~artstockfish.schema.Landmarks` by the M2 detection
  stack, fed through the unchanged critique pipeline, and rendered to an SVG overlay.
- ``GET /`` — the single static HTML page in ``web/`` (no framework).
- ``GET /healthz`` — liveness probe.

The image→landmarks step is the M2 detection stack (``artstockfish.detect``), which has
its own acceptance gates (``tests/test_detect.py``) and heavy optional dependencies
(MediaPipe, pycpd). It is injected here as the :func:`get_detector` dependency so the M4
product surface — multipart handling, the JSON ``Report`` serialization, and the SVG
overlay — can be exercised deterministically without that stack (``tests/test_api.py``
overrides the dependency); the default wiring runs real detection.

Run it::

    artstockfish web                # or: uvicorn artstockfish.server:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .annotate import render_svg
from .pipeline import critique_pair
from .schema import Finding, Landmarks, Report, Severity

_SEV_BADGE = {
    Severity.INACCURACY: "!?",
    Severity.MISTAKE: "?",
    Severity.BLUNDER: "??",
}


def web_dir() -> Path:
    """Directory holding the static page (repo ``web/``; overridable for deployment)."""
    override = os.environ.get("ARTSTOCKFISH_WEB_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "web"


# --- detection seam (injected; see module docstring) -------------------------------

class DetectedPairLike(Protocol):
    reference: Landmarks
    sketch: Landmarks
    sketch_detector: str


Detector = Callable[[bytes, bytes], DetectedPairLike]


def _decode_image(data: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode uploaded image")
    return img


def _real_detect(reference_bytes: bytes, sketch_bytes: bytes) -> DetectedPairLike:
    """Default detector: the M2 MediaPipe-reference + CPD-sketch stack."""
    from .detect import DetectionError, detect_pair  # lazy: heavy optional deps

    try:
        return detect_pair(_decode_image(reference_bytes), _decode_image(sketch_bytes))
    except DetectionError as exc:
        # No detectable face is a client-input problem, not a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def get_detector() -> Detector:
    """Image→landmarks backend. Overridden in tests via ``app.dependency_overrides``."""
    return _real_detect


# --- serialization -----------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / enum geometry into JSON-native values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Severity):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        import math

        return obj if math.isfinite(obj) else None
    return obj


def finding_to_dict(finding: Finding, sentence: str) -> dict:
    return {
        "id": finding.id,
        "level": int(finding.level),
        "level_name": finding.level.name,
        "severity": finding.severity.value,
        "badge": _SEV_BADGE.get(finding.severity, ""),
        "feature": finding.feature,
        "axis": finding.axis,
        "direction": finding.direction,
        "magnitude": float(finding.magnitude),
        "units": finding.units,
        "score": float(finding.score),
        "sentence": sentence,
        "evidence": _jsonable(finding.evidence),
    }


def report_to_dict(report: Report, sentences: tuple[str, ...]) -> dict:
    return {
        "accuracy_score": float(report.accuracy_score),
        "findings": [finding_to_dict(f, s) for f, s in zip(report.findings, sentences)],
        "transform": _jsonable(report.transform),
        "pose": _jsonable(report.pose),
    }


# --- app ---------------------------------------------------------------------------

app = FastAPI(
    title="Art Stockfish",
    description="A computer-vision drawing coach: reference + sketch → measured critique.",
    version="0.0.0",
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    page = web_dir() / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail=f"static page not found at {page}")
    return FileResponse(page)


@app.post("/critique")
async def critique(
    reference: UploadFile = File(..., description="reference image (photo / clean image)"),
    sketch: UploadFile = File(..., description="the student's sketch of it"),
    detector: Detector = Depends(get_detector),
) -> JSONResponse:
    """Critique a sketch against a reference: returns the Report JSON and SVG overlay."""
    pair = detector(await reference.read(), await sketch.read())

    result = critique_pair(pair.reference, pair.sketch)
    svg = render_svg(result.report, result.reference_points, result.aligned_sketch_points)

    return JSONResponse({
        "report": report_to_dict(result.report, result.sentences),
        "svg": svg,
        "detector": pair.sketch_detector,
    })
