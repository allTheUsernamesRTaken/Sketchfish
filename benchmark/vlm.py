"""The frontier-VLM baseline (spec §8 M5).

A VLM sees only the rendered images and is asked, with a strong fixed prompt, to
critique the sketch against the reference **in our JSON schema**. We constrain its
output to the closed finding vocabulary via structured outputs, so the comparison is
purely about *which findings it picks and how well it measures them* — never about
whether it guessed our id strings or emitted valid JSON.

Everything except the API call is provider-agnostic — the prompt, the closed
vocabulary, the JSON schema, parsing, and the on-disk cache all live on the shared
:class:`_CachedVLM` base. The concrete clients differ only in how they send two
images plus a system prompt and get back a JSON object:

- :class:`AnthropicVLM` — Claude (vision + structured output + adaptive thinking).
- :class:`OpenAIVLM` — an OpenAI vision model (chat completions + strict
  ``json_schema`` structured output).
- :class:`StubVLM` — a deterministic offline stand-in used by the smoke test and for
  exercising the full pipeline without network or an API key. Its numbers are **not**
  a frontier-VLM result and are never published as the headline.

Every real call is cached to disk keyed by the model, prompt version, both image
bytes, and the repeat index, so re-runs are free and reproducible, each of the 3
repeats is a genuinely independent sample, and the Anthropic and OpenAI caches never
collide (the model string is in the key).

Principle #1 is not at stake here: the VLM is the *baseline being measured*, not a
component of our system. Our own critique never consults it.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol

from artstockfish import config

from .scoring import ReportedFinding
from .vocab import FINDING_VOCAB, VALID_IDS, vocab_prompt_block

# Bump when the prompt or schema changes so stale cached responses are ignored.
PROMPT_VERSION = "m5-v1"

_SYSTEM_PROMPT = """\
You are an expert atelier drawing instructor. A student copied a reference face; you
critique their accuracy. You will receive two images, each a 68-point facial wireframe
drawn in the SAME coordinate frame:
  • IMAGE 1 — the REFERENCE (the target the student was copying).
  • IMAGE 2 — the SKETCH (the student's drawing).

Find the geometric errors in the SKETCH relative to the REFERENCE: features that are
mislocated, mis-sized, or tilted relative to the rest of the face.

Critique RELATIVE, internal geometry only — the way an instructor says "your eye sits
too high relative to the nose." Explicitly IGNORE any overall difference in position,
uniform size, or whole-page rotation between the two drawings: a student who drew the
entire face shifted, scaled, or tilted as a whole has NOT made an error. Judge how the
parts sit relative to each other and to the head's proportions.

Report findings using ONLY this fixed vocabulary, structural errors first:

{vocab}

Rules:
  • Use an id only when that specific error is visible; do not invent findings.
  • direction must be one allowed for that id; magnitude is a positive number in the
    id's stated units (%head_height = percent of head height; %area = percent area;
    deg = degrees; %ratio = percent deviation of the proportion from the reference).
  • Ignore differences below roughly 2% of head height or 2 degrees — within tolerance.
  • Prefer the single structural finding over many correlated local ones (if the whole
    head is turned, say so once rather than flagging every feature).
"""

_USER_INSTRUCTION = (
    "IMAGE 1 (above) is the REFERENCE; IMAGE 2 (above) is the student's SKETCH. "
    "List every relative geometric error in the sketch as findings in the required schema."
)


class VLMClient(Protocol):
    """Returns the findings a VLM reports for one (reference, sketch) image pair."""

    def critique(
        self, ref_png: bytes, sketch_png: bytes, *, case_id: str, repeat: int
    ) -> list[ReportedFinding]: ...


def response_schema() -> dict:
    """JSON schema constraining the VLM to the closed finding vocabulary.

    Satisfies both Anthropic's ``output_config.format`` and OpenAI's strict
    ``json_schema`` rules: every object sets ``additionalProperties: false`` and lists
    all of its properties as ``required``.
    """
    all_directions = sorted({d for e in FINDING_VOCAB for d in e.directions})
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "enum": sorted(VALID_IDS)},
                        "direction": {"type": "string", "enum": all_directions},
                        "magnitude": {"type": "number"},
                    },
                    "required": ["id", "direction", "magnitude"],
                },
            }
        },
        "required": ["findings"],
    }


def parse_findings(data: dict) -> list[ReportedFinding]:
    """Turn the model's JSON object into normalized findings (drop malformed rows)."""
    out: list[ReportedFinding] = []
    for row in data.get("findings", []):
        try:
            fid = str(row["id"])
            direction = str(row["direction"])
            magnitude = abs(float(row["magnitude"]))
        except (KeyError, TypeError, ValueError):
            continue
        if fid not in VALID_IDS:
            continue  # off-vocabulary: schema enums should prevent this, but be safe
        out.append(ReportedFinding(id=fid, direction=direction, magnitude=magnitude))
    return out


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.standard_b64encode(png).decode("ascii")


class _CachedVLM:
    """Shared prompt + schema + on-disk cache; subclasses implement ``_request``."""

    def __init__(self, model: str, cache_dir: str | Path | None = None, *, client=None):
        self.model = model
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path(__file__).resolve().parent / "_vlm_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client  # lazily constructed if None
        self.system_prompt = _SYSTEM_PROMPT.format(vocab=vocab_prompt_block())

    def _cache_tag(self) -> str:
        """Extra cache-key component for request settings that change the output.

        Empty by default so existing cache keys are preserved; a subclass returns a
        non-empty tag only when an opt-in setting (e.g. reasoning effort) is active.
        """
        return ""

    def _cache_path(self, ref_png: bytes, sketch_png: bytes, repeat: int) -> Path:
        h = hashlib.sha256()
        parts = [self.model.encode(), PROMPT_VERSION.encode(), str(repeat).encode()]
        tag = self._cache_tag()
        if tag:  # only when set, so default-setting keys never change
            parts.append(tag.encode())
        parts += [ref_png, sketch_png]
        for part in parts:
            h.update(hashlib.sha256(part).digest())
        return self.cache_dir / f"{h.hexdigest()}.json"

    def _request(self, ref_png: bytes, sketch_png: bytes) -> dict:
        """Call the provider once; return the parsed JSON object ``{"findings": [...]}``."""
        raise NotImplementedError

    def critique(
        self, ref_png: bytes, sketch_png: bytes, *, case_id: str, repeat: int
    ) -> list[ReportedFinding]:
        cache_path = self._cache_path(ref_png, sketch_png, repeat)
        if cache_path.exists():
            return parse_findings(json.loads(cache_path.read_text(encoding="utf-8")))

        data = self._request(ref_png, sketch_png)
        cache_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        return parse_findings(data)


class AnthropicVLM(_CachedVLM):
    """Frontier-VLM baseline via the Anthropic SDK (Claude vision + structured output)."""

    def __init__(self, model: str = config.BENCH_VLM_MODEL, cache_dir=None, *, client=None):
        super().__init__(model, cache_dir, client=client)

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy: only the real path needs the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def _content_blocks(self, ref_png: bytes, sketch_png: bytes) -> list[dict]:
        def img(b: bytes) -> dict:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(b).decode("ascii"),
                },
            }

        return [
            {"type": "text", "text": "IMAGE 1 — REFERENCE:"},
            img(ref_png),
            {"type": "text", "text": "IMAGE 2 — SKETCH:"},
            img(sketch_png),
            {"type": "text", "text": _USER_INSTRUCTION},
        ]

    def _request(self, ref_png: bytes, sketch_png: bytes) -> dict:
        client = self._ensure_client()
        with client.messages.stream(
            model=self.model,
            max_tokens=config.BENCH_VLM_MAX_TOKENS,
            system=self.system_prompt,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": response_schema()}},
            messages=[{"role": "user", "content": self._content_blocks(ref_png, sketch_png)}],
        ) as stream:
            message = stream.get_final_message()
        text = next((b.text for b in message.content if b.type == "text"), "{}")
        return json.loads(text)


class OpenAIVLM(_CachedVLM):
    """Frontier-VLM baseline via the OpenAI SDK (vision chat + strict json_schema).

    Uses the Chat Completions API with a current vision model (default ``gpt-5.5`` —
    see config; gpt-4o is superseded). Images are inlined as base64 ``data:`` URLs and
    the response is constrained with a strict ``json_schema`` response format — the
    same closed vocabulary the Anthropic path uses, so the two baselines answer the
    identical question.

    GPT-5.x specifics (researched 2026-06-14): the token cap parameter is
    ``max_completion_tokens`` (reasoning tokens count against it — hence the generous
    ``BENCH_OPENAI_MAX_TOKENS``), and ``temperature`` is left at the model default
    (reasoning models reject non-default values; the default also gives genuine
    run-to-run variance, which is the input to the consistency metric).
    """

    def __init__(
        self,
        model: str = config.BENCH_OPENAI_MODEL,
        cache_dir=None,
        *,
        reasoning_effort: str | None = config.BENCH_OPENAI_REASONING_EFFORT,
        client=None,
    ):
        super().__init__(model, cache_dir, client=client)
        self.reasoning_effort = reasoning_effort

    def _cache_tag(self) -> str:
        return f"reff={self.reasoning_effort}" if self.reasoning_effort else ""

    def _ensure_client(self):
        if self._client is None:
            import openai  # lazy: only the real path needs the SDK

            self._client = openai.OpenAI()
        return self._client

    def _messages(self, ref_png: bytes, sketch_png: bytes) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "IMAGE 1 — REFERENCE:"},
                    {"type": "image_url", "image_url": {"url": _data_url(ref_png)}},
                    {"type": "text", "text": "IMAGE 2 — SKETCH:"},
                    {"type": "image_url", "image_url": {"url": _data_url(sketch_png)}},
                    {"type": "text", "text": _USER_INSTRUCTION},
                ],
            },
        ]

    def _request(self, ref_png: bytes, sketch_png: bytes) -> dict:
        client = self._ensure_client()
        kwargs: dict = dict(
            model=self.model,
            max_completion_tokens=config.BENCH_OPENAI_MAX_TOKENS,
            messages=self._messages(ref_png, sketch_png),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "art_stockfish_findings",
                    "strict": True,
                    "schema": response_schema(),
                },
            },
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or "{}"
        return json.loads(text)


class StubVLM:
    """Deterministic offline stand-in (NOT a real VLM result).

    Delegates to a per-(case, repeat) callable so tests can script behavior — drops,
    direction flips, magnitude noise, hallucinations — to exercise scoring and the
    consistency metric without any network call.
    """

    def __init__(self, responder: Callable[[str, int], list[ReportedFinding]]):
        self._responder = responder

    def critique(
        self, ref_png: bytes, sketch_png: bytes, *, case_id: str, repeat: int
    ) -> list[ReportedFinding]:
        return list(self._responder(case_id, repeat))
