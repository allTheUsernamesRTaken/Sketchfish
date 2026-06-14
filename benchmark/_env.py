"""Minimal ``.env`` loader for the benchmark's API keys (no extra dependency).

The benchmark's VLM baselines read ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` from
the environment (the SDKs do this by default). To make that easy *and* safe to push
to GitHub, drop the key in a repo-root ``.env`` file — which is gitignored — and the
runner loads it on startup. Real environment variables always win (``setdefault``),
so this never overrides a key you exported yourself, and there is no third-party
dependency to install.

Format: ``KEY=value`` per line; ``#`` comments and blank lines ignored; surrounding
quotes on the value are stripped. See ``.env.example``.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> bool:
    """Load ``KEY=value`` pairs from ``.env`` into ``os.environ`` (existing vars win).

    Returns True if a file was found and read.
    """
    env_path = path or (_REPO_ROOT / ".env")
    if not env_path.is_file():
        return False
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    return True
