"""Pytest path setup.

``artstockfish`` is installed editable (``src/`` is on the path), but the M5
``benchmark`` package lives at the repo root and is tooling, not an installed
package. Put the repo root on ``sys.path`` so ``tests/test_benchmark.py`` can
``import benchmark`` the same way ``python -m benchmark.run`` does.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
