"""Art Stockfish — a computer-vision drawing coach.

Public surface for Wave 0 (the frozen contract + alignment spine). Downstream
waves import the schema and geometry primitives from here.
"""

from __future__ import annotations

from .schema import Finding, Landmarks, Level, Report, Severity

__all__ = [
    "Finding",
    "Landmarks",
    "Level",
    "Report",
    "Severity",
]
