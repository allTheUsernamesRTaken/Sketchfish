"""Geometric measurement layer (spec §9.3–9.5).

Each module here turns aligned landmark/contour geometry into ``Finding`` objects
via deterministic geometry only (design principle #1 — no learned model ever
produces a number in a critique). Modules are mathematically independent and own
disjoint files; see the Wave 1 split in ``IMPLEMENTATION_PLAN.md``.
"""

from __future__ import annotations
