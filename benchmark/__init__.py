"""M5 benchmark: Art Stockfish vs a frontier VLM baseline (spec §8 M5).

This is the project's headline claim. The protocol (spec §8 M5):

1. Generate a fixed set of ``(reference, distorted-sketch, ground-truth-findings)``
   triples with the labeled distortion harness (``benchmark.dataset``).
2. Run **our** deterministic pipeline on the landmark pairs, and a **frontier VLM**
   on rendered *images* of the same faces with a strong fixed prompt that requests
   critiques in our JSON schema (``benchmark.vlm``).
3. Score both systems the same way against ground truth — finding precision/recall,
   localization, magnitude accuracy, and run-to-run consistency over 3 repeats
   (``benchmark.scoring``).
4. Publish the comparison table (``benchmark.run`` → README).

The package is intentionally outside ``src/artstockfish`` (it is tooling, not part
of the shipped library) and depends only on the light core plus the optional
``anthropic`` SDK for the real VLM path. Run it with ``python -m benchmark.run``.
"""
