### Sketchfish

big fan of fish

<pre>

 _________         .    .
(..       \_    ,  |\  /|
 \       O  \  /|  \ \/ /
  \______    \/ |   \  / 
     vvvv\    \ |   /  |
     \^^^^  ==   \_/   |
      `\_   ===    \.  |
      / /\_   \ /      |
      |/   \_  \|      /
             \________/

</pre>

## Progress

**Wave 0 — the frozen contract + alignment spine (done).** The project's data
contract (`schema.py`: `Landmarks`, `Severity`, `Level`, `Finding`, `Report`, all frozen
dataclasses exactly per spec §6) is in place and is now read-only for every other agent. The
alignment spine is built and tested: `align.py` implements robust similarity Procrustes (§9.1)
— translation, rotation, uniform scale only, with IRLS trimming so one large drawing error
can't drag the fit — and `frame.py` builds the reference face frame (§9.2) that expresses every
residual as a size- and tilt-invariant "% of head height". Thresholds and importance weights
from §6/§9.5 live in `config.py` as cited, section-partitioned constants. `tests/test_align.py`
passes M0-T3 (a group displaced 25% of head height leaves all other features below the OK floor
under robust alignment, and demonstrates that naive least-squares smears the blame) and M0-T4 (a
7° page rotation is fully absorbed by the similarity transform, leaving zero residual). Run it
with `pytest tests/test_align.py -q`.
