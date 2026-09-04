# Staged candidates

The user-approved goal is structural risk reduction rather than net LOC
reduction, so candidates are ordered by behavioral risk instead of the generic
LOC-savings score.

| ID | boundary | confidence | risk | decision |
| --- | --- | ---: | ---: | --- |
| V1 | Move cohesive result adapters behind the stable `view.py` facade | 4/5 | 3/5 | accept first if monkeypatch/import identity is preserved |
| R1 | Separate Gaussian, consensus, classic-cache, and binary route bodies behind `filter_auto_k.py` | 3/5 | 5/5 | accept in smaller sub-stages with dependency injection/facade seams |
| A1 | Separate config, curve rules, and selector algorithms behind `auto_k.py` | 3/5 | 5/5 | accept last; retain class/module/pickle identity and original monkeypatch seam |

New files and a small total-LOC increase are expected because the old modules
remain stable facades. Success is lower per-file complexity/collision surface,
not fabricated LOC savings.
