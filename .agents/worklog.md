# PR #105 dependency compatibility closeout — in progress

Objective: make existing PR #105 pass its supported Python and dependency matrix without changing defaults, merging, tagging, or releasing. Preserve its runtime-evidence ancestry.

State: PR #105 is open, and its runtime evidence remains descended from clean source commit `055d9b5` through evidence commit `5ffc877`. Live CI run `36341325390` showed four failures on current dependencies and seven failures plus three warning errors on minimum dependencies. Commit `3bd0821` applies bounded compatibility fixes for mixed Periods, coarse pandas Timedeltas, old-pandas signaling-Decimal missing checks, NumPy-integer seed text, and two dependency-sensitive test fixtures. The three affected test modules pass 215 tests on both the local current stack and an isolated exact minimum-dependency stack. Runtime evidence was rerun from clean `3bd0821`: all 18 data and selection fingerprints are unchanged, and the source hash changes are exactly the three fixed source files.

Next: commit the refreshed evidence, run its binding test, push the two commits, then require PR CI to complete.
