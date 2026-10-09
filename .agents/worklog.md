# PR #105 dependency compatibility closeout — in progress

Objective: make existing PR #105 pass its supported Python and dependency matrix without changing defaults, merging, tagging, or releasing. Preserve its runtime-evidence ancestry.

State: PR #105 is open at head `3c5b9a4`, and its runtime evidence remains descended from clean source commit `055d9b5` through evidence commit `5ffc877`. Live CI run `36341325390` showed four failures on current dependencies and seven failures plus three warning errors on minimum dependencies. The bounded fixes normalize mixed-Period exceptions, avoid hashing coarse pandas Timedeltas, bypass old-pandas signaling-Decimal missing checks, stabilize NumPy-integer seed text, and make two dependency-sensitive test fixtures portable.

Next: run the focused failures on current and minimum dependency environments, commit and push the minimal fix, then require PR CI to complete.
