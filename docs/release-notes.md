# Release Notes

## Unreleased

- Expanded experimental auto-k selection with EBIC/RIC penalties,
  pseudo-posterior `k` diagnostics, calibrated CEFS+ gain stops,
  permutation-gap null envelopes, closed-form Gaussian CV curves,
  knockoff-path stopping, bootstrap stability, changepoint diagnostics,
  consensus diagnostics, and a synthetic auto-k harness.

## 0.7.0

- Added q-calibrated Gaussian-copula knockoff selection: `select_fdr`,
  `KnockoffSelector`, `sample_knockoffs`, feature-group thresholding, and
  approximate plug-in validity metadata.
- Added `benchmarks/bench_knockoffs.py` for the new knockoff timing surface.
  Its quick/full records are informational smoke data rather than promotion
  gates.
- Accelerated Gaussian cache construction by vectorizing the weighted
  rank-Gaussian transform; this benefits all Gaussian/cache selectors, not only
  knockoffs.
- Added tie-safe `statistic="cefsplus"` for knockoffs, with objective-gain
  scoring and optional `min_gain_ratio` early stopping for large runs.
- Implemented diagonal coordinate-descent `s_method="mvr"` and `"me"` optimizers
  for the MVR and maximum-entropy knockoff objectives.
- Weighted `build_cache(..., subsample=...)` now samples from positive-weight
  rows. Seeded weighted caches can choose different rows than pre-release builds;
  unweighted seeded caches preserve the old row choices.
- Knockoff noise now uses NumPy float32 standard-normal draws. Seeded knockoff
  samples can differ from pre-release builds for the same `random_state`.
- Documentation and metadata consistently frame knockoff FDR control as an
  approximate plug-in Gaussian-copula result unless the fitted feature model is
  the true Model-X distribution.
- Added standalone API, algorithm, advanced-workflow, and contributing docs
  aligned with the 0.7.0 public surface.
- Release documentation now links the `select_fdr` workflow, `KnockoffSelector`,
  `sample_knockoffs`, and the focused docs/benchmark smoke checks from the
  README, development guide, benchmark guide, and release tracker.
