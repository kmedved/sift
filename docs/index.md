# SIFT Documentation

SIFT is a Python feature-selection toolbox for fast filters, automatic-k
selection, stability selection, model importance, smart sampling, and optional
CatBoost workflows.

- Start with the [README](https://github.com/kmedved/sift#readme) for installation
  and a quick example.
- Use the [canonical manual](https://github.com/kmedved/sift/blob/main/DOCS.MD)
  for complete workflows and configuration guidance.
- Follow the [selector decision tree](choosing-a-selector.md) to match a method
  to the output contract you need.
- Work through the [tutorial](user-guide.md) for one selection job from a first
  pass through diagnostics.
- Use the [runtime and scaling guide](runtime-scaling.md) for measured costs and
  benchmark provenance.
- Use the [knockoff statistic bakeoff](knockoff-statistic-bakeoff.md) for the
  seeded relevance/lsm/ridge/cefsplus quality comparison that informs the 1.0
  default-statistic decision.
- Check the [data-type support matrix](data-type-support.md) for ndarray,
  DataFrame, categorical, sparse, datetime, weight, group, and time behavior.
- Look up SIFT-specific terms in the [glossary](glossary.md).
- Look up exact signatures and parameter documentation in the
  [generated API reference](reference/index.md).
- Read the [algorithm guide](ALGORITHMS.md) when choosing a method.
- See [troubleshooting](troubleshooting.md) for common validation errors and
  diagnostics.
