# To do

Planned work, roughly in order of value. Most items address an entry in
[`known_limitations.md`](known_limitations.md).

## Accuracy control

- [ ] **Fix the E' window of the most forward angle bin.** In
  `integrate_Ep_ridge`, take the upper end of the window from the maximum of
  E'_c(xi) + k sigma(xi) over the bin (about E + k^2 kT / 2 when the bin ends
  at xi = 1) instead of from the two bin edges, and give the stretch above E
  its own panel. The elastic-core branch sizes its panel by sigma at the
  upper bin edge, which is zero at xi = 1, so it never runs. Fallback: split
  the bin geometrically in 1 - xi until each piece has R < `ep_k_cut`.
  Validate against the `ep_k_cut=30` and narrow-bin references in
  `known_limitations.md`.
- [ ] **Per-entry error estimate.** Evaluate each matrix entry with two
  quadrature resolutions (or an embedded rule) and return the difference
  alongside the result, so users can see when a setting is too low instead
  of running convergence studies by hand.
- [ ] **Defaults that scale with the problem.** Choose `e_panel_order` from
  the largest x = E/kT in a group, so Wien and Planck weights at large x and
  cold temperatures get enough points; choose `ep_k_cut` and the E' orders
  from R until the forward-bin fix lands.
- [ ] **Configurable elastic-core rule.** Expose the order of the
  near-forward elastic-core rule (fixed at 8 points) and the double-peak
  threshold (`DOUBLE_PEAK_RATIO_THRESHOLD = 10`), or scale them with the
  other E' orders.
- [ ] **Accurate Planck normalisation.** Replace or wrap `planck_integral`
  near x ~ 2, where its relative error reaches 3.3e-5.

## Guards

- [ ] Check that a weight function's `group_boundaries` equal the kernel's
  `energy_group_boundaries`, and raise if they do not.
- [ ] Warn when the most forward angle bin has R > `ep_k_cut` (see
  `known_limitations.md`) at the highest group energy and the given
  temperature.

## Performance

- [ ] Compute sigma and dsigma/dT in one pass that shares kernel
  evaluations, instead of the three passes in `compute_dsigma_dT_matrix`.
- [ ] Release the GIL during matrix computations.
- [ ] Allow turning off the `compton_multigroup_<pid>.log` files, or choosing
  where they are written.

## Validation

- [ ] Restore a Monte Carlo vs deterministic dsigma/dT comparison with
  sample counts large enough for its tolerances (removed in `43cd06a`).
- [ ] Add regression tests for the regimes in `known_limitations.md`: Wien at
  large x, cold near-diagonal entries, and the most forward angle bin at
  R > `ep_k_cut`.

## Release

- [ ] Tag a release once the limitations above are documented and the
  published tables reference it.
- [ ] When the MACE paper is published, replace the arXiv citation in the
  README (arXiv:2610.25117) with the journal reference and DOI.
