# To do

Planned work, roughly in order of value. Most items address an entry in
[`known_limitations.md`](known_limitations.md).

## Accuracy control

- [ ] **Per-entry error estimate.** Evaluate each matrix entry with two
  quadrature resolutions (or an embedded rule) and return the difference
  alongside the result, so users can see when a setting is too low instead
  of running convergence studies by hand.
- [ ] **Defaults that scale with the problem.**
  - Choose `e_panel_order` from the largest x = E/kT in a group, so Wien and
    Planck weights at large x and cold temperatures get enough points.
  - Choose `ep_k_cut` from the ratio of the ridge widths at the two ends of
    an angle bin, and raise the E' orders with it, so wide forward bins at
    cold temperatures keep the near-forward elastic feature.
- [ ] **Configurable elastic-core rule.** Expose the order of the
  near-forward elastic-core rule (fixed at 8 points) and the double-peak
  threshold (`DOUBLE_PEAK_RATIO_THRESHOLD = 10`), or scale them with the
  other E' orders. This is the likely reason the E' integral does not
  converge with order at cold temperatures.
- [ ] **Accurate Planck normalisation.** Replace or wrap `planck_integral`
  near x ~ 2, where its relative error reaches 3.3e-5.

## Guards

- [ ] Check that a weight function's `group_boundaries` equal the kernel's
  `energy_group_boundaries`, and raise if they do not.
- [ ] Warn when a requested angle bin is much wider than the near-forward
  feature at the given energy and temperature.

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
  large x, cold near-diagonal entries, and wide forward angle bins.

## Release

- [ ] Tag a release once the limitations above are documented and the
  published tables reference it.
