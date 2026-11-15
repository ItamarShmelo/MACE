# Known limitations

MACE uses fixed-order Gauss–Legendre rules on panels placed around the
features of the kernel. It does not estimate its own quadrature error, so the
default settings are not converged in every regime. This file lists the cases
we know about, how large the effect is, and what to do about each one.
Planned fixes are in [`todo.md`](todo.md).

All numbers were measured with MACE `e542f1b` on a 128-group geometric grid
from 1e-5 to 300 keV, 100 equal bins in xi = cos(theta), and temperatures from
1e3 to 1e9 K. Errors are relative to |sigma| over entries above 1% of their
row maximum unless stated otherwise. A "row" is one incoming group summed over
outgoing groups and angle bins.

## Quadrature defaults

### E-panel order (`e_panel_order`, default 16)

The incoming-energy integral needs a higher order than the default whenever
the integrand varies on a scale much smaller than the group:

- **Wien and Planck weights at large x = E/kT.** The shifted weight
  x³exp(-(x - x_lo)) decays on the scale kT inside a group of width
  ~0.14 E, so it occupies a tiny sliver of high-energy groups. Row sums on
  4-group slices, compared with order 256:

  | T [K] | x in slice | order 16 | order 64 |
  |---|---|---|---|
  | 1e3 | ~1e5 | 37% | 3e-9 |
  | 9e3 | ~1e4 | 0.4% | 3e-9 |
  | 1.1e6 | ~1e3 | 3.5% | 9e-9 |

- **Cold temperatures, any weight.** Near-diagonal, high-energy entries are
  recoil dominated and need many E points. Uniform weight, maximum error
  compared with order 128:

  | T [K] | order 16 | order 32 | order 64 |
  |---|---|---|---|
  | 1e3 | 0.19 | 6.3e-2 | 1.3e-2 |
  | 9e3 | 7.2e-2 | — | 1.1e-3 |
  | 8.0e4 | 8.6e-3 | — | 1.3e-6 |
  | 1.1e6 | 2.4e-3 (vs 64) | — | — |
  | 1.4e8 | 2.6e-8 (vs 64) | — | — |

  Row sums are good to ~5e-7 even at order 16; the error sits in individual
  entries.

**What to do:** use `e_panel_order=64` for Wien and Planck weights at every
temperature, and 128 for kT below ~7 eV. Cost grows linearly with the order.
The published MACE tables use 128 below 8e4 K and 64 above.

The xi orders (`xi_order`, `xi_tail_order`) are converged at their defaults:
doubling them changes results by at most 1e-6.

## Most forward angle bin at cold temperatures

For each angle bin, the E' integral is split into panels around the Compton
ridge, with a window that extends `ep_k_cut` (default 5) thermal widths beyond
the ridge positions at the two edges of the bin (`integrate_Ep_ridge` in
`compton_multigroup_deterministic.cpp`). In the most forward bin the upper
edge is xi = 1, where the ridge width is zero. Once the recoil shift across
the bin exceeds `ep_k_cut` widths, the window therefore ends exactly at
E' = E. Just above E sits a narrow Doppler upscatter tail from near-forward
angles, a few kT wide; it then falls into a single panel spanning the rest of
the outgoing group and is under-resolved. The elastic-core branch meant for
this feature takes its width from the ridge at the upper bin edge, which is
zero, so it never runs.

Whether this happens is set by one number, the recoil shift across the bin
in thermal widths at its wide edge:

$$
R = \frac{E - E'_c(\xi_{lo})}{\sigma_{E'}(\xi_{lo})}
  \approx E \sqrt{\frac{1 - \xi_{lo}}{2\, kT\, m_e c^2}}
$$

where $E'_c$ is the cold Compton energy and $\sigma_{E'}$ the ridge width
(`ridge_thermal_width`). With 100 equal bins (1 - xi_lo = 0.02), R = 5 at
E ~ 10 keV for 1e3 K, 33 keV for 1e4 K, 105 keV for 1e5 K and 330 keV for
1e6 K. Only the width of the most forward bin enters.

A window of k widths is accurate (row totals within 1e-9) for R below about
k, and above that loses the same amount as the default window does at that
R. In our tests, `ep_k_cut` = 5, 10, 15 and 30 started to fail at R ~ 5, 10,
16 and 31. Error of the default window, 128 groups, 100 bins, Wien weight,
compared with `ep_k_cut=30` and E' orders 192 (the uniform weight is similar
or smaller):

| R | row total | largest entry error, relative to row maximum |
|---|---|---|
| below 5 | <= 1e-9 | <= 1e-7 |
| 5 to 7 | -2e-4 to -3e-4 | 1% to 2% |
| 10 | -9e-5 | 0.5% |
| 15 | -4e-5 | 0.25% |
| 20 | -2.5e-5 | 0.15% |
| 30 | -1e-5 | 0.06% |

The missing piece sits in the g -> g and g -> g+1 entries of the most
forward bin; at R ~ 5 to 10 the forward diagonal comes out at about half its
value. dsigma/dT has errors of the same size relative to sigma.

Other settings:

- **E' orders.** A wider window needs proportionally higher E' orders; keep
  about 5 to 6 points per unit of `ep_k_cut` (48 for 10, 96 for 15, 192 for
  30). `ep_k_cut=15` with the default orders (24) gave row errors up to
  8e-3, even where R is small. Cost grows linearly with the E' orders:
  `ep_k_cut=15` with orders 96 costs 4x the defaults.
- **Orders alone.** Raising the E' orders without widening the window only
  partly resolves the tail: at 1e3 K and 10 to 100 keV, doubling them from 24
  to 48 to 96 changed row sums by 5e-5 and then 1.6e-4 without settling.
  Where R < 5 (with 100 bins, kT above ~100 eV for E up to 300 keV),
  doubling them changes entries above 1% of the row maximum by at most 5e-7.
- **Narrower bins.** R scales as the square root of the forward bin's
  width, so a narrower forward bin works too: 26.6 keV at 1e3 K converged
  with 1000 equal bins (R ~ 4) and 134 keV at 1e3 K with 20000 (R ~ 4.5).

**What to do:**

- Compute R for the highest group energy, the coldest temperature and your
  most forward bin. If it is below 5, the defaults are fine.
- Otherwise set `ep_k_cut` above that R with E' orders of about 6 times
  `ep_k_cut`, narrow the most forward bin, or both.
- If that is too expensive, `ep_k_cut=15` with E' orders 96 limits the error
  to the R > 16 part of the curve above: row totals within ~4e-5 at any R.
- The published MACE tables (128 groups up to 300 keV, 100 bins, R up to
  ~140 at 1e3 K) use `ep_k_cut=15` with E' orders 96 below 9e5 K. Compared
  with `ep_k_cut=30` and E' orders 192 at 1e3 K and 9e3 K, their row totals
  are within 3e-5 and their entries within 0.18% of the row maximum.
- The proper fix belongs in MACE; see [`todo.md`](todo.md).

## Planck normalisation

`PlanckWeightFunction.compute_denominator` uses the external
`planck_integral` library. For groups with a width ratio of 1.14, its
relative error is up to 3.3e-5 for x_lo in [1.5, 2.5], up to ~1e-6 elsewhere
in [1, 5], and below ~2e-9 outside that range. The error enters Planck sigma
and dsigma/dT in those groups directly. `d_denominator_dT` is exact given
that denominator.

## Pointwise kernel derivative

As documented in `compton_kernel_solver.hpp`: for tau·alpha_max between about
0.025 and the series threshold, about 0.1% of `dsigma_E_dT` points carry
errors up to ~5e-3, reported in `estimated_rel_error`. The effect on group
matrices is at most ~1e-7 of the row maximum.

## Monte Carlo derivatives

The Monte Carlo dsigma/dT has no passing comparison against the deterministic
result: the tests were removed in `43cd06a` because their row-sum tolerances
failed at the fixed sample counts. Treat it as unvalidated.

## Silent failure modes

- **Weight boundaries.** `WienWeightFunction` and `PlanckWeightFunction` take
  `group_boundaries`, which must equal the kernel's
  `energy_group_boundaries`. Nothing checks this; a mismatch gives wrong cross
  sections without any error.
- **No convergence warning.** Results never report that a setting is too low
  for the regime. Check by doubling the relevant order (or halving the angle
  bin width) and comparing.

## Cost and runtime behaviour

- `compute_dsigma_dT_matrix` runs three full passes (kernel derivative, weight
  derivative, sigma) with the cutoff disabled. On 16 cores, 128 groups by 100
  bins, the hottest Wien/Planck temperatures take about a day each at
  `e_panel_order=64`. For uniform weights,
  `compute_kernel_derivative_contribution` with `cutoff_ratio=None` gives the
  same result in one pass.
- The bindings do not release the GIL, so run parallel computations as
  separate processes.
- Every matrix call appends to `compton_multigroup_<pid>.log` in the current
  directory; this cannot be turned off.
