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

### Outgoing-energy window and orders at cold temperatures

For each angle bin, the E' integral uses panels placed around the Compton
ridge, with a window of half-width `ep_k_cut` (default 5) thermal widths. A
narrow near-forward (elastic) feature is integrated with a fixed 8-point rule
that no setting changes. At cold temperatures:

- Doubling `ep_edge_order`/`ep_interior_order` (24 -> 48 -> 96) changes row
  sums by 5e-5 and then 1.6e-4 at 1e3 K for E = 10-100 keV, by 1.5e-4 and
  8.5e-5 at 9e3 K, and by 1.4e-4 (24 -> 48) at 8e4 K, without settling. At
  1.1e6 K and above, doubling changes entries above 1% of the row maximum by
  at most 5e-7.

**What to do:** treat ~1e-4 in row sums as the accuracy floor for E above
~10 keV at kT below ~10 eV, unless you also widen the E' window (next
section).

## Wide angle bins at cold temperatures

When a forward angle bin is much wider than the near-forward elastic feature,
the default E' window clips that feature. Raising the xi orders does not help
at all; only narrower bins or a wider E' window do. Forward diagonal
(xi in [0.8, 1]) and row total of a 4 keV group, Wien weight, compared with
2000 bins:

| T [K] | bins | settings | forward diagonal | row total |
|---|---|---|---|---|
| 1e3 | 10 | defaults | 0.50x | -2.0e-3 |
| 1e3 | 10 | `ep_k_cut=7` | 1.000x | -6.7e-8 |
| 1e3 | 10 | `ep_k_cut=10` | 1.000x | -2.6e-6 |
| 1e3 | 10 | `ep_k_cut=15` | 1.000x | +2.8e-4 |
| 1e3 | 10 | `ep_k_cut=10`, E' orders 48 | 1.000x | -2.8e-9 |
| 1e3 | 100 | defaults | 1.000x | -6.7e-8 |
| 1e3 | 100 | `ep_k_cut=15` | 1.000x | -4.5e-3 |
| 9e3 and above | 10 or 100 | defaults | 1.000x | <= 4e-8 |

The feature narrows as kT·mc²/E² in 1 - xi, so higher energies need narrower
bins and the problem reaches higher temperatures. At 26.6 keV, default
settings, compared with 5000 bins:

| T [K] | bins | forward diagonal | row total |
|---|---|---|---|
| 1e3 | 100 | 0.50x | -5.1e-5 |
| 1e3 | 1000 | 0.9998x | < 1e-7 |
| 9e3 | 10 | 0.50x | -4.6e-4 |
| 9e3 | 100 | 0.999x | -7e-7 |

**What to do:** at low temperature (we see the effect at kT = 0.09 eV for
4 keV and up to kT = 0.8 eV for 27 keV), widen the window *and* raise the E'
orders together. `ep_k_cut=10, ep_edge_order=48, ep_interior_order=48` was
accurate to ~1e-8 in every case tested so far (4 keV from 1e3 to 1.1e6 K,
26.6 keV at 1.1e6 K). Widening the window alone helps wide bins but costs
accuracy in narrow ones (see `ep_k_cut=15` above). Check any setting by
halving the bin width and comparing.

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
