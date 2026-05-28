# Robust Compton Kernel Solver

The `ComptonKernelSolver` provides a unified interface for evaluating the Compton
scattering kernel with empirically validated 1e-8 relative error on the calibrated
domain. It cascades through three evaluation methods (asymptotic series, power series,
quadrature) selecting the fastest method that achieves the accuracy target.

## Method Cascade

```
sigma_E(E, E', xi, tau, Ne)
  1. Validate domain (E > 0, E' > 0, tau > 0, xi in (-1,1))
  2. Compute kinematic parameters + sigma0
  3. Compute tau_alpha_max = max(tau*alpha_plus, tau*alpha_minus)
  4. If tau_alpha_max < 0.2 (dispatch threshold):
       Try asymptotic series → accept if estimated_rel_error < target
  5. Try power series (conditioning-aware error) → accept if < target
  6. Fall back to Gauss-Laguerre quadrature (NL=256), report its own error honestly
```

## Dispatch Threshold vs. Validity Threshold

These are distinct concepts:

- **Dispatch threshold** (`tau_alpha_max < 0.2`): Decides which method to *try first*.
  Based on empirical calibration of asymptotic series accuracy. This is a performance
  optimization, not a correctness check.

- **Validity threshold** (`estimated_rel_error < target_rel_tol`): Decides whether to
  *accept* a result. Applied after every series evaluation via local error estimation.
  This is the actual acceptance criterion.

A result can pass the dispatch threshold but fail the validity check (triggering
fallback to the next method), or fail the dispatch threshold but succeed via direct
power series evaluation.

## Error Estimation

### Asymptotic series
Error = smallest term magnitude / |normalized result|. This reflects the achievable
accuracy of the optimally-truncated divergent series.

### Power series (conditioning-aware)
Error = max(truncation_error, conditioning_error) where:
- `truncation_error = last_term_magnitude / |normalized_result|`
- `conditioning_error = 10 * conditioning * 2.2e-16`
- `conditioning = (|P_plus| + |P_minus| + |Psi|) / |Psi + P_plus - P_minus|`

The safety factor of 10 accounts for multi-term floating-point accumulation.

### Quadrature (NL=256)
Error = |Q(256) - Q(128)| (NL vs NL/2 discrepancy). This is not a certified bound
but an empirical convergence estimate.

## Tolerance Concepts

- `target_rel_tol` (default 1e-8): The relative error target for series acceptance.
  When the series-reported `estimated_rel_error < target_rel_tol`, the result is accepted.

- `target_abs_tol` (default 1e-300): If `|value| < target_abs_tol` after the first
  series evaluation, the kernel is considered physically negligible and returned as zero.
  This prevents wasting compute on values that cannot affect any physical observable.

- `REL_ERROR_FLOOR` (internal, 1e-300): Denominator floor used only in
  `rel_error = abs_error / max(|value|, REL_ERROR_FLOOR)` to avoid division by zero.

## SolverResult Fields

| Field | Type | Description |
|-------|------|-------------|
| value | double | Kernel value (non-negative after clamping) |
| estimated_abs_error | double | Absolute error estimate |
| estimated_rel_error | double | Relative error estimate |
| terms_used | int | Series terms used (or 256 for quadrature) |
| method_used | SolverMethod | Asymptotic, PowerSeries, or Quadrature |
| used_fallback | bool | Reserved (always false) |
| target_met | bool | True if accepted path achieved target (or result is negligible) |
| clamped | bool | True if a negative value was clamped to zero |
| tau_alpha_max | double | Diagnostic: max(tau*alpha_plus, tau*alpha_minus) |
| conditioning | double | Diagnostic: power-series conditioning number |

## target_met Semantics

- For series paths (Asymptotic or PowerSeries): `target_met = true` means the series
  achieved `estimated_rel_error < target_rel_tol`.
- For Quadrature: `target_met = (quadrature_rel_error < target_rel_tol)`. The solver
  does NOT claim 1e-8 if the quadrature cannot achieve it.
- For negligible results: `target_met = true` (zero is exact).

## Valid Parameter Domain

- E > 0 (photon energy in erg)
- E' > 0 (scattered photon energy in erg)
- tau > 0 (dimensionless electron temperature kT/m_e*c^2)
- xi strictly in (-1, 1) (cosine of scattering angle)
- Ne >= 0 (electron number density; use 1.0 for microscopic)

The solver throws `std::invalid_argument` for inputs outside this domain.
xi = +/-1 corresponds to singular kinematics (a = 1-xi = 0 or 2 boundary).

## Calibration Grid

The 1e-8 accuracy claim is empirically validated on:
- tau in [2e-5, 1.0] (T from 0.01 to 511 keV)
- Photon energies: 0.01 to 100 keV
- E'/E ratios in [0.1, 10]
- xi in [-0.9, 0.9]

Outside this grid, the solver continues to work but may report larger
`estimated_rel_error` and/or `target_met = false`.

## Performance

In the asymptotic-dominated regime (low T), the solver is ~2-3x slower than a
raw series call due to cascade overhead. In the mixed regime, it's ~4-6x slower
because some points require multiple attempts. It is always much faster than
pure quadrature evaluation.

## Usage (C++)

```cpp
#include "compton_kernel_solver/compton_kernel_solver.hpp"
using namespace compton;

ComptonKernelSolver solver(1e-8, 1e-300);
SolverResult r = solver.sigma_E(E, E_prime, xi, tau, Ne);
if (r.target_met) {
    // Use r.value with confidence
}
```

## Usage (Python)

```python
from _compton_kernel_solver import ComptonKernelSolver
solver = ComptonKernelSolver(target_rel_tol=1e-8)
r = solver.sigma_E(E, E_prime, xi, tau, Ne)
# r.value, r.target_met, r.method_used, etc.

# Vectorized:
values, errors, methods, terms, fallbacks, target_mets = \
    solver.sigma_E_vec(E, E_prime_array, xi, tau, Ne)
```
