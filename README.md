# Compton Cross Section Calculator

C++ implementation (with Python bindings) of the thermal Compton scattering
kernel $\Sigma_E(E \to E', \xi;\, T, N_e)$ following Kershaw, Prasad, and
Beason (1986). The kernel describes photon energy redistribution
off a relativistic Maxwell-Juttner electron distribution.

Three evaluation methods are provided:

- **Gauss-Laguerre quadrature** -- direct numerical integration.
- **Power series** -- fast convergent expansion for hot plasmas.
- **Asymptotic series** -- divergent expansion truncated optimally for cold plasmas.

An **Auto** dispatch mode selects the appropriate series method based on the
scattering kinematics.

## Dependencies

| Dependency | Role |
|------------|------|
| CMake >= 3.22 | Build system |
| C++20 compiler | Language standard |
| [pybind11](https://pybind11.readthedocs.io/) | Python bindings |
| [Boost](https://www.boost.org/) | Modified Bessel functions ($K_1$, $K_2$) |
| [doubledouble](https://github.com/WarrenWeckesser/doubledouble) | ~31-digit arithmetic for HP power series (fetched automatically by CMake) |

**Python** (for tests and usage): `numpy`, `scipy`, `pytest`.

## Compilation

```bash
cmake -S . -B build
cmake --build build -j
```

This produces four pybind11 extension modules in `cpp_modules/`:

| Module | Contents |
|--------|----------|
| `_compton_common` | `SigmaResult` result type |
| `_compton_kernel_quadrature` | Gauss-Laguerre quadrature evaluator |
| `_compton_kernel_series` | Power / asymptotic / auto series evaluator |
| `_units` | Physical constants in CGS (`kev`, `kev_kelvin`, `me_c2`, ...) |

## Example usage

```python
import sys
sys.path.insert(0, "cpp_modules")

import _compton_kernel_quadrature as cq
import _compton_kernel_series as cs
from _units import kev, kev_kelvin

# Quadrature (256-point Gauss-Laguerre, post-IBP form)
quad = cq.ComptonKernelQuadrature(256, cq.QuadratureForm.PostIBP)
r = quad.sigma_E(1*kev, 2*kev, 0.0, kev_kelvin, 1.0)
print(f"quadrature: {r.value:.6e}  (rel_err ~ {r.estimated_rel_error:.1e})")

# Series (auto dispatch)
series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
r = series.sigma_E(1*kev, 2*kev, 0.0, 1*kev_kelvin, 1.0)
print(f"series:     {r.value:.6e}  (rel_err ~ {r.estimated_rel_error:.1e})")
```

Both return a `SigmaResult` with fields `value`, `estimated_abs_error`, and
`estimated_rel_error`. Temperature derivatives `dsigma_E_dT` and vectorized
`sigma_E_vec` / `dsigma_E_dT_vec` are also available.

## Equations

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $E$, $E'$ | `E`, `E_prime` | Incident / scattered photon energy [erg] |
| $\xi = \cos\theta$ | `xi` | Scattering angle cosine, strictly in $(-1, 1)$ |
| $\tau = kT / (m_e c^2)$ | `tau` | Dimensionless electron temperature |
| $N_e$ | `Ne` | Electron number density [$\mathrm{cm}^{-3}$] |

Dimensionless photon energies: $\gamma = E / (m_e c^2)$, $\gamma' = E' / (m_e c^2)$.

### Kinematic parameters

With $a = 1 - \xi$, $s = 1/\gamma + 1/\gamma'$, and $\Delta\gamma = \gamma' - \gamma$:

$$q^2 = \Delta\gamma^2 + 2\gamma\gamma' a, \quad s = \frac{1}{\gamma} + \frac{1}{\gamma'}$$

$$\lambda_+ = \frac{as + q}{2a}, \quad \omega^2 = \frac{1 + \xi}{a}$$

$$\rho_+ = \lambda_+ + \gamma, \quad \rho_- = \lambda_+ - \gamma'$$

$$\alpha_\pm = \frac{1}{\sqrt{\rho_\pm^2 + \omega^2}}$$

plus boundary/coefficient terms $G$, $A_\pm$, $\Psi$ derived from these
(see `compute_params` in `compton_common.hpp`).

### Common prefactor

$$\Sigma_0 = \frac{N_e\, r_e^2\, m_e c^2}{4 E^2 \tau} \cdot \frac{e^{-(\lambda_+ - 1)/\tau}}{\tilde{K}_2(1/\tau)}$$

where $r_e^2 = 3\sigma_T / (8\pi)$ is the classical electron radius squared.

**Scaled Bessel functions.** The Maxwell-Juttner normalization involves
$K_2(1/\tau)$ and the Boltzmann factor $e^{-\lambda_+/\tau}$, which
individually overflow or underflow at extreme temperatures. The code uses the
scaled form $\tilde{K}_2(x) = e^x K_2(x)$, which absorbs the $e^{1/\tau}$
factor from the denominator into the numerator:

$$\frac{e^{-\lambda_+/\tau}}{K_2(1/\tau)} = \frac{e^{-(\lambda_+ - 1)/\tau}}{\tilde{K}_2(1/\tau)}$$

This is where the $\lambda_+ - 1$ in the exponent comes from -- it keeps both
factors numerically moderate across all temperature regimes.

### Quadrature

The kernel involves an integral over the electron Lorentz factor $\lambda$
from $\lambda_+$ to $\infty$ against the Maxwell-Juttner distribution
$e^{-\lambda/\tau}$. After extracting $\Sigma_0$ (which carries the
$e^{-\lambda_+/\tau}$ suppression), the remaining integral over the shifted
variable $\rho = \lambda - \lambda_+$ has an $e^{-\rho/\tau}$ weight. The
substitution $x = \rho / \tau$ transforms this into the standard
Gauss-Laguerre form:

$$\int_0^\infty f(x)\, e^{-x}\, dx \approx \sum_{i=1}^{N_L} w_i\, f(x_i)$$

where $x_i$ and $w_i$ are precomputed Gauss-Laguerre nodes and weights.
Supported quadrature orders are $N_L = 64, 128, 256$. The error is estimated
Richardson-style by comparing $N_L$ vs $N_L / 2$.

Two algebraically equivalent integrand forms are implemented:

**Post-IBP (default):** Smoother $O(1/\sqrt{R})$ integrand obtained by
integration by parts, producing a boundary term $\Psi$:

$$\Sigma_E = \Sigma_0 \left(\Psi + \tau \int_0^\infty H(\tau x)\, e^{-x}\, dx\right)$$

with $r_\pm = \rho + \rho_\pm$, $R_\pm = r_\pm^2 + \omega^2$, and

$$H(\rho) = \frac{A_+ - r_+ / (\tau a)}{\sqrt{R_+}} + \frac{-A_- + r_- / (\tau a)}{\sqrt{R_-}}$$

**Pre-IBP:** Original $O(1/R^{3/2})$ integrand, sharper but with no
$\Psi$ cancellation:

$$\Sigma_E = \Sigma_0 \cdot \tau \int_0^\infty F(\tau x)\, e^{-x}\, dx$$

$$F(\rho) = \frac{2\gamma\gamma'}{q} + \frac{1}{a^2}\left[\frac{r_- s + 1 + \xi}{R_-^{3/2}} + \frac{r_+ s - 1 - \xi}{R_+^{3/2}}\right] + G\left(\frac{1}{\sqrt{R_+}} - \frac{1}{\sqrt{R_-}}\right)$$

The two forms satisfy the identity $\Psi + I_Q^{\text{post}} = I_Q^{\text{pre}}$.

### Power series

The normalized kernel is expanded as:

$$\frac{\Sigma_E}{\Sigma_0} = \Psi + P_+ - P_-$$

$$P_\pm = \sum_{n=0}^{N} w_n^\pm\, c_n^\pm\, \hat{E}_{n+1}(x_\pm)$$

where:

- $\hat{E}_m(x) = e^x E_m(x)$ is the scaled exponential integral, evaluated
  via continued fraction and advanced via the recurrence
  $\hat{E}_{n+1}(x) = (1 - x \hat{E}_n(x)) / n$.
- **Poisson weights:** $w_0^\pm = e^{-y_\pm}$, $w_{n+1}^\pm = w_n^\pm \cdot y_\pm / (n+1)$.
- **Coefficients:** $c_n^\pm = A_\pm + 2n / a$.
- $x_\pm$ and $y_\pm$ are derived from a hyperbolic substitution
  on $\rho_\pm$ and $\omega$.

The series converges when the Poisson weights decay, which requires the
temperature to be sufficiently high relative to the scattering kinematics.

**Double-double precision.** The result involves a cancellation $P_+ - P_-$
where both terms can be individually much larger than their difference. At low
photon energies ($\gamma < 0.02$, roughly $E < 10$ keV), catastrophic
cancellation causes the double-precision power series to lose significant
digits. In this regime the computation is promoted to double-double arithmetic
(~31 decimal digits) using the `doubledouble` library, recovering full
accuracy. The Auto dispatch selects this path automatically.

### Asymptotic series

$$\frac{\Sigma_E}{\Sigma_0} = \frac{2\tau\gamma\gamma'}{q} + S_+ + S_-$$

where $S_\pm$ are Legendre polynomial expansions in powers of
$(-\tau \alpha_\pm)^n$, with $P_n(\zeta_\pm)$ Legendre polynomials evaluated
via the standard recurrence.

**Asymptotic truncation.** This is a divergent asymptotic series -- the terms
initially decrease but eventually grow without bound. The implementation
monitors consecutive term magnitudes and truncates the sum at the smallest
term once two consecutive magnitude increases are detected. This gives the
optimal asymptotic approximation. The series is only used in the cold where $\tau \cdot \max(\alpha_+, \alpha_-)$ is small and the optimal
truncation point is reached quickly.

### Series Auto dispatch

```
tau_alpha_max = tau * max(alpha_plus, alpha_minus)

if tau_alpha_max < 0.025:
    method = Asymptotic                # cold regime
elif min(gamma, gamma_prime) >= 0.02:
    method = PowerSeries               # double precision
else:
    method = PowerSeriesHighPrecision   # double-double precision
```

The thresholds are defined in `compton_common.hpp`:

- `ASYMP_TAU_ALPHA_THRESHOLD = 0.025` -- below this, the asymptotic series
  reaches its optimal truncation with few terms.
- `GAMMA_DOUBLE_PRECISION_SAFE = 0.02` -- above this, the $P_+ - P_-$
  cancellation is mild enough for double precision.

## Tests

```bash
pytest tests/
```

The test suite validates all series methods (PowerSeries, PowerSeriesHighPrecision,
Asymptotic, Auto) against the Q256 Gauss-Laguerre quadrature reference across
a grid of photon energies, scattering angles, and temperatures spanning both
the hot-plasma (power series) and cold-plasma (asymptotic) regimes.

## Reference

D. S. Kershaw, M. K. Prasad, and J. D. Beason,
*"A simple and fast method for computing the relativistic Compton scattering kernel for radiative transfer,"*
Journal of Quantitative Spectroscopy and Radiative Transfer 36(4):273-282, 1986.
doi:[10.1016/0022-4073(86)90050-6](https://doi.org/10.1016/0022-4073(86)90050-6).
