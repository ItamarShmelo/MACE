# Compton Cross Section Calculator

C++ implementation (with Python bindings) of the thermal Compton scattering
kernel $\Sigma_E(E \to E', \xi;\, T, N_e)$ following Kershaw, Prasad, and
Beason (1986). The kernel describes photon energy redistribution
off a relativistic Maxwell-Juttner electron distribution.

Three evaluation methods are provided:

- **Gauss-Laguerre quadrature** -- direct numerical integration.
- **Power series** -- fast convergent expansion for hot plasmas.
- **Asymptotic series** -- divergent expansion truncated optimally for cold plasmas.

The **ComptonKernelSolver** adaptively selects the fastest accurate method
at each phase-space point based on the scattering kinematics.

## Dependencies

| Dependency | Role |
|------------|------|
| CMake >= 3.25 | Build system |
| C++23 compiler | Language standard |
| [pybind11](https://pybind11.readthedocs.io/) | Python bindings |
| [Boost](https://www.boost.org/) | Modified Bessel functions ($K_1$, $K_2$) |
| [doubledouble](https://github.com/WarrenWeckesser/doubledouble) | ~31-digit arithmetic for HP power series (fetched automatically by CMake) |
| [planck_integral](https://github.com/menahemkrief/planck_integral) | Planck integral for weight-function denominators (fetched automatically by CMake) |
| OpenMP (optional) | Parallel multigroup integration (enabled by default via `COMPTON_ENABLE_OMP`) |
| [LLVM 18+](https://llvm.org/) (clang-tidy, clang++) | C++ static analysis (`just lint-cpp`), auto-discovered from PATH |

**System prerequisites:** Boost development headers and (optionally) OpenMP must be
installed system-wide. On RHEL/CentOS: `dnf install boost-devel`. The
`planck_integral` dependency is fetched via SSH (`git@github.com:...`), so an
SSH key with access to that repository is required at build time.

**Python** (for tests and usage): `numpy`, `scipy`, `pytest`.

## Installation

The recommended workflow uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                  # create venv and install dependencies
uv pip install -e .      # editable install (triggers CMake build)
```

This builds all pybind11 extension modules and installs them inside the
`compton_matrix` Python package.

### Manual CMake build (for C++ development / IDE support)

```bash
cmake --preset dev
cmake --build --preset dev
```

Or equivalently:

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

## Development

The project uses [just](https://github.com/casey/just) as a task runner.
Install it with `cargo install just`, `brew install just`, or see the
[just installation docs](https://just.systems/man/en/packages.html).

Prepare the local development environment:

```bash
just setup
```

Run tests:

```bash
just test
```

Format Python and C++ code:

```bash
just format
```

Run all lint checks (ruff, clang-format, clang-tidy):

```bash
just lint
```

Run the full local validation suite (lint + test + build):

```bash
just check
```

Build the Python package:

```bash
just build
```

**Note:** `just lint-cpp` auto-discovers `clang-tidy`, `clang++` (both >= 18),
and the GCC install directory (>= 15) from PATH. It configures a separate
`build-tidy` directory with clang++ and OpenMP disabled, producing a
`compile_commands.json` tailored for clang-tidy analysis. The `dev` preset
remains on GCC with OpenMP for production builds. `clang-format` must also be
available on PATH. To override the discovered tools:

```bash
just clang_tidy=/path/to/clang-tidy clang_cxx=/path/to/clang++ \
     gcc_install_dir=/path/to/gcc/lib/gcc/triplet/version lint-cpp
```

**CI:** No CI pipeline exists yet. The CMake build fetches `planck_integral`
over SSH, which requires deploy keys or switching to HTTPS before CI can be
set up. This is a planned follow-up.

### Extension modules

The package contains the following extension modules (accessible as
`compton_matrix._module_name`):

| Module | Contents |
|--------|----------|
| `_compton_common` | `ComptonResult` result type |
| `_compton_differential_cross_section` | All kernel evaluators: power series, asymptotic series, quadrature, and adaptive solver |
| `_compton_multigroup` | Multigroup deterministic + Monte Carlo integration, weight functions |
| `_compton_kernel_multipliers` | `EnergyTransferMultiplier` for multigroup integrals |
| `_utilities` | Gauss-Legendre / Gauss-Laguerre quadrature rules |
| `_units` | Physical constants in CGS (`kev`, `kev_kelvin`, `me_c2`, ...) |

## Example usage

```python
import compton_matrix._compton_differential_cross_section as cds
from compton_matrix._units import kev, kev_kelvin

# Quadrature (256-point Gauss-Laguerre, post-IBP form)
quad = cds.ComptonKernelQuadrature(256, cds.QuadratureForm.PostIBP)
r = quad.sigma_E(1*kev, 2*kev, 0.0, kev_kelvin, 1.0)
print(f"quadrature: {r.value:.6e}  (rel_err ~ {r.estimated_rel_error:.1e})")

# Solver (adaptive dispatch: asymptotic / power series / DD fallback)
solver = cds.ComptonKernelSolver()
r = solver.sigma_E(1*kev, 2*kev, 0.0, 1*kev_kelvin, 1.0)
print(f"solver:     {r.value:.6e}  (rel_err ~ {r.estimated_rel_error:.1e})")
```

Both return a `ComptonResult` with fields `value`, `estimated_abs_error`, and
`estimated_rel_error`. Temperature derivatives `dsigma_E_dT` and vectorized
`sigma_E_vec` / `dsigma_E_dT_vec` are also available.

## Multigroup kernel

The `_compton_multigroup` module computes the Planck-weighted multigroup-multiangle
Compton scattering matrix by numerically integrating the point-wise kernel over
energy groups and angle bins:

$$
\sigma(g \to g', [\xi_i, \xi_{i+1}]; T)
= \frac{2\pi \int_{\Delta E_g} \int_{\Delta E_{g'}} \int_{\xi_i}^{\xi_{i+1}}
        w(E,T)\, \Sigma_E(E, E', \xi)\; d\xi\, dE'\, dE}
       {\int_{\Delta E_g} w(E,T)\, dE}
$$

The $2\pi$ factor accounts for azimuthal symmetry ($d\Omega = 2\pi\, d\xi$).

**Weight function.** The weighting function $w(E,T)$ is user-selectable via
the `weight_function` constructor argument.

**API summary.**

- **Constructor**: `ComptonMultigroupKernel(energy_group_boundaries, weight_function, config=MGIntegrationConfig())` -- boundaries are G+1 strictly increasing values in [erg], all > 0. `weight_function` is a `WeightFunction` subclass (e.g. `PlanckWeightFunction`, `UniformWeightFunction`, `WienWeightFunction`). `config` controls quadrature orders, adaptive refinement depth, and convergence tolerance.
- **`compute_sigma_matrix(kernel, num_angle_bins, T, Ne)`** -- returns a `(G, G, N_angles)` NumPy array.
- **`compute_sigma_matrix(kernel, T, Ne)`** -- angle-integrated, returns a `(G, G)` NumPy array.
- **`compute_kernel_derivative_contribution`** -- same signatures, using the temperature-derivative kernel.

The `kernel` argument is a `ComptonKernelSolver` instance.

**Example:**

```python
import compton_matrix._compton_multigroup as cm
import compton_matrix._compton_differential_cross_section as cds
from compton_matrix._units import kev, kev_kelvin

kernel = cds.ComptonKernelSolver()
mg = cm.ComptonMultigroupKernel(
    energy_group_boundaries=[0.1*kev, 0.5*kev, 1*kev, 5*kev, 10*kev],
    weight_function=cm.PlanckWeightFunction(cap_x=25.0),
    config=cm.MGIntegrationConfig(
        xi_order=8,
        xi_tail_order=8,
        ep_edge_order=8,
        ep_interior_order=8,
        e_panel_order=8))

# Angle-integrated G x G matrix
S = mg.compute_sigma_matrix(kernel, T=10*kev_kelvin, Ne=1.0)

# Multiangle G x G x N_angles tensor
S_angle = mg.compute_sigma_matrix(kernel, num_angle_bins=8, T=10*kev_kelvin, Ne=1.0)
```

### Monte Carlo multigroup kernel

The same module also provides `ComptonMonteCarloKernel`, which computes the
multigroup matrix by direct Klein-Nishina thermal sampling rather than
deterministic quadrature of the point-wise kernel. It does not require a
`ComptonKernelSolver` instance.

- **Constructor**: `ComptonMonteCarloKernel(energy_group_boundaries, weight_function, config=MCIntegrationConfig())` -- same boundary and weight-function conventions as the deterministic kernel. `MCIntegrationConfig` controls `num_samples` (default 1,000,000), `seed` (default -1 for time-based), and `discard_out_of_grid` (default `True`).
- **`compute_sigma_matrix`** / **`compute_kernel_derivative_contribution`** -- same overloads as the deterministic kernel (angle-resolved and angle-integrated).

## Equations

This section gives the final equations used by the evaluators. The code returns
the energy-space kernel
$\Sigma_E(E\to E',\xi;T,N_e)$, where $E=h\nu$ and
$E'=h\nu'$. It is related to the frequency-space kernel in the paper by
$\Sigma_E = \Sigma_\nu/h$.

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| $E$, $E'$ | `E`, `E_prime` | Incident / scattered photon energy [erg] |
| $\xi=\cos\theta$ | `xi` | Scattering-angle cosine, with $-1<\xi<1$ |
| $T$ | `T` | Electron temperature [K] |
| $\tau=k_B T/(m_e c^2)$ | `tau` | Dimensionless electron temperature |
| $N_e$ | `Ne` | Electron number density [$\mathrm{cm}^{-3}$] |

The dimensionless photon energies are

$$
\gamma = \frac{E}{m_e c^2},
\qquad
\gamma' = \frac{E'}{m_e c^2}.
$$

### Kinematic quantities

Define

$$
a = 1-\xi,
\qquad
\Delta = \gamma'-\gamma,
\qquad
s = \frac{1}{\gamma}+\frac{1}{\gamma'},
$$

$$
q = \sqrt{\gamma^2+\gamma'^2-2\gamma\gamma'\xi}
  = \sqrt{\Delta^2+2\gamma\gamma'a},
\qquad
\omega^2 = \frac{1+\xi}{a}.
$$

The minimum electron Lorentz factor allowed by the scattering kinematics is

$$
\lambda_+
= \frac{\Delta}{2} + \sqrt{\left(1+\frac{\gamma\gamma'a}{2}\right)\left(1+\frac{\Delta^2}{2\gamma\gamma'a}\right)}
= \frac{\Delta+q\sqrt{1+2/(\gamma\gamma'a)}}{2}.
$$

The shifted lower-limit quantities are

$$
\rho_+ = \lambda_+ + \gamma,
\qquad
\rho_- = \lambda_+ - \gamma',
$$

$$
\alpha_\pm = \frac{1}{\sqrt{\rho_\pm^2+\omega^2}},
\qquad
\chi_\pm = \rho_\pm\alpha_\pm.
$$

For quadrature, the integration variable is shifted from
$\lambda\in[\lambda_+,\infty)$ to $\rho=\lambda-\lambda_+\in[0,\infty)$.
For either sign,

$$
r_\pm(\rho)=\rho+\rho_\pm,
\qquad
R_\pm(\rho)=r_\pm(\rho)^2+\omega^2.
$$

The common coefficient used in all representations is

$$
G = -\gamma\gamma' + \frac{2}{a} + \frac{2}{\gamma\gamma'a^2}.
$$

The post-integration-by-parts coefficients are

$$
\Lambda_+ = G - \frac{s}{\tau a^2},
\qquad
\Lambda_- = G + \frac{s}{\tau a^2}.
$$

The boundary term is

$$
\Psi = \frac{2\tau\gamma\gamma'}{q} + \frac{s}{a^2}\left(\alpha_+ + \alpha_-\right) + \frac{\rho_+\alpha_+ - \rho_-\alpha_-}{a}.
$$

### Common prefactor

The energy-space kernel is written as $\Sigma_E=\Sigma_0\mathcal{M}$.
The mathematically direct prefactor is

$$
\Sigma_0
= \frac{N_e r_e^2 m_e c^2}{4E^2\tau}
  \frac{\exp(-\lambda_+/\tau)}{K_2(1/\tau)}.
$$

For numerical stability the implementation uses the scaled Bessel function

$$
\widetilde K_2(x)=e^xK_2(x),
$$

so that

$$
\frac{\exp(-\lambda_+/\tau)}{K_2(1/\tau)}
= \frac{\exp[-(\lambda_+-1)/\tau]}{\widetilde K_2(1/\tau)}.
$$

Equivalently, the prefactor evaluated in the code is

$$
\Sigma_0
= \frac{N_e r_e^2 m_e c^2}{4E^2\tau}
  \frac{\exp[-(\lambda_+-1)/\tau]}{\widetilde K_2(1/\tau)},
\qquad
r_e^2=\frac{3\sigma_T}{8\pi}.
$$

### Gauss-Laguerre quadrature

The substitution $x=\rho/\tau$ converts the remaining Maxwell-Juttner weight
into the standard Gauss-Laguerre weight:

$$
\int_0^\infty f(x)e^{-x}\,dx
\approx
\sum_{i=1}^{N_L} w_i f(x_i),
$$

where $x_i$ and $w_i$ are Gauss-Laguerre nodes and weights.
Supported quadrature orders are $N_L=32,64,128,256$.
The estimated error is obtained by comparing the selected order with
$N_L/2$.

Two algebraically equivalent integrand forms are implemented.

**Post-IBP form, the default:**

$$
\Sigma_E
= \Sigma_0\left[\Psi + \tau\int_0^\infty H(\tau x)e^{-x}\,dx\right],
$$

with

$$
H(\rho)
= \frac{\Lambda_+ - r_+(\rho)/(\tau a)}{\sqrt{R_+(\rho)}} + \frac{r_-(\rho)/(\tau a)-\Lambda_-}{\sqrt{R_-(\rho)}}.
$$

**Pre-IBP form:**

$$
\Sigma_E
= \Sigma_0\,\tau\int_0^\infty F(\tau x)e^{-x}\,dx,
$$

with

$$
F(\rho)
= \frac{2\gamma\gamma'}{q} + \frac{1}{a^2}
    \left[
      \frac{s r_-(\rho)+1+\xi}{R_-(\rho)^{3/2}} + \frac{s r_+(\rho)-1-\xi}{R_+(\rho)^{3/2}}
    \right] + G\left[\frac{1}{\sqrt{R_+(\rho)}} -\frac{1}{\sqrt{R_-(\rho)}}\right].
$$

The two forms satisfy

$$
\Psi + \tau\int_0^\infty H(\tau x)e^{-x}\,dx
= \tau\int_0^\infty F(\tau x)e^{-x}\,dx.
$$

### Power series

Define

$$
b=\frac{\omega}{2\tau},
\qquad
\theta_\pm=\sinh^{-1}\!\left(\frac{\rho_\pm}{\omega}\right),
$$

$$
x_\pm=b e^{\theta_\pm},
\qquad
y_\pm=b e^{-\theta_\pm}.
$$

The generalized exponential integral and its scaled form are

$$
E_m(x)=\int_1^\infty e^{-xt}t^{-m}\,dt,
\qquad
\widehat E_m(x)=e^xE_m(x).
$$

The scaled recurrence used for advancing the exponential integrals is

$$
\widehat E_{n+1}(x)
=\frac{1-x\widehat E_n(x)}{n},
\qquad n\ge 1.
$$

The normalized kernel is

$$
\frac{\Sigma_E}{\Sigma_0}
= \Psi + P_+ - P_-,
$$

where

$$
P_\pm
= \sum_{n=0}^{N}
  w_n^\pm
  \left(\Lambda_\pm+\frac{2n}{a}\right)
  \widehat E_{n+1}(x_\pm).
$$

The Poisson weights are

$$
w_0^\pm=e^{-y_\pm},
\qquad
w_{n+1}^\pm=w_n^\pm\frac{y_\pm}{n+1},
$$

or equivalently $w_n^\pm=e^{-y_\pm}y_\pm^n/n!$.

The power series converges when the Poisson weights decay rapidly enough,
which corresponds to sufficiently hot temperatures relative to the scattering
kinematics.

**Double-double precision.** The result involves a cancellation $P_+-P_-$,
where the two terms can be individually much larger than their difference.
At low photon energies ($\gamma<0.02$, roughly $E<10$ keV), the power-series
calculation is promoted to double-double arithmetic using the `doubledouble`
library.  The solver dispatch selects this path automatically.  Both double
and DD backends use `eps_rel = 1e-8`; DD's value is the 31-digit intermediate
precision that prevents cancellation catastrophe, not a tighter convergence
criterion.

### Asymptotic series

For cold regimes, the code uses the divergent low-temperature asymptotic
series and truncates it near the smallest term. Define

$$
\eta_+
= \alpha_+\left(\frac{s}{a^2}+\frac{\rho_+}{a}\right),
\qquad
\eta_-
= \alpha_-\left(-\frac{s}{a^2}+\frac{\rho_-}{a}\right).
$$

Then

$$
\frac{\Sigma_E}{\Sigma_0}
\sim
\frac{2\tau\gamma\gamma'}{q} + S_+ + S_-,
$$

with

$$
S_+
= \sum_{n=0}^{\infty}(-\tau\alpha_+)^{n+1}
  \left[
    \left(-G n!+\frac{(n+1)!}{a}\right)P_n(\chi_+)
    - \eta_+(n+1)!P_{n+1}(\chi_+)
  \right],
$$

$$
S_-
= \sum_{n=0}^{\infty}(-\tau\alpha_-)^{n+1}
  \left[
    \left(G n!-\frac{(n+1)!}{a}\right)P_n(\chi_-)
    + \eta_-(n+1)!P_{n+1}(\chi_-)
  \right].
$$

Here $P_n$ is the Legendre polynomial of degree $n$, evaluated by

$$
P_0(z)=1,
\qquad
P_1(z)=z,
\qquad
(n+1)P_{n+1}(z)=(2n+1)zP_n(z)-nP_{n-1}(z).
$$

The terms of this asymptotic series initially decrease and eventually grow.
The implementation truncates at the smallest term after detecting consecutive
term-magnitude increases.

### Adaptive dispatch (ComptonKernelSolver)

```
tau_alpha_max = tau * max(alpha_plus, alpha_minus)

if tau_alpha_max < 0.035:              -- Asymptotic regime
    A1: AsymptoticSeries (double)    -- accept if rel_error < 1e-7
    A2: AsymptoticSeries (DD)        -- accept if rel_error < 1e-3
else:                                  -- Power series regime
    P1: PowerSeries (double)         -- accept if rel_error < 1e-7 and non-negative
    P2: PowerSeries (DD, n_max=500)  -- accept if rel_error < 1e-3 and non-negative

Throw if no backend passes its tolerance.
```

The two regimes are mutually exclusive (strict `if/else`).  DD escalation
within the asymptotic regime is purely error-driven: the roundoff-aware error
estimator detects cancellation at ultra-low gamma and reports large self-error,
triggering the A1-to-A2 escalation automatically.

The thresholds are configurable at construction time (defaults from `compton_kernel_solver.hpp`):

- `asymp_tau_alpha_threshold = 0.035` -- below this, the asymptotic series
  reaches its optimal truncation with few terms.
- `power_series_self_tol = 1e-7` -- accept double power-series result when its
  self-reported relative error is below this tolerance.
- `asymp_self_tol = 1e-7` -- reject the double asymptotic result when its
  self-reported error exceeds this; escalates to DD within the regime.
- `dd_power_series_self_tol = 1e-3` -- accept DD power-series result when its
  self-reported error is below this (looser) tolerance.
- `dd_asymp_self_tol = 1e-3` -- accept DD asymptotic result when its
  self-reported error is below this (looser) tolerance.

## Tests

```bash
uv run pytest
```

The suite covers point-wise kernels, multigroup integration, and utilities:

- **`test_series_vs_quadrature`** -- `ComptonPowerSeries` (double and DD),
  `ComptonKernelAsymptoticSeries`, and `ComptonKernelSolver` against Q256
  Gauss-Laguerre reference across hot-plasma and cold-plasma regimes.
- **`test_kernel_solver`** -- adaptive dispatch regime selection, custom
  thresholds, and `sigma_E_vec` / `dsigma_E_dT_vec` vectorized APIs.
- **`test_multigroup`** -- deterministic `ComptonMultigroupKernel`, Planck
  denominator sanity, adaptive tolerance convergence, and group cutoff.
- **`test_monte_carlo`** -- `ComptonMonteCarloKernel`, seed reproducibility,
  and weight-function invariance.
- **`test_weight_function`** -- `PlanckWeightFunction`, `UniformWeightFunction`,
  `WienWeightFunction` against analytic formulae and SciPy quadrature.
- **`test_integration_functions`** -- Gauss-Legendre and Gauss-Laguerre node/weight
  properties, polynomial exactness, and adaptive integrators vs SciPy.
- **`test_openmp`** -- OpenMP bitwise reproducibility (deterministic) and
  statistical consistency (Monte Carlo) across thread counts.

## Reference

D. S. Kershaw, M. K. Prasad, and J. D. Beason,
*"A simple and fast method for computing the relativistic Compton scattering kernel for radiative transfer,"*
Journal of Quantitative Spectroscopy and Radiative Transfer 36(4):273-282, 1986.
doi:[10.1016/0022-4073(86)90050-6](https://doi.org/10.1016/0022-4073(86)90050-6).
