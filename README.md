# MACE

**MACE** (Multigroup and Angle-resolved Compton Evaluator) computes the exact
thermal Compton scattering kernel and reduces it to multigroup, angle-resolved
redistribution matrices for radiation-transport calculations. The numerical
core is written in C++23 and exposed through Python bindings.

## Capabilities

- Pointwise thermal Compton kernels and temperature derivatives.
- Adaptive evaluation using convergent and asymptotic series.
- Deterministic multigroup and multiangle integration.
- Independent Monte Carlo matrix estimation.
- Full multigroup temperature derivatives, including the spectral weight and
  normalization terms.
- Planck, Wien, capped, and uniform spectral weights, plus angular and energy
  moment multipliers.

All cross sections are microscopic, per free electron. Multiply by the free
electron number density to obtain macroscopic coefficients.

## Online tables

The [MACE web interface](https://itamarshmelo.github.io/MACE-Website/) generates
tables from precomputed results without requiring a local installation.

## Installation

MACE requires Python 3.11 or newer, CMake 3.25 or newer, a C++23 compiler,
Boost, and optionally OpenMP.

```bash
uv sync
uv pip install -e .
```

The `planck_integral` build dependency is fetched over SSH, so the current
build configuration requires GitHub SSH access to that repository.

## Quick start

```python
import compton_matrix._compton_differential_cross_section as cds
from compton_matrix._units import kev, kev_kelvin

kernel = cds.ComptonKernelSolver()
result = kernel.sigma_E(
    E=10.0 * kev,
    E_prime=12.0 * kev,
    xi=0.0,
    T=10.0 * kev_kelvin,
)

print(result.value, result.estimated_rel_error)
```

## Examples

The [`examples`](examples/) directory contains runnable demonstrations:

| Script | Demonstrates |
|---|---|
| `point_kernel.py` | Pointwise kernel and temperature derivative |
| `multigroup_matrix.py` | Deterministic angle-resolved matrices and full temperature derivatives |
| `monte_carlo_comparison.py` | Deterministic and Monte Carlo matrix comparison |

Run an example from the repository root:

```bash
uv run python examples/point_kernel.py
```

Plots and NumPy data files are written to `examples/output/`.

## Development

```bash
just test
just check
```

## Reference

D. S. Kershaw, M. K. Prasad, and J. D. Beason, "A simple and fast method for
computing the relativistic Compton scattering kernel for radiative transfer,"
*Journal of Quantitative Spectroscopy and Radiative Transfer* 36(4), 273-282
(1986). [doi:10.1016/0022-4073(86)90050-6](https://doi.org/10.1016/0022-4073(86)90050-6)
