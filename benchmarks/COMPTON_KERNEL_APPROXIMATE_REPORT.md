# Approximate Compton kernel benchmark and validation report

## Summary

`ComptonKernelApproximate` implements the explicit finite order-five Sazonov
coefficients, the `[3/2]` and `[2/3]` Padé continuations, and their
pole-suppressed blend. It deliberately uses no jet arithmetic and no
logarithmic kernel. The Maxwell-Jüttner exponential is evaluated directly, so
ordinary IEEE underflow returns zero.

The explicit coefficient formula is exceptionally fast where its cancellation
diagnostic passes. It is not reliable across the complete low-energy domain.
`ComptonKernelApproximateSolver` therefore uses a calibrated three-backend
selection:

1. asymptotic series in its cold expansion domain
2. explicit approximation when its coefficient and Padé diagnostics pass
3. convergent power series everywhere else

The series backends retain their double-double fallbacks. The complete solver
had no failures and stayed below one percent error in every structured and
randomized validation point described below.

## Environment

- CPU: Intel Xeon Platinum 8573C
- visible cores: 9
- operating system: Linux 6.18.35
- compiler: GCC 13.3.0
- build flags: `-std=c++23 -O3 -march=native -flto -DNDEBUG`
- fast math: disabled

## Point-kernel performance

The structured benchmark has 3,024 points:

- temperatures: 0.01, 0.1, 1, 5, 10, 25, 50, 70, and 100 keV
- incident ratios `E/kT`: `1e-3`, `1e-2`, `0.1`, `1`, `3`, and `10`
- eight scattering angles from `-0.999` through `0.999`
- seven outgoing-energy offsets around the recoil-shifted line

Each timing sample performs 362,880 kernel evaluations. Nine samples are
collected and the median is reported.

| Evaluator | Median time | Relative speed |
|---|---:|---:|
| Existing `ComptonKernelSolver` | 2,303 ns | 1.00x |
| `ComptonKernelApproximateSolver` | 1,229 ns | 1.87x |
| Accepted `ComptonKernelApproximate` points | 159 ns | 14.45x |

The explicit approximation accepted 1,881 of 3,024 points. Its diagnostic
rejected 946 points and returned a failure sentinel at 197 cancellation-damaged
points. The robust solver redirected every rejected or failed point to a series
backend.

## Point-kernel accuracy

The exact reference evaluates the monoenergetic redistribution factor under
the Maxwell-Jüttner integral with Gauss-Laguerre nodes. These figures compare
the complete approximate solver to that integral, not merely to the existing
MACE solver.

### Structured grid

| Metric | Relative error |
|---|---:|
| points | 3,024 |
| median | `5.48e-8` |
| 95th percentile | `2.51e-4` |
| 99th percentile | `1.02e-3` |
| maximum | `4.02e-3` |
| points at or above 1% | 0 |

### Randomized validation

The randomized sweep contains 5,000 deterministic samples over:

- `kT` from 0.01 through 300 keV, log-uniform
- `E/kT` from `1e-4` through 30, log-uniform
- `xi` from `-0.999` through `0.999`
- outgoing energy from minus six through plus six local thermal widths

One exact reference value underflowed in binary64, leaving 4,999 comparisons.

| Metric | Relative error |
|---|---:|
| median | `1.20e-8` |
| 95th percentile | `4.98e-4` |
| 99th percentile | `1.77e-3` |
| 99.9th percentile | `5.78e-3` |
| maximum | `8.17e-3` |
| points at or above 1% | 0 |

The analytic temperature derivative was independently checked at 2,000
randomized points against a centered derivative of the exact thermal integral.
Its maximum relative error was `9.77e-3`, with no point at or above one percent.
Points whose exact derivative was less than `1e-3 * sigma/T` are excluded from
this relative derivative metric.

## Multigroup-multiangle matrices

Both solvers were passed through the same production
`ComptonMultigroupKernel` integration path. The benchmark uses four energy
groups with boundaries 0.1, 1, 10, 100, and 1000 keV, four angle bins, uniform
incident weighting, no group cutoff, and orders 24, 24, 24, 16, and 8 for the
configured integration rules.

### Single thread

| kT | Existing solver | Approximate solver | Speedup | Matrix L1 error | Maximum significant-cell error | Maximum row-sum error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 keV | 597.46 ms | 182.31 ms | 3.28x | `2.07e-8` | `1.01e-7` | `4.38e-8` |
| 10 keV | 304.21 ms | 96.45 ms | 3.15x | `7.27e-8` | `5.34e-7` | `2.37e-7` |
| 100 keV | 1247.15 ms | 169.91 ms | 7.34x | `2.86e-4` | `8.22e-4` | `1.43e-4` |

### Nine OpenMP threads

| kT | Existing solver | Approximate solver | Speedup |
|---:|---:|---:|---:|
| 1 keV | 398.40 ms | 72.09 ms | 5.53x |
| 10 keV | 177.45 ms | 46.41 ms | 3.82x |
| 100 keV | 732.73 ms | 76.44 ms | 9.59x |

Matrix errors use the existing solver with identical quadrature as the
reference. A significant cell is one whose magnitude exceeds `1e-8` times the
largest matrix entry.

## Reproduction

Configure the optional benchmark targets after installing the normal project
dependencies:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOMPTON_BUILD_BENCHMARKS=ON
cmake --build build -j

./build/compton_kernel_approximate_benchmark
OMP_NUM_THREADS=1 ./build/compton_multigroup_approximate_benchmark
OMP_NUM_THREADS=9 ./build/compton_multigroup_approximate_benchmark
```

Timing depends on the compiler, CPU, and system load. Accuracy and dispatch
tests are deterministic.
