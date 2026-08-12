# Approximate Compton kernel benchmark and validation report

## Summary

`ComptonKernelApproximate` implements the explicit finite order-five Sazonov
coefficients, the `[3/2]` and `[2/3]` Padé continuations, and their
pole-suppressed blend. It deliberately uses no jet arithmetic and no
logarithmic kernel. The Maxwell-Jüttner exponential is evaluated directly, so
ordinary IEEE underflow returns zero.

The explicit coefficient formula is exceptionally fast but is not uniformly
accurate over the complete point-kernel domain. The default
`ComptonKernelApproximateSolver` profile is now calibrated for maximum
multigroup-multiangle throughput under a strict 5% significant-cell error
budget. It uses three backends:

1. asymptotic series when `tau * max(alpha_plus, alpha_minus) < 0.035` and
   `tau < 0.02`
2. explicit approximation when `tau <= 0.45`,
   `max(gamma, gamma_prime) <= 2.4`, and the explicit evaluation is finite
3. convergent power series otherwise

Cold points that fail the explicit path also use the asymptotic series. The
series backends retain their double-double fallbacks. The temperature cutoff
corresponds to about 229.95 keV and the photon-energy cutoff corresponds to
about 1.226 MeV. These limits were selected by sweeping both boundaries against
the production matrix integrator.

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
| Existing `ComptonKernelSolver` | 2,400 ns | 1.00x |
| Tuned `ComptonKernelApproximateSolver` | 748 ns | 3.21x |
| Raw approximation on its diagnostic-safe subset | 169 ns | 14.21x |

The tuned solver had no failures on the 3,024-point grid. Relative to the
previous strict 1% dispatcher, its median point throughput improved from
1,229 ns to 748 ns, or 1.64x.

## Point-kernel accuracy

The tuned profile deliberately optimizes integrated matrices rather than the
maximum error of every point evaluation. On the structured grid, comparison to
the existing solver gives:

| Metric | Relative error |
|---|---:|
| points | 3,024 |
| median | `5.36e-7` |
| 95th percentile | `8.81e-4` |
| 99th percentile | `1.17e-2` |
| maximum | `8.70e-1` |

The large pointwise maximum occurs in a very small redistribution tail. It is
why the tuned profile is specified and tested by matrix error, not by a global
pointwise error promise. The analytic temperature derivative keeps its strict
Padé disagreement gate of `1e-6` and was not relaxed by this tuning.

## Multigroup-multiangle matrices

Both solvers were passed through the same production
`ComptonMultigroupKernel` integration path. The benchmark uses four energy
groups with boundaries 0.1, 1, 10, 100, and 1000 keV, four angle bins, uniform
incident weighting, no group cutoff, and orders 24, 24, 24, 16, and 8 for the
configured integration rules.

### Single thread

| kT | Existing solver | Approximate solver | Speedup | Matrix L1 error | Maximum significant-cell error | Maximum row-sum error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 keV | 813.32 ms | 205.38 ms | 3.96x | `2.07e-8` | `1.01e-7` | `4.38e-8` |
| 10 keV | 382.89 ms | 110.84 ms | 3.45x | `7.27e-8` | `5.34e-7` | `2.37e-7` |
| 100 keV | 1321.16 ms | 125.65 ms | 10.52x | `5.30e-4` | `2.77e-3` | `2.00e-4` |

### Nine OpenMP threads

| kT | Existing solver | Approximate solver | Speedup |
|---:|---:|---:|---:|
| 1 keV | 416.08 ms | 74.01 ms | 5.62x |
| 10 keV | 166.01 ms | 44.88 ms | 3.70x |
| 100 keV | 602.57 ms | 66.32 ms | 9.09x |

Matrix errors use the existing solver with identical quadrature as the
reference. A significant cell is one whose magnitude exceeds `1e-8` times the
largest matrix entry.

## Five-percent stress calibration

The stress benchmark covers four matrix families:

- the standard four-group, four-angle uniform matrix
- an eight-group, eight-angle uniform matrix
- a six-group, eight-angle uniform matrix spanning 0.01 keV to 10 MeV
- an eight-group, eight-angle Wien-weighted matrix

At 229.9 keV, immediately below the selected temperature cutoff, all four
families remain below 5% maximum significant-cell error:

| Matrix family | L1 error | Maximum significant-cell error | Row-sum error |
|---|---:|---:|---:|
| Standard uniform | `5.30e-3` | `2.59e-2` | `2.88e-3` |
| Fine uniform | `5.94e-3` | `4.66e-2` | `2.92e-3` |
| Broad uniform | `5.77e-3` | `4.50e-2` | `4.35e-3` |
| Fine Wien | `6.02e-3` | `4.9796e-2` | `3.26e-3` |

At 230 keV the power-series path is selected. The worst significant-cell error
then drops below `1.24e-4`. Across the full committed stress sweep from 1 to
250 keV, the maximum observed significant-cell error is `4.9796e-2`.

The two closest rejected tuning candidates were:

| Candidate | Observed failure |
|---|---:|
| Maximum photon gamma `2.5` | `5.1706e-2` broad-matrix cell error |
| Asymptotic domain `0.10` | `3.2754e-1` cell error at 50 keV |

The selected maximum photon gamma is therefore `2.4`. Widening the cold
asymptotic domain was also slower around 15 to 30 keV, so its original `0.035`
limit remains optimal in this sweep.

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
OMP_NUM_THREADS=9 ./build/compton_multigroup_approximate_stress_benchmark
```

Timing depends on the compiler, CPU, and system load. Accuracy and dispatch
tests are deterministic.
