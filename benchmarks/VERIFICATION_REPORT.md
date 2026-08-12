# Verification report: Compton kernel approximate benchmark

## Environment

- CPU: INTEL(R) XEON(R) GOLD 6534
- visible cores: 16
- operating system: Linux 5.14.0-362.8.1.el9_3.x86_64
- compiler: gcc (GCC) 15.1.0
- Python: 3.12.1
- interface: pybind11 (Python bindings)

## Point-kernel accuracy

Comparison of `ComptonKernelApproximateSolver` against `ComptonKernelSolver` on the structured 3024-point grid:

| Metric | Value |
|---|---:|
| points | 3024 |
| solver failures | 0 |
| median relative error | `5.53e-07` |
| 95th percentile | `8.84e-04` |
| 99th percentile | `1.17e-02` |
| maximum | `8.70e-01` |

Raw `ComptonKernelApproximate` diagnostic statistics:

| Metric | Value |
|---|---:|
| accepted points | 1881 |
| failures | 199 |
| rejections (self-error >= 3e-4) | 944 |
| max relative error (accepted) | `4.02e-03` |

## Point-kernel performance

Timing via `sigma_E_vec` (vectorized C++ loop, 9 samples of 120 repeats):

| Evaluator | Median time | Relative speed |
|---|---:|---:|
| `ComptonKernelSolver` | 1947 ns | 1.00x |
| `ComptonKernelApproximateSolver` | 709 ns | 2.75x |
| Raw `ComptonKernelApproximate` (accepted subset) | 319 ns | 6.10x |

## Multigroup-multiangle matrices

Configuration: 4 groups [0.1, 1, 10, 100, 1000] keV, 4 angle bins, uniform weight, cutoff disabled.

### Single thread

| kT | Existing solver | Approximate solver | Speedup | Matrix L1 error | Maximum significant-cell error | Maximum row-sum error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 keV | 458.77 ms | 132.82 ms | 3.45x | `3.44e-08` | `1.74e-07` | `3.13e-08` |
| 10 keV | 226.67 ms | 70.28 ms | 3.23x | `1.39e-07` | `1.13e-06` | `4.53e-07` |
| 100 keV | 940.03 ms | 102.11 ms | 9.21x | `5.30e-04` | `2.77e-03` | `2.00e-04` |

### Fifteen OpenMP threads

| kT | Existing solver | Approximate solver | Speedup |
|---:|---:|---:|---:|
| 1 keV | 283.37 ms | 35.24 ms | 8.04x |
| 10 keV | 125.53 ms | 34.36 ms | 3.65x |
| 100 keV | 442.13 ms | 49.87 ms | 8.87x |

## Five-percent stress calibration

Four matrix families swept across 14 temperatures from 1 to 250 keV. Pass criterion: maximum significant-cell error < 5%.

### standard_uniform

| kT (keV) | L1 error | Max significant-cell error | Row-sum error | Pass |
|---:|---:|---:|---:|:---:|
| 1.0 | `3.44e-08` | `1.74e-07` | `3.13e-08` | yes |
| 10.0 | `1.39e-07` | `1.13e-06` | `4.53e-07` | yes |
| 50.0 | `3.46e-05` | `1.76e-04` | `1.45e-05` | yes |
| 100.0 | `5.30e-04` | `2.77e-03` | `2.00e-04` | yes |
| 150.0 | `1.87e-03` | `9.02e-03` | `8.47e-04` | yes |
| 200.0 | `3.87e-03` | `1.82e-02` | `1.99e-03` | yes |
| 220.0 | `4.81e-03` | `2.36e-02` | `2.58e-03` | yes |
| 224.0 | `4.99e-03` | `2.45e-02` | `2.70e-03` | yes |
| 225.0 | `5.05e-03` | `2.48e-02` | `2.72e-03` | yes |
| 228.0 | `5.19e-03` | `2.55e-02` | `2.82e-03` | yes |
| 229.0 | `5.24e-03` | `2.57e-02` | `2.85e-03` | yes |
| 229.9 | `5.30e-03` | `2.59e-02` | `2.88e-03` | yes |
| 230.0 | `1.29e-07` | `1.97e-06` | `2.79e-07` | yes |
| 250.0 | `6.39e-07` | `1.20e-05` | `9.53e-07` | yes |

### fine_uniform

| kT (keV) | L1 error | Max significant-cell error | Row-sum error | Pass |
|---:|---:|---:|---:|:---:|
| 1.0 | `1.33e-07` | `3.50e-06` | `2.71e-07` | yes |
| 10.0 | `1.86e-07` | `3.03e-06` | `1.25e-06` | yes |
| 50.0 | `3.79e-05` | `4.38e-04` | `1.84e-05` | yes |
| 100.0 | `5.48e-04` | `5.00e-03` | `2.09e-04` | yes |
| 150.0 | `1.98e-03` | `1.62e-02` | `8.59e-04` | yes |
| 200.0 | `4.26e-03` | `3.37e-02` | `2.03e-03` | yes |
| 220.0 | `5.34e-03` | `4.22e-02` | `2.61e-03` | yes |
| 224.0 | `5.58e-03` | `4.39e-02` | `2.73e-03` | yes |
| 225.0 | `5.63e-03` | `4.44e-02` | `2.77e-03` | yes |
| 228.0 | `5.81e-03` | `4.57e-02` | `2.86e-03` | yes |
| 229.0 | `5.88e-03` | `4.62e-02` | `2.89e-03` | yes |
| 229.9 | `5.94e-03` | `4.66e-02` | `2.92e-03` | yes |
| 230.0 | `8.09e-07` | `1.96e-05` | `1.06e-06` | yes |
| 250.0 | `8.83e-07` | `7.27e-05` | `2.61e-06` | yes |

### broad_uniform

| kT (keV) | L1 error | Max significant-cell error | Row-sum error | Pass |
|---:|---:|---:|---:|:---:|
| 1.0 | `5.44e-06` | `6.58e-05` | `5.65e-06` | yes |
| 10.0 | `9.01e-06` | `1.05e-04` | `3.48e-05` | yes |
| 50.0 | `5.32e-04` | `1.67e-02` | `2.17e-03` | yes |
| 100.0 | `9.39e-04` | `1.69e-02` | `1.99e-03` | yes |
| 150.0 | `2.25e-03` | `2.81e-02` | `2.49e-03` | yes |
| 200.0 | `4.23e-03` | `4.13e-02` | `3.16e-03` | yes |
| 220.0 | `5.22e-03` | `4.43e-02` | `3.77e-03` | yes |
| 224.0 | `5.42e-03` | `4.49e-02` | `3.95e-03` | yes |
| 225.0 | `5.52e-03` | `4.53e-02` | `4.19e-03` | yes |
| 228.0 | `5.58e-03` | `4.54e-02` | `3.89e-03` | yes |
| 229.0 | `5.70e-03` | `4.58e-02` | `4.25e-03` | yes |
| 229.9 | `5.75e-03` | `4.50e-02` | `4.28e-03` | yes |
| 230.0 | `8.75e-06` | `1.46e-04` | `2.05e-05` | yes |
| 250.0 | `7.70e-06` | `1.23e-04` | `8.04e-06` | yes |

### fine_wien

| kT (keV) | L1 error | Max significant-cell error | Row-sum error | Pass |
|---:|---:|---:|---:|:---:|
| 1.0 | `4.80e-16` | `2.95e-14` | `4.23e-07` | yes |
| 10.0 | `5.74e-08` | `1.57e-06` | `1.56e-07` | yes |
| 50.0 | `3.64e-05` | `4.13e-04` | `1.88e-05` | yes |
| 100.0 | `5.46e-04` | `5.11e-03` | `2.07e-04` | yes |
| 150.0 | `2.07e-03` | `1.52e-02` | `1.10e-03` | yes |
| 200.0 | `4.30e-03` | `3.41e-02` | `2.03e-03` | yes |
| 220.0 | `5.43e-03` | `4.44e-02` | `2.72e-03` | yes |
| 224.0 | `5.66e-03` | `4.65e-02` | `2.91e-03` | yes |
| 225.0 | `5.72e-03` | `4.71e-02` | `2.97e-03` | yes |
| 228.0 | `5.91e-03` | `4.87e-02` | `3.14e-03` | yes |
| 229.0 | `5.97e-03` | `4.93e-02` | `3.20e-03` | yes |
| 229.9 | `6.02e-03` | `4.98e-02` | `3.26e-03` | yes |
| 230.0 | `4.09e-07` | `5.86e-05` | `1.48e-06` | yes |
| 250.0 | `3.22e-07` | `2.95e-05` | `6.49e-07` | yes |

## Summary

**PASSED**: All significant-cell errors remain below 5%.

- Worst significant-cell error: `4.98e-02` (fine_wien at 229.9 keV)
- Point-kernel solver speedup: 2.75x
- Multigroup speedup (single-thread, T=100 keV): 9.21x
- Multigroup speedup (15 threads, T=100 keV): 8.87x
