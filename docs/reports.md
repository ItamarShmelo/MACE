# Report Generation Scripts

This document describes the report generation scripts in `reports/` that produce validation reports as markdown files with embedded plots.

---

## General Usage

All report scripts are run from the project root:

```bash
python3 reports/<script_name>.py
```

Reports and their figures are written to `reports/generated/`:
```
reports/generated/
├── convergence_analysis.md
├── python_cpp_comparison.md
├── series_validation.md
└── figs/
    ├── *.png  (plots embedded in reports)
```

The `reports/generated/` directory is gitignored.

### Prerequisites

Before running any report:

1. Build the C++ modules:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
   ```

2. Build CMMC (for reports that compare against Monte Carlo):
   ```bash
   cmake -S external/CMMC -B external/CMMC/build -DCMAKE_BUILD_TYPE=Release
   cmake --build external/CMMC/build -j
   ```

3. Python dependencies: `numpy`, `scipy`, `matplotlib`, `pytest` (for tests)

---

## convergence_analysis.py

**Output:** `reports/generated/convergence_analysis.md`

Analyzes the convergence and numerical properties of the Gauss-Laguerre quadrature module.

### Sections

1. **Quadrature Order Convergence** — Compares NL=64, 128, 256 across test points.  Shows how the result changes with increasing quadrature order.

2. **Scipy Outer Integration Tolerance** — Sensitivity of the outer (energy/angle) integration to `epsrel` settings.

3. **Post-IBP vs Pre-IBP Agreement** — Verifies algebraic equivalence of the two quadrature forms across temperature regimes, highlighting where post-IBP cancellation degrades.

4. **Pointwise Error Estimates** — Examines the Richardson-style N vs N/2 error indicator returned by the C++ kernel.

5. **Performance Benchmarks** — Timing comparison of different quadrature orders and forms.

6. **Power Series Convergence** — Shows how the power series partial sum converges as `n_max` increases for warm/hot plasma cases.  Each case plots relative error vs quadrature reference as a function of the number of terms.

7. **Asymptotic Series Convergence** — Demonstrates the non-convergent (asymptotic) nature of the low-temperature series.  Top panels show relative error vs quadrature reference; bottom panels show the characteristic U-shaped term magnitude, revealing the optimal truncation point.

---

## python_cpp_comparison.py

**Output:** `reports/generated/python_cpp_comparison.md`

Compares the pure Python (`pycompton`) and C++ implementations against each other and against CMMC Monte Carlo.

### Sections

1. **Pointwise Kernel Agreement** — Python (fixed GL and adaptive) vs C++ quadrature at individual (E, E', ξ, τ) points.

2. **Pomraning Multigroup Comparison** — Multigroup scattering matrix S[g,g'] computed by Python, C++, and CMMC for the Pomraning benchmark problem (1 keV photon, 1 keV electrons, 30-group structure).

3. **Timing Comparison** — Wall-clock performance of Python vs C++ for quadrature evaluations.

4. **Python vs C++ Series Agreement** — Pointwise comparison of `pycompton.sigma_E_series` against `ComptonKernelSeries` for both power and asymptotic series methods.

5. **Series in Pomraning Plots** — Pomraning multigroup results with series overlaid on quadrature and CMMC curves.

6. **Series Timing** — Performance comparison of C++ quadrature vs C++ series vs Python series.

---

## series_validation.py

**Output:** `reports/generated/series_validation.md`

Validates the C++ series module against the C++ quadrature module and CMMC Monte Carlo.

### Sections

1. **Pointwise Series vs Quadrature** — Compares C++ series (Auto mode) against C++ quadrature (NL=256) at representative (E, E', ξ, τ) points.  Includes relative errors, method selection, convergence flags, and timing.

2. **Series vs Quadrature Spectra** — Plots σ(E') curves from both series and quadrature for several benchmark configurations.  Visual verification of spectral shape agreement.

3. **Pomraning Multigroup 3-Way Comparison** — Multigroup S[g,g'] computed by series, quadrature, and CMMC for the Pomraning problem.  Demonstrates that series results are indistinguishable from quadrature and consistent with Monte Carlo.

4. **Convergence Diagnostics** — Analyzes which series method Auto mode selects across the parameter space.  Shows terms used, convergence flags, and error estimates as a function of temperature and energy.

5. **Aggregate Timing Summary** — Performance comparison across all four methods: C++ quadrature, C++ series (Auto), Python quadrature, and Python series (Auto).  Bar chart visualization.

---

## derivative_validation.py

**Output:** `reports/generated/derivative_validation.md`

Validates the temperature derivative `dsigma_E_dtau` implementation.

### Sections

1. **Derivative GL Convergence** — Compares NL=64, 128, 256 Richardson error estimates for both pre-IBP and post-IBP derivative forms across representative test points.

2. **Finite-Difference Comparison** — Log-log plot of |FD − analytic| / |analytic| vs step size h, plus a table of Richardson-extrapolated FD vs analytic derivative at each test point.

3. **Pre-IBP vs Post-IBP Derivative Agreement** — Relative difference between the two forms as a function of temperature, showing where post-IBP degrades.

4. **Kappa Ratio Validation** — Relative error of C++ `scaled_K1` vs scipy `kve(1, x)`, and kappa(τ) vs cold/hot asymptotic limits.

5. **Small-tau Stability** — Table of derivative values and error estimates at low temperatures, verifying finiteness.

---

## Adding New Reports

Follow the conventions in existing scripts:

1. Place the script in `reports/`
2. Write output to `reports/generated/<name>.md` and figures to `reports/generated/figs/`
3. Use `matplotlib.use('Agg')` for headless rendering
4. Document the script in this file
5. Use the shared constants (`ME_C2`, `KEV`, `XI_EPS`) for unit conversions
