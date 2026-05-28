# Project Architecture

## Directory Layout

```
compton_cross_section/
├── src/
│   ├── compton_common/                     # Shared kinematics and normalization
│   │   ├── compton_common.hpp             # KershawParams, compute_params, etc.
│   │   └── compton_common.cpp             # Implementations
│   ├── compton_kernel_quadrature/          # C++ quadrature kernel
│   │   ├── compton_kernel_quadrature.hpp   # QuadratureForm, ComptonKernelQuadrature
│   │   ├── compton_kernel_quadrature.cpp   # Gauss-Laguerre evaluation
│   │   ├── gauss_laguerre.hpp              # Custom Gauss-Laguerre quadrature (header-only)
│   │   └── bind/
│   │       └── _compton_kernel_quadrature.cpp  # pybind11 bindings
│   ├── compton_kernel_series/              # C++ series kernel
│   │   ├── compton_kernel_series.hpp       # SeriesMethod, SeriesResult, ComptonKernelSeries
│   │   ├── compton_kernel_series.cpp       # Power series + asymptotic series
│   │   └── bind/
│   │       └── _compton_kernel_series.cpp  # pybind11 bindings
│   ├── compton_kernel_solver/              # Robust adaptive solver
│   │   ├── compton_kernel_solver.hpp       # SolverMethod, SolverResult, ComptonKernelSolver
│   │   ├── compton_kernel_solver.cpp       # Cascade logic: asymptotic → power → quadrature
│   │   └── bind/
│   │       └── _compton_kernel_solver.cpp  # pybind11 bindings
│   └── python/
│       └── pycompton/                      # Pure Python reimplementation
│           ├── __init__.py
│           ├── compton_kernel_quadrature.py
│           └── compton_kernel_series.py
├── cpp_modules/                        # Build output (shared libraries)
│   ├── _compton_kernel_quadrature.cpython-*.so
│   ├── _compton_kernel_series.cpython-*.so
│   └── _compton_kernel_solver.cpython-*.so
├── external/
│   └── CMMC/                           # Reference Monte Carlo implementation
│       ├── src/
│       │   ├── compton_matrix_mc.{hpp,cpp}
│       │   ├── units/units.hpp
│       │   └── planck_integral/
│       └── cpp_modules/                # CMMC build output
├── tests/
│   ├── conftest.py                     # pytest --run-slow flag
│   ├── test_deterministic.py           # Fast unit tests (+ GL nodes vs scipy)
│   ├── test_python_vs_cpp.py           # Python vs C++ pointwise comparison
│   ├── test_series_python.py           # Python series validation (Phase 2f)
│   ├── test_series.py                  # C++ series tests (Phase 4)
│   ├── test_vs_mc.py                   # Slow MC comparison tests (+ Pomraning)
│   ├── plot_comparison.py              # Generate comparison plots
│   └── output/                         # Plot outputs (PNG, PDF)
├── scripts/
│   └── benchmark_python_cpp_mc.py      # Timing + accuracy benchmark
├── reports/
│   ├── convergence_analysis.py         # Convergence report generator (with plots)
│   ├── python_cpp_comparison.py        # Python vs C++ report generator (with plots)
│   ├── series_validation.py            # Series validation report (with plots)
│   ├── solver_validation.py            # Solver validation report (with plots)
│   └── generated/                      # Generated reports and plots (gitignored)
├── docs/
│   ├── quadrature.md                   # Main documentation
│   ├── gauss_laguerre.md               # Quadrature algorithm details
│   ├── numerical_stability.md          # Stability techniques
│   ├── edge_cases.md                   # Failure modes and safe operating envelope
│   ├── final_equations.md              # Step-by-step equations matching code
│   ├── cmmc_comparison.md              # CMMC comparison and artifacts
│   ├── python_implementation.md        # Pure Python pycompton package
│   ├── series.md                       # Series methods documentation
│   ├── solver.md                       # Robust solver documentation
│   ├── reports.md                      # Report generation scripts
│   └── architecture.md                 # This file
├── CMakeLists.txt                      # Build configuration
└── build/                              # CMake build directory
```

---

## Module Dependency Graph

```
units/units.hpp  (from CMMC: physical constants)
       │
       ▼
compton_common.hpp  (shared kinematics & normalization)
       │
       ├────────────────────────────────┐
       ▼                                ▼
compton_kernel_quadrature.hpp    compton_kernel_series.hpp
       │                                │
       ├────────┐                       ├────────┐
       ▼        ▼                       ▼        ▼
gauss_laguerre  boost/bessel      boost/expint   <cmath>
       │                                │
       ▼                                ▼
*.cpp (implementation)           *.cpp (implementation)
       │                                │
       ▼                                ▼
bind/_compton_kernel_quadrature  bind/_compton_kernel_series
       │                                │
       ▼                                ▼
_compton_kernel_quadrature.so    _compton_kernel_series.so

       ├────────────────────────────────┤
       ▼                                ▼
       compton_kernel_solver.hpp ────────┘
                │
                ▼
       compton_kernel_solver.cpp (cascade logic)
                │
                ▼
       bind/_compton_kernel_solver.cpp
                │
                ▼
       _compton_kernel_solver.so
```

---

## Source Files

### Shared Kinematics (`src/compton_common/`)

#### `compton_common.hpp`

**Role:** Shared interface for kinematics and normalization used by both quadrature and series modules.

Declares:
- `scaled_K2(x)` — scaled Bessel function K̃₂(x) = exp(x) K₂(x)
- `KershawParams` — kinematic intermediates struct
- `SigmaResult` — return type with value + error estimates
- `compute_params(gamma, gamma_p, xi, tau)` — free function deriving all kinematic quantities
- `stable_sigma0_E(E, tau, lambda_plus, Ne)` — prefactor computation

#### `compton_common.cpp`

**Role:** Implementations of scaled_K2, compute_params, and stable_sigma0_E.

### C++ Quadrature Kernel (`src/compton_kernel_quadrature/`)

#### `compton_kernel_quadrature.hpp`

**Role:** Public interface for quadrature evaluation.

Declares:
- `QuadratureForm` — enum selecting post-IBP vs pre-IBP
- `ComptonKernelQuadrature` — main evaluation class

Includes `compton_common.hpp` for shared types.

#### `compton_kernel_quadrature.cpp`

**Role:** Gauss-Laguerre quadrature evaluation.

Contains:
- `compute_IQ_post_ibp` and `compute_IQ_pre_ibp` performing the quadrature
- `sigma_E` tying prefactor and quadrature together
- Static rule cache (`get_rule`)
- Gauss-Laguerre integration helper template

### C++ Series Kernel (`src/compton_kernel_series/`)

#### `compton_kernel_series.hpp`

**Role:** Public interface for series evaluation.

Declares:
- `SeriesMethod` — enum: PowerSeries, Asymptotic, Auto
- `SeriesResult` — struct with value, error estimates, terms_used, method_used, converged
- `ComptonKernelSeries` — main evaluation class
- `ehat_expn(m, x)` — scaled exponential integral Ê_m(x) = exp(x) E_m(x)

#### `compton_kernel_series.cpp`

**Role:** Power series and asymptotic series implementations.

Contains:
- `ehat_expn` with two-regime strategy (direct for x<50, asymptotic for x≥50)
- Power series loop with Poisson weights and per-term Boost `expint` calls
- Asymptotic series with Legendre recurrence and smallest-term truncation
- Auto switching based on tau*alpha threshold

#### `gauss_laguerre.hpp`

**Role:** Header-only quadrature rule generation.

Self-contained implementation of:
- `GaussLaguerreRule` struct (nodes + weights)
- `detail::tql2` — implicit QL eigenvalue algorithm
- `compute_gauss_laguerre(N)` — Golub-Welsch algorithm

No external dependencies beyond the standard library.

#### `bind/_compton_kernel_quadrature.cpp`

**Role:** Python ↔ C++ bridge via pybind11.

Exposes:
- `QuadratureForm` enum → Python enum
- `SigmaResult` struct → Python object with readonly attributes
- `ComptonKernelQuadrature` class → Python class with `sigma_E` and `sigma_E_vec`
- `scaled_K2` → standalone Python function
- `gauss_laguerre_rule(N)` → returns (nodes, weights) tuple for testing

The vectorized `sigma_E_vec` loops in C++ over an input NumPy array, avoiding per-element Python overhead.  GIL is held (no threading benefit) for simplicity.

### Robust Solver (`src/compton_kernel_solver/`)

#### `compton_kernel_solver.hpp`

**Role:** Public interface for the adaptive solver.

Declares:
- `SolverMethod` — enum: Asymptotic, PowerSeries, Quadrature
- `SolverResult` — struct with value, error estimates, method_used, target_met, diagnostics
- `ComptonKernelSolver` — main class with cascade evaluation logic

#### `compton_kernel_solver.cpp`

**Role:** Cascade method selection for robust kernel evaluation.

The solver tries methods in order of preference (asymptotic first for low tau_alpha_max,
then power series, then asymptotic as fallback for cancellation cases, then quadrature).
Each attempt is locally validated via estimated_rel_error before acceptance. Includes
non-negative clamping, negligibility detection, and diagnostic reporting.

See [Solver Documentation](solver.md) for full algorithmic details.

#### `bind/_compton_kernel_solver.cpp`

**Role:** Python bindings for the solver via pybind11.

Exposes `ComptonKernelSolver` class with scalar `sigma_E` and vectorized `sigma_E_vec`,
plus `SolverResult` and `SolverMethod` types.

### Pure Python (`src/python/pycompton/`)

#### `compton_kernel_quadrature.py`

**Role:** Complete reimplementation of the C++ quadrature kernel using only numpy and scipy.

Provides two integration modes:
- **Fixed Gauss-Laguerre** (`method="fixed"`): uses `scipy.special.roots_laguerre` for direct comparison with C++
- **Adaptive quadrature** (`method="adaptive"`): uses `scipy.integrate.quad` with explicit e^{-x} weight as a diagnostic cross-check

See [Python Implementation](python_implementation.md) for details.

#### `compton_kernel_series.py`

**Role:** Pure Python reimplementation of the C++ series kernel.

Key components:
- `ehat_expn(m, x)` — scaled exponential integral with two-regime strategy (direct for x<50, asymptotic for x≥50)
- `sigma_E_series(gamma, gamma_p, xi, tau, ...)` — top-level function with power series, asymptotic series, and auto-switching
- `SeriesResult` dataclass — output with value, error estimates, convergence flag, method used, terms count

See [Series Methods](series.md) for algorithmic details.

---

## Build System

### CMakeLists.txt

- Requires CMake ≥ 3.22 and a C++20 compiler
- Finds Boost (header-only) and pybind11 via `find_package`
- Includes `external/CMMC/src` (for `units/units.hpp`)
- Produces three shared library targets: `_compton_kernel_quadrature`, `_compton_kernel_series`, and `_compton_kernel_solver`
- Each links against `compton_common.cpp` (compiled into each module)
- The solver module links both series and quadrature source files
- Output goes to `cpp_modules/` for direct Python import
- Compiler flags: `-O3 -Wall -Wextra -Wpedantic`

### Building

```bash
# Quadrature module
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CMMC reference (needed for comparison tests)
cmake -S external/CMMC -B external/CMMC/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/CMMC/build -j
```

---

## Testing Architecture

### Layer 1: Fast Deterministic Tests (`test_deterministic.py`)

Run in < 5 seconds.  No randomness, no external dependencies beyond the C++ module.

| Test | What it verifies |
|------|------------------|
| `TestScaledK2` | scaled_K2 matches scipy.special.kve for all x ranges |
| `TestFiniteOutput` | No NaN/Inf for representative points |
| `TestPositivity` | Σ_E ≥ 0 (within tolerance) |
| `TestDetailedBalance` | E² e^{−γ/τ} Σ(E→E') = E'² e^{−γ'/τ} Σ(E'→E) |
| `TestNLConvergence` | NL=128 and NL=256 agree to 5×10⁻⁶ |
| `TestPostVsPreIBP` | Both forms agree (relaxed tolerance for small τ) |
| `TestAngularNormalization` | Total cross section ~ σ_Thomson for low-energy photons |
| `TestGaussLaguerreVsScipy` | C++ GL nodes/weights match scipy to rtol=1e-11 |

### Layer 2: Python vs C++ Tests (`test_python_vs_cpp.py`)

Pointwise correctness comparison between the pure Python and C++ implementations.

| Test | What it verifies |
|------|------------------|
| `TestFixedRuleAgreement` | Python fixed-GL matches C++ for post/pre-IBP at NL=64,128,256 |
| `TestFixedRuleAgreement.test_error_estimates` | Richardson error estimates agree in order of magnitude |
| `TestAdaptiveDiagnostic` | Python adaptive quad agrees with C++ within 1e-4 |

### Layer 2b: Python Series Validation (`test_series_python.py`)

Python-first tests that validate the series formulas against quadrature, without requiring the C++ series module.

| Test | What it verifies |
|------|------------------|
| `TestEhatExpn` | `ehat_expn` matches scipy across all regimes and orders |
| `TestPowerSeriesVsQuadrature` | Python power series agrees with Python quadrature (warm/hot τ) |
| `TestAsymptoticSeriesVsQuadrature` | Python asymptotic series agrees at low τ |
| `TestAutoSeriesVsQuadrature` | Auto switching produces correct results across all test points |
| `TestDetailedBalanceSeries` | Detailed balance holds for series results |
| `TestPositivitySeries` | Series returns non-negative values |

### Layer 2c: C++ Series Tests (`test_series.py`)

C++ series validation, including cross-module and cross-language comparisons.

| Test | What it verifies |
|------|------------------|
| `TestEhatExpn` | C++ `ehat_expn` matches scipy for all orders and x ranges |
| `TestSeriesVsQuadrature` | C++ series vs C++ quadrature agreement |
| `TestPythonVsCpp` | Python and C++ series produce consistent results |
| `TestDetailedBalance` | Detailed balance for C++ series |
| `TestPositivity` | Non-negativity for C++ series |
| `TestConvergenceFlags` | Convergence flags are consistent with method selection |

### Layer 2d: Solver Tests (`test_solver.py`)

Validates the adaptive solver's cascade logic, accuracy, and edge-case handling.

| Test | What it verifies |
|------|------------------|
| `TestDomainValidation` | Exceptions for invalid inputs (E<=0, tau<=0, xi=+-1) |
| `TestMethodSelection` | Correct dispatch (low tau → asymptotic, high tau → power/quad) |
| `TestAccuracy` | Solver vs Q256 agreement; target_met semantics |
| `TestEdgeCases` | xi near +-1, near-elastic, sigma0 underflow, high tau |
| `TestOutOfDomain` | Beyond calibration grid: still works, reports honestly |
| `TestPhysicalConsistency` | Non-negativity; Q256/Q128 convergence cross-check |
| `TestVectorized` | sigma_E_vec matches scalar sigma_E |

### Layer 3: Slow MC Comparison Tests (`test_vs_mc.py`)

Run with `--run-slow`.  Compare multigroup S-matrices against CMMC.

- T = 100 keV: off-diagonal elements, wide bins
- T = 20 keV: neighboring groups, wide bins
- T = 1 keV: diagonal elements only (exponentially suppressed off-diagonal)
- Pomraning cases: T=1 keV (low/high) and T=20 keV (low/high) with coarse bins

### Layer 4: Comparison Plots (`plot_comparison.py`)

Visual validation.  Generates σ(E') vs E' for multiple incident energies at each temperature.

### Layer 5: Reports (`reports/`)

Report-generating scripts that produce markdown documents with embedded plots.  See [Reports](reports.md) for details.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| C++20 | `std::numbers::pi`, structured bindings, constexpr improvements |
| Header-only Gauss-Laguerre | Avoids extra translation unit; code is small and inlined |
| Static rule cache | Rules are expensive (O(N²)) but only needed at a few fixed N |
| pybind11 (not ctypes/cython) | Clean C++ → Python bridge with NumPy support |
| Boost for K₂ only | Minimal dependency; rest is standard library |
| No GIL release | Simplicity over threading; vectorization in C++ loop suffices |
| Pre- and post-IBP forms | Cross-validation and regime-appropriate convergence |
| Richardson error estimate | Cheap (one extra half-order evaluation), practical indicator |
| compton_common extraction | Shared kinematics avoids duplication between quadrature and series modules |
| Stepwise-guarded forward recurrence for ehat_expn | In-loop recurrence `Ê_{m+1} = (1 - x·Ê_m)/m` with per-chain amplification budget (1e2). After fallback, `amp` resets to allow recurrence to resume from fresh seed. Standalone `ehat_expn()` retained for external use and as per-step fallback. |
| Legendre three-term recurrence | In-loop `P_{n+2} = ((2n+3)·z·P_{n+1} - (n+1)·P_n) / (n+2)` replaces per-term `boost::math::legendre_p` calls. Boost Legendre header removed from series module. |
| Python-first validation | Series formulas validated in Python (scipy) before C++ port, catching formula bugs early |
| Series-or-fail Auto | Auto mode returns `converged = false` rather than silently falling back to quadrature, keeping module boundaries clean |
| Smallest-term truncation | Standard optimal truncation for asymptotic (non-convergent) series, with two-consecutive-increase safeguard |
| Solver cascade with local validation | Dispatch threshold decides what to try first (performance); validity threshold decides acceptance (correctness). Two distinct concepts documented separately. |
| Conditioning-aware power series error | Error = max(truncation, conditioning * eps * safety_factor). Prevents accepting results with hidden cancellation-driven inaccuracy. |
| Honest quadrature error reporting | Solver does not claim 1e-8 if quadrature fallback cannot achieve it; `target_met` flag exposes this to callers. |
