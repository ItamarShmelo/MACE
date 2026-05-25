# Project Architecture

## Directory Layout

```
compton_cross_section/
├── src/
│   ├── compton_kernel_quadrature/          # C++ kernel implementation
│   │   ├── compton_kernel_quadrature.hpp   # Public API: classes, structs, enums
│   │   ├── compton_kernel_quadrature.cpp   # Implementation of the Compton kernel
│   │   ├── gauss_laguerre.hpp              # Custom Gauss-Laguerre quadrature (header-only)
│   │   └── bind/
│   │       └── _compton_kernel_quadrature.cpp  # pybind11 bindings
│   └── python/
│       └── pycompton/                      # Pure Python reimplementation
│           ├── __init__.py
│           └── compton_kernel_quadrature.py
├── cpp_modules/                        # Build output (shared libraries)
│   └── _compton_kernel_quadrature.cpython-*.so
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
│   ├── test_vs_mc.py                   # Slow MC comparison tests (+ Pomraning)
│   ├── plot_comparison.py              # Generate comparison plots
│   └── output/                         # Plot outputs (PNG, PDF)
├── scripts/
│   └── benchmark_python_cpp_mc.py      # Timing + accuracy benchmark
├── reports/
│   ├── convergence_analysis.py         # Convergence report generator (with plots)
│   ├── python_cpp_comparison.py        # Python vs C++ report generator (with plots)
│   └── generated/                      # Generated reports and plots (gitignored)
├── docs/
│   ├── quadrature.md                   # Main documentation
│   ├── gauss_laguerre.md               # Quadrature algorithm details
│   ├── numerical_stability.md          # Stability techniques
│   ├── edge_cases.md                   # Failure modes and safe operating envelope
│   ├── final_equations.md              # Step-by-step equations matching code
│   ├── cmmc_comparison.md              # CMMC comparison and artifacts
│   ├── python_implementation.md        # Pure Python pycompton package
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
compton_kernel_quadrature.hpp  (API declarations)
       │
       ├──────────────────────┐
       ▼                      ▼
gauss_laguerre.hpp     boost/math/bessel.hpp
       │                      │
       └──────────┬───────────┘
                  ▼
compton_kernel_quadrature.cpp  (implementation)
                  │
                  ▼
bind/_compton_kernel_quadrature.cpp  (pybind11)
                  │
                  ▼
_compton_kernel_quadrature.so  (Python module)
```

---

## Source Files

### C++ Kernel (`src/compton_kernel_quadrature/`)

#### `compton_kernel_quadrature.hpp`

**Role:** Public interface.

Declares:
- `r_e2` — classical electron radius squared (constexpr)
- `scaled_K2(x)` — scaled Bessel function
- `KershawParams` — kinematic intermediates struct
- `SigmaResult` — return type with value + error estimates
- `QuadratureForm` — enum selecting post-IBP vs pre-IBP
- `ComptonKernelQuadrature` — main evaluation class

No implementation details are exposed.  The header depends only on `<cmath>`, `<numbers>`, `<stdexcept>`, and `units/units.hpp` (for physical constants).

#### `compton_kernel_quadrature.cpp`

**Role:** Core physics and numerics.

Contains all the heavy computation:
- `scaled_K2` with Boost + asymptotic paths
- `compute_params` deriving kinematic quantities
- `stable_sigma0_E` computing the prefactor
- `compute_IQ_post_ibp` and `compute_IQ_pre_ibp` performing the quadrature
- `sigma_E` tying everything together
- Static rule cache (`get_rule`)
- Gauss-Laguerre integration helper template

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

### Pure Python (`src/python/pycompton/`)

#### `compton_kernel_quadrature.py`

**Role:** Complete reimplementation of the C++ kernel using only numpy and scipy.

Provides two integration modes:
- **Fixed Gauss-Laguerre** (`method="fixed"`): uses `scipy.special.roots_laguerre` for direct comparison with C++
- **Adaptive quadrature** (`method="adaptive"`): uses `scipy.integrate.quad` with explicit e^{-x} weight as a diagnostic cross-check

See [Python Implementation](python_implementation.md) for details.

---

## Build System

### CMakeLists.txt

- Requires CMake ≥ 3.22 and a C++20 compiler
- Finds Boost (header-only) and pybind11 via `find_package`
- Includes `external/CMMC/src` (for `units/units.hpp`)
- Produces a single shared library target: `_compton_kernel_quadrature`
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
