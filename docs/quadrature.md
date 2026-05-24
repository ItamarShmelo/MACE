# Direct Gauss-Laguerre Quadrature for the Compton Kernel

## Overview

This module implements the **Kershaw-Prasad-Beason (1986)** thermal Compton scattering frequency/energy kernel using direct Gauss-Laguerre quadrature.  The implementation is in C++ (C++20) with pybind11 bindings for Python.

The kernel describes photon scattering off a thermal Maxwell-Jüttner electron distribution.  It is fully relativistic and valid for arbitrary photon energies and electron temperatures.

**Reference:**  D. Kershaw, M. Prasad, and J. Beason, "Photon Transport in a Compton Scattering Medium," Technical Report UCRL-94345, 1986.

---

## Physical Setup

A photon with energy `E` scatters off a thermal electron gas at temperature `T`, emerging at energy `E'` and scattering angle `θ` (measured from the incident direction).

The **differential scattering kernel** gives the cross-section per unit final energy per unit solid angle:

```
Σ_E(E → E', ξ; τ, N_e)   [cm²/erg]  or  [1/(cm·erg)]
```

where:
- `ξ = cos(θ)` — scattering angle cosine
- `τ = kT / (m_e c²)` — dimensionless electron temperature
- `N_e` — electron number density (set to 1 for microscopic cross-section)

---

## Energy-Based API

The kernel is expressed in energy units (erg) rather than frequency.  Dimensionless photon energies:

```
γ  = E  / (m_e c²)
γ' = E' / (m_e c²)
```

All internal kinematic quantities depend only on `(γ, γ', ξ, τ)`.

---

## Kernel Factorization

The kernel factorizes into a prefactor and an integral:

```
Σ_E = σ₀(E, τ, λ₊, N_e)  ×  I_Q(γ, γ', ξ, τ)
```

### Prefactor σ₀

```
σ₀ = N_e · r_e² · m_e c² / (4 E² τ)  ·  exp(−(λ₊ − 1)/τ)  /  K̃₂(1/τ)
```

where:
- `r_e² = σ_Thomson / (8π/3)` — classical electron radius squared
- `K̃₂(x) = exp(x) · K₂(x)` — exponentially scaled modified Bessel function
- `λ₊` — minimum electron Lorentz factor capable of producing the transition

The exponential factor `exp(−(λ₊−1)/τ)` controls the kernel magnitude:
- **Elastic scattering** (E ≈ E', ξ → 1): λ₊ → 1, no suppression
- **Large energy transfer**: λ₊ ≫ 1, exponentially suppressed

### Integral I_Q

A semi-infinite integral over electron momentum, evaluated by Gauss-Laguerre quadrature after the change of variable `ρ = τx + ρ_offset`:

```
I_Q = ∫₀^∞  f(x) · e^{−x} dx  ≈  Σᵢ w_i · f(x_i)
```

Two mathematically equivalent forms are provided (see [Quadrature Forms](#quadrature-forms)).

---

## Kinematic Quantities

### Stable q² computation

To avoid catastrophic cancellation near ξ ~ 1 and γ ~ γ':

```
a    = 1 − ξ                              ∈ (0, 2]
dg   = γ' − γ
q²   = dg² + 2 · γ · γ' · a              (always ≥ 0)
q    = √(q²)
```

### Delta and λ₊

```
Δ  = √[(1 + γγ'a/2) · (1 + dg²/(2γγ'a))]
λ₊ = dg/2 + Δ
```

Physically `λ₊ ≥ 1` (minimum electron energy for the transition).  Numerical roundoff violations within 1e-12 are clamped; larger violations raise an error.

### Derived quantities

```
ρ₊     = λ₊ + γ
ρ₋     = λ₊ − γ'
ω²     = (1 + ξ) / a
s      = 1/γ + 1/γ'
α₊     = 1 / √(ρ₊² + ω²)
α₋     = 1 / √(ρ₋² + ω²)
G      = −γγ' + 2/a + 2/(γγ'a²)
A₊     = G − s/(τa²)
A₋     = G + s/(τa²)
Ψ      = 2τγγ'/q + s/a²·(α₊ + α₋) + (ρ₊α₊ − ρ₋α₋)/a
```

---

## Quadrature Forms

### Post-Integration-by-Parts (default)

After applying IBP to the original integral, the 1/R^{3/2} integrand becomes 1/√R (smoother, faster convergence), plus a boundary term Ψ:

```
Σ_E = σ₀ · [Ψ + I_Q^{post}]

I_Q^{post} = τ · Σᵢ w_i · H(τ · x_i)

H(ρ) = (A₊ − (ρ+ρ₊)/(τa)) / √R₊  +  (−A₋ + (ρ+ρ₋)/(τa)) / √R₋
```

where `R± = (ρ + ρ±)² + ω²`.

**Pros:** Fewer quadrature points needed for convergence at moderate/high τ.  
**Cons:** At very low τ, Ψ and I_Q nearly cancel (catastrophic cancellation).

### Pre-Integration-by-Parts (cross-check mode)

The original integral before IBP, containing 1/R±^{3/2} terms:

```
Σ_E = σ₀ · I_Q^{pre}

I_Q^{pre} = τ · Σᵢ w_i · F(τ · x_i)

F(ρ) = 2γγ'/q + [numerator±]/R±^{3/2}/a² + G·(1/√R₊ − 1/√R₋)
```

**Pros:** No boundary term, converges uniformly across all τ. Useful for validation.  
**Cons:** Slightly more terms per evaluation.

---

## Scaled K₂ Implementation

`scaled_K2(x)` computes `K̃₂(x) = exp(x) · K₂(x)`:

| Regime | Method | Rationale |
|--------|--------|-----------|
| x < 50 | `exp(x) · boost::cyl_bessel_k(2, x)` | Boost is accurate; exp(x) doesn't overflow |
| x ≥ 50 | Hankel asymptotic (5 terms) | Direct K₂ underflows, but K̃₂ ~ √(π/2x) |

The asymptotic expansion:
```
K̃_ν(x) ~ √(π/(2x)) · Σ_{k=0}^{4} Π_{j=0}^{k-1}(μ−(2j+1)²) / (k! · (8x)^k)
```
with μ = 4ν² = 16 for ν = 2.  Five terms give relative error < 10⁻¹⁵ for x ≥ 50.

---

## Error Estimation

Each evaluation returns a `SigmaResult` with:

| Field | Description |
|-------|-------------|
| `value` | The kernel value Σ_E |
| `estimated_abs_error` | `|σ₀| · |I_Q(N) − I_Q(N/2)|` |
| `estimated_rel_error` | `abs_error / (|value| + 10⁻³⁰⁰)` |

This is a Richardson-extrapolation-style heuristic: comparing the integral at full order N and half order N/2.  It is a convergence indicator, not a rigorous error bound.

---

## Angular Normalization

The kernel Σ_E is differential in both energy and solid angle.  To form the multigroup scattering matrix S[g,g'] (as used in transport codes), one integrates:

```
S[g, g'] = C_Ω · ∫₋₁¹ ∫_{E_lo}^{E_hi} Σ_E(E_g, E', ξ, τ) dE' dξ
```

The angular normalization factor C_Ω depends on the convention:

| Convention | C_Ω | Use case |
|------------|------|----------|
| Zeroth angular moment | 2π | Matches CMMC output |
| Angular average | 1/2 | Gives ⟨σ⟩ averaged over all directions |

**CMMC returns the zeroth angular moment** (factor 2π).

---

## Public Python API

```python
from cpp_modules._compton_kernel_quadrature import (
    ComptonKernelQuadrature, QuadratureForm, SigmaResult, scaled_K2
)

# Create engine with NL quadrature points (64, 128, or 256)
engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

# Scalar evaluation
result = engine.sigma_E(E, E_prime, xi, tau, Ne=1.0)
# result.value              -- kernel value [cm²/erg or 1/(cm·erg)]
# result.estimated_abs_error
# result.estimated_rel_error

# Vectorized over E'
values, errors = engine.sigma_E_vec(E, E_prime_array, xi, tau, Ne=1.0)

# Standalone scaled Bessel function
kve2 = scaled_K2(x)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `E` | float | Incoming photon energy [erg] |
| `E_prime` | float | Outgoing photon energy [erg] |
| `xi` | float | Cosine of scattering angle, strictly in (−1, 1) |
| `tau` | float | Dimensionless temperature kT/(m_e c²) |
| `Ne` | float | Electron density [1/cm³] or 1.0 for microscopic |
| `NL` | int | Gauss-Laguerre quadrature order: 64, 128, or 256 |

### Unit conversions

```python
ME_C2 = 9.109383713928e-28 * (2.99792458e10)**2  # m_e c² in erg
KEV   = 1.602176634e-9                            # 1 keV in erg

tau = T_kev * KEV / ME_C2       # temperature → dimensionless
E   = E_kev * KEV               # energy → erg
```

---

## Build Instructions

```bash
# From project root:

# 1. Build CMMC (for MC comparison tests)
cmake -S external/CMMC -B external/CMMC/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/CMMC/build -j

# 2. Build quadrature module
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Run fast deterministic tests
python3 -m pytest tests/test_deterministic.py

# 4. Run slow MC comparison
python3 -m pytest tests/test_vs_mc.py --run-slow

# 5. Generate comparison plots
python3 tests/plot_comparison.py
```

## Further Documentation

- [Gauss-Laguerre Algorithm Details](gauss_laguerre.md) — how nodes/weights are computed
- [Numerical Stability Design](numerical_stability.md) — techniques addressing overflow, cancellation, etc.
- [Edge Cases and Pathological Regimes](edge_cases.md) — where the code fails or degrades
- [CMMC Comparison and Artifacts](cmmc_comparison.md) — validation against Monte Carlo
- [Project Architecture](architecture.md) — source layout, build system, testing layers

## Dependencies

- C++20 compiler (GCC 12+, Clang 14+)
- Boost (header-only: `boost/math/special_functions/bessel.hpp` for `cyl_bessel_k`)
- pybind11
- Python 3.10+: NumPy, SciPy, pytest, matplotlib (for test plots)
