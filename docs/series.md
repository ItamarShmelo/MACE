# Series Methods for the Compton Kernel

## Overview

This module implements the **Kershaw-Prasad-Beason** thermal Compton scattering kernel using closed-form series expansions rather than numerical quadrature.  The same physical kernel $\Sigma_E$ is evaluated, but instead of Gauss-Laguerre integration over electron momentum, the electron integral is expressed analytically in terms of special functions.

Two complementary series are provided:
- **Power series**: sums Poisson-weighted scaled exponential integrals $\hat{E}_m(x)$
- **Low-temperature asymptotic series**: expands in powers of $\tau \cdot \alpha_\pm$ with Legendre polynomial coefficients

An **auto mode** selects the appropriate series based on the parameter regime.

**Reference:** D. Kershaw, M. Prasad, and J. Beason, "Photon Transport in a Compton Scattering Medium," Technical Report UCRL-94345, 1986, Section 4.

---

## Physical Setup

Identical to the [quadrature module](quadrature.md): a photon with energy $E$ scatters off a thermal Maxwell-Jüttner electron gas at temperature $\tau = kT / (m_e c^2)$, emerging at energy $E'$ and angle $\xi = \cos\theta$.

The kernel factorizes as:

```
Σ_E = σ₀(E, τ, λ₊, N_e) × [normalized ratio]
```

The prefactor $\sigma_0$ and all kinematic quantities ($\lambda_+$, $\rho_\pm$, $\omega^2$, $\alpha_\pm$, $\Psi$, $G$, $A_\pm$, etc.) are shared with the quadrature module and computed by `compton_common`.  Only the evaluation of the normalized ratio differs.

---

## Power Series Method

### When It Works Best

The power series converges rapidly when the Poisson parameters $y_\pm$ are moderate (roughly $y_\pm < 100$).  This corresponds to warm-to-hot plasmas ($\tau \gtrsim 0.01$) with moderate energy transfers.

### Mathematical Basis

The electron momentum integral is expanded using a hyperbolic substitution that separates the integral into exponential integrals weighted by Poisson-like terms:

```
Σ_E / σ₀ = Ψ + P₊ − P₋
```

where $P_\pm = \sum_n w_n^\pm \cdot c_n^\pm \cdot \hat{E}_{n+1}(x_\pm)$, with:
- $w_n^\pm = e^{-y_\pm} \cdot y_\pm^n / n!$ — Poisson weights (updated incrementally)
- $c_n^\pm = A_\pm + 2n/a$ — kinematic coefficients
- $\hat{E}_m(x) = e^x \cdot E_m(x)$ — scaled exponential integrals

The parameters $x_\pm$ and $y_\pm$ are derived from the hyperbolic substitution:

```
b = ω / (2τ)
θ± = arcsinh(ρ± / ω)
x± = b · exp(θ±)
y± = b · exp(−θ±)
```

### Convergence Properties

The Poisson weights $w_n^\pm$ peak at $n \approx y_\pm$ and then decay rapidly.  The series typically converges in $O(y_\pm + \sqrt{y_\pm})$ terms.  When $y_\pm$ is small (high temperature or small energy transfer), convergence is very fast (5–20 terms).

### Failure Mode

When $y_\pm > 500$, `exp(−y)` underflows and the series cannot produce useful results.  The code returns `converged = false` immediately.

See [Final Equations: Power Series](final_equations.md#power-series) for the exact formulas.

---

## Low-Temperature Asymptotic Series

### When It Works Best

The asymptotic series is most accurate when $\tau \cdot \alpha_\pm$ is small — i.e., low temperatures and/or kinematics where $\alpha_\pm$ (the inverse of $\sqrt{\rho_\pm^2 + \omega^2}$) is moderate.  It excels precisely where the power series struggles (cold plasmas).

### Mathematical Basis

The normalized ratio is expanded in powers of $(-\tau \alpha_\pm)$:

```
Σ_E / σ₀ = 2τγγ'/q + S₊ + S₋
```

Each term involves:
- A power $(-\tau \alpha_\pm)^{n+1}$ — provides exponential suppression when $\tau \alpha_\pm < 1$
- Factorials $n!$ and $(n+1)!$ — eventually cause divergence
- Legendre polynomials $P_n(\zeta_\pm)$ — encode the angular structure

The Legendre polynomials are computed via the stable three-term recurrence:

```
P₀(z) = 1,   P₁(z) = z
P_{k+1}(z) = [(2k+1)·z·Pₖ(z) − k·Pₖ₋₁(z)] / (k+1)
```

### Convergence Properties

This is an **asymptotic** series, not a convergent one.  The terms first decrease (geometric suppression from $(\tau \alpha)^n$ dominates), reach a minimum, then increase without bound (factorial growth dominates).  The best approximation is obtained by truncating at the smallest term.

The accuracy at the truncation point scales roughly as $\exp(-1/(\tau \alpha))$ — exponentially good for small $\tau \alpha$.

### Stopping Rule

The two-consecutive-increase rule: if the term magnitude increases for two consecutive terms (after at least $n_{\min}$ terms), the partial sum at the smallest-term point is returned.  This provides robust truncation even when the minimum is broad.

See [Final Equations: Asymptotic Series](final_equations.md#low-temperature-asymptotic-series) for the exact formulas.

---

## Auto Switching Logic

The `Auto` method selects between power and asymptotic series based on:

```
τ · max(α₊, α₋)  < 0.05  →  Asymptotic
τ · max(α₊, α₋)  ≥ 0.05  →  Power
```

The threshold 0.05 ensures the asymptotic series achieves at least ~10⁻⁸ relative accuracy.

Auto mode does **not** fall back to quadrature.  If the selected series returns `converged = false`, the caller must handle the fallback.  This prevents an implicit dependency on the quadrature module.

---

## API Reference

### C++ (`compton_kernel_series.hpp`)

```cpp
namespace compton {

enum class SeriesMethod { PowerSeries, Asymptotic, Auto };

struct SeriesResult {
    double value;                // Kernel value Σ_E
    double estimated_abs_error;  // |σ₀| × (last or smallest term)
    double estimated_rel_error;  // abs_error / (|value| + 1e-300)
    int    terms_used;           // Number of terms summed
    SeriesMethod method_used;    // Which method was actually used
    bool   converged;            // Whether convergence criteria were met
};

double ehat_expn(int m, double x);  // Ê_m(x) = exp(x) E_m(x)

class ComptonKernelSeries {
public:
    ComptonKernelSeries(
        SeriesMethod method = SeriesMethod::Auto,
        double eps_rel = 1e-12,
        int n_min = 4,
        int n_max = 200
    );
    SeriesResult sigma_E(double E, double E_prime, double xi,
                         double tau, double Ne) const;
};
}
```

### Python (`cpp_modules._compton_kernel_series`)

```python
from cpp_modules._compton_kernel_series import (
    ComptonKernelSeries, SeriesMethod, SeriesResult, ehat_expn
)

engine = ComptonKernelSeries(
    method=SeriesMethod.Auto,  # or PowerSeries, Asymptotic
    eps_rel=1e-12,
    n_min=4,
    n_max=200
)

result = engine.sigma_E(E, E_prime, xi, tau, Ne)
# result.value, result.estimated_abs_error, result.estimated_rel_error
# result.terms_used, result.method_used, result.converged
```

### Pure Python mirror (`pycompton.compton_kernel_series`)

```python
from pycompton import sigma_E_series, ehat_expn, SeriesResult

result = sigma_E_series(E, E_prime, xi, tau, Ne=1.0, method="auto")
# Same fields as C++ SeriesResult
```

---

## Comparison with Quadrature

| Aspect | Series | Quadrature |
|--------|--------|------------|
| **Speed** | Typically faster (no precomputed nodes) | Requires GL node cache |
| **Accuracy** | Machine precision when converged | Machine precision with enough nodes |
| **Low τ** | Asymptotic series excels | Pre-IBP works; post-IBP has cancellation |
| **High τ** | Power series works well | Reliable with enough nodes |
| **Extreme kinematics** | May fail (`converged = false`) | More robust in edge cases |
| **Dependencies** | `boost::expint` + `doubledouble` for power series | `boost::cyl_bessel_k` + GL nodes |
| **Error estimate** | Last/smallest term magnitude | N vs N/2 Richardson comparison |

### Recommended Usage

For production transport codes:
1. Try `SeriesMethod::Auto` first (fastest).
2. If `converged = false`, fall back to `ComptonKernelQuadrature` with `PreIBP`.

For validation and debugging:
- Compare series and quadrature results to verify mutual consistency.

---

## Recurrence Optimizations

Both series loops now use in-loop recurrence relations to avoid per-term library calls:

**Power series (ehat_expn forward recurrence with stepwise guard):**
During summation at term n, `ehat_curr` holds Ê_{n+1}(x). After the convergence
check, the implementation advances via `ehat_next = (1 - x * ehat_curr) / (n+1)`,
guarded by a running amplification tracker (`amp *= x/(n+1)`). If cumulative
amplification exceeds `EHAT_AMPLIFICATION_BUDGET` (= 1e2, derived from
rel_tol / eps_machine / safety = 1e-13 / 1e-16 / 10), the advance falls back to
a direct `ehat_expn(n+2, x)` library call, then resets `amp = 1.0` to allow
recurrence to resume from the fresh seed. The plus and minus chains are guarded
independently.

**Asymptotic series (Legendre three-term recurrence):**
During summation at term n, the implementation maintains `P_prev = P_n(z)` and
`P_curr = P_{n+1}(z)`. After the convergence/truncation check, it advances to
(P_{n+1}, P_{n+2}) using the three-term recurrence at k=n+1:
`P_next = ((2n+3)*z*P_curr - (n+1)*P_prev) / (n+2)`.
A runtime bounds check clamps |zeta| to 1.0 if roundoff exceeds the bound.

Measured speedups (C++ vectorized throughput): 3-13x for asymptotic series,
3-11x for power series.

---

## Double-Double Precision

The power series loop performs all arithmetic in **double-double precision** (~32 significant digits) to handle the severe cancellation in $P_+ - P_-$.  This uses the [WarrenWeckesser/doubledouble](https://github.com/WarrenWeckesser/doubledouble) library (fetched at build time via CMake FetchContent), with two domain-specific extensions in `dd_extras.hpp`:

- `dd_asinh(x)` — needed for the hyperbolic substitution $\theta_\pm = \text{arcsinh}(\rho_\pm / \omega)$
- `dd_ehat_cf(m, x)` — Ehat_m via modified Lentz continued fraction (DLMF 8.9.2)

The kinematic parameters are also computed in double-double precision (`compute_params_dd()`) to avoid seeding the series with truncated inputs.

---

## Error Estimation

The power series reports a two-component error:

```
rel_error = max(trunc_rel, round_rel)
```

- **Truncation** (`last_term_mag / |result|`): the magnitude of the last terms added to P+ and P-, divided by the result. This is the standard convergent-series bound.
- **Rounding** (`N * eps_DD * max(|P+|, |P-|) / |result|`): models accumulated double-double roundoff (one eps_DD per iteration) amplified by any cancellation in P+ - P-.

In practice, for the Kershaw parameterization used here, the conditioning (`max(|P+|, |P-|) / |result|`) is benign (1-3x) across the entire calibrated domain. The rounding term is always negligible (~10⁻³⁰), and the truncation term dominates at ~10⁻¹³ to 10⁻¹⁶.

### Series vs Quadrature Accuracy at High τ

For τ > 20, the power series converges in 6-7 terms (due to rapidly decaying Poisson weights) and is *more accurate* than Gauss-Laguerre quadrature:

| τ | Series terms | Series-vs-Quad diff | Quad self-reported error |
|---|---|---|---|
| 5 | 9 | 6×10⁻⁹ | 4×10⁻⁸ |
| 20 | 7 | 8×10⁻⁶ | 2×10⁻⁷ |
| 50 | 6 | 4×10⁻⁴ | 7×10⁻⁴ |
| 100 | 6 | 2×10⁻³ | 1×10⁻⁴ |

At τ=100, the discrepancy is dominated by the quadrature's difficulty with sharply peaked integrands — both PostIBP and PreIBP forms at 256 points give different answers, and neither matches the series. The series is the trusted result in this regime.

---

## Known Limitations

1. **No vectorized interface** — unlike quadrature's `sigma_E_vec`, the series module evaluates one point at a time.
2. **Poisson weight underflow** — the power series cannot represent the Poisson distribution for $y > 500$.  A log-space implementation could extend this range.
3. **Asymptotic accuracy ceiling** — the asymptotic series accuracy is bounded by the smallest term magnitude.  For $\tau \alpha \sim 0.1$, this may be only $10^{-4}$.

---

## Further Documentation

- [Quadrature Module](quadrature.md) — the complementary evaluation method
- [Final Equations](final_equations.md) — step-by-step equations for both series
- [Numerical Stability](numerical_stability.md) — `ehat_expn`, Poisson guard, asymptotic truncation
- [Edge Cases](edge_cases.md) — series-specific failure modes and fallback strategy
- [Architecture](architecture.md) — module layout and build system
