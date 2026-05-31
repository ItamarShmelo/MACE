# Edge Cases and Pathological Regimes

This document identifies parameter regimes where the direct Gauss-Laguerre quadrature is expected to produce unreliable, degraded, or physically meaningless results.  It serves as a "failure mode" guide for users.

---

## Hard Failures (Code Throws)

These parameter combinations will raise exceptions:

| Condition | Exception | Reason |
|-----------|-----------|--------|
| ξ ≥ 1 − 10⁻¹⁴ | `invalid_argument` | a = 1−ξ underflows; ω² = ∞ |
| ξ ≤ −1 or ξ ≥ 1 | `invalid_argument` | Outside physical range |
| E ≤ 0 or E' ≤ 0 | `invalid_argument` | Non-physical energy |
| τ ≤ 0 | `invalid_argument` | Non-physical temperature |
| N_e = NaN/Inf | `invalid_argument` | Invalid density |
| λ₊ < 1 − 10⁻¹² | `runtime_error` | Kinematic constraint violated (likely a bug) |

---

## Exponential Underflow to True Zero

### Regime: Large energy transfer at low temperature

When `(λ₊ − 1) / τ > ~709`, the factor `exp(−(λ₊−1)/τ)` underflows to exactly 0.0 in IEEE 754 double precision.  The kernel returns exactly zero.

**When this happens:**
- Upscattering at cold temperatures: T = 1 keV, E_in = 10 keV, E' = 50 keV → λ₊ ≈ 5, (λ₊−1)/τ ≈ 2000 → underflow
- Any transition requiring an electron with energy ≫ kT

**Impact:** The returned value is 0.0 with zero error estimate.  This is physically correct (the transition probability is genuinely negligible — e.g., 10⁻⁸⁷⁰), but the code cannot distinguish "exactly zero" from "too small to represent."

**Workaround:** None needed.  These values are physically negligible.  If sub-double-precision results are required, use extended precision arithmetic (not implemented).

---

## Post-IBP Catastrophic Cancellation

### Regime: τ < 0.01 (T < 5 keV)

The post-IBP form computes `Ψ + I_Q` where both terms are large (∝ 1/τ) and nearly cancel.

| τ | Digits lost to cancellation | Remaining precision |
|---|------|------|
| 0.1 | ~1 | ~15 digits |
| 0.01 | ~2 | ~14 digits |
| 0.001 | ~5–6 | ~10 digits |
| 0.0001 | ~8–9 | ~7 digits |
| 0.00001 | ~11–12 | ~4 digits (unreliable) |

**Symptom:** Post-IBP and pre-IBP forms disagree.  Post-IBP may show noise or wrong sign.

**Workaround:** Use `QuadratureForm::PreIBP` for τ < 0.01.  The pre-IBP form has no cancellation and maintains full precision at all temperatures.

**Why post-IBP exists:** At moderate/high τ, post-IBP converges faster (fewer quadrature points needed) because the integrand is smoother (1/√R vs 1/R^{3/2}).

---

## Slow Quadrature Convergence at Very High Temperature

### Regime: τ > 10 (T > 5 MeV)

At ultra-relativistic temperatures, the electron distribution is very broad.  The Gauss-Laguerre integrand `f(τx + ρ_offset)` varies on a scale ∝ 1/τ in the variable x, meaning significant structure exists at x ∝ 1/τ ≪ 1.  Meanwhile, the Gauss-Laguerre nodes are distributed according to the Laguerre polynomial zeros, which are concentrated near x ~ N for large N.

**Impact:**
- N = 64 may be insufficient for convergence
- N = 256 should still work for τ ≤ 50 based on testing, but the error estimate (N vs N/2 comparison) becomes less reliable as a convergence indicator

**Symptom:** `estimated_rel_error` is large (> 10⁻³).  The N=128 and N=256 results may differ by more than 10⁻⁴.

**Workaround:** Use N = 256 and verify the error estimate.  For τ > 50, consider adaptive quadrature instead of fixed-order Gauss-Laguerre (not currently implemented).

---

## Ultra-Relativistic Photons (γ > 100)

### Regime: E > 50 MeV

At very high photon energies, the kinematic structure of the kernel becomes more complex:
- The recoil effect is extreme (Compton wavelength ≪ photon wavelength)
- λ₊ can be very large even for small angular changes
- The kernel has sharper features in angle

**Impact:** The quadrature itself still converges (since Gauss-Laguerre handles the electron integral), but:
- The outer integration over ξ (done in Python via `scipy.integrate.quad`) may need tighter tolerances
- The outer integration over E' may have very narrow features that sparse energy grids miss

**Workaround:** Increase `epsrel` for scipy integration, use finer energy grids, verify with higher N.

---

## Extreme Forward Scattering (ξ → 1)

### Regime: 1 − ξ < 10⁻⁸

As ξ → 1:
- `a = 1 − ξ → 0`
- `ω² = (1+ξ)/a → 2/a → ∞`
- Several quantities involve division by `a` or `a²`

The code enforces `1 − ξ ≥ 10⁻¹⁴` as a hard check, but even for `1 − ξ ~ 10⁻⁸`:
- Terms like `2/(γγ'a²)` and `s/(τa²)` become extremely large
- Cancellation between G, A±, and Ψ terms increases
- Numerical noise in the integrand grows

**Impact:** Results for `1 − ξ < 10⁻⁸` may have degraded precision (few correct digits).

**Workaround:** For the physical problem (integration over all ξ), the contribution from ξ > 1 − 10⁻⁸ is negligible because `dξ` is infinitesimal.  The exclusion `XI_EPS = 1e-10` is sufficient.  If pointwise evaluation at extreme forward angles is needed, consider a dedicated asymptotic expansion (not implemented).

---

## Degenerate Elastic Limit (E = E', ξ = 1)

### Regime: γ = γ' AND ξ → 1 simultaneously

At the exact elastic forward-scattering point:
- `q → 0` (zero momentum transfer)
- `λ₊ → 1` (any electron can participate)
- The kernel has a δ-function-like peak (infinite differential cross-section in zero solid angle)

The Kershaw kernel is well-defined for E = E' at ξ < 1, and for ξ → 1 at E ≠ E', but the combined limit E = E' AND ξ = 1 is singular.

**Impact:** Cannot evaluate at this exact point.  The code will throw for ξ too close to 1.

**Physical interpretation:** This singularity integrates to a finite contribution (the total cross-section is finite).  It is handled correctly by the outer ξ-integration which avoids the endpoint.

---

## Extremely Disparate Energies (E'/E > 10⁶ or < 10⁻⁶)

### Regime: γ'/γ or γ/γ' > 10⁶

Scattering from keV to GeV (or vice versa) requires an electron with enormous Lorentz factor.  The kernel is exponentially suppressed (see "Exponential Underflow" above), but even before complete underflow:

- λ₊ is very large → `exp(−(λ₊−1)/τ)` is tiny
- The kinematic quantities may lose relative precision
- The meaningful signal is buried in the exp suppression

**Impact:** Result is either exactly zero (underflow) or a very small number with limited relative precision.

**Workaround:** Not an issue in practice — such extreme energy transfers have negligible cross-section.

---

## Pair Production Not Included

### Regime: E + E' > 2 m_e c² at head-on angles

For center-of-mass energies above the pair threshold, electron-positron pair production becomes possible.  The Kershaw kernel describes **Compton scattering only** and does not include:
- Pair production
- Triplet production
- Higher-order QED processes

**Impact:** The kernel is physically incomplete for hard gamma-ray scattering off ultra-relativistic electrons.  The total cross-section should include pair contributions for accurate transport.

**Boundary:** Pair production becomes relevant when γ · γ' · (1−ξ) > 2 (in the electron rest frame, the photon CM energy exceeds 2m_e c²).  For thermal electrons this is unlikely unless τ > 1 AND E > m_e c².

---

## Non-Maxwellian Electron Distributions

The Kershaw kernel assumes a **relativistic Maxwell-Jüttner distribution** f(p) ∝ exp(−γ_e/τ).  It does not apply to:
- Beam distributions (mono-energetic electrons)
- Power-law distributions (cosmic ray electrons)
- Two-temperature plasmas (hot tail + cold core)
- Degenerate Fermi-Dirac distributions (white dwarfs, neutron stars)

For non-thermal distributions, one must return to the original single-electron Klein-Nishina cross-section and integrate numerically over the actual electron distribution function.

---

## Finite Quadrature Order Limitations

The Gauss-Laguerre rule with N points is exact for polynomial integrands of degree ≤ 2N−1.  The actual integrand is not a polynomial — it contains square roots and rational functions.  Convergence depends on how well the integrand can be approximated by the polynomial basis.

**When convergence is slow:**
- Very small τ AND post-IBP form (integrand oscillates relative to smooth approximation)
- Integrand has sharp features at x ≈ 0 (small τ regime)
- Very large argument of R± (large ρ_offset when λ₊ ≫ 1)

**Practical limits (tested):**

| N | Typical relative accuracy | Regime |
|---|--------------------------|--------|
| 64 | 10⁻⁸ to 10⁻¹² | τ > 0.01, moderate energies |
| 128 | 10⁻¹⁰ to 10⁻¹⁴ | All typical cases |
| 256 | 10⁻¹² to 10⁻¹⁶ | Conservative choice |

Use the built-in error estimate (`estimated_rel_error`) to detect cases where convergence is insufficient.

---

---

## Series-Specific Failure Modes

The series module (`ComptonKernelSeries`) provides an alternative evaluation method.  It has its own failure modes distinct from quadrature.

### Power Series Non-Convergence

**Regime:** Very low τ (cold plasmas) with moderate-to-large energy transfers.

At low temperatures, the power series terms involve large, nearly-cancelling contributions (Ψ and the P± sums have opposite signs).  This is the same catastrophic cancellation that affects post-IBP quadrature, but manifests differently:

- **Poisson weight underflow:** When `y± > 500`, the Poisson weights `exp(−y) · y^n / n!` underflow to zero.  The series returns `converged = false` immediately (with only the boundary term Ψ as a partial result).
- **Slow convergence / many terms:** Even when weights don't underflow, large `y±` values require many terms before the Poisson distribution peaks, leading to slow convergence or hitting `n_max`.

**Symptom:** `converged = false`, or large `estimated_rel_error`, or disagreement with quadrature.

**Workaround:** Auto mode selects the asymptotic series for low τ, which avoids this regime.  If forced to use the power series (via `method = "power"`), fall back to quadrature when `converged = false`.

### Asymptotic Series Non-Convergence

**Regime:** Moderate-to-high τ (warm/hot plasmas).

The asymptotic series diverges by design (it is an asymptotic, not convergent, expansion).  The useful accuracy depends on how small `τ · α±` is:

| τ · max(α₊, α₋) | Typical accuracy | Behavior |
|-------------------|-----------------|----------|
| < 0.01 | 10⁻¹² or better | Rapid convergence, few terms |
| 0.01 – 0.05 | 10⁻⁸ to 10⁻¹² | Good, more terms needed |
| 0.05 – 0.2 | 10⁻⁴ to 10⁻⁸ | Marginal accuracy |
| > 0.2 | Unreliable | Series diverges before reaching useful accuracy |

**Symptom:** `converged = false` (reached `n_max` without meeting stopping criteria), or factorial overflow, or large error estimate.

**Workaround:** Auto mode uses `τ · max(α₊, α₋) < 0.05` as the decision boundary.  Above this threshold, the power series is selected.

### The `converged` Flag

The `converged` flag in `SeriesResult` has precise semantics:

| Value | Meaning | Action |
|-------|---------|--------|
| `true` | Series met its convergence criterion (relative tolerance or smallest-term truncation) | Result is reliable |
| `false` | Series failed: hit `n_max`, Poisson weight underflow, or factorial overflow | Fall back to quadrature |

Auto mode never silently falls back to quadrature — it always returns `SeriesResult` with the `converged` flag, and the **caller** decides on fallback.

### Quadrature Fallback Strategy

For production use, the recommended pattern is:

```python
result = series_engine.sigma_E(E, E_prime, xi, tau, Ne)
if not result.converged:
    result = quadrature_engine.sigma_E(E, E_prime, xi, tau, Ne)
```

The series module is complementary to quadrature:
- **Series excels** at moderate temperatures and standard kinematics (fast, no quadrature nodes needed).
- **Quadrature excels** at extreme parameters where series methods break down.

---

## Temperature Derivative Edge Cases

The `dsigma_E_dtau` method inherits all edge cases of `sigma_E` and adds:

### Post-IBP Derivative Cancellation

At low τ, the post-IBP derivative suffers from the same Ψ-cancellation as the kernel itself.  The pre-IBP derivative avoids this entirely and is recommended for all temperatures.

### Kappa Ratio at Extreme τ

At very small τ (< 10⁻⁴), κ(τ) approaches 1 and the log-derivative `d(ln σ₀)/dτ` is dominated by the `(λ₊ − 1)/τ²` term.  At very large τ (> 100), κ ≈ 1/(2τ) and the log-derivative is moderate.  Both limits are handled correctly by the scaled Bessel implementation.

### Recommended Form

Use pre-IBP for `dsigma_E_dtau` at all temperatures.  The post-IBP form provides a useful cross-check at moderate τ but degrades at low τ.

---

## Summary: Safe Operating Envelope

### Quadrature

| Parameter | Safe range | Marginal | Unreliable/Fails |
|-----------|-----------|----------|-------------------|
| τ (post-IBP) | > 0.01 | 0.001 – 0.01 | < 0.001 |
| τ (pre-IBP) | > 10⁻⁶ | 10⁻⁸ – 10⁻⁶ | < 10⁻⁸ (underflow) |
| τ (high) | < 10 | 10 – 50 | > 50 (slow convergence) |
| ξ | (−1, 0.99999999) | (0.99999999, 1−10⁻¹⁰) | > 1 − 10⁻¹⁴ (throws) |
| E'/E ratio | 10⁻³ – 10³ | 10⁻⁶ – 10⁻³ or 10³ – 10⁶ | > 10⁶ (underflow) |
| γ | 10⁻⁶ – 100 | 100 – 1000 | > 1000 (untested) |
| N (quadrature order) | 128–256 | 64 | 32 (inaccurate) |

**The recommended default:** N = 64 with `PreIBP` form covers the widest range of conditions with acceptable accuracy.  Switch to N = 128 or 256 for precision-critical applications.

### Series

| Parameter | Power series | Asymptotic series | Auto |
|-----------|-------------|-------------------|------|
| τ < 0.01 | Unreliable (cancellation) | Best regime | Selects asymptotic |
| 0.01 < τ < 0.1 | Good (fast) | Good | Depends on α± |
| τ > 0.1 | Best regime | May diverge early | Selects power |
| y± > 500 | Fails (underflow) | N/A | Falls through |
| τ·α± > 0.2 | N/A | Unreliable | Avoids asymptotic |

**The recommended default:** `SeriesMethod::Auto` with fallback to quadrature when `converged = false`.
