# Numerical Stability Design

This document catalogues the numerical challenges encountered in evaluating the Kershaw-Prasad-Beason kernel and the techniques used to address them.

For parameter regimes where the code is expected to produce unreliable or degraded results (even with these mitigations), see [Edge Cases and Pathological Regimes](edge_cases.md).

---

## Challenge 1: Bessel Function Overflow/Underflow

### Problem

The normalization involves K₂(1/τ), the modified Bessel function of the second kind.  For cold plasmas (τ ≪ 1, i.e., 1/τ ≫ 1):

```
K₂(x) ~ √(π/(2x)) · e^{−x}       (x → ∞)
```

At τ = 0.002 (T = 1 keV): x = 1/τ = 500, so K₂(500) ~ 10⁻²¹⁹.  This underflows to zero in double precision.  Meanwhile the prefactor contains exp(−(λ₊−1)/τ) which may also be extremely small.

### Solution: Scaled Bessel Function

Work with `K̃₂(x) = exp(x) · K₂(x)` which remains O(√(π/(2x))) for all x > 0.

The full prefactor becomes:
```
σ₀ = ... · exp(−(λ₊ − 1)/τ) / K̃₂(1/τ)  · exp(−1/τ + 1/τ)
   = ... · exp(−(λ₊ − 1)/τ) / K̃₂(1/τ)
```

The exponential factor exp(−(λ₊−1)/τ) naturally handles the cancellation: it's the ratio of two exponentially small quantities (the Boltzmann factor for the minimum electron energy vs. the partition function).

### Implementation

| x range | Method | Error |
|---------|--------|-------|
| x < 50 | `exp(x) · boost::cyl_bessel_k(2, x)` | Machine precision |
| x ≥ 50 | Hankel asymptotic (5 terms) | < 10⁻¹⁵ relative |

---

## Challenge 2: Cancellation in q²

### Problem

The momentum transfer q² = γ² + γ'² − 2γγ'ξ suffers catastrophic cancellation when γ ≈ γ' and ξ → 1 (near-elastic forward scattering).

### Solution: Two-Term Reformulation

```
q² = (γ' − γ)² + 2γγ'(1 − ξ)
```

Both terms are individually non-negative.  Even when γ' ≈ γ, the second term 2γγ'(1−ξ) remains well-conditioned (a = 1−ξ is computed exactly when ξ is stored directly).

---

## Challenge 3: λ₊ Roundoff Below 1

### Problem

Physically, λ₊ ≥ 1 always (minimum electron Lorentz factor).  But the formula:

```
λ₊ = (γ'−γ)/2 + √[(1 + γγ'a/2)(1 + (γ'−γ)²/(2γγ'a))]
```

can produce values like 0.9999999999998 due to floating-point arithmetic.

### Solution: Clamping with Guard

```cpp
if (lambda_plus < 1.0 - 1e-12)
    throw std::runtime_error("lambda_plus significantly below 1");
if (lambda_plus < 1.0)
    lambda_plus = 1.0;  // Clamp small roundoff violations
```

This distinguishes between acceptable roundoff (< 10⁻¹²) and genuine bugs.

---

## Challenge 4: Post-IBP Cancellation at Low τ

### Problem

In the post-IBP form:

```
Σ_E = σ₀ · (Ψ + I_Q)
```

For very cold electrons (τ ≪ 0.01), both Ψ and I_Q are large and nearly equal in magnitude but opposite in sign.  Their sum is the physically meaningful (small) result, but computing it via subtraction loses precision.

Example at τ = 0.002:
- Ψ ≈ 1.5 × 10⁶
- I_Q ≈ −1.5 × 10⁶ + O(1)
- Σ_E ∝ O(1) (the meaningful result)

This requires ~6 digits of cancellation, leaving only ~10 significant digits.

### Solution: Pre-IBP as Cross-Check

The pre-IBP form computes I_Q directly without the Ψ splitting:

```
Σ_E = σ₀ · I_Q^{pre}
```

No cancellation occurs because the integrand directly represents the physical result.  The cost is a slightly more complex integrand (1/R^{3/2} terms vs 1/√R).

**Recommendation:** Use pre-IBP (`QuadratureForm::PreIBP`) when τ < 0.01 (T < 5 keV).

---

## Challenge 5: Integration Domain Transformation

### Problem

The electron momentum integral runs over [λ₊, ∞) with a Boltzmann weight exp(−λ/τ):

```
I = ∫_{λ₊}^∞  f(λ) · exp(−λ/τ) dλ
```

This is not in standard Gauss-Laguerre form.

### Solution: Change of Variable

Substitute ρ = λ₊ + τ·x (so x = (ρ − λ₊)/τ, dx = dρ/τ):

```
I = τ · exp(−λ₊/τ) · ∫₀^∞  f(λ₊ + τx) · exp(−x) dx
```

The exp(−λ₊/τ) factor is absorbed into σ₀.  The remaining integral is in standard Gauss-Laguerre form with weight e^{−x} on [0,∞).

The factor τ in front accounts for the Jacobian and also scales the integrand: at small τ, the integrand is very peaked near x=0 (low electron momenta dominate), which is exactly where Gauss-Laguerre places its densest nodes.

---

## Challenge 6: ω² Singularity at ξ = ±1

### Problem

```
ω² = (1 + ξ) / (1 − ξ)
```

- At ξ → +1: ω² → ∞ (forward scattering, integrand vanishes but slowly)
- At ξ → −1: ω² → 0 (backscattering, no issue)

The integration over ξ therefore requires endpoints strictly inside (−1, 1).

### Solution: Endpoint Exclusion

The Python-level integration uses:
```python
XI_EPS = 1e-10
quad(integrand, -1 + XI_EPS, 1 - XI_EPS, ...)
```

The C++ code throws `std::invalid_argument` if `1 − ξ < 10⁻¹⁴`, providing a hard safety check.  The physical contribution from ξ extremely close to ±1 is negligible in practice.

---

## Challenge 7: R± Positivity

### Problem

The functions R± = (ρ + ρ±)² + ω² appear in denominators (as √R or R^{3/2}).

### Non-Issue

R± > 0 always because ω² ≥ 0 and the squared term is non-negative.  No special handling is needed.  This is a consequence of the algebraic structure of the kernel — the denominators never vanish for physical kinematics.

---

---

## Challenge 8: Scaled Exponential Integral Overflow (Series Module)

### Problem

The power series requires the **scaled exponential integral**:

```
Ê_m(x) = exp(x) · E_m(x)
```

For large `x`, `exp(x)` overflows while `E_m(x)` underflows — the naive product `exp(x) * expn(m, x)` is `∞ × 0 = NaN`.

### Solution: Two-Regime Strategy

| x range | Method | Error |
|---------|--------|-------|
| x < 50 | Direct product: `exp(x) · boost::expint(m, x)` | Machine precision |
| x ≥ 50 | Asymptotic expansion (15 terms max) | < 10⁻¹⁵ relative |

The asymptotic expansion for large x:

```
Ê_m(x) ~ (1/x) · [1 − m/x + m(m+1)/x² − m(m+1)(m+2)/x³ + ...]
```

Terms are accumulated until `|term| < 10⁻¹⁵ |partial sum|` or 15 terms are reached.  This mirrors the two-regime approach used for `scaled_K2`.

### Implementation

- Python: `ehat_expn(m, x)` in `pycompton/compton_kernel_series.py`
- C++: `ehat_expn(m, x)` in `compton_kernel_series.cpp` (uses `boost::math::expint`)

---

## Challenge 9: Poisson Weight Underflow (Series Module)

### Problem

The power series involves Poisson-like weights:

```
w_n^± = exp(−y±) · y±^n / n!
```

When `y±` is very large (cold plasmas with large ω²), `exp(−y±)` underflows to zero, making all subsequent terms exactly zero.  This silent underflow produces incorrect results rather than a NaN or exception.

### Solution: Early Bail-Out

Before entering the summation loop, check:

```
if y_plus > 500  or  y_minus > 500:
    return Ψ (boundary term only), converged = false
```

The threshold 500 is chosen because `exp(−500) ≈ 7 × 10⁻²¹⁸`, which is near the IEEE 754 double-precision underflow boundary (`~5 × 10⁻³²⁴`).  With the `y^n / n!` factor, a few terms can still be represented, but the series is numerically meaningless.

When the bail-out triggers, `converged = false` signals the caller to fall back to quadrature or an alternative method.

---

## Challenge 10: Asymptotic Series Divergence (Series Module)

### Problem

The low-temperature asymptotic series is an **asymptotic** (non-convergent) expansion:

```
Σ_E / σ₀ = base_term + Σ_{n=0}^∞  c_n · (−τ α±)^{n+1} · n! · (...)
```

The `n!` growth ensures that terms eventually increase without bound.  The series first decreases, reaches a minimum, then diverges.

### Solution: Smallest-Term Truncation

The partial sum is truncated at the optimal point where terms are smallest, using a **two-consecutive-increase rule**:

1. Track the smallest term magnitude seen so far, along with the corresponding partial sum.
2. After each new term, check if term magnitude has increased compared to the previous term.
3. If two consecutive increases are observed (after `n_min` terms), return the partial sum at the best (smallest-term) point.  This is reported as `converged = true` because the asymptotic truncation is intentional and the result is as accurate as the series can provide.
4. If `n_max` is reached without triggering the two-increase rule, return `converged = false`.
5. If `n!` overflows (`factorial_n` becomes non-finite), break early and return the best partial sum with `converged = false`.

The error estimate is the magnitude of the smallest term — a standard heuristic for asymptotic series.

### When This Works

The asymptotic series is most accurate when `τ · α±` is small (low temperature, moderate kinematics).  The auto-switching criterion `τ · max(α₊, α₋) < 0.05` selects the asymptotic series only when this product is small enough for rapid initial convergence.

---

## Summary Table

| Challenge | Technique | Where |
|-----------|-----------|-------|
| K₂ overflow/underflow | Scaled Bessel K̃₂ = e^x K₂(x) | `scaled_K2()` |
| q² cancellation | Two-term formula (dg² + 2γγ'a) | `compute_params()` |
| λ₊ roundoff | Clamp with 10⁻¹² guard | `compute_params()` |
| Ψ + I_Q cancellation | Pre-IBP form for low τ | `compute_IQ_pre_ibp()` |
| Integration domain | ρ = λ₊ + τx substitution | Gauss-Laguerre integration |
| ξ = ±1 singularity | Endpoint exclusion (10⁻¹⁰) | Python integration layer |
| R± positivity | Guaranteed by ω² ≥ 0 | No handling needed |
| Ê_m overflow | Two-regime `ehat_expn` (direct × asymptotic) | `ehat_expn()` |
| Poisson weight underflow | Bail-out when y± > 500 | Power series loop |
| Asymptotic divergence | Smallest-term truncation with 2-increase rule | Asymptotic series loop |
