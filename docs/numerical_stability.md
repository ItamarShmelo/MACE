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
