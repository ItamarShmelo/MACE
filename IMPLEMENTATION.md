# Implementation Decisions

This document records the numerical choices made in the implementation of the
Kershaw–Prasad–Beason (1986) thermal Compton scattering kernel and its
multigroup reduction.  The focus is on *why* certain methods were chosen and
what numerical pitfalls they address, not on the class hierarchy or API.


## Point Kernel Evaluation

The point-wise differential scattering kernel factorizes as

$$\Sigma_E(E \to E',\, \xi;\, \tau,\, N_e) \;=\; \Sigma_0 \times \mathcal{M}(\gamma,\gamma',\xi,\tau)$$

where $\gamma = E / m_e c^2$, $\tau = k_B T / m_e c^2$, and $\Sigma_0$
carries the prefactor (electron density, exponential suppression, Bessel
normalization).  Three complementary methods evaluate $\mathcal{M}$.

### Gauss–Laguerre Quadrature

The integral over electron momentum has the form
$\int_0^\infty f(x)\, e^{-x}\, dx$ after the substitution $\rho = \tau x + \rho_+$,
which maps the Maxwell–Jüttner tail onto the standard Laguerre weight.
Gauss–Laguerre rules of order 64, 128, or 256 are computed at construction via
the Golub–Welsch algorithm (implicit QL diagonalization of the Laguerre Jacobi
matrix).

Two algebraically equivalent integrand forms are available:

- **Post-IBP** (default): integration by parts, leaving an $O(1/\sqrt{R})$ integrand.
- **Pre-IBP**: the original $O(1/R^{3/2})$ form; kept for validation.

The post-IBP form converges with fewer nodes because it is smoother.

### Power Series (Convergent)

For hot plasmas the kernel admits a convergent Poisson-weighted expansion:

$$\Sigma_E / \Sigma_0 = \Psi + P_+ - P_-$$

with $P_\pm = \sum_n w_n^\pm \cdot c_n^\pm \cdot \hat{E}_{n+1}(x_\pm)$, where
$w_n^\pm$ are Poisson weights in $y_\pm$ and $\hat{E}_m(x) = e^x E_m(x)$ is the
scaled exponential integral.

A **hyperbolic substitution** converts the raw arguments into $(x_\pm, y_\pm)$:

$$b = \omega/(2\tau),\quad z_\pm = \rho_\pm/\omega$$

$$x_\pm = b(z_\pm + r_\pm),\quad y_\pm = b/(z_\pm + r_\pm)$$

where $r_\pm = \sqrt{z_\pm^2 + 1}$.  The formula $y = b/(z+r)$ avoids
catastrophic cancellation that would occur in computing $r - z$ when $z \gg 1$.

**Convergence criterion:** at least 4 terms (`n_min`), up to 200 (`n_max`);
stop when the relative change in $P_+ - P_-$ drops below $\varepsilon_\text{rel} = 10^{-12}$.
The error estimate is `max(truncation_error, roundoff_error)`, where the
roundoff contribution scales as $N \cdot \epsilon_\text{mach} \cdot \max(|P_+|, |P_-|)$
to account for catastrophic cancellation in the difference.

### Asymptotic Series (Divergent)

For cold plasmas ($\tau \to 0$) the expansion in powers of $(-\tau\alpha_\pm)$
is divergent but asymptotic:

$$\Sigma_E / \Sigma_0 \sim \frac{2\tau\gamma\gamma'}{q} + S_+ + S_-$$

Terms involve factorials and Legendre polynomials $P_n(\zeta_\pm)$.  Because the
series diverges, it is truncated at the **smallest term** (optimal truncation):

1. Track the smallest term magnitude seen so far.
2. After `n_min` terms, if 2 consecutive terms grow, return the partial sum
   accumulated up to the smallest-term index.
3. Also exit early if term/norm drops below $10^{-12}$.

The series is best suited for $\tau \cdot \max(\alpha_+, \alpha_-) < 0.05$.

**Legendre stability:** Mathematically $|\rho_\pm \alpha_\pm| < 1$ always (since
$\omega^2 > 0$ for $\xi < 1$), but floating-point rounding can push the computed
ratio to $\pm 1$ or slightly beyond when $\omega$ is tiny.  The clamp
$\zeta_\pm = \text{clamp}(\rho_\pm \alpha_\pm,\, -1,\, 1)$ is a floating-point
precaution that prevents the Legendre three-term recurrence from exciting the
exponentially growing solution.

**Double-double arithmetic:** Like the power series, the asymptotic series
supports double-double (~31 digits) arithmetic via the `high_precision`
constructor flag.  The internal summation is templatized on type `T` (double or
DD); the prefactor $\Sigma_0$ remains in double since it passes through
`exp`/`K_2` which are double-only.

At ultra-low $\gamma$ ($E \lesssim 0.05$ keV), the kinematic parameters $q$,
$\rho_\pm$, and $\alpha_\pm$ involve differences of nearly-equal quantities
(e.g.\ $q = \sqrt{(\gamma'-\gamma)^2 + 2\gamma\gamma'a}$ with tiny $\gamma$),
and the alternating factorial/power accumulation suffers catastrophic
cancellation.  Empirical measurement shows double vs DD relative error rising to
$\sim 10^{-4}$ at $\gamma = 10^{-4}$ across all cold temperatures tested
(0.01--5 keV), whereas at $\gamma > 0.01$ the two agree to $< 10^{-9}$.

The solver dispatch (see below) now includes a DD asymptotic path: when the
asymptotic regime is active *and* $\min(\gamma, \gamma') < 0.002$ (~1 keV),
the solver automatically switches to DD arithmetic.  The threshold of 0.002 is
chosen so double-vs-DD error stays below $10^{-6}$ with margin (empirically
$7 \times 10^{-7}$ at $\gamma = 10^{-3}$, $10^{-9}$ at $\gamma = 10^{-2}$).

### Solver Dispatch

`ComptonKernelSolver` selects the fastest accurate method at each phase-space
point:

```
tau_alpha_max = tau * max(alpha_+, alpha_-)
gamma_min     = min(gamma, gamma')

1a. if tau_alpha_max < 0.025 AND gamma_min >= 0.002
                                   --> Asymptotic series (double)
1b. if tau_alpha_max < 0.025 AND gamma_min <  0.002
                                   --> Asymptotic series (double-double)
2.  elif gamma_min >= 0.02         --> Power series (double)
3.  else try Q64 Gauss-Laguerre:
        if self-error < 1e-6       --> Accept Q64
        else                       --> Power series (double-double)
```

The thresholds are empirically validated:

| Constant | Value | Rationale |
|----------|-------|-----------|
| `ASYMP_TAU_ALPHA_THRESHOLD` | 0.025 | Asymptotic series achieves < 1e-3 relative error vs Q256 |
| `ASYMP_GAMMA_DD_THRESHOLD` | 0.002 (~1 keV) | Worst-case double vs DD asymptotic error is 7e-7 at this boundary |
| `GAMMA_DOUBLE_PRECISION_SAFE` | 0.02 (~10 keV) | Worst-case double vs DD power-series error is 3.15e-7 at this boundary |
| `quadrature_self_tol` | 1e-6 | Accepts Q64 only when its Richardson error estimate is tight |


## Precision Strategy

### Double vs Double-Double Arithmetic

Both the power series and asymptotic series support double-double (~31 digits)
arithmetic, implemented via the `doubledouble` library and controlled by a
`high_precision` constructor flag on each class.  Internally, the series
summation loops are templatized on type `T` (double or DD); inputs and outputs
remain `double` in all cases.

**Power series:** computes $P_+ - P_-$, a difference of two nearly equal
quantities at low photon energies.  When $\gamma \ll 1$ (say E < 10 keV), up to
15 significant digits cancel, making double precision (~15 digits) inadequate.
The solver uses Q64 as a fast first attempt in the DD regime; only when the
quadrature's self-reported error exceeds $10^{-6}$ does it fall through to the
expensive DD power series.

**Asymptotic series:** the factorial/power accumulation $\sum_n (-\tau\alpha_\pm)^{n+1}$
is similarly susceptible to cancellation at ultra-low $\gamma$ (< 1e-3), where
$q$, $\rho_\pm$, $\alpha_\pm$ become differences of nearly-equal numbers.  DD
reduces relative error from $\sim 10^{-4}$ to machine precision at
$\gamma = 10^{-4}$.  See `reports/asymptotic_dd_precision_report.py` for the
empirical sweep.

Guard: `POISSON_Y_MAX = 500` rejects evaluations where the Poisson weight
$y_\pm$ is so large that `exp(-y)` would underflow even in double-double.

### Scaled Bessel Functions

The prefactor $\Sigma_0$ contains $K_2(1/\tau)$ which overflows for small $\tau$
(cold plasmas: $1/\tau \gg 1$).  All Bessel functions are used in their
**scaled** form $\tilde{K}_\nu(x) = e^x K_\nu(x)$:

- $x < 50$: Boost `cyl_bessel_k` multiplied by $e^x$.
- $x \geq 50$: 5-term Hankel asymptotic expansion
  $\tilde{K}_\nu(x) \sim \sqrt{\pi/(2x)} \sum_{k=0}^{4} a_k / (8x)^k$,
  which avoids computing $e^x$ and $K_\nu$ separately.

The crossover at $x = 50$ is chosen so that the 5-term Hankel expansion has
relative error below $10^{-15}$.

The ratio $\kappa(\tau) = K_1(1/\tau)/K_2(1/\tau)$ appears in temperature
derivatives and is computed from the scaled forms (the exponentials cancel).

### Scaled Exponential Integrals

$\hat{E}_m(x) = e^x E_m(x)$ is evaluated via the modified Lentz continued
fraction (DLMF 8.9.2):

$$\hat{E}_m(x) = \cfrac{1}{x + m - \cfrac{m \cdot 1}{x + m + 2 - \cfrac{(m+1) \cdot 2}{x + m + 4 - \cdots}}}$$

Convergence tolerances: $10^{-14}$ (double), $10^{-25}$ (DD), with maximum
1000 / 2000 iterations respectively.

When successive $\hat{E}_{n+1}$ values are needed, a **downward recurrence**
$\hat{E}_{n+1} = (1 - x\hat{E}_n)/n$ is cheaper than restarting the CF at
every order.  However the recurrence amplifies errors; it is used only while the
accumulated amplification stays below an **amplitude budget**: $10^3$ for double,
$10^{10}$ for DD.  Once the budget is exceeded the CF is restarted to reset
error.


## Multigroup Integration

### Adaptive Recursive Gauss–Legendre Quadrature

The multigroup cross section

$$\sigma(g \to g') = \frac{2\pi \int_{\Delta E_g} \int_{\Delta E_{g'}} \int_{\mu_i}^{\mu_{i+1}} w(E,T)\,\Sigma_E\, d\mu\, dE'\, dE}{\int_{\Delta E_g} w(E,T)\, dE}$$

is evaluated by **adaptive recursive Gauss–Legendre** quadrature on all three
finite intervals $(E, E', \mu)$.  Each axis uses the same base GL rule (default
order 16, configurable via `base_order`) and recursively bisects panels until
the error estimate drops below a per-axis tolerance.

**Error estimation:** For each panel $[a, b]$, compute $I_\text{whole}$ using the
base rule, then compute $I_\text{halves} = I_\text{left} + I_\text{right}$ by
splitting at the midpoint.  The error estimate is $|I_\text{halves} - I_\text{whole}|$.
If this exceeds `tol * |I_halves|`, recurse independently on each half.  A maximum
recursion depth of 15 prevents runaway subdivision.

**Tolerance hierarchy:** Inner integrals use progressively tighter tolerances to
keep the total error within the user-specified budget:

| Axis | Tolerance |
|------|-----------|
| E (outer) | `tol` |
| E' (middle) | `tol * 0.1` |
| μ (inner) | `tol * 0.01` |

This ensures inner-axis quadrature noise, when propagated through outer-axis
summations, stays below the overall target.  The default `tol = 1e-3` was chosen
as a practical balance between accuracy and speed for the full 3-axis adaptive
approach.

Group centers are placed at the geometric mean $\sqrt{E_\text{lo} \cdot E_\text{hi}}$.
Angle bins partition $[-1, 1]$ into $N$ equal segments of width $2/N$.  The $2\pi$
azimuthal factor is included so that summing over angle bins recovers the total
group-to-group cross section.

### Weight Functions and Denominators

Three weight functions are supported:

| Weight | $w(E,T)$ | Denominator |
|--------|-----------|-------------|
| **Planck** | $x^3/(e^x - 1)$, capped at $x = $ `cap_x` | Clark (1987) polylogarithm series |
| **Wien** | $x^3 e^{-x}$, capped at `cap_x` | Taylor / closed-form (see below) |
| **Uniform** | 1 | $E_\text{right} - E_\text{left}$ |

The Planck cap avoids evaluating $x^3/(e^x - 1)$ in the exponential tail where
it underflows; above `cap_x` the weight is held constant at its value at the cap.
The denominator is computed **analytically** (not by quadrature) to avoid
introducing quadrature noise into the normalization.

#### Wien Denominator — Taylor Series for Small x

The Wien denominator antiderivative $G(x) = \int_0^x t^3 e^{-t}\,dt$ equals
$6 - e^{-x}(x^3 + 3x^2 + 6x + 6)$.  For small $x$ this expression suffers
from catastrophic cancellation because $e^{-x}(\cdots) \approx 6$; at
$x = 0.01$ roughly 9 decimal digits are lost, and at $x = 10^{-6}$ the result
rounds to zero.

For $x \le 0.1$ we instead evaluate the Taylor expansion

$$G(x) = x^4 \sum_{n=0}^{6} \frac{(-x)^n}{(n+4)\,n!}$$

via Horner's method (7 terms).  At the threshold $x = 0.1$ this gives relative
error below $10^{-9}$; for smaller $x$ convergence is even faster.  Above the
threshold the closed form is used, where cancellation is mild (< 1 digit lost at
$x = 0.2$).


## Error Estimation

All kernel evaluations return a `SigmaResult` containing `value` and
`estimated_rel_error`.  These are **heuristic** estimates, not rigorous bounds:

- **Quadrature:** Richardson-style comparison of order $N_L$ vs $N_L/2$:
  `rel_err = |I_hi - I_lo| / (|I_hi| + 1e-300)`.
- **Power series:** `max(last_term_magnitude, N * eps_mach * max(|P+|, |P-|))`
  normalized by the result.
- **Asymptotic series:** magnitude of the smallest term (at optimal truncation)
  divided by the accumulated sum.

Tests anchor accuracy against Q256 post-IBP Gauss–Laguerre as the numerical
ground truth.  Multigroup accuracy is validated against the CMMC Monte Carlo
code (row sums and angular CDFs, since element-wise agreement is affected by
CMMC's linear energy redistribution).


## Temperature Derivatives

All kernel modules provide analytic $\partial\Sigma_E / \partial T$ by
differentiating through their respective formulations:

- **Quadrature:** each node's integrand is multiplied by a $\tau$-derivative
  weight involving $\kappa(\tau)$.
- **Power series:** per-term $\partial P_\pm / \partial\tau$ tracking both
  $\hat{E}_{n+1}$ and $\hat{E}_n$.
- **Asymptotic series:** each term times $((\lambda_+ - \kappa)/\tau^2 + (n-2)/\tau)$.

The chain rule factor $d\tau/dT = k_B / (m_e c^2)$ is applied at the API
boundary.  The multigroup `dsigma_dT` plugs this derivative kernel into the
same weighted integral; it is **not** the full $\partial\sigma/\partial T$ of
the ratio (which would need quotient-rule terms for the denominator).
