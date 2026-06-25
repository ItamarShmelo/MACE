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
Gauss–Laguerre rules of order 16, 32, 64, 128, or 256 are computed at construction via
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

1. Track the smallest rearranged term magnitude `combined_mag` seen so far,
   together with the partial sum at that index.
2. After `n_min` terms, if `combined_mag / norm` drops below `eps_rel`, return
   immediately with error = $|\Sigma_0|$ × `combined_mag`.
3. If 2 consecutive `combined_mag` values grow, return the partial sum
   accumulated up to the smallest-`combined_mag` index.

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

The solver dispatch (see below) includes a DD asymptotic path: when the
double-precision asymptotic series reports a self-error above the tolerance
(driven by the roundoff-aware error estimator detecting cancellation), the
solver automatically escalates to DD arithmetic.  This is purely error-driven
and requires no hard-coded $\gamma$ threshold.

**Rearranged accumulation for G-cancellation.**
At ultra-low $\gamma$ the kinematic parameter
$G = -\gamma\gamma' + 2/a + 2/(\gamma\gamma' a^2)$ is dominated by the last
term and reaches $\sim 10^{16}$ at $\gamma \sim 10^{-8}$.  Each term
$T_n^\pm$ in the series is proportional to $G \cdot n!$, and $S_+ + S_-$
cancels $\sim 16$ digits.  In DD (31 digits) this leaves only $\sim 7$ good
digits---marginal when the normalised result is $\sim 10^{-10}$.

The implementation avoids this by grouping the $G$ contribution analytically:

$$T_n^+ + T_n^- = \bigl(G - (n{+}1)/a\bigr)\, n!\, D_n + (n{+}1)!\, F_{n+1}$$

where $D_n = (-\tau\alpha_-)^{n+1} P_n(\zeta_-) - (-\tau\alpha_+)^{n+1} P_n(\zeta_+)$
and $F_{n+1}$ groups the $\eta_\pm$ Legendre terms.
$D_n$ involves an $O(\gamma)$ subtraction that loses $\sim 8$ digits,
so the combined formula preserves $\sim 15$ good digits in DD---a gain
of $\sim 8$ digits over the original two-accumulator approach.

Convergence/truncation tracks a single metric: the rearranged term magnitude
`combined_mag` $= |T_n^+ + T_n^-|$, which reflects the actual per-term
contribution after analytical G-cancellation.  The partial sum at the index
where `combined_mag` was smallest is recorded for use by the divergence exit.
The original per-term magnitudes $|T_n^+| + |T_n^-|$ are not tracked
separately---since `combined_mag` $\leq$ `term_mag` by the triangle inequality,
`combined_mag` always produces a tighter (and more accurate) error bound.
Empirical validation at 50-digit precision confirms up to 74$\times$
improvement in the DD result accuracy.

### Solver Dispatch

`ComptonKernelSolver` selects the fastest accurate method at each phase-space
point via a purely error-driven cascade:

```
tau_alpha_max = tau * max(alpha_+, alpha_-)

if tau_alpha_max < 0.035:              -- Asymptotic regime
    A1: try Asymptotic series (double)
        accept if self-error < asymp_self_tol (1e-7).
        The roundoff-aware error estimator naturally flags cancellation
        at ultra-low gamma, triggering escalation to DD.
    A2: try Asymptotic series (DD)
        accept if self-error < dd_asymp_self_tol (1e-3).
else:                                  -- Power series regime
    P1: try Power series (double)
        accept if self-error < power_series_self_tol (1e-7)
        AND (dsigma_dT or value >= 0).
    P2: try Power series (DD, n_max=500)
        accept if self-error < dd_power_series_self_tol (1e-3)
        AND (dsigma_dT or value >= 0).

Throw if no backend passes its tolerance.
```

**P3 removal and n_max increase.**  The previous dispatch included a P3 step
that tried the asymptotic DD series as a last resort in the power-series regime.
Analysis showed that at ultra-low $\gamma$ (< $10^{-6}$) the DD power series
with the default `n_max=200` converged prematurely due to extreme cancellation
in P+ − P−, reporting self-error just above $10^{-3}$.  The asymptotic DD
happened to pass because it is geometrically accurate when $\gamma \to 0$
(its expansion parameter $\tau\alpha$ is gamma-independent).  However, raising
`n_max` to 500 allows the DD power series to converge properly at these points
(reaching the DD roundoff floor of ~$10^{-6}$), eliminating the need for P3.
This is preferable because the asymptotic series is a divergent expansion and
its self-reported error is not a reliable bound outside its intended regime
($\tau\alpha < 0.035$).

**Error-driven DD escalation.**  The previous dispatch used a hard-coded
$\gamma_{\min}$ threshold to decide when DD arithmetic was needed in the
asymptotic regime.  This was fragile: the threshold was empirically tuned and
couldn't adapt to the actual cancellation at each point.  The roundoff-aware
error estimator (see Error Estimation below) now tracks per-term
floating-point cancellation in the $D_n$ and $F_{n+1}$ subtractions and
reports it as part of the self-error.  When cancellation exhausts double
precision, the reported error exceeds `asymp_self_tol`, and the cascade
naturally escalates to DD where the roundoff contribution is negligible.

**Asymptotic quality gate.**  At high temperature and ultra-low
photon energy ($\gamma \ll 1$), the quantity $\tau \cdot \max(\alpha_+, \alpha_-)$
depends on the scattering angle $\xi$.  Near the dispatch boundary the
asymptotic series can report self-errors much larger than the value, producing
garbage that corrupts the multigroup angular integral.  The quality gate
detects this (self-error > `asymp_self_tol`) and falls through to the power
series.  The entire asymptotic branch is wrapped in `try/catch` because the
series can throw "failed to converge" near the dispatch boundary when factorial
overflow occurs before the optimal truncation point is reached.

**Non-negative check (P1/P2).**  For `sigma_E`, the power series result is
accepted only if non-negative.  A negative value indicates catastrophic
cancellation at the $\Psi + P_+ - P_-$ level, and the solver escalates to a
higher-precision method.

The thresholds are empirically validated:

| Constant | Value | Rationale |
|----------|-------|-----------|
| `ASYMP_TAU_ALPHA_THRESHOLD` | 0.035 | Widened from 0.02; the asymptotic series is highly reliable in this range and the power series (via DD) safely handles points above the threshold |
| `power_series_self_tol` | 1e-7 | Tightened from 5e-6; the DD fallback handles points where the double power series cannot meet this tolerance |
| `asymp_self_tol` | 1e-7 | Tightened from 1e-3; forces DD escalation for any asymptotic result with self-error above 0.00001%; guards dispatch boundary and roundoff-dominated points |


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
The solver falls through to DD power series when the speculative double PS
fails.  The DD inner loop uses precomputed
reciprocals and strength-reduced coefficient updates to minimize per-iteration
cost (~18-21 µs per call at typical red-zone points).

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
relative error below $2 \times 10^{-9}$ (first neglected term $\sim 1.6 \times
10^{-9}$ for $K_2$, $\sim 8.8 \times 10^{-10}$ for $K_1$).

The ratio $\kappa(\tau) = K_1(1/\tau)/K_2(1/\tau)$ appears in temperature
derivatives and is computed from the scaled forms (the exponentials cancel).

### Scaled Exponential Integrals

$\hat{E}_m(x) = e^x E_m(x)$ is evaluated via the modified Lentz continued
fraction (DLMF 8.9.2):

$$\hat{E}_m(x) = \cfrac{1}{x + m - \cfrac{m \cdot 1}{x + m + 2 - \cfrac{(m+1) \cdot 2}{x + m + 4 - \cdots}}}$$

Convergence tolerances: $10^{-14}$ (double), $10^{-20}$ (DD), with maximum
1000 / 2000 iterations respectively.

When successive $\hat{E}_{n+1}$ values are needed, a **downward recurrence**
$\hat{E}_{n+1} = (1 - x\hat{E}_n)/n$ is cheaper than restarting the CF at
every order.  However the recurrence amplifies errors; it is used only while the
accumulated amplification stays below an **amplitude budget**: $10^3$ for double,
$10^{15}$ for DD.  Once the budget is exceeded the CF is restarted to reset
error.  The DD budget is set high because DD has ~31 digits of precision; even
after 15 orders of magnitude of amplification, ~16 significant digits remain.

**Taylor series fallback for small $x$ (m=1).**  The CF converges slowly for
small $x$: at $x \approx 0.05$ it requires $>5000$ iterations (the partial
quotient ratio $|a_j|/b_j \approx j/2$ grows without bound).  For $m = 1$ and
$x < 4$, the function is evaluated via the Taylor series instead:

$$E_1(x) = -\gamma - \ln x + \sum_{k=1}^{N} \frac{(-x)^k}{k \cdot k!}$$

$$\hat{E}_1(x) = e^x \cdot E_1(x)$$

The crossover at $x = 4$ was chosen because the Taylor series remains cheaper
than the CF up to this point (approximately 30 terms at $x = 4$ for double,
~55 for DD, while the CF still needs $>50$ iterations).  For $x < 1$ the sum
$-\gamma - \ln x + S(x)$ has no cancellation since all terms are positive; in
$[1, 4)$ the alternating series introduces mild cancellation that is well
within double precision (~2 digits lost at $x = 4$).  Only $m = 1$ needs the
fallback; higher-order $\hat{E}_{n+1}$ are computed via the downward recurrence
which is stable for small $x$ (amplitude factor $x/(n+1) < 1$).


## Multigroup Integration

### Gauss–Legendre Quadrature Strategy

The multigroup cross section

$$\sigma(g \to g') = \frac{2\pi \int_{\Delta E_g} \int_{\Delta E_{g'}} \int_{\xi_i}^{\xi_{i+1}} w(E,T)\,\Sigma_E\, d\xi\, dE'\, dE}{\int_{\Delta E_g} w(E,T)\, dE}$$

is evaluated by Gauss–Legendre quadrature on three finite intervals $(E, E', \xi)$.
**Only the E' peak region uses adaptive refinement**; all other axes and sub-regions
use single-panel GL quadrature with appropriate coordinate mappings.

**Rationale:** The Compton kernel has a sharp recoil-band peak in $E'$ that benefits
from adaptive bisection.  The tails decay smoothly (exponentially away from the
peak boundary) and are well resolved by the log/rlog change of variable alone.
The $E$ and $\xi$ integrands, being weighted integrals over the $E'$ axis result,
are already smooth.  Removing adaptivity from these axes dramatically reduces
function evaluations, allowing higher base quadrature orders without performance
penalty.

**Adaptive error estimation (E' peak only):** For each panel $[a, b]$, compute
$I_\text{whole}$ using the base rule, then compute
$I_\text{halves} = I_\text{left} + I_\text{right}$ by splitting at the midpoint.
The error estimate is $|I_\text{halves} - I_\text{whole}|$.  If this exceeds
`peak_tol * |I_halves|`, recurse independently on each half.  A maximum recursion
depth (configurable via `MGIntegrationConfig::peak_max_depth`) prevents runaway
subdivision.

**Peak tolerance:** `integration_tolerance * 0.1` controls adaptive refinement of
the E' peak region.  Accuracy of other axes is controlled by increasing `base_order`.

**Cold-temperature regime:** Below 0.005 keV the Compton kernel narrows to
near-Thomson scattering, requiring more quadrature nodes for convergence.
When $T <$ `COLD_TEMPERATURE_THRESHOLD` (defined in `compton_common.hpp`),
the integrator automatically substitutes `cold_temperature_order` (default 48)
for `base_order` on the E and $\xi$ axes.  The threshold was determined
empirically: `base_order=24` produces < 0.05% self-convergence error above
0.002 keV but degrades to ~8% at 0.0001 keV.

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


### Peak-Aware E' Integration

The E' (middle) axis uses a peak-aware quadrature scheme that exploits Compton
kinematics to concentrate quadrature effort where the integrand is large.

#### Cold Compton Recoil Band

For an angle bin $[\xi_\text{lo}, \xi_\text{hi}]$ and incoming energy $E$, the
cold-electron recoil band in $E'$ is

$$a = \frac{E}{1 + \gamma(1 - \xi_\text{lo})}, \quad
  b = \frac{E}{1 + \gamma(1 - \xi_\text{hi})}$$

where $\gamma = E / m_e c^2$.  Inside this band there exists a scattering angle
for which a cold (rest-frame) electron can produce the observed $E'$; outside,
the kernel is exponentially suppressed by the Boltzmann factor
$\sim\exp(-(\lambda_\mathrm{min} - 1)/\tau)$.  This kinematic band depends only on $E$
and the $\xi$-bin endpoints, not on temperature.

At finite temperature `peak_limits` extends each edge by one thermal Doppler
half-width $\Delta E = E\sqrt{2\tau}$; below `COLD_TEMPERATURE_THRESHOLD` the
padding is widened to $5\Delta E$ because the kernel is extremely narrow and the
recoil band may not otherwise span a full group.

#### Direction-Aware Log-Space Integrators

Two single-panel log-space GL quadrature variants handle the exponentially
decaying tails:

- **`log_legendre_integrate`** (right tail): substitution $u = \log(x)$
  clusters nodes near the lower end $a$ (the peak boundary).
- **`rlog_legendre_integrate`** (left tail): reflected substitution
  $u = \log(a + b - x)$ applied to $y = a + b - x$ clusters nodes near the
  upper end $b$ (the peak boundary).

Both accept the integrand in the original $E'$ space and handle the change of
variable internally.  The log-space mapping alone concentrates quadrature nodes
where the exponentially decaying integrand is largest, making adaptive refinement
unnecessary for these smooth tails.

#### `integrate_Ep_group` Splitting Logic

Each target group $[E'_\text{lo}, E'_\text{hi}]$ is classified by its overlap
with the recoil band $[a, b]$ using `std::clamp`:

```
overlap_lo = clamp(a, Ep_lo, Ep_hi)
overlap_hi = clamp(b, Ep_lo, Ep_hi)
```

- If `overlap_lo >= overlap_hi`: the group is **far** (no overlap with peak).
  Uses single-panel **log or rlog** GL with the far rule: `log_legendre_integrate`
  when the peak is below the group ($E'_\text{peak} \le E'_\text{lo}$), and
  `rlog_legendre_integrate` otherwise.  The log-space mapping concentrates
  nodes at the group edge closest to the peak, which is critical when the
  kernel decays rapidly across the group width.
- Otherwise the group is split into up to three sub-intervals:
  - `[Ep_lo, overlap_lo]` **left tail**: single-panel reflected-log GL (nodes near peak boundary).
  - `[overlap_lo, overlap_hi]` **peak**: adaptive GL with tight tolerance.
  - `[overlap_hi, Ep_hi]` **right tail**: single-panel log GL (nodes near peak boundary).

The peak can span any number of groups: one group (both boundaries inside),
two groups (boundary in each), or three+ groups (interior groups entirely within
the peak).  All cases are handled uniformly by the clamp logic.

#### `MGIntegrationConfig`

The `MGIntegrationConfig` struct consolidates all multigroup integration
parameters: GL orders, adaptive refinement depth, integration tolerance, and
the outward-from-peak cutoff ratio.  All parameter validation is performed by
the constructor so that invalid configurations are rejected early.

| Field | Default | Description |
|-------|---------|-------------|
| `base_order` | 24 | GL panel order for E, xi, and E'-peak axes |
| `cold_temperature_order` | 48 | GL order for E/xi axes when T < 0.005 keV |
| `peak_max_depth` | 5 | Maximum recursion depth for adaptive E' peak |
| `tail_order` | `nullopt` → `base_order` | GL order for E' tail (log/rlog) regions |
| `far_order` | `nullopt` → `base_order` | GL order for E' far-from-peak regions |
| `xi_order` | `nullopt` → `base_order` | GL order for the ξ peak panel |
| `xi_peak_k` | 10.0 | Number of FWHMs for the ξ peak-focused splitting window |
| `integration_tolerance` | 1e-3 | Overall relative tolerance for the outer integral |
| `cutoff_ratio` | 1e-8 | Outward-from-peak early-termination ratio |
| `flat_ep` | `nullopt` | Optional flat E' density config (replaces adaptive E' with single-pass GL) |

Only the peak region uses adaptive quadrature; tails and far use single-panel
GL with their respective coordinate mappings:

| Region | Order | Quadrature | Adaptive |
|--------|-------|------------|----------|
| Peak   | `base_order` | linear GL | Yes (configurable depth) |
| Tail   | `tail_order` (default: `base_order`) | log/rlog GL | No |
| Far    | `far_order` (default: `base_order`) | log/rlog GL | No |

The tail and far integrands decay exponentially away from the peak boundary.
The log-space change of variable concentrates nodes where the integrand is
largest, making adaptive refinement unnecessary.  For far groups the mapping
direction is chosen so that nodes cluster at the group edge closest to the
peak (log for groups above the peak, rlog for groups below).

#### ξ Peak-Focused Splitting

For non-elastic scattering (|E'/E − 1| > 0.05), the ξ integrand has a sharp
peak at the Compton angle $\xi_c = 1 - (1/\gamma)(1/r - 1)$ where $r = E'/E$.
The FWHM of this peak is:

$$\text{FWHM}(\xi) = 2\sqrt{2\ln 2}\;\frac{\sqrt{\tau\,|\gamma'-\gamma|\,(2+|\gamma-\gamma'|)}}{\gamma\,\gamma'}$$

This expression is obtained from a Gaussian fit to the thermal broadening of
the Compton kernel in ξ, retaining the exact relativistic recoil kinematics
(the $(2+|\Delta\gamma|)$ factor) rather than the non-relativistic
approximation used previously.

The integrator splits the ξ interval into three panels:
1. Left tail: [ξ_lo, ξ_c − k·FWHM] with 8-point GL
2. Peak: [ξ_c − k·FWHM, ξ_c + k·FWHM] with `xi_order`-point GL
3. Right tail: [ξ_c + k·FWHM, ξ_hi] with 8-point GL

where k = `xi_peak_k` (default 10) is the half-width in units of FWHM (total
window = 2k·FWHM). This splitting is applied only when the peak window is
narrower than 80% of the full ξ interval, ensuring the tails are
exponentially small and well-resolved by just 8 points. For near-elastic
scatter (|r − 1| < 0.05) or when the peak fills most of the interval, the
integrator falls back to log/rlog mapping or standard linear GL.

#### Recommended Production Configurations

Two factory methods provide validated high-accuracy configurations:

**`MGIntegrationConfig::cold_adaptive()`** — for T < 0.1 keV:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `base_order` | 192 | Resolves narrow Compton recoil band at cold T |
| `peak_max_depth` | 9 | Deep recursion for extremely narrow E' peaks |
| `xi_order` | 512 | Resolves sharp ξ peak (FWHM ∝ √(τ·|Δγ|·(2+|Δγ|))/(γγ'), very narrow at high E + cold T) |
| `xi_peak_k` | 10 | 10× FWHM half-width (20 FWHM total window) captures >99.99% of ξ peak area |
| `integration_tolerance` | 1e-8 | Tight tolerance for adaptive refinement |
| `cutoff_ratio` | 1e-12 | Conservative group cutoff |
| `cold_temperature_order` | 192 | Matches base_order |

Validated accuracy: MC converges as 1/√N against this reference with no
detectable det bias. At N=10^9: mid-group row-sum error ~5e-5 (pure MC
noise), element-wise RMS ~3e-3. Runtime: ~300–600s per 24-group matrix
(with bo=48; full bo=192 takes ~600–1300s).

**`MGIntegrationConfig::warm_flat()`** — for T ≥ 0.1 keV:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `base_order` | 96 | High-order GL for E axis |
| `xi_order` | 96 | Adequate for broader ξ peaks at warm T |
| `xi_peak_k` | 10 | 10× FWHM half-width (20 FWHM total window) |
| `flat_ep` | density=512, ppd, max=8192 | Dense flat E' (no adaptive recursion needed at warm T) |
| `flat_E` | false | Keeps E-axis boundary layer log-mapping |
| `flat_xi` | false | Keeps ξ peak-focused splitting |
| `cutoff_ratio` | 1e-12 | Conservative group cutoff |

Validated accuracy: MC converges as 1/√N with no bias. At N=10^9:
mid-group row-sum error < 1e-4 for T ≥ 1 keV, ~5e-4 at T=0.1 keV.
Runtime: ~30–120s per 24-group matrix.

**Temperature switch at T = 0.1 keV:** Below this threshold, the Compton
kernel narrows dramatically, requiring high adaptive resolution (bo=192) to
resolve. Above, the kernel is broad enough that flat E' with 512 points/decade
is sufficient and much faster. Both configs use `xi_peak_k=10` and keep
`flat_xi=false` to leverage the analytically-derived FWHM peak splitting.

Python usage:
```python
import _compton_multigroup as cm

# Select config based on temperature
if T_kev < 0.1:
    cfg = cm.MGIntegrationConfig.cold_adaptive()
else:
    cfg = cm.MGIntegrationConfig.warm_flat()

det = cm.ComptonMultigroupKernel(bounds_erg, wf, cfg)
S = det.compute_sigma_matrix(kernel=kernel, T=T_K, Ne=1.0)
```

### Outward-from-Peak Group Cutoff

For grids with many groups, most target groups $g'$ have negligible scattering
from a given incoming group $g$ because Compton scattering peaks near the
elastic energy $E' \approx E$ and is exponentially suppressed for large energy
transfers.  The outward-from-peak cutoff exploits this sparsity.

**Algorithm:** For each incoming group $g$:

1. Identify the **peak target group** $g'_\text{peak}$ as the group containing
   the geometric-mean center energy of group $g$.  This approximates the
   elastic forward-scattering peak.
2. Integrate $g'_\text{peak}$ first, summing the absolute values across all
   angle bins to obtain `peak_sum`.
3. Expand rightward ($g' = g'_\text{peak}+1, g'_\text{peak}+2, \ldots$): for
   each group, compute all angle bins and sum absolute values.  Stop when the
   sum drops below `cutoff_ratio * peak_sum`.  Remaining groups stay at zero.
4. Expand leftward ($g' = g'_\text{peak}-1, g'_\text{peak}-2, \ldots$):
   same stopping criterion.

**Why this is valid:** Compton scattering conserves total rate per incoming
group (row sums $\sum_{g'} \sigma(g \to g') \approx \sigma_T$).  The kernel
is exponentially suppressed by the Boltzmann factor $\sim\exp(-\Delta E / kT)$
for large energy transfers, so the omitted groups contribute a fraction below
`cutoff_ratio` of the peak.  For `cutoff_ratio = 1e-8`, the row-sum error is
negligible compared to quadrature tolerances.

**Default:** `cutoff_ratio = 1e-8`, set at construction time via
`MGIntegrationConfig`.


## Error Estimation

All kernel evaluations return a `ComptonResult` containing `value` and
`estimated_rel_error`.  These are **heuristic** estimates, not rigorous bounds:

- **Quadrature:** Richardson-style comparison of order $N_L$ vs $N_L/2$:
  `rel_err = |I_hi - I_lo| / (|I_hi| + 1e-300)`.
- **Power series:** cascaded cancellation condition number:
  $\varepsilon_\text{round} = N \cdot \varepsilon_\text{mach} \cdot C_P \cdot C_\Psi$,
  where $C_P = (|P_+| + |P_-|) / |P_+ - P_-|$ measures cancellation in the
  $P_+ - P_-$ subtraction and $C_\Psi = (|\Psi| + |P_+ - P_-|) / |\Psi + P_+ - P_-|$
  measures cancellation in the final sum.  For the derivative, the result has
  two parallel computational paths that combine:
  $\text{deriv} = (d\Psi + dP_+ - dP_-) + d\ln\Sigma_0 \cdot (\Psi + P_+ - P_-)$.
  Each path carries a two-stage cancellation chain: the **dP path** has
  $C_{dP} \cdot C_{d\Psi}$ (subtraction then addition with $d\Psi$), and the
  **P path** (inside $\sigma_\text{term}$) has $C_P \cdot C_\Psi$ (same as
  non-derivative).  The rounding error is weighted by each path's magnitude:
  $\varepsilon_\text{round} = N \varepsilon_\text{mach}
  (|d\Psi + \text{ddiff}| \cdot C_{dP} C_{d\Psi} + |\sigma_\text{term}| \cdot C_P C_\Psi)
  / |\text{deriv}|$.
  This avoids the over-conservative `max` pathology where a negligible path with
  high internal cancellation would inflate the estimate.  The truncation error
  also accounts for the P-series tail propagated through $d\ln\Sigma_0$:
  $\varepsilon_\text{trunc} = (|dt_\pm^\text{last}| + |d\ln\Sigma_0| \cdot |t_\pm^\text{last}|) / |\text{deriv}|$.
  The final error is `max(truncation, round)`.
- **Asymptotic series:** `max(truncation, roundoff)`.  The truncation
  component is the smallest rearranged term magnitude `combined_mag` at optimal
  truncation (see above).  The roundoff component accumulates per-term
  floating-point cancellation from the $D_n$ and $F_{n+1}$ subtractions:
  $$\varepsilon_\text{roundoff} = \sum_n \left( |c_{D,n}| \cdot \varepsilon_T (|d_n^+| + |d_n^-|) + |c_{F,n}| \cdot \varepsilon_T (|f_n^+| + |f_n^-|) \right)$$
  where $d_n^\pm$ are the subtraction operands of $D_n$, $f_n^\pm$ those of
  $F_{n+1}$, $c_{D,n} = (G - (n+1)/a) \cdot n!$ and $c_{F,n} = (n+1)!$ are the
  amplifying coefficients, and $\varepsilon_T$ is the unit roundoff for the
  arithmetic type ($\sim 1.1 \times 10^{-16}$ for double, $\sim 1.2 \times 10^{-32}$
  for DD via `MachineEps<T>`).  For the derivative, each term's roundoff is
  additionally multiplied by $|w_n|$ (the derivative weight).
  The convergence gate (`combined_mag / norm < eps_rel`) intentionally does not
  consider roundoff --- continuing to iterate cannot reduce accumulated roundoff,
  so stopping when additional terms are negligible is correct.  The roundoff
  is captured only at the return site via `max(truncation, roundoff)`.

**Why the original power series error formula was wrong.**  The prior formula
used $N \cdot \varepsilon_\text{mach} \cdot \max(|P_+|, |P_-|) / |\text{result}|$,
which only accounts for one stage of cancellation.  When both $P_+ \approx P_-$
*and* $\Psi \approx -(P_+ - P_-)$, the actual condition number is the product of
two large ratios.  At ultra-low $\gamma$ and extreme forward scattering, the old
formula reported self-errors of $\sim 10^{-7}$ when the true error was 17% or
worse, causing the solver to trust catastrophically wrong DD power series results.

**Why the original asymptotic series error formula was wrong.**  After the
rearranged accumulation was introduced (see above), the convergence tracker
still used $|T_n^+| + |T_n^-|$ — the magnitude of the *original* un-cancelled
terms.  These are $\sim 10^{18}\times$ larger than the actual contribution
$|T_n^+ + T_n^-|$ after analytical cancellation of the $G$ terms.  The
self-reported error was therefore massively inflated, causing the solver to
reject correct asymptotic results and dispatch to the (broken) power series.
The fix switched the convergence tracker to use `combined_mag` $= |T_n^+ + T_n^-|$,
which reports the true error of the rearranged sum.  The original `term_mag`
tracking was subsequently removed as dead code (since `combined_mag` $\leq$
`term_mag` always, the `term_mag` path could never be selected).

**Roundoff-aware asymptotic error.**  The `combined_mag` truncation metric
tracks series convergence but is blind to floating-point cancellation in the
$D_n = d_n^- - d_n^+$ and $F_{n+1} = f_n^- - f_n^+$ subtractions.  At
ultra-low $\gamma$ (< 0.002), $\alpha_+$ and $\alpha_-$ are nearly equal, so
$d_n^+$ and $d_n^-$ differ by $\sim 10^{-8}$ relative to their magnitude.  The
subtraction loses $\sim 8$ digits, and the large $G$ coefficient ($\sim 10^8$
at $\gamma \sim 10^{-4}$) then amplifies this roundoff, exhausting double
precision.  The series converges to a wrong answer with a falsely small
`combined_mag`.  Empirically, at $\gamma \sim 2 \times 10^{-5}$ the actual
double-vs-DD error is $\sim 3\%$ while `combined_mag` reports $\sim 5 \times 10^{-14}$
--- an under-report by $\sim 10^{11}$.  The per-term roundoff accumulator
(`roundoff_sum`) tracks the cancellation-amplified roundoff from each $D_n$ and
$F_{n+1}$ subtraction, and the final error is `max(truncation, roundoff)`.
In DD arithmetic, $\varepsilon_T \sim 10^{-32}$ makes the roundoff contribution
negligible even after $G$-amplification, so DD results are unaffected.

**Strict regime dispatch.**  Each step in the cascade is wrapped in
`try/catch` because any backend can throw on underflow or non-convergence at
extreme parameters.  The asymptotic and power-series regimes are mutually
exclusive (strict `if/else`): there is no fallthrough from the asymptotic
regime to the power series.  If no backend within the selected regime passes
its acceptance tolerance, the solver throws `runtime_error`.  The public
`sigma_E` and `dsigma_E_dT` methods catch the throw and return
`{value=0, error=1}` with a stderr warning, so callers in integration loops
are not interrupted.  The DD power series uses a separate, looser tolerance
(`dd_power_series_self_tol`, default $10^{-3}$) because it is the last resort
in the power-series regime and its DD precision makes even modestly-converged
results reliable.

Tests anchor accuracy against Q256 post-IBP Gauss–Laguerre as the numerical
ground truth.


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

- **Monte Carlo:** uses a direct per-sample derivative weight.  Each MC
  sample's Klein-Nishina contribution is multiplied by the derivative weight
  $(\lambda_i - \kappa)/\tau^2 - 3/\tau$ (where $\lambda_i$ is the sampled
  Lorentz factor and $\kappa = K_1(1/\tau)/K_2(1/\tau)$) and accumulated
  into a single tally.  The derivative is then

  $$\frac{\partial\sigma}{\partial T}\bigg|_{g_0,g'} = \text{norm} \sum_i \sigma_i \left[\frac{\lambda_i - \kappa(\tau)}{\tau^2} - \frac{3}{\tau}\right] \frac{d\tau}{dT}$$

  This is algebraically equivalent to the ratio form
  $\sigma \cdot [J_1/(\tau^2 J_0) - 3/\tau - \kappa/\tau^2]$, but avoids
  the catastrophic cancellation that occurs at cold temperatures ($\tau \ll 1$)
  where $J_1/J_0 \approx \kappa \approx 1$ and dividing their stochastic
  difference by $\tau^2$ amplifies MC noise.  The direct form computes
  $\lambda_i - \kappa$ per sample where $\lambda_i$ is exact, eliminating the
  noisy-ratio problem.  The same normalization (`beta_avg`, `weight_avg`) is
  applied, matching the multigroup convention (kernel-only derivative, no
  quotient-rule terms).
