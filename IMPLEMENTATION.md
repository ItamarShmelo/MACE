# Implementation Decisions

This document records the numerical choices made in the implementation of the
Kershaw–Prasad–Beason (1986) thermal Compton scattering kernel and its
multigroup reduction.  The focus is on *why* certain methods were chosen and
what numerical pitfalls they address, not on the class hierarchy or API.


## Point Kernel Evaluation

The point-wise differential scattering kernel factorizes as

$$\Sigma_E(E \to E',\, \xi;\, \tau) \;=\; \Sigma_0 \times \mathcal{M}(\gamma,\gamma',\xi,\tau)$$

where $\gamma = E / m_e c^2$, $\tau = k_B T / m_e c^2$, and $\Sigma_0$
carries the microscopic prefactor (exponential suppression, Bessel
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
The E axis uses **feature-aware sub-interval GL** with per-sub-interval coordinate
mapping selection (linear, log, or rlog).  The E' axis uses a ridge-based
three-region split (see [Ridge-Based E' Integration](#ridge-based-e-integration)
below).  The $\xi$ axis uses peak-focused splitting.

The deterministic implementation is organized as three nested integration
helpers:

- `integrate_xi_bin` integrates over one $\xi$ bin for fixed $(E, E')$.
- `integrate_Ep_xi_bin` integrates over an outgoing-energy interval and one
  $\xi$ bin for fixed $E$, reusing the ridge-based $E'$ split.
- `integrate_E_Ep_xi_bin` integrates the weighted numerator over the incoming
  energy group for one selected $(g, g', \xi_i)$ bin, subdividing the E-axis
  at boundary layers and the weight-function peak, and selecting the
  appropriate coordinate mapping for each sub-interval.

**E-axis sub-interval splitting.**  For each incoming energy group $g$,
`integrate_E_Ep_xi_bin` subdivides the integration domain
$[E_\text{lo}, E_\text{hi}]$ into up to four sub-intervals using two
feature sources:

1. **Boundary layers.**  The Compton ridge has significant thermal width
   near the group edges.  Boundary-layer points
   $\text{bl\_lo} = E_\text{lo} + k_b \cdot \sigma(E_\text{lo})$ and
   $\text{bl\_hi} = E_\text{hi} - k_b \cdot \sigma(E_\text{hi})$
   (where $\sigma$ is `ridge_thermal_width` at $\xi = -1$ and $k_b$ is
   `e_boundary_k`) isolate thin edge regions from the interior.
   When the boundary layers overlap ($\text{bl\_lo} \ge \text{bl\_hi}$),
   no edge split is applied.
2. **Weight function peak.**  If the weight-function peak energy (e.g.
   $2.821\,k_B T$ for Planck) falls strictly inside the middle region,
   that region is split at the peak so each piece has a monotone weight.

Each sub-interval $[a, b]$ is integrated with `e_panel_order` GL points
(default 12) using a coordinate mapping chosen by the weight profile:

| Condition | Mapping | Rationale |
|-----------|---------|-----------|
| $b/a \le$ `log_e_panel_ratio` | **Linear** | Narrow sub-interval; uniform nodes adequate |
| $w(a,T) \ge w(b,T)$ | **LogLower** | Weight falls toward $b$; cluster nodes near $a$ |
| $w(a,T) < w(b,T)$ | **LogUpper** | Weight rises toward $b$; cluster nodes near $b$ |

Zero-width sub-intervals (where $b \le a$) are skipped, which naturally
handles cases where boundary-layer points coincide with group edges.

**Endpoint-localized ξ regime:** When the Compton peak is genuinely
localised near $\xi = 1$ and narrow, the integrator routes to reflected-log
quadrature for improved resolution.  The condition is controlled by the
analytic endpoint-localization test (see ξ Peak-Focused Splitting below).

Group centers are placed at the geometric mean $\sqrt{E_\text{lo} \cdot E_\text{hi}}$.
Angle bins partition $[-1, 1]$ into $N$ equal segments of width $2/N$.  The $2\pi$
azimuthal factor is included so that summing over angle bins recovers the total
group-to-group cross section.

### Weight Functions and Denominators

Three weight functions are supported:

| Weight | $w(E,T)$ | Peak energy | Analytic denominator |
|--------|-----------|-------------|---------------------|
| **Planck** | $x^3/(e^x - 1)$, capped at $x = $ `cap_x` | $2.821439 \cdot k_B T$ | Clark (1987) polylogarithm series |
| **Wien** | $x^3 e^{-x}$, capped at `cap_x` | $3.0 \cdot k_B T$ | Taylor / closed-form (see below) |
| **Uniform** | 1 | none (`std::nullopt`) | $E_\text{right} - E_\text{left}$ |

The Planck cap avoids evaluating $x^3/(e^x - 1)$ in the exponential tail where
it underflows; above `cap_x` the weight is held constant at its value at the cap.

Each weight function provides a `peak_energy(T)` method that returns the energy
at which $w(E,T)$ attains its maximum (or `std::nullopt` for the uniform weight).
This is used by the E-axis sub-interval splitting to place a split point at the
weight peak, ensuring each sub-interval has a monotone weight function.

**Analytic denominator.**  The denominator $D(g) = \int_{\Delta E_g} w(E,T)\,dE$
is computed analytically via `WeightFunction::compute_denominator()`, which uses
Clark (1987) polylogarithm series for the Planck weight and closed-form
expressions for Wien and uniform weights.

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


### Ridge-Based E' Integration

The E' (middle) axis uses a ridge-based quadrature scheme that exploits Compton
kinematics to concentrate quadrature effort where the integrand is large.

#### Cold Compton Ridge

For an angle bin $[\xi_\text{lo}, \xi_\text{hi}]$ and incoming energy $E$, the
cold-electron recoil band in $E'$ is

$$E'_\text{cold}(\xi) = \frac{E}{1 + \gamma(1 - \xi)}, \qquad \gamma = E / m_e c^2$$

The cold ridge endpoints for the bin are:

$$\text{cold\_lo} = E'_\text{cold}(\xi_\text{lo}), \qquad \text{cold\_hi} = E'_\text{cold}(\xi_\text{hi})$$

Inside this band there exists a scattering angle for which a cold (rest-frame)
electron can produce the observed $E'$; outside, the kernel is exponentially
suppressed by the Boltzmann factor $\sim\exp(-(\lambda_+ - 1)/\tau)$.

#### Ridge Thermal Width

The local thermal width at scattering angle $\xi$ is derived from the curvature
of $\lambda_+$ at the cold-Compton saddle:

$$\sigma_{\gamma'}(\xi) = \frac{\gamma}{[1+\gamma(1-\xi)]^2} \sqrt{\tau(1-\xi)\left[2 + 2\gamma(1-\xi) + \gamma^2(1-\xi)\right]}$$

Converted to energy units: $\sigma_{E'}(\xi) = \sigma_{\gamma'}(\xi) \cdot m_e c^2$.
This is implemented as `ridge_thermal_width(E, xi, T)`.

This is a local width (depends on $\xi$) and is Gaussian-curvature based, so it
may be less accurate if prefactors vary strongly in the wings.

The old `thermal_half_width` formula ($E\sqrt{2\tau}$, with a 5x cold multiplier)
was an ad-hoc, $\xi$-independent estimate and has been removed.  The
`ridge_thermal_width` formula is derived from the kernel's curvature and
provides a tighter, angle-dependent width used by the E' ridge quadrature.

#### Retained Interval and Truncation

The retained interval extends $k_\text{cut} \cdot \sigma$ outward from each
cold ridge endpoint:

$$[\text{cold\_lo} - k_\text{cut} \cdot \sigma_\text{lo},\; \text{cold\_hi} + k_\text{cut} \cdot \sigma_\text{hi}]$$

clipped to the target group boundaries $[E'_\text{lo}, E'_\text{hi}]$.

The truncation is justified by $\lambda_+ - 1 \ge N\tau$ where $N = k^2/2$:
- $k_\text{cut} = 5$: $e^{-N} = e^{-12.5} \approx 3.7 \times 10^{-6}$
- $k_\text{cut} = 6$: $e^{-18} \approx 1.5 \times 10^{-8}$

The FWHM half-width is not used for truncation because it removes too much area.

Groups entirely outside the retained interval return exactly zero in default mode.
When the outward-from-peak group cutoff is active (`cutoff_ratio` is set), it
breaks immediately on these zeros, providing an automatic performance benefit.

#### Right Tail

Beyond the retained interval, the remaining target-group tail
$[\text{keep\_hi}, E'_\text{hi}]$ is still integrated when non-empty.  It uses
log Gauss-Legendre quadrature with `ep_edge_order` nodes when
$E'_\text{hi}/\text{keep\_hi} > 2$, and ordinary linear GL otherwise.  There is
no separate tail-width configuration; the retained interval is controlled by
`ep_k_cut`.

#### Three-Region Split

The retained interval is split into three sub-regions:

- **Left edge**: $[\text{keep\_lo},\; \text{cold\_lo} + k_\text{in} \cdot \sigma_\text{lo}]$
- **Ridge interior**: $[\text{cold\_lo} + k_\text{in} \cdot \sigma_\text{lo},\; \text{cold\_hi} - k_\text{in} \cdot \sigma_\text{hi}]$
- **Right edge**: $[\text{cold\_hi} - k_\text{in} \cdot \sigma_\text{hi},\; \text{keep\_hi}]$

All three use ordinary (linear) Gauss-Legendre quadrature in $E'$.  The edges
contain the sharpest rolloffs, so convergence should be checked by increasing
`ep_edge_order` first.

**Overlap collapse**: when $\text{cold\_lo} + k_\text{in} \cdot \sigma_\text{lo} \ge
\text{cold\_hi} - k_\text{in} \cdot \sigma_\text{hi}$, the three regions collapse
to a single central region $[\text{keep\_lo}, \text{keep\_hi}]$ integrated with
`ep_interior_order`.

#### Double-Peak Elastic Endpoint Handling

When the last angular bin includes near-forward scattering ($\xi_\text{hi} \to 1$),
the $\xi$-integrated $E'$ integrand contains two numerical features at very
different scales:

1. A **broad thermally-broadened Compton ridge** centred near $\text{cold\_lo}$,
   width $\sim \sigma_\text{lo}$ (order 0.1--3 keV depending on $T$).
2. A **narrow near-forward elastic endpoint feature** centred at $\text{cold\_hi}$
   ($\approx E$), width $\sim \sigma_\text{hi}$ (order 0.006--0.4 meV depending
   on $T$ and $\xi_\text{hi}$).

The standard three-region scheme collapses to a single interior GL panel when
$\text{edge\_lo} \ge \text{edge\_hi}$, which always happens for the last bin
because $\sigma_\text{hi} \ll \sigma_\text{lo}$.  A single GL panel spanning
several keV cannot resolve a meV-scale feature.  As `ep_order` changes, GL nodes
shift relative to this narrow feature, causing persistent 1--3% oscillation.

**4-region scheme.**  When $\sigma_\text{lo} / \sigma_\text{hi} >$ `DOUBLE_PEAK_RATIO_THRESHOLD` (= 10) and the elastic-core panel
$[\text{cold\_hi} - k_\text{cut} \cdot \sigma_\text{hi},\; \text{cold\_hi} + k_\text{cut} \cdot \sigma_\text{hi}]$
intersects $[E'_\text{lo}, E'_\text{hi}]$, the integration switches from the
three-region path to a four-region split:

| Region | Bounds | Quadrature | Rule |
|--------|--------|------------|------|
| Broad-left | $[\text{keep\_lo}, \text{ec\_lo}]$ | linear GL | `ep_edge_order` |
| Elastic-core | $[\text{ec\_lo}, \text{ec\_hi}]$ | linear GL | `ep_edge_order` |
| Broad-right | $[\text{ec\_hi}, \text{keep\_hi}]$ | linear GL | `ep_interior_order` |
| Far-tail | $[\text{keep\_hi}, \text{tail\_cap}]$ | log or linear GL | `ep_edge_order` |

where $\text{ec\_lo} = \max(\text{keep\_lo},\; \text{cold\_hi} - k_\text{cut} \cdot \sigma_\text{hi})$
and $\text{ec\_hi} = \min(\text{keep\_hi},\; \text{cold\_hi} + k_\text{cut} \cdot \sigma_\text{hi})$.

The elastic-core spans only $2 \cdot k_\text{cut} \cdot \sigma_\text{hi}$ (0.02--0.4 meV),
so even a modest GL order of 16 gives micro-eV node spacing that trivially
resolves the narrow feature.  The remaining regions contain smooth integrands
and converge normally.

The far-tail uses log-GL when $\text{tail\_cap} / \text{keep\_hi} > 2$ (wide
multiplicative range) and linear GL otherwise.

**Activation conditions:**  The double-peak path activates when
$\sigma_\text{lo} / \sigma_\text{hi} > 10$.  Non-last bins have a sigma ratio
of 1--3 and remain on the standard three-region path unchanged.  All region
boundaries are defensively clipped, and empty intervals are skipped.

#### `MGIntegrationConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `xi_order` | `nullopt` → 48 | GL order for the ξ peak panel |
| `xi_peak_k` | 5.0 | Half-width of the ξ peak window in sigma units |
| `xi_tail_order` | `nullopt` → 16 | GL order for ξ tail sub-intervals |
| `cutoff_ratio` | 1e-8 | Outward-from-peak early-termination ratio (must be > 0 when set; `nullopt`/`None` disables) |
| `ep_k_cut` | 5.0 | E' truncation width in sigma units (must be > 0) |
| `ep_k_in` | 2.0 | E' interior-edge separator in sigma units (must be >= 0, < `ep_k_cut`) |
| `ep_edge_order` | `nullopt` → 24 | GL order for E' edge regions |
| `ep_interior_order` | `nullopt` → 24 | GL order for E' ridge interior |
| `e_panel_order` | `nullopt` → 12 | GL order for E-axis sub-intervals |
| `log_e_panel_ratio` | 2.0 | Sub-interval width ratio threshold for log/rlog-E mapping (must be > 1) |
| `e_boundary_k` | 5.0 | Incoming-E boundary-layer width in sigma units |

All E' sub-regions use fixed-order GL (no adaptive refinement):

| Region | Order | Quadrature | Notes |
|--------|-------|------------|-------|
| Left edge | `ep_edge_order` | linear GL | Standard 3-region path |
| Interior | `ep_interior_order` | linear GL | Standard 3-region path |
| Right edge | `ep_edge_order` | linear GL | Standard 3-region path |
| Broad-left | `ep_edge_order` | linear GL | Double-peak 4-region path |
| Elastic-core | `ep_edge_order` | linear GL | Double-peak 4-region path |
| Broad-right | `ep_interior_order` | linear GL | Double-peak 4-region path |
| Right tail (always-on) | `ep_edge_order` | log or linear GL | Both paths |

#### Migration from Previous Scheme

The previous E' integration had two modes: adaptive (peak bisection + log/rlog
tails + log/rlog far groups) and flat (single dense GL rule per target group,
`FlatEpConfig`).  Both are superseded by the unified ridge scheme.

| Old parameter | New parameter | Notes |
|---|---|---|
| `peak_max_depth` | removed | No adaptive E' refinement; convergence via GL order |
| `tail_order` | `ep_edge_order` | Edge regions replace log/rlog tails |
| `far_order` | removed | Diagnostic tail mode removed |
| `flat_ep` (`FlatEpConfig`) | removed | Ridge scheme supersedes both adaptive and flat modes |
| `flat_E` | removed | E-axis always uses boundary layers |
| `warm_flat()` | removed | Factory presets removed; construct `MGIntegrationConfig` directly |

#### ξ Peak-Focused Splitting

The ξ integrand has a sharp peak at the Compton angle

$$\xi_\text{pk} = 1 - \frac{|\gamma'-\gamma|}{\gamma\,\gamma'}$$

with curvature-based Gaussian width

$$\sigma_\xi = \frac{\sqrt{\tau\,|\gamma'-\gamma|\,(2+|\gamma'-\gamma|)}}{\gamma\,\gamma'}$$

and FWHM $= 2\sqrt{2\ln 2}\;\sigma_\xi$.  The peak window is
$[\xi_\text{pk} - k \cdot \sigma_\xi,\; \xi_\text{pk} + k \cdot \sigma_\xi]$
where $k$ = `xi_peak_k` (default 5) is the half-width in $\sigma_\xi$ units
(total window = $2k\,\sigma_\xi$).  The raw, unclamped $\xi_\text{pk}$ determines
which integration regime applies — it is not clamped to the bin domain.

**Endpoint-localized reflected-log condition.**  The reflected-log ξ quadrature
(clustering nodes near $\xi = 1$) activates when either of two conditions holds:

**(A) Thermal endpoint-localisation:** the peak is close to $\xi = 1$ AND narrow:

$$\frac{|\gamma'-\gamma|}{\gamma\,\gamma'} \;\le\; \sigma_\xi
\quad\text{AND}\quad \sigma_\xi \;\le\; \varepsilon_\xi$$

**(B) Near-elastic kinematic cusp:** the fractional energy transfer is small,
so the Klein–Nishina forward peak at $\xi = 1$ dominates the last angular bin
regardless of thermal width:

$$\frac{|\Delta\gamma|}{\gamma} \;\le\; \varepsilon_\xi$$

where $\varepsilon_\xi$ = `XI_ENDPOINT_EPS` = 0.1 and
$\tau_\text{cusp}$ = `XI_CUSP_TAU` = 0.001 are compile-time constants calibrated
against a 50-point temperature sweep (1e-5 to 1e3 keV).

Condition (A) handles the narrow-peak regime (cold/moderate T), where the
thermal peak is genuinely localised near $\xi = 1$.  Condition (B) handles
same-group scattering at hot T where $\sigma_\xi \gg 1$ but the KN forward
cusp at $\xi = 1$ still requires rlog resolution.  Together, they capture
both the thermal and kinematic sources of endpoint structure.

When either condition is met, the integrator uses `rlog_legendre_integrate` on
shifted coordinates $s \in [\varepsilon, \text{span}]$ with $\xi = \xi_\text{lo} + s$,
clustering nodes near $\xi_\text{hi}$ (toward $\xi = 1$).

The exactly elastic case ($\gamma = \gamma'$) satisfies both conditions: (A)
evaluates as $0 \le 0$ and $0 \le \varepsilon_\xi$; (B) as $0 \le 0$.
Non-strict inequalities ($\le$) are required for this case.

This replaces the previous temperature-tiered `elastic_threshold(τ)` step
function, which used 6 empirically calibrated thresholds for different
temperature regimes.  The analytic two-part condition requires no hand-tuning
and correctly handles both the narrow-peak and broad-kernel regimes.  The
standalone predicate `endpoint_localized_xi(gamma, gamma_p, tau, xi_endpoint_eps)`
is available for direct testing.

**Integration regimes (non-endpoint-localized):**

1. **Peak window entirely left** ($\xi_\text{pk} + k\cdot\sigma_\xi \le \xi_\text{lo}$):
   the integrand decays away from $\xi_\text{lo}$.  Plain `legendre_integrate`
   over the full bin $[\xi_\text{lo}, \xi_\text{hi}]$.

2. **Peak window entirely right** ($\xi_\text{pk} - k\cdot\sigma_\xi \ge \xi_\text{hi}$):
   the integrand decays away from $\xi_\text{hi}$.  Plain `legendre_integrate`
   over the full bin $[\xi_\text{lo}, \xi_\text{hi}]$.

3. **Peak overlaps bin** (three-region split):
   - Left tail $[\xi_\text{lo}, \text{core\_lo}]$: plain `legendre_integrate`
     with `xi_tail_order` (default 16).
   - Peak core $[\text{core\_lo}, \text{core\_hi}]$: ordinary GL with
     `xi_order`.
   - Right tail $[\text{core\_hi}, \xi_\text{hi}]$: plain `legendre_integrate`
     with `xi_tail_order` (default 16).

   When the peak window covers the entire bin (common at high temperature),
   core_lo = ξ_lo and core_hi = ξ_hi, so no tail integrals are computed and
   the result is a single GL pass — equivalent to plain linear GL.

**Rationale for plain GL on non-elastic branches.** The log/rlog coordinate
mappings concentrate nodes near one endpoint, which in theory improves accuracy
for exponentially decaying integrands.  However, the mappings require positive
arguments (introducing shifted coordinates and eps guards), and the exp/log
transforms can amplify floating-point noise on very narrow sub-intervals.
Since the peak window already captures the dominant contribution (5σ on
each side = 10σ total), the far-bin and tail integrands are smooth and small,
making plain GL adequate with the existing quadrature orders.  This improves
robustness without measurable accuracy loss.

**Defensive guards.** Each tail is skipped if its span is less than
$10^{-14} \cdot \text{bin\_span}$ (prevents zero-width quadrature).
The peak core is skipped if core_hi $\le$ core_lo (floating-point edge case).

**Removed mechanisms:**
- The temperature-tiered `elastic_threshold(τ)` step function and all 12
  `XI_ELASTIC_*` constants are replaced by the analytic endpoint-localized
  condition above.
- The 5% ratio guard (`XI_PEAK_RATIO_THRESHOLD = 0.05`) is superseded by
  the endpoint-localized condition.
- The 80% width check is removed; splitting always applies and naturally
  degrades when the peak window covers the bin.
- The ratio-based log/rlog fallback (`LOG_XI_EP_RATIO_THRESHOLD = 1.5`) is
  removed; the peak geometry (left/right/overlapping) handles all cases.
- The `flat_xi` bypass is removed; peak-aware splitting always applies and
  naturally degrades to full-bin GL when the peak window covers the entire bin.
- Log/rlog mappings for non-endpoint-localized far-bin and tail sub-intervals
  are replaced with plain GL for robustness (avoids exp/log transforms on
  narrow intervals).

#### Recommended Production Configurations

The `cold_adaptive()` and `warm_default()` factory presets have been removed.
Construct `MGIntegrationConfig` directly with the desired parameters.
For cold temperatures (T < 0.1 keV), use higher GL orders (e.g.
`xi_order=512`, `ep_edge_order=192`, `ep_interior_order=192`,
`cutoff_ratio=1e-12`).  For warm temperatures, moderate orders suffice
(e.g. `xi_order=96`, `ep_edge_order=96`, `ep_interior_order=96`).

Python usage:
```python
import compton_matrix._compton_multigroup as cm

cfg = cm.MGIntegrationConfig(
    cutoff_ratio=1e-12,
    xi_order=96,
    ep_edge_order=96,
    ep_interior_order=96,
)
det = cm.ComptonMultigroupKernel(bounds_erg, wf, cfg)
S = det.compute_sigma_matrix(kernel=kernel, T=T_K)
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
3. When `cutoff_ratio` is set: expand rightward
   ($g' = g'_\text{peak}+1, g'_\text{peak}+2, \ldots$), computing all angle
   bins and summing absolute values.  Stop when the sum drops below
   `cutoff_ratio * peak_sum`.  Remaining groups stay at zero.
4. Expand leftward ($g' = g'_\text{peak}-1, g'_\text{peak}-2, \ldots$):
   same stopping criterion.
5. When `cutoff_ratio` is `nullopt` (`None` in Python), all target groups are
   evaluated unconditionally.

**Why this is valid:** Compton scattering conserves total rate per incoming
group (row sums $\sum_{g'} \sigma(g \to g') \approx \sigma_T$).  The kernel
is exponentially suppressed by the Boltzmann factor $\sim\exp(-\Delta E / kT)$
for large energy transfers, so the omitted groups contribute a fraction below
`cutoff_ratio` of the peak.  For `cutoff_ratio = 1e-8`, the row-sum error is
negligible compared to quadrature tolerances.

**Default:** `cutoff_ratio = 1e-8`, set at construction time via
`MGIntegrationConfig`.  Pass `nullopt` (`None`) to disable cutoff entirely.


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

### Full Multigroup Temperature Derivative

The methods above (`compute_kernel_derivative_contribution`) only differentiate the kernel
$\Sigma_E$ with respect to $T$.  The full derivative of the multigroup cross
section also includes the temperature dependence of the weight function $w(E,T)$
and the denominator $D(g) = \int_{\Delta E_g} w(E,T)\,dE$:

$$\frac{d\sigma}{dT}\bigg|_{g \to g'} = \frac{1}{D}\frac{dN_{\text{kd}}}{dT} + \frac{1}{D}\frac{dN_{\text{wd}}}{dT} - \frac{\sigma}{D}\frac{dD}{dT}$$

where

- $dN_{\text{kd}}/dT$ = numerator integral with the kernel derivative
  $\partial\Sigma_E/\partial T$ (same as `compute_kernel_derivative_contribution`),
- $dN_{\text{wd}}/dT$ = numerator integral with $\frac{\partial \ln w}{\partial T} \cdot \Sigma_E$
  (weight-function derivative contribution),
- $dD/dT$ = analytic derivative of the denominator.

**Multiplier trick.**  The weight-derivative term is computed without modifying
the integration routines.  A lightweight `WeightDerivMultiplier` wraps the
user-provided `KernelMultiplier` and multiplies each integrand evaluation by
$d(\ln w)/dT$ at the incoming energy.  This is passed to the same
`compute_matrix_impl` (deterministic) or `mc_integrate` (MC) used for the
kernel-only path.  The `d(\ln w)/dT` form avoids dividing by $w$ when $w$ is
tiny (above the cap energy, both $w$ and $dw/dT$ are zero, and $d(\ln w)/dT$
returns zero without computing a ratio).

**Cutoff disabled.**  The outward-from-peak group cutoff is disabled for the
full derivative (all $G^2$ pairs are evaluated).  The cutoff exploits the
assumption that the integrand is dominated by near-elastic scattering, which
may not hold for the weight-derivative term where $d(\ln w)/dT$ varies
across groups.

**Cost.**  `compute_dsigma_dT_matrix` makes 3× the integration calls of
`compute_sigma_matrix`: once for the kernel derivative, once for the
weight-derivative multiplier, and once for the original $\sigma$ (needed for
the denominator correction term).  Future optimization could fuse the weight
and kernel-derivative integrands into one pass.

**Deterministic implementation.**  Three `compute_matrix_impl` calls with
`effective_cutoff = std::nullopt`, combined element-wise via the quotient rule.
The denominator and its derivative are computed analytically by the weight
function.

**MC implementation.**  Three independent `mc_integrate` calls, each consuming
a separate RNG base seed, combined with the analytic $dD/dT \,/\, D$ ratio.
The MC estimator is consistent (converges as $N \to \infty$).

**Multiplier contract.**  The `KernelMultiplier` interface is purely kinematic:
`operator()(E, Ep, xi)`.  It must not depend on temperature; temperature
dependence is handled entirely by the kernel and weight function.

**Cap kink.**  The Planck and Wien weight functions have a cap at $x = $ `cap_x`,
above which $w$ is held constant.  At the cap boundary, `d_weight_dT` returns 0
(capped-side convention), introducing a kink in the temperature derivative.
The Leibniz-rule derivation of `d_denominator_dT` is valid because $w$ itself
is continuous at the cap.
