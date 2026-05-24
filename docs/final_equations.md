# Final Equations Used in the Kernel Integration

This document lists the exact equations implemented in `src/compton_kernel_quadrature.cpp`,
in the order they are evaluated, with variable names matching the code.

---

## Input Variables

| Symbol | Code name | Definition | Units |
|---|---|---|---|
| $E$ | `E` | Incident photon energy | erg |
| $E'$ | `E_prime` | Scattered photon energy | erg |
| $\xi$ | `xi` | $\cos(\theta)$, scattering angle cosine, in $(-1, 1)$ | dimensionless |
| $\tau$ | `tau` | $kT / (m_e c^2)$ | dimensionless |
| $N_e$ | `Ne` | Electron number density (or 1.0 for microscopic) | cm$^{-3}$ |

---

## Step 1: Dimensionless Energies

$$
\gamma = \frac{E}{m_e c^2}, \qquad \gamma' = \frac{E'}{m_e c^2}
$$

---

## Step 2: Kinematic Parameters (`compute_params`)

### Primary quantities

$$
a = 1 - \xi, \qquad s = \frac{1}{\gamma} + \frac{1}{\gamma'}, \qquad \Delta\gamma = \gamma' - \gamma
$$

### Momentum transfer (stable form)

$$
q^2 = (\Delta\gamma)^2 + 2\,\gamma\,\gamma'\,a
$$

$$
q = \sqrt{q^2}
$$

### Angular parameter

$$
\omega^2 = \frac{1 + \xi}{a} = \frac{1+\xi}{1-\xi}
$$

### Minimum electron Lorentz factor

$$
\Delta = \sqrt{\left(1 + \frac{\gamma\gamma' a}{2}\right)\left(1 + \frac{(\Delta\gamma)^2}{2\,\gamma\gamma' a}\right)}
$$

$$
\lambda_+ = \frac{\Delta\gamma}{2} + \Delta
$$

Clamped: if $\lambda_+ < 1$ due to roundoff, set $\lambda_+ = 1$.

### Shifted momentum parameters

$$
\rho_+ = \lambda_+ + \gamma, \qquad \rho_- = \lambda_+ - \gamma'
$$

### Boundary evaluation terms

$$
\alpha_+ = \frac{1}{\sqrt{\rho_+^2 + \omega^2}}, \qquad \alpha_- = \frac{1}{\sqrt{\rho_-^2 + \omega^2}}
$$

### Combined constants

$$
G = -\gamma\gamma' + \frac{2}{a} + \frac{2}{\gamma\gamma' a^2}
$$

$$
A_+ = G - \frac{s}{\tau\, a^2}, \qquad A_- = G + \frac{s}{\tau\, a^2}
$$

$$
\Psi = \frac{2\tau\,\gamma\gamma'}{q} + \frac{s}{a^2}\left(\alpha_+ + \alpha_-\right) + \frac{\rho_+\,\alpha_+ - \rho_-\,\alpha_-}{a}
$$

---

## Step 3: Prefactor (`stable_sigma0_E`)

$$
\sigma_0 = \frac{N_e\, r_e^2\, m_e c^2}{4\, E^2\, \tau} \cdot \frac{\exp\!\left(-\dfrac{\lambda_+ - 1}{\tau}\right)}{\tilde{K}_2\!\left(\dfrac{1}{\tau}\right)}
$$

where:

$$
r_e^2 = \frac{\sigma_T}{8\pi/3}
$$

$$
\tilde{K}_2(x) = e^x \, K_2(x) \quad \text{(exponentially scaled modified Bessel function)}
$$

### Origin of $\exp(-(\lambda_+ - 1)/\tau)$

The original (un-scaled) formula has two separate factors:

1. The change of variable in the electron momentum integral pulls out $e^{-\lambda_+/\tau}$
2. The Maxwell-Juttner normalization denominator is $K_2(1/\tau) = e^{-1/\tau} \cdot \tilde{K}_2(1/\tau)$

Their ratio:

$$
\frac{e^{-\lambda_+/\tau}}{K_2(1/\tau)} = \frac{e^{-\lambda_+/\tau}}{e^{-1/\tau} \cdot \tilde{K}_2(1/\tau)} = \frac{e^{-(\lambda_+ - 1)/\tau}}{\tilde{K}_2(1/\tau)}
$$

The $-1$ arises from absorbing $e^{-1/\tau}$ (from the Bessel function) into
the numerator.  Physically, $\lambda_+ - 1$ is the minimum electron **kinetic energy**
(above rest mass, in units of $m_e c^2$) required for the transition.  The exponential
$e^{-(\lambda_+-1)/\tau}$ is the Boltzmann suppression for finding such an electron
in the thermal distribution.

### $\tilde{K}_2$: The Scaled Bessel Function

$K_2(x)$ is the modified Bessel function of the second kind, order 2.  It arises
from the normalization of the relativistic Maxwell-Juttner electron distribution:

$$
\int_1^\infty \lambda\sqrt{\lambda^2 - 1}\; e^{-\lambda/\tau}\, d\lambda = \tau^2 \, K_2(1/\tau)
$$

This integral is the partition function for relativistic thermal electrons
(density of states times Boltzmann weight).  It appears in the denominator
because the kernel is proportional to the fraction of electrons at a given
momentum.

At low temperatures, $K_2(1/\tau)$ underflows (e.g., $K_2(500) \sim 10^{-217}$),
but the scaled form $\tilde{K}_2(x) = e^x K_2(x)$ remains $O(\sqrt{\pi/(2x)})$
and is always representable in double precision.

In scipy this function is `kve(2, x)`.  In the code it is implemented as
`scaled_K2()` using Boost for $x < 50$ and a Hankel asymptotic series for $x \geq 50$.

---

## Step 4: Change of Variable and Gauss-Laguerre Quadrature

The original integral over electron Lorentz factor $\lambda$ runs from $\lambda_+$ to $\infty$
with a Boltzmann weight:

$$
I = \int_{\lambda_+}^{\infty} f(\lambda)\, e^{-\lambda/\tau}\, d\lambda
$$

We substitute $\lambda = \lambda_+ + \tau\, x$, so $d\lambda = \tau\, dx$ and as
$\lambda \to \infty$, $x \to \infty$:

$$
I = \tau\, e^{-\lambda_+/\tau} \int_0^{\infty} f(\lambda_+ + \tau\, x)\, e^{-x}\, dx
$$

The factor $e^{-\lambda_+/\tau}$ is absorbed into $\sigma_0$ (combining with $K_2$ to
give $e^{-(\lambda_+-1)/\tau}/\tilde{K}_2$ as shown above).  The remaining integral
is in standard Gauss-Laguerre form with weight $e^{-x}$ on $[0, \infty)$.

**The variable $x$ is therefore $(\lambda - \lambda_+)/\tau$** -- the electron Lorentz
factor measured from its minimum value $\lambda_+$, in units of the thermal scale $\tau$.

The integral is evaluated as:

$$
I_Q = \sum_{i=1}^{N_L} w_i \, f(x_i)
$$

where $\{x_i, w_i\}_{i=1}^{N_L}$ are the $N_L$-point Gauss-Laguerre nodes and weights
(zeros of $L_{N_L}(x)$ and associated Christoffel numbers), exact for:

$$
\int_0^\infty g(x)\, e^{-x}\, dx = \sum_{i=1}^{N_L} w_i\, g(x_i)
$$

when $g$ is a polynomial of degree $\leq 2N_L - 1$.

---

## Step 5a: Post-IBP Integrand (default form)

For each quadrature node $x_i$, define:

$$
\rho = \tau\, x_i
$$

$$
r_+ = \rho + \rho_+, \qquad r_- = \rho + \rho_-
$$

$$
R_+ = r_+^2 + \omega^2, \qquad R_- = r_-^2 + \omega^2
$$

The integrand:

$$
H(x_i) = \left(A_+ - \frac{r_+}{\tau\, a}\right) \frac{1}{\sqrt{R_+}} + \left(-A_- + \frac{r_-}{\tau\, a}\right) \frac{1}{\sqrt{R_-}}
$$

$$
f(x_i) = \tau \cdot H(x_i)
$$

**Final result:**

$$
\boxed{\Sigma_E = \sigma_0 \cdot \left(\Psi + I_Q\right)}
$$

---

## Step 5b: Pre-IBP Integrand (alternative form)

Constants computed once:

$$
C = \frac{2\,\gamma\,\gamma'}{q}, \qquad a^2 = a \cdot a, \qquad (1+\xi) = 2 - a
$$

For each quadrature node $x_i$, define:

$$
t_+ = \tau\, x_i + \rho_+, \qquad t_- = \tau\, x_i + \rho_-
$$

$$
R_+ = t_+^2 + \omega^2, \qquad R_- = t_-^2 + \omega^2
$$

Numerators:

$$
n_+ = t_- \cdot s + (1+\xi), \qquad n_- = t_+ \cdot s - (1+\xi)
$$

The integrand:

$$
F(x_i) = C + \frac{1}{a^2}\left(\frac{n_+}{R_-^{3/2}} + \frac{n_-}{R_+^{3/2}}\right) + G\left(\frac{1}{\sqrt{R_+}} - \frac{1}{\sqrt{R_-}}\right)
$$

$$
f(x_i) = \tau \cdot F(x_i)
$$

**Final result:**

$$
\boxed{\Sigma_E = \sigma_0 \cdot I_Q}
$$

Note: no $\Psi$ term in the pre-IBP form.

---

## Step 6: Error Estimate

Two evaluations are performed: $I_Q(N_L)$ at full order and $I_Q(N_L/2)$ at half order.

$$
\varepsilon_{\text{abs}} = |\sigma_0| \cdot |I_Q(N_L) - I_Q(N_L/2)|
$$

$$
\varepsilon_{\text{rel}} = \frac{\varepsilon_{\text{abs}}}{|\Sigma_E| + 10^{-300}}
$$

---

## Step 7: Multigroup Scattering Matrix (Python layer)

To form the group-to-group scattering matrix element:

$$
S[g, g'] = 2\pi \int_{-1}^{1} \int_{E'_{\text{lo}}}^{E'_{\text{hi}}} \Sigma_E\!\left(E_g,\, E',\, \xi,\, \tau,\, N_e\right)\, dE'\, d\xi
$$

where:
- $E_g$ = center energy of incident group $g$
- $[E'_{\text{lo}},\, E'_{\text{hi}}]$ = energy boundaries of scattered group $g'$
- The factor $2\pi$ gives the zeroth angular moment (matching CMMC convention)
- Both integrals are computed by nested `scipy.integrate.quad`

The differential cross-section per unit energy (for plotting) is:

$$
\sigma(E') = \frac{S[g, g']}{E'_{\text{hi}} - E'_{\text{lo}}} \quad [\text{cm}^2/\text{erg}]
$$

---

## Constants Used

| Constant | Value | Source |
|---|---|---|
| $m_e c^2$ | $8.187 \times 10^{-7}$ erg | `units::me_c2` |
| $\sigma_T$ | $6.652 \times 10^{-25}$ cm$^2$ | `units::sigma_thomson` |
| $r_e^2$ | $\sigma_T / (8\pi/3) = 7.941 \times 10^{-26}$ cm$^2$ | derived |

---

## Relation Between Forms

The two forms are algebraically identical:

$$
\Psi + I_Q^{\text{post}} = I_Q^{\text{pre}}
$$

They differ only in numerical conditioning:
- **Post-IBP**: smooth integrand ($1/\sqrt{R}$), but $\Psi$ and $I_Q^{\text{post}}$ cancel at low $\tau$
- **Pre-IBP**: sharper integrand ($1/R^{3/2}$), but no cancellation at any $\tau$
