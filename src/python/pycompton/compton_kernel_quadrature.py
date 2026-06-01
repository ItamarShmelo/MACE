"""
Pure Python implementation of the Kershaw-Prasad-Beason Compton frequency
kernel via Gauss-Laguerre quadrature.

This mirrors the C++ implementation in
src/compton_kernel_quadrature/compton_kernel_quadrature.cpp
using only numpy and scipy -- no custom integration schemes.

Two quadrature modes are available:
  - "fixed"    : fixed-order Gauss-Laguerre via scipy.special.roots_laguerre
                 (matches the C++ implementation for apples-to-apples comparison)
  - "adaptive" : scipy.integrate.quad on [0, inf) with explicit e^{-x} weight
                 (independent diagnostic cross-check)
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import kve, roots_laguerre
from scipy.integrate import quad

from _units import me, clight, me_c2, k_boltz, sigma_thomson, r_e2, kev

# ═══════════════════════════════════════════════════════════════════════════════
# Scaled Bessel function
# ═══════════════════════════════════════════════════════════════════════════════

def scaled_K2(x):
    """Compute kve(2, x) = exp(x) * K_2(x) using scipy."""
    if not (x > 0.0 and math.isfinite(x)):
        raise ValueError("scaled_K2 requires finite x > 0")
    return kve(2, x)


# ═══════════════════════════════════════════════════════════════════════════════
# Gauss-Laguerre node/weight cache
# ═══════════════════════════════════════════════════════════════════════════════

_LAGUERRE_CACHE = {}

def _laguerre_nodes_weights(NL):
    if NL not in _LAGUERRE_CACHE:
        nodes, weights = roots_laguerre(NL)
        _LAGUERRE_CACHE[NL] = (nodes, weights)
    return _LAGUERRE_CACHE[NL]


# ═══════════════════════════════════════════════════════════════════════════════
# Kinematic parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KershawParams:
    a: float = 0.0
    s: float = 0.0
    q: float = 0.0
    omega2: float = 0.0
    Delta: float = 0.0
    lambda_plus: float = 0.0
    rho_plus: float = 0.0
    rho_minus: float = 0.0
    alpha_plus: float = 0.0
    alpha_minus: float = 0.0
    G: float = 0.0
    A_plus: float = 0.0
    A_minus: float = 0.0
    Psi: float = 0.0


def compute_params(gamma, gamma_p, xi, tau):
    p = KershawParams()

    p.a = 1.0 - xi
    p.s = 1.0 / gamma + 1.0 / gamma_p

    dg = gamma_p - gamma
    q2 = dg * dg + 2.0 * gamma * gamma_p * p.a
    p.q = math.sqrt(q2)

    p.omega2 = (1.0 + xi) / p.a

    gg_a = gamma * gamma_p * p.a
    factor1 = 1.0 + gg_a / 2.0
    factor2 = 1.0 + (dg * dg) / (2.0 * gg_a)
    p.Delta = math.sqrt(factor1 * factor2)

    p.lambda_plus = dg / 2.0 + p.Delta
    if p.lambda_plus < 1.0 - 1e-12:
        raise RuntimeError("lambda_plus significantly below 1")
    if p.lambda_plus < 1.0:
        p.lambda_plus = 1.0

    p.rho_plus = p.lambda_plus + gamma
    p.rho_minus = p.lambda_plus - gamma_p

    Rp0 = p.rho_plus**2 + p.omega2
    Rm0 = p.rho_minus**2 + p.omega2
    p.alpha_plus = 1.0 / math.sqrt(Rp0)
    p.alpha_minus = 1.0 / math.sqrt(Rm0)

    a2 = p.a * p.a
    p.G = -gamma * gamma_p + 2.0 / p.a + 2.0 / (gamma * gamma_p * a2)

    s_over_tau_a2 = p.s / (tau * a2)
    p.A_plus = p.G - s_over_tau_a2
    p.A_minus = p.G + s_over_tau_a2

    p.Psi = (2.0 * tau * gamma * gamma_p / p.q
             + p.s / a2 * (p.alpha_plus + p.alpha_minus)
             + (p.rho_plus * p.alpha_plus
                - p.rho_minus * p.alpha_minus) / p.a)

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Prefactor
# ═══════════════════════════════════════════════════════════════════════════════

def stable_sigma0_E(E, tau, lambda_plus, Ne):
    return (Ne * r_e2 * me_c2
            / (4.0 * E * E * tau)
            * math.exp(-(lambda_plus - 1.0) / tau)
            / scaled_K2(1.0 / tau))


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed Gauss-Laguerre quadrature (primary mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _integrand_post_ibp(x, p, tau):
    """Post-IBP integrand, vectorized over x (numpy array)."""
    rho = tau * x
    rp = rho + p.rho_plus
    rm = rho + p.rho_minus

    Rp = rp * rp + p.omega2
    Rm = rm * rm + p.omega2

    inv_sqrt_Rp = 1.0 / np.sqrt(Rp)
    inv_sqrt_Rm = 1.0 / np.sqrt(Rm)

    tau_a = tau * p.a
    H = ((p.A_plus - rp / tau_a) * inv_sqrt_Rp
         + (-p.A_minus + rm / tau_a) * inv_sqrt_Rm)

    return tau * H


def _integrand_pre_ibp(x, p, tau, gamma_val, gamma_p_val, const_term, a2,
                        one_plus_xi):
    """Pre-IBP integrand, vectorized over x (numpy array)."""
    t_plus = tau * x + p.rho_plus
    t_minus = tau * x + p.rho_minus

    Rp = t_plus * t_plus + p.omega2
    Rm = t_minus * t_minus + p.omega2

    inv_sqrt_Rp = 1.0 / np.sqrt(Rp)
    inv_sqrt_Rm = 1.0 / np.sqrt(Rm)
    inv_Rp_32 = inv_sqrt_Rp / Rp
    inv_Rm_32 = inv_sqrt_Rm / Rm

    num_plus = t_minus * p.s + one_plus_xi
    num_minus = t_plus * p.s - one_plus_xi

    bracket_32 = (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2
    bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm)

    F = const_term + bracket_32 + bracket_12
    return tau * F


def compute_IQ_post_ibp(p, tau, NL):
    nodes, weights = _laguerre_nodes_weights(NL)
    vals = _integrand_post_ibp(nodes, p, tau)
    return np.sum(weights * vals)


def compute_IQ_pre_ibp(p, tau, NL):
    gamma_val = p.rho_plus - p.lambda_plus
    gamma_p_val = p.lambda_plus - p.rho_minus
    const_term = 2.0 * gamma_val * gamma_p_val / p.q
    a2 = p.a * p.a
    one_plus_xi = 2.0 - p.a

    nodes, weights = _laguerre_nodes_weights(NL)
    vals = _integrand_pre_ibp(nodes, p, tau, gamma_val, gamma_p_val,
                               const_term, a2, one_plus_xi)
    return np.sum(weights * vals)


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive quadrature diagnostic mode
# ═══════════════════════════════════════════════════════════════════════════════

def compute_IQ_post_ibp_adaptive(p, tau, epsabs=1e-30, epsrel=1e-12,
                                  limit=200):
    def f(x):
        rho = tau * x
        rp = rho + p.rho_plus
        rm = rho + p.rho_minus

        Rp = rp * rp + p.omega2
        Rm = rm * rm + p.omega2

        inv_sqrt_Rp = 1.0 / math.sqrt(Rp)
        inv_sqrt_Rm = 1.0 / math.sqrt(Rm)

        tau_a = tau * p.a
        H = ((p.A_plus - rp / tau_a) * inv_sqrt_Rp
             + (-p.A_minus + rm / tau_a) * inv_sqrt_Rm)

        return math.exp(-x) * tau * H

    val, err = quad(f, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return val, err


def compute_IQ_pre_ibp_adaptive(p, tau, epsabs=1e-30, epsrel=1e-12,
                                 limit=200):
    gamma_val = p.rho_plus - p.lambda_plus
    gamma_p_val = p.lambda_plus - p.rho_minus
    const_term = 2.0 * gamma_val * gamma_p_val / p.q
    a2 = p.a * p.a
    one_plus_xi = 2.0 - p.a

    def f(x):
        t_plus = tau * x + p.rho_plus
        t_minus = tau * x + p.rho_minus

        Rp = t_plus * t_plus + p.omega2
        Rm = t_minus * t_minus + p.omega2

        inv_sqrt_Rp = 1.0 / math.sqrt(Rp)
        inv_sqrt_Rm = 1.0 / math.sqrt(Rm)
        inv_Rp_32 = inv_sqrt_Rp / Rp
        inv_Rm_32 = inv_sqrt_Rm / Rm

        num_plus = t_minus * p.s + one_plus_xi
        num_minus = t_plus * p.s - one_plus_xi

        bracket_32 = (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2
        bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm)

        F = const_term + bracket_32 + bracket_12
        return math.exp(-x) * tau * F

    val, err = quad(f, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return val, err


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level API
# ═══════════════════════════════════════════════════════════════════════════════

def sigma_E(E, E_prime, xi, T, Ne=1.0, form="post_ibp", NL=128,
            method="fixed"):
    """
    Evaluate the Compton frequency kernel Sigma_E(E -> E', xi; T, Ne).

    Parameters
    ----------
    E        : Incident photon energy [erg]
    E_prime  : Scattered photon energy [erg]
    xi       : cos(scattering angle), strictly in (-1, 1)
    T        : Electron temperature [K]
    Ne       : Electron number density [cm^-3] (use 1.0 for microscopic)
    form     : "post_ibp" or "pre_ibp"
    NL       : Gauss-Laguerre order (used only for method="fixed")
    method   : "fixed" (Gauss-Laguerre) or "adaptive" (scipy.integrate.quad)

    Returns
    -------
    (value, abs_error, rel_error)
    """
    if not (E > 0.0 and math.isfinite(E)):
        raise ValueError("E must be finite and > 0")
    if not (E_prime > 0.0 and math.isfinite(E_prime)):
        raise ValueError("E_prime must be finite and > 0")
    if not (T > 0.0 and math.isfinite(T)):
        raise ValueError("T must be finite and > 0")
    if not (-1.0 < xi < 1.0 and math.isfinite(xi)):
        raise ValueError("xi must be finite and strictly inside (-1, 1)")
    if not math.isfinite(Ne):
        raise ValueError("Ne must be finite")
    if 1.0 - xi < 1e-14:
        raise ValueError("xi too close to 1 for direct quadrature")

    tau = T * k_boltz / me_c2
    gamma = E / me_c2
    gamma_p = E_prime / me_c2

    p = compute_params(gamma, gamma_p, xi, tau)
    sigma0 = stable_sigma0_E(E, tau, p.lambda_plus, Ne)

    tiny_scale = 1e-300

    if method == "fixed":
        if form == "post_ibp":
            IQ_hi = compute_IQ_post_ibp(p, tau, NL)
            IQ_lo = compute_IQ_post_ibp(p, tau, NL // 2)
            value = sigma0 * (p.Psi + IQ_hi)
        else:
            IQ_hi = compute_IQ_pre_ibp(p, tau, NL)
            IQ_lo = compute_IQ_pre_ibp(p, tau, NL // 2)
            value = sigma0 * IQ_hi

        abs_error = abs(sigma0) * abs(IQ_hi - IQ_lo)
        rel_error = abs_error / (abs(value) + tiny_scale)
        return value, abs_error, rel_error

    elif method == "adaptive":
        if form == "post_ibp":
            IQ_val, IQ_err = compute_IQ_post_ibp_adaptive(p, tau)
            value = sigma0 * (p.Psi + IQ_val)
        else:
            IQ_val, IQ_err = compute_IQ_pre_ibp_adaptive(p, tau)
            value = sigma0 * IQ_val

        abs_error = abs(sigma0) * IQ_err
        rel_error = abs_error / (abs(value) + tiny_scale)
        return value, abs_error, rel_error

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'fixed' or 'adaptive'.")
