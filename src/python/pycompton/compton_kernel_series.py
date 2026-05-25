"""
Pure Python implementation of the Kershaw-Prasad-Beason Compton frequency
kernel via Section 4 series expansions: power series and low-temperature
asymptotic series.

This mirrors the C++ implementation in
src/compton_kernel_series/compton_kernel_series.cpp
using only numpy and scipy.

Two series methods are available:
  - "power"      : power series using scaled exponential integrals Ehat_m
  - "asymptotic" : low-temperature asymptotic series using Legendre polynomials
  - "auto"       : selects method based on tau*alpha_pm threshold
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import expn, eval_legendre

from .compton_kernel_quadrature import (
    compute_params, stable_sigma0_E, me_c2, KershawParams,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Scaled exponential integral: Ehat_m(x) = exp(x) * E_m(x)
# ═══════════════════════════════════════════════════════════════════════════════

def _ehat_asymptotic(m, x, n_terms=15):
    """Large-x asymptotic: Ehat_m(x) ~ (1/x)[1 - m/x + m(m+1)/x^2 - ...]"""
    inv_x = 1.0 / x
    result = 1.0
    term = 1.0
    for k in range(1, n_terms):
        term *= -(m + k - 1) * inv_x
        result += term
        if abs(term) < 1e-15 * abs(result):
            break
    return inv_x * result


def ehat_expn(m, x):
    """
    Compute Ehat_m(x) = exp(x) * E_m(x) safely.

    For x < 50: direct product exp(x) * expn(m, x).
    For x >= 50: asymptotic expansion (avoids exp overflow).
    """
    if x <= 0.0:
        raise ValueError("ehat_expn requires x > 0")
    if m < 1:
        raise ValueError("ehat_expn requires m >= 1")

    if x < 50.0:
        return math.exp(x) * float(expn(m, x))
    return _ehat_asymptotic(m, x)


# ═══════════════════════════════════════════════════════════════════════════════
# Series result type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SeriesResult:
    value: float
    estimated_abs_error: float
    estimated_rel_error: float
    terms_used: int
    method_used: str      # "power", "asymptotic", or "auto"
    converged: bool


# ═══════════════════════════════════════════════════════════════════════════════
# Power series
# ═══════════════════════════════════════════════════════════════════════════════

_POISSON_Y_MAX = 500.0

def _power_series_normalized(p: KershawParams, gamma: float, gamma_p: float,
                              tau: float, eps_rel: float = 1e-12,
                              n_min: int = 4, n_max: int = 200):
    """
    Compute the normalized ratio Sigma_E / Sigma_0 via power series.

    Returns (normalized_ratio, estimated_normalized_error, terms_used, converged).
    """
    a = p.a
    omega = math.sqrt(p.omega2)
    b = omega / (2.0 * tau)

    theta_plus = math.asinh(p.rho_plus / omega)
    theta_minus = math.asinh(p.rho_minus / omega)

    x_plus = b * math.exp(theta_plus)
    y_plus = b * math.exp(-theta_plus)
    x_minus = b * math.exp(theta_minus)
    y_minus = b * math.exp(-theta_minus)

    if y_plus > _POISSON_Y_MAX or y_minus > _POISSON_Y_MAX:
        return p.Psi, 0.0, 0, False

    if x_plus <= 0.0 or x_minus <= 0.0:
        return p.Psi, 0.0, 0, False

    w_plus = math.exp(-y_plus)
    w_minus = math.exp(-y_minus)

    P_plus = 0.0
    P_minus = 0.0

    eps_tiny = 1e-300

    last_term_mag = 0.0
    terms_used = 0

    for n in range(n_max + 1):
        coeff_plus = p.A_plus + 2.0 * n / a
        coeff_minus = p.A_minus + 2.0 * n / a

        ehat_p = ehat_expn(n + 1, x_plus)
        ehat_m = ehat_expn(n + 1, x_minus)

        t_plus = w_plus * coeff_plus * ehat_p
        t_minus = w_minus * coeff_minus * ehat_m

        P_plus += t_plus
        P_minus += t_minus

        term_mag = abs(t_plus) + abs(t_minus)
        last_term_mag = term_mag
        terms_used = n + 1

        S_n = abs(P_plus) + abs(P_minus)
        if n >= n_min and term_mag / (S_n + eps_tiny) < eps_rel:
            break

        if n < n_max:
            w_plus *= y_plus / (n + 1)
            w_minus *= y_minus / (n + 1)

    converged = terms_used <= n_max
    normalized_ratio = p.Psi + P_plus - P_minus
    return normalized_ratio, last_term_mag, terms_used, converged


# ═══════════════════════════════════════════════════════════════════════════════
# Asymptotic series
# ═══════════════════════════════════════════════════════════════════════════════

def _asymptotic_series_normalized(p: KershawParams, gamma: float, gamma_p: float,
                                   tau: float, eps_rel: float = 1e-12,
                                   n_min: int = 4, n_max: int = 200):
    """
    Compute the normalized ratio Sigma_E / Sigma_0 via low-temperature
    asymptotic series.

    Returns (normalized_ratio, estimated_normalized_error, terms_used, converged).
    """
    a = p.a
    a2 = a * a

    zeta_plus = p.rho_plus * p.alpha_plus
    zeta_minus = p.rho_minus * p.alpha_minus

    eta_plus = p.alpha_plus * (p.s / a2 + p.rho_plus / a)
    eta_minus = p.alpha_minus * (-p.s / a2 + p.rho_minus / a)

    base_term = 2.0 * tau * gamma * gamma_p / p.q

    neg_tau_alpha_plus = -tau * p.alpha_plus
    neg_tau_alpha_minus = -tau * p.alpha_minus

    S_plus = 0.0
    S_minus = 0.0

    smallest_term_mag = math.inf
    best_S_plus = 0.0
    best_S_minus = 0.0
    best_terms = 0
    increase_count = 0
    prev_term_mag = math.inf

    factorial_n = 1.0
    power_plus = neg_tau_alpha_plus
    power_minus = neg_tau_alpha_minus

    terms_used = 0

    for n in range(n_max + 1):
        if n > 0:
            factorial_n *= n
            power_plus *= neg_tau_alpha_plus
            power_minus *= neg_tau_alpha_minus

        factorial_n1 = factorial_n * (n + 1)

        Pp_n = float(eval_legendre(n, zeta_plus))
        Pp_n1 = float(eval_legendre(n + 1, zeta_plus))
        Pm_n = float(eval_legendre(n, zeta_minus))
        Pm_n1 = float(eval_legendre(n + 1, zeta_minus))

        term_plus = power_plus * (
            (-p.G * factorial_n + factorial_n1 / a) * Pp_n
            - eta_plus * factorial_n1 * Pp_n1
        )

        term_minus = power_minus * (
            (p.G * factorial_n - factorial_n1 / a) * Pm_n
            + eta_minus * factorial_n1 * Pm_n1
        )

        S_plus += term_plus
        S_minus += term_minus
        terms_used = n + 1

        term_mag = abs(term_plus) + abs(term_minus)

        if term_mag < smallest_term_mag:
            smallest_term_mag = term_mag
            best_S_plus = S_plus
            best_S_minus = S_minus
            best_terms = terms_used

        S_n = abs(S_plus) + abs(S_minus)
        if n >= n_min and term_mag / (S_n + 1e-300) < eps_rel:
            normalized = base_term + S_plus + S_minus
            return normalized, term_mag, terms_used, True

        if n >= n_min and term_mag > prev_term_mag:
            increase_count += 1
            if increase_count >= 2:
                normalized = base_term + best_S_plus + best_S_minus
                return normalized, smallest_term_mag, best_terms, True
        else:
            increase_count = 0

        prev_term_mag = term_mag

        if not math.isfinite(factorial_n) or not math.isfinite(term_mag):
            break

    normalized = base_term + best_S_plus + best_S_minus
    return normalized, smallest_term_mag, best_terms, False


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level API
# ═══════════════════════════════════════════════════════════════════════════════

def sigma_E_series(E, E_prime, xi, tau, Ne=1.0, method="auto",
                   eps_rel=1e-12, n_min=4, n_max=200):
    """
    Evaluate the Compton frequency kernel via Section 4 series.

    Parameters
    ----------
    E        : Incident photon energy [erg]
    E_prime  : Scattered photon energy [erg]
    xi       : cos(scattering angle), strictly in (-1, 1)
    tau      : Dimensionless electron temperature kT/(m_e c^2)
    Ne       : Electron number density [cm^-3] (use 1.0 for microscopic)
    method   : "power", "asymptotic", or "auto"
    eps_rel  : Relative tolerance for convergence (default 1e-12)
    n_min    : Minimum terms before checking convergence (default 4)
    n_max    : Maximum terms (default 200)

    Returns
    -------
    SeriesResult with value, error estimates, diagnostics, and convergence flag.
    """
    if not (E > 0.0 and math.isfinite(E)):
        raise ValueError("E must be finite and > 0")
    if not (E_prime > 0.0 and math.isfinite(E_prime)):
        raise ValueError("E_prime must be finite and > 0")
    if not (tau > 0.0 and math.isfinite(tau)):
        raise ValueError("tau must be finite and > 0")
    if not (-1.0 < xi < 1.0 and math.isfinite(xi)):
        raise ValueError("xi must be finite and strictly inside (-1, 1)")
    if not math.isfinite(Ne):
        raise ValueError("Ne must be finite")
    if 1.0 - xi < 1e-14:
        raise ValueError("xi too close to 1")

    gamma = E / me_c2
    gamma_p = E_prime / me_c2

    p = compute_params(gamma, gamma_p, xi, tau)
    sigma0 = stable_sigma0_E(E, tau, p.lambda_plus, Ne)

    if method == "auto":
        tau_alpha_max = max(tau * p.alpha_plus, tau * p.alpha_minus)
        if tau_alpha_max < 0.05:
            chosen = "asymptotic"
        else:
            chosen = "power"
    else:
        chosen = method

    if chosen == "power":
        norm_ratio, norm_err, terms, converged = _power_series_normalized(
            p, gamma, gamma_p, tau, eps_rel, n_min, n_max)
    elif chosen == "asymptotic":
        norm_ratio, norm_err, terms, converged = _asymptotic_series_normalized(
            p, gamma, gamma_p, tau, eps_rel, n_min, n_max)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    value = sigma0 * norm_ratio
    abs_error = abs(sigma0) * norm_err
    rel_error = abs_error / (abs(value) + 1e-300)

    method_label = chosen if method != "auto" else f"auto({chosen})"

    return SeriesResult(
        value=value,
        estimated_abs_error=abs_error,
        estimated_rel_error=rel_error,
        terms_used=terms,
        method_used=method_label,
        converged=converged,
    )
