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

from _units import me_c2, k_boltz
from .compton_kernel_quadrature import (
    compute_params, stable_sigma0_E, KershawParams,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Double-double (dd) arithmetic helpers
# ═══════════════════════════════════════════════════════════════════════════════
#
# dd values are represented as (hi, lo) tuples of floats.

_DD_ZERO = (0.0, 0.0)
_DD_ONE  = (1.0, 0.0)

_has_fma = hasattr(math, 'fma')

def _two_sum(a, b):
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return (s, e)

def _two_prod(a, b):
    p = a * b
    if _has_fma:
        e = math.fma(a, b, -p)
    else:
        # Dekker splitting
        _SPLIT = 134217729.0  # 2^27 + 1
        ca = _SPLIT * a; ah = ca - (ca - a); al = a - ah
        cb = _SPLIT * b; bh = cb - (cb - b); bl = b - bh
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return (p, e)

def _dd_from_double(x):
    return (float(x), 0.0)

def _dd_to_double(a):
    return a[0] + a[1]

def _dd_add_scalar(a, b):
    s = _two_sum(a[0], b)
    lo = s[1] + a[1]
    return _two_sum(s[0], lo)

def _dd_add(a, b):
    s = _two_sum(a[0], b[0])
    t = _two_sum(a[1], b[1])
    lo = s[1] + t[0]
    s = _two_sum(s[0], lo)
    lo2 = s[1] + t[1]
    return _two_sum(s[0], lo2)

def _dd_sub(a, b):
    return _dd_add(a, (-b[0], -b[1]))

def _dd_mul(a, b):
    p = _two_prod(a[0], b[0])
    lo = p[1] + a[0] * b[1] + a[1] * b[0]
    return _two_sum(p[0], lo)

def _dd_mul_scalar(a, b):
    p = _two_prod(a[0], b)
    lo = p[1] + a[1] * b
    return _two_sum(p[0], lo)

def _dd_div(a, b):
    q1 = a[0] / b[0]
    r = _dd_sub(a, _dd_mul_scalar(b, q1))
    q2 = r[0] / b[0]
    r = _dd_sub(r, _dd_mul_scalar(b, q2))
    q3 = r[0] / b[0]
    q = _two_sum(q1, q2)
    return _dd_add_scalar(q, q3)

def _dd_div_scalar(a, b):
    return _dd_div(a, _dd_from_double(b))

def _dd_abs(a):
    return (-a[0], -a[1]) if a[0] < 0.0 else a

# ln(2) in dd
_DD_LN2 = (6.931471805599452862e-01, 2.319046813023978449e-17)

def _dd_exp(a):
    if a[0] > 700.0:
        return (math.inf, 0.0)
    if a[0] < -700.0:
        return _DD_ZERO

    k = round(a[0] / _DD_LN2[0])
    r = _dd_sub(a, _dd_mul_scalar(_DD_LN2, k))

    s = _DD_ONE
    term = _DD_ONE
    for i in range(1, 13):
        term = _dd_div_scalar(_dd_mul(term, r), float(i))
        s = _dd_add(s, term)
        if abs(term[0]) < 1e-32 * abs(s[0]):
            break

    ki = int(k)
    return (math.ldexp(s[0], ki), math.ldexp(s[1], ki))

def _dd_log(a):
    y0 = math.log(a[0])
    ey = _dd_exp((-y0, 0.0))
    aey = _dd_mul(a, ey)
    corr = _dd_sub(aey, _DD_ONE)
    return _dd_add(_dd_from_double(y0), corr)

def _dd_sqrt(a):
    if a[0] <= 0.0 and a[1] == 0.0:
        return _DD_ZERO
    y0 = math.sqrt(a[0])
    q = _dd_div(a, _dd_from_double(y0))
    return _dd_mul_scalar(_dd_add(_dd_from_double(y0), q), 0.5)

def _dd_asinh(x):
    x2 = _dd_mul(x, x)
    arg = _dd_sqrt(_dd_add_scalar(x2, 1.0))
    return _dd_log(_dd_add(x, arg))

def _dd_ehat_cf(m, x):
    """Ehat_m(x) via modified Lentz CF in dd."""
    TINY = 1e-300
    CF_TOL = 1e-31
    MAX_ITER = 200

    b = _dd_add_scalar(x, float(m))
    if abs(b[0]) < TINY:
        b = (TINY, b[1])

    f = b
    C = b
    D = _DD_ZERO

    for j in range(1, MAX_ITER + 1):
        aj = -float(m + j - 1) * float(j)
        bj = _dd_add_scalar(x, float(m + 2 * j))

        D = _dd_add(bj, _dd_mul_scalar(D, aj))
        if abs(D[0]) < TINY:
            D = (TINY, D[1])
        D = _dd_div(_DD_ONE, D)

        C = _dd_add(bj, _dd_div(_dd_from_double(aj), C))
        if abs(C[0]) < TINY:
            C = (TINY, C[1])

        delta = _dd_mul(C, D)
        f = _dd_mul(f, delta)

        if abs(delta[0] - 1.0) + abs(delta[1]) < CF_TOL:
            break

    return _dd_div(_DD_ONE, f)


# ═══════════════════════════════════════════════════════════════════════════════
# DD-precision compute_params
# ═══════════════════════════════════════════════════════════════════════════════

class _KershawParamsDD:
    __slots__ = ('a', 's', 'q', 'omega2', 'Delta', 'lambda_plus',
                 'rho_plus', 'rho_minus', 'alpha_plus', 'alpha_minus',
                 'G', 'A_plus', 'A_minus', 'Psi')

def _compute_params_dd(gamma, gamma_p, xi, tau):
    p = _KershawParamsDD()
    gamma_dd   = _dd_from_double(gamma)
    gamma_p_dd = _dd_from_double(gamma_p)
    xi_dd      = _dd_from_double(xi)
    tau_dd     = _dd_from_double(tau)
    one = _DD_ONE
    two = (2.0, 0.0)

    p.a = _dd_sub(one, xi_dd)
    p.s = _dd_add(_dd_div(one, gamma_dd), _dd_div(one, gamma_p_dd))

    dg  = _dd_sub(gamma_p_dd, gamma_dd)
    dg2 = _dd_mul(dg, dg)
    gg  = _dd_mul(gamma_dd, gamma_p_dd)
    q2  = _dd_add(dg2, _dd_mul(_dd_mul(two, gg), p.a))
    p.q = _dd_sqrt(q2)

    p.omega2 = _dd_div(_dd_add(one, xi_dd), p.a)

    gg_a    = _dd_mul(gg, p.a)
    factor1 = _dd_add(one, _dd_div(gg_a, two))
    factor2 = _dd_add(one, _dd_div(dg2, _dd_mul(two, gg_a)))
    p.Delta = _dd_sqrt(_dd_mul(factor1, factor2))

    p.lambda_plus = _dd_add(_dd_div(dg, two), p.Delta)

    if p.lambda_plus[0] < 1.0 - 1e-12:
        raise RuntimeError("lambda_plus significantly below 1")
    if p.lambda_plus[0] < 1.0:
        p.lambda_plus = one

    p.rho_plus  = _dd_add(p.lambda_plus, gamma_dd)
    p.rho_minus = _dd_sub(p.lambda_plus, gamma_p_dd)

    Rp0 = _dd_add(_dd_mul(p.rho_plus, p.rho_plus), p.omega2)
    Rm0 = _dd_add(_dd_mul(p.rho_minus, p.rho_minus), p.omega2)
    p.alpha_plus  = _dd_div(one, _dd_sqrt(Rp0))
    p.alpha_minus = _dd_div(one, _dd_sqrt(Rm0))

    a2 = _dd_mul(p.a, p.a)
    p.G = _dd_add(_dd_sub(_DD_ZERO, gg),
                  _dd_add(_dd_div(two, p.a),
                          _dd_div(two, _dd_mul(gg, a2))))

    s_over_tau_a2 = _dd_div(p.s, _dd_mul(tau_dd, a2))
    p.A_plus  = _dd_sub(p.G, s_over_tau_a2)
    p.A_minus = _dd_add(p.G, s_over_tau_a2)

    term1 = _dd_div(_dd_mul(_dd_mul(two, tau_dd), gg), p.q)
    term2 = _dd_mul(_dd_div(p.s, a2),
                    _dd_add(p.alpha_plus, p.alpha_minus))
    term3 = _dd_div(_dd_sub(_dd_mul(p.rho_plus, p.alpha_plus),
                            _dd_mul(p.rho_minus, p.alpha_minus)),
                    p.a)
    p.Psi = _dd_add(_dd_add(term1, term2), term3)

    return p


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
class SigmaResult:
    value: float
    estimated_abs_error: float
    estimated_rel_error: float


# ═══════════════════════════════════════════════════════════════════════════════
# Power series
# ═══════════════════════════════════════════════════════════════════════════════

_POISSON_Y_MAX = 500.0
_ACCUMULATION_SAFETY_FACTOR = 10.0
COND_ERROR_COEFF = _ACCUMULATION_SAFETY_FACTOR * np.finfo(np.float64).eps
DD_COND_ERROR_COEFF = _ACCUMULATION_SAFETY_FACTOR * np.finfo(np.float64).eps ** 2

EHAT_AMPLIFICATION_BUDGET = 1e2

def _power_series_normalized(p: KershawParams, gamma: float, gamma_p: float,
                              tau: float, eps_rel: float = 1e-12,
                              n_min: int = 4, n_max: int = 200,
                              xi: float = None):
    """
    Compute the normalized ratio Sigma_E / Sigma_0 via power series
    using full double-double arithmetic.

    Returns (normalized_ratio, estimated_normalized_error, terms_used, converged).
    """
    if xi is None:
        xi = 1.0 - p.a

    pd = _compute_params_dd(gamma, gamma_p, xi, tau)

    omega_dd = _dd_sqrt(pd.omega2)
    tau_dd   = _dd_from_double(tau)
    b_dd     = _dd_div(omega_dd, _dd_mul_scalar(tau_dd, 2.0))

    theta_plus_dd  = _dd_asinh(_dd_div(pd.rho_plus, omega_dd))
    theta_minus_dd = _dd_asinh(_dd_div(pd.rho_minus, omega_dd))

    neg_theta_plus  = (-theta_plus_dd[0], -theta_plus_dd[1])
    neg_theta_minus = (-theta_minus_dd[0], -theta_minus_dd[1])

    x_plus_dd  = _dd_mul(b_dd, _dd_exp(theta_plus_dd))
    y_plus_dd  = _dd_mul(b_dd, _dd_exp(neg_theta_plus))
    x_minus_dd = _dd_mul(b_dd, _dd_exp(theta_minus_dd))
    y_minus_dd = _dd_mul(b_dd, _dd_exp(neg_theta_minus))

    if y_plus_dd[0] > _POISSON_Y_MAX or y_minus_dd[0] > _POISSON_Y_MAX:
        return p.Psi, 0.0, 0, False

    if x_plus_dd[0] <= 0.0 or x_minus_dd[0] <= 0.0:
        return p.Psi, 0.0, 0, False

    w_plus_dd  = _dd_exp((-y_plus_dd[0], -y_plus_dd[1]))
    w_minus_dd = _dd_exp((-y_minus_dd[0], -y_minus_dd[1]))

    P_plus_dd  = _DD_ZERO
    P_minus_dd = _DD_ZERO

    eps_tiny = 1e-300
    last_diff_change = 0.0
    terms_used = 0

    ehat_plus_dd  = _dd_ehat_cf(1, x_plus_dd)
    ehat_minus_dd = _dd_ehat_cf(1, x_minus_dd)
    amp_plus_dd  = _DD_ONE
    amp_minus_dd = _DD_ONE

    for n in range(n_max + 1):
        coeff_plus_dd  = _dd_add(pd.A_plus,
                                 _dd_div(_dd_from_double(2.0 * n), pd.a))
        coeff_minus_dd = _dd_add(pd.A_minus,
                                 _dd_div(_dd_from_double(2.0 * n), pd.a))

        t_plus_dd  = _dd_mul(_dd_mul(w_plus_dd, coeff_plus_dd), ehat_plus_dd)
        t_minus_dd = _dd_mul(_dd_mul(w_minus_dd, coeff_minus_dd), ehat_minus_dd)

        prev_diff = _dd_to_double(_dd_sub(P_plus_dd, P_minus_dd))
        P_plus_dd  = _dd_add(P_plus_dd, t_plus_dd)
        P_minus_dd = _dd_add(P_minus_dd, t_minus_dd)
        curr_diff = _dd_to_double(_dd_sub(P_plus_dd, P_minus_dd))
        last_diff_change = abs(curr_diff - prev_diff)

        term_mag = abs(t_plus_dd[0]) + abs(t_minus_dd[0])
        terms_used = n + 1

        S_n = abs(P_plus_dd[0]) + abs(P_minus_dd[0])
        if n >= n_min and term_mag / (S_n + eps_tiny) < eps_rel:
            partial = abs(_dd_to_double(_dd_add(pd.Psi,
                          _dd_sub(P_plus_dd, P_minus_dd))))
            if last_diff_change / (partial + eps_tiny) < eps_rel:
                break

        if n < n_max:
            w_plus_dd  = _dd_div(_dd_mul(w_plus_dd, y_plus_dd),
                                 _dd_from_double(n + 1.0))
            w_minus_dd = _dd_div(_dd_mul(w_minus_dd, y_minus_dd),
                                 _dd_from_double(n + 1.0))

            amp_plus_dd = _dd_mul(amp_plus_dd,
                                  _dd_div(x_plus_dd, _dd_from_double(n + 1.0)))
            if amp_plus_dd[0] < EHAT_AMPLIFICATION_BUDGET:
                ehat_plus_dd = _dd_div_scalar(
                    _dd_sub(_DD_ONE, _dd_mul(x_plus_dd, ehat_plus_dd)),
                    n + 1)
            else:
                ehat_plus_dd = _dd_ehat_cf(n + 2, x_plus_dd)
                amp_plus_dd = _DD_ONE

            amp_minus_dd = _dd_mul(amp_minus_dd,
                                   _dd_div(x_minus_dd, _dd_from_double(n + 1.0)))
            if amp_minus_dd[0] < EHAT_AMPLIFICATION_BUDGET:
                ehat_minus_dd = _dd_div_scalar(
                    _dd_sub(_DD_ONE, _dd_mul(x_minus_dd, ehat_minus_dd)),
                    n + 1)
            else:
                ehat_minus_dd = _dd_ehat_cf(n + 2, x_minus_dd)
                amp_minus_dd = _DD_ONE

    converged = terms_used <= n_max
    diff = _dd_sub(P_plus_dd, P_minus_dd)
    normalized_dd = _dd_add(pd.Psi, diff)
    normalized_ratio = _dd_to_double(normalized_dd)

    psi_abs = abs(pd.Psi[0]) + abs(pd.Psi[1])
    sum_abs = abs(P_plus_dd[0]) + abs(P_minus_dd[0]) + psi_abs
    norm_abs = abs(normalized_ratio) + eps_tiny
    conditioning = sum_abs / norm_abs
    cond_error = DD_COND_ERROR_COEFF * conditioning
    trunc_error = last_diff_change / norm_abs
    rel_error = max(trunc_error, cond_error)
    norm_err = rel_error * norm_abs

    return normalized_ratio, norm_err, terms_used, converged


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

    if zeta_plus > 1.0:
        zeta_plus = 1.0
    elif zeta_plus < -1.0:
        zeta_plus = -1.0
    if zeta_minus > 1.0:
        zeta_minus = 1.0
    elif zeta_minus < -1.0:
        zeta_minus = -1.0

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

    Pp_prev, Pp_curr = 1.0, zeta_plus
    Pm_prev, Pm_curr = 1.0, zeta_minus

    terms_used = 0

    for n in range(n_max + 1):
        if n > 0:
            factorial_n *= n
            power_plus *= neg_tau_alpha_plus
            power_minus *= neg_tau_alpha_minus

        factorial_n1 = factorial_n * (n + 1)

        Pp_n = Pp_prev
        Pp_n1 = Pp_curr
        Pm_n = Pm_prev
        Pm_n1 = Pm_curr

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

        norm_so_far = abs(base_term + S_plus + S_minus)
        if n >= n_min and term_mag / (norm_so_far + 1e-300) < eps_rel:
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

        Pp_next = ((2.0*n + 3.0) * zeta_plus * Pp_curr - (n + 1.0) * Pp_prev) / (n + 2.0)
        Pp_prev = Pp_curr
        Pp_curr = Pp_next

        Pm_next = ((2.0*n + 3.0) * zeta_minus * Pm_curr - (n + 1.0) * Pm_prev) / (n + 2.0)
        Pm_prev = Pm_curr
        Pm_curr = Pm_next

    normalized = base_term + best_S_plus + best_S_minus
    return normalized, smallest_term_mag, best_terms, False


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level API
# ═══════════════════════════════════════════════════════════════════════════════

def sigma_E_series(E, E_prime, xi, T, Ne=1.0, method="auto",
                   eps_rel=1e-12, n_min=4, n_max=200):
    """
    Evaluate the Compton frequency kernel via Section 4 series.

    Parameters
    ----------
    E        : Incident photon energy [erg]
    E_prime  : Scattered photon energy [erg]
    xi       : cos(scattering angle), strictly in (-1, 1)
    T        : Electron temperature [K]
    Ne       : Electron number density [cm^-3] (use 1.0 for microscopic)
    method   : "power", "asymptotic", or "auto"
    eps_rel  : Relative tolerance for convergence (default 1e-12)
    n_min    : Minimum terms before checking convergence (default 4)
    n_max    : Maximum terms (default 200)

    Returns
    -------
    SigmaResult with value and error estimates.

    Raises
    ------
    RuntimeError if the series fails to converge.
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
        raise ValueError("xi too close to 1")

    tau = T * k_boltz / me_c2
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
            p, gamma, gamma_p, tau, eps_rel, n_min, n_max, xi=xi)
    elif chosen == "asymptotic":
        norm_ratio, norm_err, terms, converged = _asymptotic_series_normalized(
            p, gamma, gamma_p, tau, eps_rel, n_min, n_max)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if not converged:
        raise RuntimeError(
            f"{chosen} series failed to converge after {terms} terms")

    value = sigma0 * norm_ratio
    abs_error = abs(sigma0) * norm_err
    rel_error = abs_error / (abs(value) + 1e-300)

    return SigmaResult(
        value=value,
        estimated_abs_error=abs_error,
        estimated_rel_error=rel_error,
    )
