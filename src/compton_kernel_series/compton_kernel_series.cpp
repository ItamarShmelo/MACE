/**
 * @file compton_kernel_series.cpp
 * @brief Power series and asymptotic series for the Compton kernel.
 *
 * Direct port of the validated Python implementation in
 * pycompton/compton_kernel_series.py.
 */

#include "compton_kernel_series.hpp"

#include <boost/math/special_functions/expint.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace compton {

// ═══════════════════════════════════════════════════════════════════════════
// ehat_expn: Ehat_m(x) = exp(x) * E_m(x)
// ═══════════════════════════════════════════════════════════════════════════

static double ehat_asymptotic(int m, double x, int n_terms = 15) {
    const double inv_x = 1.0 / x;
    double result = 1.0;
    double term = 1.0;
    for (int k = 1; k < n_terms; ++k) {
        term *= -(static_cast<double>(m + k - 1)) * inv_x;
        result += term;
        if (std::abs(term) < 1e-15 * std::abs(result))
            break;
    }
    return inv_x * result;
}

double ehat_expn(int m, double x) {
    if (!(x > 0.0) || !std::isfinite(x))
        throw std::invalid_argument("ehat_expn requires finite x > 0");
    if (m < 1)
        throw std::invalid_argument("ehat_expn requires m >= 1");

    if (x < 50.0) {
        return std::exp(x) * boost::math::expint(m, x);
    }
    return ehat_asymptotic(m, x);
}

// ═══════════════════════════════════════════════════════════════════════════
// ComptonKernelSeries
// ═══════════════════════════════════════════════════════════════════════════

ComptonKernelSeries::ComptonKernelSeries(
    SeriesMethod method, double eps_rel, int n_min, int n_max)
    : method_(method), eps_rel_(eps_rel), n_min_(n_min), n_max_(n_max)
{}

// ─────────────────────────────────────────────────────────────────────────
// Power series
// ─────────────────────────────────────────────────────────────────────────

static constexpr double POISSON_Y_MAX = 500.0;

// rel_tol / eps_machine / safety_factor = 1e-13 / 1e-16 / 10
static constexpr double EHAT_AMPLIFICATION_BUDGET = 1e2;

SeriesResult ComptonKernelSeries::power_series(
    const KershawParams<double>& p, double gamma, double gamma_p, double xi,
    double tau, double sigma0) const
{
    KershawParams<DD> pd = compute_params<DD>(gamma, gamma_p, xi, tau);

    const DD omega_dd = pd.omega2.sqrt();
    const DD tau_dd(tau);
    const DD b_dd = omega_dd / (tau_dd * 2.0);

    const DD theta_plus_dd  = dd_asinh(pd.rho_plus / omega_dd);
    const DD theta_minus_dd = dd_asinh(pd.rho_minus / omega_dd);

    const DD x_plus_dd  = b_dd * theta_plus_dd.exp();
    const DD y_plus_dd  = b_dd * (-theta_plus_dd).exp();
    const DD x_minus_dd = b_dd * theta_minus_dd.exp();
    const DD y_minus_dd = b_dd * (-theta_minus_dd).exp();

    if (y_plus_dd.upper > POISSON_Y_MAX || y_minus_dd.upper > POISSON_Y_MAX ||
        x_plus_dd.upper <= 0.0 || x_minus_dd.upper <= 0.0) {
        double value = sigma0 * p.Psi;
        return SeriesResult{value, 0.0, 0.0, 0, SeriesMethod::PowerSeries, false};
    }

    DD w_plus_dd  = (-y_plus_dd).exp();
    DD w_minus_dd = (-y_minus_dd).exp();

    DD P_plus_dd(0.0);
    DD P_minus_dd(0.0);

    constexpr double eps_tiny = 1e-300;
    double last_diff_change = 0.0;
    double last_term_mag = 0.0;
    int terms_used = 0;

    DD ehat_plus_dd  = dd_ehat_cf(1, x_plus_dd);
    DD ehat_minus_dd = dd_ehat_cf(1, x_minus_dd);
    DD amp_plus_dd(1.0);
    DD amp_minus_dd(1.0);

    for (int n = 0; n <= n_max_; ++n) {
        const DD coeff_plus_dd  = pd.A_plus + DD(2.0 * n) / pd.a;
        const DD coeff_minus_dd = pd.A_minus + DD(2.0 * n) / pd.a;

        const DD t_plus_dd  = w_plus_dd * coeff_plus_dd * ehat_plus_dd;
        const DD t_minus_dd = w_minus_dd * coeff_minus_dd * ehat_minus_dd;

        const double prev_diff = dd_to_double(P_plus_dd - P_minus_dd);
        P_plus_dd  = P_plus_dd + t_plus_dd;
        P_minus_dd = P_minus_dd + t_minus_dd;
        const double curr_diff = dd_to_double(P_plus_dd - P_minus_dd);
        last_diff_change = std::abs(curr_diff - prev_diff);

        const double term_mag = std::abs(t_plus_dd.upper) + std::abs(t_minus_dd.upper);
        last_term_mag = term_mag;
        terms_used = n + 1;

        const double S_n = std::abs(P_plus_dd.upper) + std::abs(P_minus_dd.upper);
        if (n >= n_min_ && term_mag / (S_n + eps_tiny) < eps_rel_) {
            double partial = std::abs(dd_to_double(
                pd.Psi + (P_plus_dd - P_minus_dd)));
            if (last_diff_change / (partial + eps_tiny) < eps_rel_)
                break;
        }

        if (n < n_max_) {
            w_plus_dd  = w_plus_dd * y_plus_dd / (n + 1.0);
            w_minus_dd = w_minus_dd * y_minus_dd / (n + 1.0);

            amp_plus_dd = amp_plus_dd * (x_plus_dd / (n + 1.0));
            if (amp_plus_dd.upper < EHAT_AMPLIFICATION_BUDGET) {
                ehat_plus_dd = (DD(1.0) - x_plus_dd * ehat_plus_dd) / (n + 1.0);
            } else {
                ehat_plus_dd = dd_ehat_cf(n + 2, x_plus_dd);
                amp_plus_dd = DD(1.0);
            }

            amp_minus_dd = amp_minus_dd * (x_minus_dd / (n + 1.0));
            if (amp_minus_dd.upper < EHAT_AMPLIFICATION_BUDGET) {
                ehat_minus_dd = (DD(1.0) - x_minus_dd * ehat_minus_dd) / (n + 1.0);
            } else {
                ehat_minus_dd = dd_ehat_cf(n + 2, x_minus_dd);
                amp_minus_dd = DD(1.0);
            }
        }
    }

    const bool converged = terms_used <= n_max_;
    DD diff = P_plus_dd - P_minus_dd;
    DD normalized_dd = pd.Psi + diff;
    const double normalized_ratio = dd_to_double(normalized_dd);
    const double value = sigma0 * normalized_ratio;

    constexpr double DD_EPS = std::numeric_limits<double>::epsilon()
                            * std::numeric_limits<double>::epsilon();
    const double norm_abs = std::abs(normalized_ratio) + eps_tiny;
    const double max_accum = std::max(std::abs(P_plus_dd.upper), std::abs(P_minus_dd.upper));
    const double trunc_rel = last_term_mag / norm_abs;
    const double round_rel = terms_used * DD_EPS * max_accum / norm_abs;
    const double rel_error = std::max(trunc_rel, round_rel);
    const double abs_error = rel_error * std::abs(value);

    return SeriesResult{value, abs_error, rel_error, terms_used,
                        SeriesMethod::PowerSeries, converged};
}

// ─────────────────────────────────────────────────────────────────────────
// Asymptotic series
// ─────────────────────────────────────────────────────────────────────────

SeriesResult ComptonKernelSeries::asymptotic_series(
    const KershawParams<double>& p, double gamma, double gamma_p,
    double tau, double sigma0) const
{
    const double a = p.a;
    const double a2 = a * a;

    double zeta_plus = p.rho_plus * p.alpha_plus;
    double zeta_minus = p.rho_minus * p.alpha_minus;

    if (zeta_plus > 1.0) zeta_plus = 1.0;
    else if (zeta_plus < -1.0) zeta_plus = -1.0;
    if (zeta_minus > 1.0) zeta_minus = 1.0;
    else if (zeta_minus < -1.0) zeta_minus = -1.0;

    const double eta_plus = p.alpha_plus * (p.s / a2 + p.rho_plus / a);
    const double eta_minus = p.alpha_minus * (-p.s / a2 + p.rho_minus / a);

    const double base_term = 2.0 * tau * gamma * gamma_p / p.q;

    const double neg_tau_alpha_plus = -tau * p.alpha_plus;
    const double neg_tau_alpha_minus = -tau * p.alpha_minus;

    double S_plus = 0.0;
    double S_minus = 0.0;

    double smallest_term_mag = std::numeric_limits<double>::infinity();
    double best_S_plus = 0.0;
    double best_S_minus = 0.0;
    int best_terms = 0;
    int increase_count = 0;
    double prev_term_mag = std::numeric_limits<double>::infinity();

    double factorial_n = 1.0;
    double power_plus = neg_tau_alpha_plus;
    double power_minus = neg_tau_alpha_minus;

    double Pp_prev = 1.0, Pp_curr = zeta_plus;
    double Pm_prev = 1.0, Pm_curr = zeta_minus;

    int terms_used = 0;

    for (int n = 0; n <= n_max_; ++n) {
        if (n > 0) {
            factorial_n *= n;
            power_plus *= neg_tau_alpha_plus;
            power_minus *= neg_tau_alpha_minus;
        }

        const double factorial_n1 = factorial_n * (n + 1);

        const double Pp_n  = Pp_prev;
        const double Pp_n1 = Pp_curr;
        const double Pm_n  = Pm_prev;
        const double Pm_n1 = Pm_curr;

        const double term_plus = power_plus * (
            (-p.G * factorial_n + factorial_n1 / a) * Pp_n
            - eta_plus * factorial_n1 * Pp_n1
        );

        const double term_minus = power_minus * (
            (p.G * factorial_n - factorial_n1 / a) * Pm_n
            + eta_minus * factorial_n1 * Pm_n1
        );

        S_plus += term_plus;
        S_minus += term_minus;
        terms_used = n + 1;

        const double term_mag = std::abs(term_plus) + std::abs(term_minus);

        if (term_mag < smallest_term_mag) {
            smallest_term_mag = term_mag;
            best_S_plus = S_plus;
            best_S_minus = S_minus;
            best_terms = terms_used;
        }

        const double norm_so_far = std::abs(base_term + S_plus + S_minus);
        if (n >= n_min_ && term_mag / (norm_so_far + 1e-300) < eps_rel_) {
            const double normalized = base_term + S_plus + S_minus;
            const double value = sigma0 * normalized;
            const double abs_error = std::abs(sigma0) * term_mag;
            const double rel_error = abs_error / (std::abs(value) + 1e-300);
            return SeriesResult{value, abs_error, rel_error, terms_used,
                                SeriesMethod::Asymptotic, true};
        }

        if (n >= n_min_ && term_mag > prev_term_mag) {
            ++increase_count;
            if (increase_count >= 2) {
                const double normalized = base_term + best_S_plus + best_S_minus;
                const double value = sigma0 * normalized;
                const double abs_error = std::abs(sigma0) * smallest_term_mag;
                const double rel_error = abs_error / (std::abs(value) + 1e-300);
                return SeriesResult{value, abs_error, rel_error, best_terms,
                                    SeriesMethod::Asymptotic, true};
            }
        } else {
            increase_count = 0;
        }

        prev_term_mag = term_mag;

        if (!std::isfinite(factorial_n) || !std::isfinite(term_mag))
            break;

        const double Pp_next = ((2.0*n + 3.0) * zeta_plus * Pp_curr - (n + 1.0) * Pp_prev) / (n + 2.0);
        Pp_prev = Pp_curr;
        Pp_curr = Pp_next;

        const double Pm_next = ((2.0*n + 3.0) * zeta_minus * Pm_curr - (n + 1.0) * Pm_prev) / (n + 2.0);
        Pm_prev = Pm_curr;
        Pm_curr = Pm_next;
    }

    const double normalized = base_term + best_S_plus + best_S_minus;
    const double value = sigma0 * normalized;
    const double abs_error = std::abs(sigma0) * smallest_term_mag;
    const double rel_error = abs_error / (std::abs(value) + 1e-300);
    return SeriesResult{value, abs_error, rel_error, best_terms,
                        SeriesMethod::Asymptotic, false};
}

// ─────────────────────────────────────────────────────────────────────────
// Top-level sigma_E
// ─────────────────────────────────────────────────────────────────────────

SeriesResult ComptonKernelSeries::sigma_E(
    double E, double E_prime, double xi, double tau, double Ne) const
{
    if (!(E > 0.0) || !std::isfinite(E))
        throw std::invalid_argument("E must be finite and > 0");
    if (!(E_prime > 0.0) || !std::isfinite(E_prime))
        throw std::invalid_argument("E_prime must be finite and > 0");
    if (!(tau > 0.0) || !std::isfinite(tau))
        throw std::invalid_argument("tau must be finite and > 0");
    if (!(xi > -1.0 && xi < 1.0) || !std::isfinite(xi))
        throw std::invalid_argument("xi must be finite and strictly inside (-1, 1)");
    if (!std::isfinite(Ne))
        throw std::invalid_argument("Ne must be finite");
    if (1.0 - xi < 1e-14)
        throw std::invalid_argument("xi too close to 1");

    const double gamma = E / units::me_c2;
    const double gamma_p = E_prime / units::me_c2;

    KershawParams<double> p = compute_params<double>(gamma, gamma_p, xi, tau);
    const double sigma0_val = stable_sigma0_E(E, tau, p.lambda_plus, Ne);

    SeriesMethod chosen;
    if (method_ == SeriesMethod::Auto) {
        const double tau_alpha_max = std::max(tau * p.alpha_plus,
                                              tau * p.alpha_minus);
        chosen = (tau_alpha_max < 0.05) ? SeriesMethod::Asymptotic
                                        : SeriesMethod::PowerSeries;
    } else {
        chosen = method_;
    }

    if (chosen == SeriesMethod::PowerSeries)
        return power_series(p, gamma, gamma_p, xi, tau, sigma0_val);
    else
        return asymptotic_series(p, gamma, gamma_p, tau, sigma0_val);
}

} // namespace compton
