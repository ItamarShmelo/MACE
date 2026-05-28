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
    const KershawParams& p, double /*gamma*/, double /*gamma_p*/,
    double tau, double sigma0) const
{
    const double omega = std::sqrt(p.omega2);
    const double b = omega / (2.0 * tau);

    const double theta_plus = std::asinh(p.rho_plus / omega);
    const double theta_minus = std::asinh(p.rho_minus / omega);

    const double x_plus = b * std::exp(theta_plus);
    const double y_plus = b * std::exp(-theta_plus);
    const double x_minus = b * std::exp(theta_minus);
    const double y_minus = b * std::exp(-theta_minus);

    if (y_plus > POISSON_Y_MAX || y_minus > POISSON_Y_MAX ||
        x_plus <= 0.0 || x_minus <= 0.0) {
        double value = sigma0 * p.Psi;
        return SeriesResult{value, 0.0, 0.0, 0, SeriesMethod::PowerSeries, false};
    }

    double w_plus = std::exp(-y_plus);
    double w_minus = std::exp(-y_minus);

    double P_plus = 0.0;
    double P_minus = 0.0;

    constexpr double eps_tiny = 1e-300;
    double last_term_mag = 0.0;
    int terms_used = 0;

    double ehat_plus_curr = ehat_expn(1, x_plus);
    double ehat_minus_curr = ehat_expn(1, x_minus);
    double amp_plus = 1.0;
    double amp_minus = 1.0;

    for (int n = 0; n <= n_max_; ++n) {
        const double coeff_plus = p.A_plus + 2.0 * n / p.a;
        const double coeff_minus = p.A_minus + 2.0 * n / p.a;

        const double t_plus = w_plus * coeff_plus * ehat_plus_curr;
        const double t_minus = w_minus * coeff_minus * ehat_minus_curr;

        P_plus += t_plus;
        P_minus += t_minus;

        const double term_mag = std::abs(t_plus) + std::abs(t_minus);
        last_term_mag = term_mag;
        terms_used = n + 1;

        const double S_n = std::abs(P_plus) + std::abs(P_minus);
        if (n >= n_min_ && term_mag / (S_n + eps_tiny) < eps_rel_)
            break;

        if (n < n_max_) {
            w_plus *= y_plus / (n + 1);
            w_minus *= y_minus / (n + 1);

            amp_plus *= x_plus / (n + 1);
            if (amp_plus < EHAT_AMPLIFICATION_BUDGET) {
                ehat_plus_curr = (1.0 - x_plus * ehat_plus_curr) / (n + 1);
            } else {
                ehat_plus_curr = ehat_expn(n + 2, x_plus);
                amp_plus = 1.0;
            }

            amp_minus *= x_minus / (n + 1);
            if (amp_minus < EHAT_AMPLIFICATION_BUDGET) {
                ehat_minus_curr = (1.0 - x_minus * ehat_minus_curr) / (n + 1);
            } else {
                ehat_minus_curr = ehat_expn(n + 2, x_minus);
                amp_minus = 1.0;
            }
        }
    }

    const bool converged = terms_used <= n_max_;
    const double normalized_ratio = p.Psi + P_plus - P_minus;
    const double value = sigma0 * normalized_ratio;

    const double sum_abs = std::abs(P_plus) + std::abs(P_minus) + std::abs(p.Psi);
    const double norm_abs = std::abs(normalized_ratio) + eps_tiny;
    const double conditioning = sum_abs / norm_abs;
    const double cond_error = COND_ERROR_COEFF * conditioning;
    const double trunc_error = last_term_mag / norm_abs;
    const double rel_error = std::max(trunc_error, cond_error);
    const double abs_error = std::abs(sigma0) * rel_error * norm_abs;

    return SeriesResult{value, abs_error, rel_error, terms_used,
                        SeriesMethod::PowerSeries, converged};
}

// ─────────────────────────────────────────────────────────────────────────
// Asymptotic series
// ─────────────────────────────────────────────────────────────────────────

SeriesResult ComptonKernelSeries::asymptotic_series(
    const KershawParams& p, double gamma, double gamma_p,
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

    KershawParams p = compute_params(gamma, gamma_p, xi, tau);
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
        return power_series(p, gamma, gamma_p, tau, sigma0_val);
    else
        return asymptotic_series(p, gamma, gamma_p, tau, sigma0_val);
}

} // namespace compton
