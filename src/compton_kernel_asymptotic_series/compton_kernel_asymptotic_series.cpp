#include "compton_kernel_asymptotic_series.hpp"
#include "compton_common/compton_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace compton {

ComptonKernelAsymptoticSeries::ComptonKernelAsymptoticSeries(
    double const eps_rel,
    int const n_min,
    int const n_max)
    : eps_rel_(eps_rel)
    , n_min_(n_min)
    , n_max_(n_max)
{}

SigmaResult ComptonKernelAsymptoticSeries::asymptotic_series(
    double const gamma,
    double const gamma_p,
    double const xi,
    double const tau,
    double const E,
    double const Ne) const
{
    KershawParams<double> const p = compute_params<double>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, p.lambda_plus, Ne);

    double const a = p.a;
    double const a2 = a * a;

    double const zeta_plus = std::clamp(p.rho_plus * p.alpha_plus, -1.0, 1.0);
    double const zeta_minus = std::clamp(p.rho_minus * p.alpha_minus, -1.0, 1.0);

    double const eta_plus = p.alpha_plus * (p.s / a2 + p.rho_plus / a);
    double const eta_minus = p.alpha_minus * (-p.s / a2 + p.rho_minus / a);

    double const base_term = 2.0 * tau * gamma * gamma_p / p.q;

    double const neg_tau_alpha_plus = -tau * p.alpha_plus;
    double const neg_tau_alpha_minus = -tau * p.alpha_minus;

    double S_plus = 0.0;
    double S_minus = 0.0;

    double smallest_term_mag = std::numeric_limits<double>::infinity();
    double best_S_plus = 0.0;
    double best_S_minus = 0.0;
    int increase_count = 0;
    double prev_term_mag = std::numeric_limits<double>::infinity();

    double factorial_n = 1.0;
    double power_plus = neg_tau_alpha_plus;
    double power_minus = neg_tau_alpha_minus;

    double Pp_prev = 1.0;
    double Pp_curr = zeta_plus;
    double Pm_prev = 1.0;
    double Pm_curr = zeta_minus;

    for (int n = 0; n <= n_max_; ++n) {
        if (n > 0) {
            factorial_n *= n;
            power_plus *= neg_tau_alpha_plus;
            power_minus *= neg_tau_alpha_minus;
        }

        double const factorial_n1 = factorial_n * (n + 1);

        double const Pp_n  = Pp_prev;
        double const Pp_n1 = Pp_curr;
        double const Pm_n  = Pm_prev;
        double const Pm_n1 = Pm_curr;

        double const term_plus = power_plus * (
            (-p.G * factorial_n + factorial_n1 / a) * Pp_n
            - eta_plus * factorial_n1 * Pp_n1
        );

        double const term_minus = power_minus * (
            (p.G * factorial_n - factorial_n1 / a) * Pm_n
            + eta_minus * factorial_n1 * Pm_n1
        );

        S_plus += term_plus;
        S_minus += term_minus;

        double const term_mag = std::abs(term_plus) + std::abs(term_minus);

        if (term_mag < smallest_term_mag) {
            smallest_term_mag = term_mag;
            best_S_plus = S_plus;
            best_S_minus = S_minus;
        }

        double const norm_so_far = std::abs(base_term + S_plus + S_minus);
        if (n >= n_min_ && term_mag / (norm_so_far + constants::REL_ERROR_TINY_SCALE) < eps_rel_) {
            double const normalized = base_term + S_plus + S_minus;
            double const value = sigma0 * normalized;
            double const abs_error = std::abs(sigma0) * term_mag;
            double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
            return SigmaResult{value, abs_error, rel_error};
        }

        if (n >= n_min_ && term_mag > prev_term_mag) {
            ++increase_count;
            if (increase_count >= 2) {
                double const normalized = base_term + best_S_plus + best_S_minus;
                double const value = sigma0 * normalized;
                double const abs_error = std::abs(sigma0) * smallest_term_mag;
                double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
                return SigmaResult{value, abs_error, rel_error};
            }
        } else {
            increase_count = 0;
        }

        prev_term_mag = term_mag;

        if (!std::isfinite(factorial_n) || !std::isfinite(term_mag))
            break;

        double const Pp_next = ((2.0*n + 3.0) * zeta_plus * Pp_curr - (n + 1.0) * Pp_prev) / (n + 2.0);
        Pp_prev = Pp_curr;
        Pp_curr = Pp_next;

        double const Pm_next = ((2.0*n + 3.0) * zeta_minus * Pm_curr - (n + 1.0) * Pm_prev) / (n + 2.0);
        Pm_prev = Pm_curr;
        Pm_curr = Pm_next;
    }

    throw std::runtime_error("asymptotic series failed to converge");
}

SigmaResult ComptonKernelAsymptoticSeries::asymptotic_series_derivative(
    double const gamma,
    double const gamma_p,
    double const xi,
    double const tau,
    double const E,
    double const Ne) const
{
    KershawParams<double> const p = compute_params<double>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, p.lambda_plus, Ne);
    double const kappa_val = kappa_ratio(tau);

    double const a = p.a;
    double const a2 = a * a;

    double const zeta_plus = std::clamp(p.rho_plus * p.alpha_plus, -1.0, 1.0);
    double const zeta_minus = std::clamp(p.rho_minus * p.alpha_minus, -1.0, 1.0);

    double const eta_plus = p.alpha_plus * (p.s / a2 + p.rho_plus / a);
    double const eta_minus = p.alpha_minus * (-p.s / a2 + p.rho_minus / a);

    double const lk = p.lambda_plus - kappa_val;
    double const base_deriv = 2.0 * gamma * gamma_p / p.q * (lk / tau - 2.0);

    double const neg_tau_alpha_plus = -tau * p.alpha_plus;
    double const neg_tau_alpha_minus = -tau * p.alpha_minus;

    double dS_plus = 0.0;
    double dS_minus = 0.0;

    double smallest_dterm_mag = std::numeric_limits<double>::infinity();
    double best_dS_plus = 0.0;
    double best_dS_minus = 0.0;
    int increase_count = 0;
    double prev_dterm_mag = std::numeric_limits<double>::infinity();

    double factorial_n = 1.0;
    double power_plus = neg_tau_alpha_plus;
    double power_minus = neg_tau_alpha_minus;

    double Pp_prev = 1.0;
    double Pp_curr = zeta_plus;
    double Pm_prev = 1.0;
    double Pm_curr = zeta_minus;

    for (int n = 0; n <= n_max_; ++n) {
        if (n > 0) {
            factorial_n *= n;
            power_plus *= neg_tau_alpha_plus;
            power_minus *= neg_tau_alpha_minus;
        }

        double const factorial_n1 = factorial_n * (n + 1);

        double const Pp_n  = Pp_prev;
        double const Pp_n1 = Pp_curr;
        double const Pm_n  = Pm_prev;
        double const Pm_n1 = Pm_curr;

        double const Cn_plus = (-p.G * factorial_n + factorial_n1 / a) * Pp_n
                             - eta_plus * factorial_n1 * Pp_n1;

        double const Cn_minus = (p.G * factorial_n - factorial_n1 / a) * Pm_n
                              + eta_minus * factorial_n1 * Pm_n1;

        double const weight = lk / (tau * tau) + (n - 2.0) / tau;

        double const dterm_plus = weight * power_plus * Cn_plus;
        double const dterm_minus = weight * power_minus * Cn_minus;

        dS_plus += dterm_plus;
        dS_minus += dterm_minus;

        double const dterm_mag = std::abs(dterm_plus) + std::abs(dterm_minus);

        if (dterm_mag < smallest_dterm_mag) {
            smallest_dterm_mag = dterm_mag;
            best_dS_plus = dS_plus;
            best_dS_minus = dS_minus;
        }

        double const norm_so_far = std::abs(base_deriv + dS_plus + dS_minus);
        if (n >= n_min_ && dterm_mag / (norm_so_far + constants::REL_ERROR_TINY_SCALE) < eps_rel_) {
            double const normalized = base_deriv + dS_plus + dS_minus;
            double const value = sigma0 * normalized;
            double const abs_error = std::abs(sigma0) * dterm_mag;
            double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
            return SigmaResult{value, abs_error, rel_error};
        }

        if (n >= n_min_ && dterm_mag > prev_dterm_mag) {
            ++increase_count;
            if (increase_count >= 2) {
                double const normalized = base_deriv + best_dS_plus + best_dS_minus;
                double const value = sigma0 * normalized;
                double const abs_error = std::abs(sigma0) * smallest_dterm_mag;
                double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
                return SigmaResult{value, abs_error, rel_error};
            }
        } else {
            increase_count = 0;
        }

        prev_dterm_mag = dterm_mag;

        if (!std::isfinite(factorial_n) || !std::isfinite(dterm_mag))
            break;

        double const Pp_next = ((2.0*n + 3.0) * zeta_plus * Pp_curr - (n + 1.0) * Pp_prev) / (n + 2.0);
        Pp_prev = Pp_curr;
        Pp_curr = Pp_next;

        double const Pm_next = ((2.0*n + 3.0) * zeta_minus * Pm_curr - (n + 1.0) * Pm_prev) / (n + 2.0);
        Pm_prev = Pm_curr;
        Pm_curr = Pm_next;
    }

    throw std::runtime_error("asymptotic series derivative failed to converge");
}

SigmaResult ComptonKernelAsymptoticSeries::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau     = T * units::k_boltz / units::me_c2;
    double const gamma   = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    return asymptotic_series(gamma, gamma_p, xi, tau, E, Ne);
}

SigmaResult ComptonKernelAsymptoticSeries::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau     = T * units::k_boltz / units::me_c2;
    double const dtau_dT = units::k_boltz / units::me_c2;
    double const gamma   = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    SigmaResult const dtau_result =
        asymptotic_series_derivative(gamma, gamma_p, xi, tau, E, Ne);

    return SigmaResult{
        dtau_result.value * dtau_dT,
        dtau_result.estimated_abs_error * dtau_dT,
        dtau_result.estimated_rel_error};
}

} // namespace compton
