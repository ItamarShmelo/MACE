/**
 * @file compton_kernel_series.cpp
 * @brief Power series and asymptotic series for the Compton kernel.
 *
 * Direct port of the validated Python implementation in
 * pycompton/compton_kernel_series.py.
 */

#include "compton_kernel_series.hpp"
#include "compton_common/compton_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace compton {

ComptonKernelSeries::ComptonKernelSeries(
    SeriesMethod method, 
    double eps_rel, 
    int n_min, 
    int n_max)
    :   method_(method), 
        eps_rel_(eps_rel), 
        n_min_(n_min), 
        n_max_(n_max)
{}

template<typename T>
SigmaResult ComptonKernelSeries::power_series(
    double const gamma,
    double const gamma_p,
    double const xi,
    double const tau,
    double const E,
    double const Ne) const
{
    using namespace details;
    using namespace constants;

    KershawParams<T> p = compute_params<T>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, to_double(p.lambda_plus), Ne);

    T const omega = param_sqrt(p.omega2);
    T const tau_t(tau);
    T const b = omega / (tau_t * 2.0);

    T const theta_plus  = param_asinh(p.rho_plus / omega);
    T const theta_minus = param_asinh(p.rho_minus / omega);

    T const x_plus  = b * param_exp(theta_plus);
    T const y_plus  = b * param_exp(-theta_plus);
    T const x_minus = b * param_exp(theta_minus);
    T const y_minus = b * param_exp(-theta_minus);

    // Poisson weight exp(-y) underflows when y exceeds this threshold.
    if (y_plus > POISSON_Y_MAX || y_minus > POISSON_Y_MAX) {
        throw std::runtime_error("power series: Poisson weight underflow");
    }

    T w_plus  = param_exp(-y_plus);
    T w_minus = param_exp(-y_minus);

    T P_plus(0.0);
    T P_minus(0.0);

    constexpr double eps_tiny = 1e-300;
    T last_diff_change(0.0);
    T last_term_mag(0.0);
    int terms_used = 0;

    T ehat_plus  = ehat(1, x_plus);
    T ehat_minus = ehat(1, x_minus);
    T amp_plus(1.0);
    T amp_minus(1.0);

    for (int n = 0; n <= n_max_; ++n) {
        T const coeff_plus  = p.A_plus + T(2.0 * n) / p.a;
        T const coeff_minus = p.A_minus + T(2.0 * n) / p.a;

        T const t_plus  = w_plus * coeff_plus * ehat_plus;
        T const t_minus = w_minus * coeff_minus * ehat_minus;

        T const prev_diff = P_plus - P_minus;
        P_plus  = P_plus + t_plus;
        P_minus = P_minus + t_minus;
        T const curr_diff = P_plus - P_minus;
        last_diff_change = param_abs(curr_diff - prev_diff);

        T const term_mag = param_abs(t_plus) + param_abs(t_minus);
        last_term_mag = term_mag;
        terms_used = n + 1;

        // Converge on the actual result magnitude, not the inflated accumulators.
        if (n >= n_min_ &&
            last_diff_change / (param_abs(p.Psi + (P_plus - P_minus)) + eps_tiny) < eps_rel_)
            break;

        if (n < n_max_) {
            w_plus  = w_plus * y_plus / (n + 1.0);
            w_minus = w_minus * y_minus / (n + 1.0);

            // Advance ehat via recurrence Ehat_{n+1}(x) = (1 - x*Ehat_n(x)) / n
            // unless cumulative amplification (product of x/(k+1) factors) exceeds
            // budget, in which case restart from the continued fraction to curb
            // round-off error growth.
            amp_plus = amp_plus * (x_plus / (n + 1.0));
            if (amp_plus < EHAT_AMPLIFICATION_BUDGET) {
                ehat_plus = (T(1.0) - x_plus * ehat_plus) / (n + 1.0);
            } else {
                ehat_plus = ehat(n + 2, x_plus);
                amp_plus = T(1.0);
            }

            amp_minus = amp_minus * (x_minus / (n + 1.0));
            if (amp_minus < EHAT_AMPLIFICATION_BUDGET) {
                ehat_minus = (T(1.0) - x_minus * ehat_minus) / (n + 1.0);
            } else {
                ehat_minus = ehat(n + 2, x_minus);
                amp_minus = T(1.0);
            }
        }
    }

    if (terms_used > n_max_) {
        throw std::runtime_error("power series failed to converge");
    }

    T const diff = P_plus - P_minus;
    T const normalized = p.Psi + diff;
    T const value = sigma0 * normalized;

    constexpr double T_EPS = details::MachineEps<T>::value;
    T const norm_abs = param_abs(normalized) + eps_tiny;
    T const max_accum = std::max(param_abs(P_plus), param_abs(P_minus));
    T const trunc_rel = last_term_mag / norm_abs;
    T const round_rel = T(terms_used) * T_EPS * max_accum / norm_abs;
    T const rel_error = std::max(trunc_rel, round_rel);
    T const abs_error = rel_error * param_abs(value);

    return SigmaResult{
        to_double(value),
        to_double(abs_error),
        to_double(rel_error)};
}

template SigmaResult ComptonKernelSeries::power_series<double>(
    double, double, double, double, double, double) const;
template SigmaResult ComptonKernelSeries::power_series<DD>(
    double, double, double, double, double, double) const;

SigmaResult ComptonKernelSeries::asymptotic_series(
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

        // Converged: relative term magnitude dropped below tolerance.
        double const norm_so_far = std::abs(base_term + S_plus + S_minus);
        if (n >= n_min_ && term_mag / (norm_so_far + constants::REL_ERROR_TINY_SCALE) < eps_rel_) {
            double const normalized = base_term + S_plus + S_minus;
            double const value = sigma0 * normalized;
            double const abs_error = std::abs(sigma0) * term_mag;
            double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
            return SigmaResult{value, abs_error, rel_error};
        }

        // Asymptotic divergence: terms growing; truncate at smallest-term point.
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

        // Legendre polynomial recurrence: (n+2) P_{n+2}(x) = (2n+3) x P_{n+1}(x) - (n+1) P_n(x)
        double const Pp_next = ((2.0*n + 3.0) * zeta_plus * Pp_curr - (n + 1.0) * Pp_prev) / (n + 2.0);
        Pp_prev = Pp_curr;
        Pp_curr = Pp_next;

        double const Pm_next = ((2.0*n + 3.0) * zeta_minus * Pm_curr - (n + 1.0) * Pm_prev) / (n + 2.0);
        Pm_prev = Pm_curr;
        Pm_curr = Pm_next;
    }

    throw std::runtime_error("asymptotic series failed to converge");
}

SigmaResult ComptonKernelSeries::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const tau,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, tau, Ne);

    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    KershawParams<double> const p = compute_params<double>(gamma, gamma_p, xi, tau);

    SeriesMethod chosen;
    if (method_ == SeriesMethod::Auto) {
        double const tau_alpha_max = std::max(tau * p.alpha_plus,
                                              tau * p.alpha_minus);
        if (tau_alpha_max < 0.05) {
            chosen = SeriesMethod::Asymptotic;
        } else if (std::min(gamma, gamma_p) >= constants::GAMMA_DOUBLE_PRECISION_SAFE) {
            chosen = SeriesMethod::PowerSeries;
        } else {
            chosen = SeriesMethod::PowerSeriesHighPrecision;
        }
    } else {
        chosen = method_;
    }

    if (chosen == SeriesMethod::PowerSeriesHighPrecision)
        return power_series<DD>(gamma, gamma_p, xi, tau, E, Ne);
    else if (chosen == SeriesMethod::PowerSeries)
        return power_series<double>(gamma, gamma_p, xi, tau, E, Ne);
    else
        return asymptotic_series(gamma, gamma_p, xi, tau, E, Ne);
}

double ComptonKernelSeries::sigma_E_precision_check(
    double const E,
    double const E_prime,
    double const xi,
    double const tau,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, tau, Ne);

    double const gamma   = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    SigmaResult const dd_res  = power_series<DD>(gamma, gamma_p, xi, tau, E, Ne);
    SigmaResult const dbl_res = power_series<double>(gamma, gamma_p, xi, tau, E, Ne);

    return std::abs(dd_res.value - dbl_res.value)
         / (std::abs(dd_res.value) + constants::REL_ERROR_TINY_SCALE);
}

} // namespace compton
