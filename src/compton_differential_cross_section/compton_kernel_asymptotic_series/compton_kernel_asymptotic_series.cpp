#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"
#include "compton_common/compton_common.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace compton {

ComptonKernelAsymptoticSeries::ComptonKernelAsymptoticSeries(
    bool const high_precision,
    double const eps_rel,
    int const n_min,
    int const n_max)
    : high_precision_(high_precision)
    , eps_rel_(eps_rel)
    , n_min_(n_min)
    , n_max_(n_max)
{}

template<typename T>
ComptonResult ComptonKernelAsymptoticSeries::asymptotic_series(
    double const gamma,
    double const gamma_p,
    double const xi,
    double const tau,
    double const E,
    double const Ne) const
{
    using namespace details;

    KershawParams<T> const p = compute_params<T>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, to_double(p.lambda_plus), Ne);

    T const a = p.a;
    T const a2 = a * a;

    T const one = static_cast<T>(1.0);
    T const neg_one = static_cast<T>(-1.0);

    T const zeta_plus  = param_clamp(p.rho_plus * p.alpha_plus, neg_one, one);
    T const zeta_minus = param_clamp(p.rho_minus * p.alpha_minus, neg_one, one);

    T const eta_plus  = p.alpha_plus * (p.s / a2 + p.rho_plus / a);
    T const eta_minus = p.alpha_minus * (-p.s / a2 + p.rho_minus / a);

    T const gamma_t   = static_cast<T>(gamma);
    T const gamma_p_t = static_cast<T>(gamma_p);
    T const tau_t     = static_cast<T>(tau);

    T const base_term = static_cast<T>(2.0) * tau_t * gamma_t * gamma_p_t / p.q;

    T const neg_tau_alpha_plus  = -tau_t * p.alpha_plus;
    T const neg_tau_alpha_minus = -tau_t * p.alpha_minus;

    T S_combined = static_cast<T>(0.0);

    double smallest_term_mag = std::numeric_limits<double>::infinity();
    T best_S = static_cast<T>(0.0);
    int increase_count = 0;
    double prev_combined_mag = std::numeric_limits<double>::infinity();

    T factorial_n    = one;
    T power_plus     = neg_tau_alpha_plus;
    T power_minus    = neg_tau_alpha_minus;

    T Pp_prev = one;
    T Pp_curr = zeta_plus;
    T Pm_prev = one;
    T Pm_curr = zeta_minus;

    for (int n = 0; n <= n_max_; ++n) {
        if (n > 0) {
            factorial_n *= static_cast<T>(n);
            power_plus  *= neg_tau_alpha_plus;
            power_minus *= neg_tau_alpha_minus;
        }

        T const factorial_n1 = factorial_n * static_cast<T>(n + 1);

        T const Pp_n  = Pp_prev;
        T const Pp_n1 = Pp_curr;
        T const Pm_n  = Pm_prev;
        T const Pm_n1 = Pm_curr;

        // Rearranged accumulation: groups the G contribution so that
        // the massive G*n! terms cancel analytically rather than
        // numerically.  See docs/asymptotic_rearrangement_derivation.md.
        T const D_n = power_minus * Pm_n - power_plus * Pp_n;
        T const F_n1 = eta_minus * power_minus * Pm_n1
                     - eta_plus * power_plus * Pp_n1;
        T const combined =
            (p.G - static_cast<T>(n + 1) / a) * factorial_n * D_n
            + factorial_n1 * F_n1;

        S_combined += combined;
        double const combined_mag = to_double(param_abs(combined));

        if (combined_mag < smallest_term_mag) {
            smallest_term_mag = combined_mag;
            best_S = S_combined;
        }

        double const norm_so_far = to_double(param_abs(base_term + S_combined));

        if (n >= n_min_ && combined_mag / (norm_so_far + constants::REL_ERROR_TINY_SCALE) < eps_rel_) {
            double const normalized = to_double(base_term + S_combined);
            double const value = sigma0 * normalized;
            double const abs_error = std::abs(sigma0) * combined_mag;
            double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
            return ComptonResult{value, abs_error, rel_error, n + 1};
        }

        if (n >= n_min_ && combined_mag > prev_combined_mag) {
            ++increase_count;
            if (increase_count >= 2) {
                double const normalized = to_double(base_term + best_S);
                double const value = sigma0 * normalized;
                double const abs_error = std::abs(sigma0) * smallest_term_mag;
                double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
                return ComptonResult{value, abs_error, rel_error, n + 1};
            }
        } else {
            increase_count = 0;
        }

        prev_combined_mag = combined_mag;

        if (!param_isfinite(factorial_n) || !std::isfinite(combined_mag))
            break;

        T const n_dbl = static_cast<T>(n);
        T const Pp_next = ((static_cast<T>(2.0) * n_dbl + static_cast<T>(3.0)) * zeta_plus * Pp_curr
                         - (n_dbl + one) * Pp_prev) / (n_dbl + static_cast<T>(2.0));
        Pp_prev = Pp_curr;
        Pp_curr = Pp_next;

        T const Pm_next = ((static_cast<T>(2.0) * n_dbl + static_cast<T>(3.0)) * zeta_minus * Pm_curr
                         - (n_dbl + one) * Pm_prev) / (n_dbl + static_cast<T>(2.0));
        Pm_prev = Pm_curr;
        Pm_curr = Pm_next;
    }

    throw std::runtime_error("asymptotic series failed to converge");
}

template ComptonResult ComptonKernelAsymptoticSeries::asymptotic_series<double>(
    double, double, double, double, double, double) const;
template ComptonResult ComptonKernelAsymptoticSeries::asymptotic_series<DD>(
    double, double, double, double, double, double) const;

template<typename T>
ComptonResult ComptonKernelAsymptoticSeries::asymptotic_series_derivative(
    double const gamma,
    double const gamma_p,
    double const xi,
    double const tau,
    double const E,
    double const Ne) const
{
    using namespace details;

    KershawParams<T> const p = compute_params<T>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, to_double(p.lambda_plus), Ne);
    double const kappa_val = kappa_ratio(tau);

    T const a = p.a;
    T const a2 = a * a;

    T const one = static_cast<T>(1.0);
    T const neg_one = static_cast<T>(-1.0);

    T const zeta_plus  = param_clamp(p.rho_plus * p.alpha_plus, neg_one, one);
    T const zeta_minus = param_clamp(p.rho_minus * p.alpha_minus, neg_one, one);

    T const eta_plus  = p.alpha_plus * (p.s / a2 + p.rho_plus / a);
    T const eta_minus = p.alpha_minus * (-p.s / a2 + p.rho_minus / a);

    T const gamma_t   = static_cast<T>(gamma);
    T const gamma_p_t = static_cast<T>(gamma_p);
    T const tau_t     = static_cast<T>(tau);

    T const lk = p.lambda_plus - static_cast<T>(kappa_val);
    T const base_deriv = static_cast<T>(2.0) * gamma_t * gamma_p_t / p.q
                       * (lk / tau_t - static_cast<T>(2.0));

    T const neg_tau_alpha_plus  = -tau_t * p.alpha_plus;
    T const neg_tau_alpha_minus = -tau_t * p.alpha_minus;

    T dS_combined = static_cast<T>(0.0);

    double smallest_dterm_mag = std::numeric_limits<double>::infinity();
    T best_dS = static_cast<T>(0.0);
    int increase_count = 0;
    double prev_dcombined_mag = std::numeric_limits<double>::infinity();

    T factorial_n    = one;
    T power_plus     = neg_tau_alpha_plus;
    T power_minus    = neg_tau_alpha_minus;

    T Pp_prev = one;
    T Pp_curr = zeta_plus;
    T Pm_prev = one;
    T Pm_curr = zeta_minus;

    for (int n = 0; n <= n_max_; ++n) {
        if (n > 0) {
            factorial_n *= static_cast<T>(n);
            power_plus  *= neg_tau_alpha_plus;
            power_minus *= neg_tau_alpha_minus;
        }

        T const factorial_n1 = factorial_n * static_cast<T>(n + 1);

        T const Pp_n  = Pp_prev;
        T const Pp_n1 = Pp_curr;
        T const Pm_n  = Pm_prev;
        T const Pm_n1 = Pm_curr;

        T const weight = lk / (tau_t * tau_t) + (static_cast<T>(n) - static_cast<T>(2.0)) / tau_t;

        T const D_n = power_minus * Pm_n - power_plus * Pp_n;
        T const F_n1 = eta_minus * power_minus * Pm_n1
                     - eta_plus * power_plus * Pp_n1;
        T const combined =
            weight * ((p.G - static_cast<T>(n + 1) / a) * factorial_n * D_n
                      + factorial_n1 * F_n1);

        dS_combined += combined;
        double const dcombined_mag = to_double(param_abs(combined));

        if (dcombined_mag < smallest_dterm_mag) {
            smallest_dterm_mag = dcombined_mag;
            best_dS = dS_combined;
        }

        double const norm_so_far = to_double(param_abs(base_deriv + dS_combined));

        if (n >= n_min_ && dcombined_mag / (norm_so_far + constants::REL_ERROR_TINY_SCALE) < eps_rel_) {
            double const normalized = to_double(base_deriv + dS_combined);
            double const value = sigma0 * normalized;
            double const abs_error = std::abs(sigma0) * dcombined_mag;
            double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
            return ComptonResult{value, abs_error, rel_error, n + 1};
        }

        if (n >= n_min_ && dcombined_mag > prev_dcombined_mag) {
            ++increase_count;
            if (increase_count >= 2) {
                double const normalized = to_double(base_deriv + best_dS);
                double const value = sigma0 * normalized;
                double const abs_error = std::abs(sigma0) * smallest_dterm_mag;
                double const rel_error = abs_error / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
                return ComptonResult{value, abs_error, rel_error, n + 1};
            }
        } else {
            increase_count = 0;
        }

        prev_dcombined_mag = dcombined_mag;

        if (!param_isfinite(factorial_n) || !std::isfinite(dcombined_mag))
            break;

        T const n_dbl = static_cast<T>(n);
        T const Pp_next = ((static_cast<T>(2.0) * n_dbl + static_cast<T>(3.0)) * zeta_plus * Pp_curr
                         - (n_dbl + one) * Pp_prev) / (n_dbl + static_cast<T>(2.0));
        Pp_prev = Pp_curr;
        Pp_curr = Pp_next;

        T const Pm_next = ((static_cast<T>(2.0) * n_dbl + static_cast<T>(3.0)) * zeta_minus * Pm_curr
                         - (n_dbl + one) * Pm_prev) / (n_dbl + static_cast<T>(2.0));
        Pm_prev = Pm_curr;
        Pm_curr = Pm_next;
    }

    throw std::runtime_error("asymptotic series derivative failed to converge");
}

template ComptonResult ComptonKernelAsymptoticSeries::asymptotic_series_derivative<double>(
    double, double, double, double, double, double) const;
template ComptonResult ComptonKernelAsymptoticSeries::asymptotic_series_derivative<DD>(
    double, double, double, double, double, double) const;

ComptonResult ComptonKernelAsymptoticSeries::sigma_E(
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

    if (high_precision_)
        return asymptotic_series<DD>(gamma, gamma_p, xi, tau, E, Ne);
    else
        return asymptotic_series<double>(gamma, gamma_p, xi, tau, E, Ne);
}

double ComptonKernelAsymptoticSeries::sigma_E_precision_check(
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

    ComptonResult const dd_res  = asymptotic_series<DD>(gamma, gamma_p, xi, tau, E, Ne);
    ComptonResult const dbl_res = asymptotic_series<double>(gamma, gamma_p, xi, tau, E, Ne);

    return std::abs(dd_res.value - dbl_res.value)
         / (std::abs(dd_res.value) + constants::REL_ERROR_TINY_SCALE);
}

ComptonResult ComptonKernelAsymptoticSeries::dsigma_E_dT(
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

    ComptonResult dtau_result;
    if (high_precision_)
        dtau_result = asymptotic_series_derivative<DD>(gamma, gamma_p, xi, tau, E, Ne);
    else
        dtau_result = asymptotic_series_derivative<double>(gamma, gamma_p, xi, tau, E, Ne);

    return ComptonResult{
        dtau_result.value * dtau_dT,
        dtau_result.estimated_abs_error * dtau_dT,
        dtau_result.estimated_rel_error,
        dtau_result.terms_used};
}

double ComptonKernelAsymptoticSeries::dsigma_E_dT_precision_check(
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

    ComptonResult const dd_res  = asymptotic_series_derivative<DD>(gamma, gamma_p, xi, tau, E, Ne);
    ComptonResult const dbl_res = asymptotic_series_derivative<double>(gamma, gamma_p, xi, tau, E, Ne);

    return std::abs(dd_res.value - dbl_res.value)
         / (std::abs(dd_res.value) + constants::REL_ERROR_TINY_SCALE);
}

} // namespace compton
