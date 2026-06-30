#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_common/compton_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace compton {

namespace constants {
constexpr double POISSON_Y_MAX = 500.0;
} // namespace constants

ComptonPowerSeries::ComptonPowerSeries(
    bool const high_precision,
    double const eps_rel,
    int const n_min,
    int const n_max)
    : high_precision_(high_precision),
      eps_rel_(eps_rel),
      n_min_(n_min),
      n_max_(n_max)
{}

template <typename T>
ComptonResult ComptonPowerSeries::power_series(
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

    // exp(asinh(z)) = z + sqrt(z^2+1);  since (z+r)(r-z) = 1, use y = b/(z+r)
    // to avoid cancellation in the subtraction r - z.
    T const z_plus = p.rho_plus / omega;
    T const z_minus = p.rho_minus / omega;
    T const r_plus = param_sqrt(z_plus * z_plus + T(1.0));
    T const r_minus = param_sqrt(z_minus * z_minus + T(1.0));
    T const x_plus = b * (z_plus + r_plus);
    T const y_plus = b / (z_plus + r_plus);
    T const x_minus = b * (z_minus + r_minus);
    T const y_minus = b / (z_minus + r_minus);

    if (y_plus > POISSON_Y_MAX || y_minus > POISSON_Y_MAX) {
        throw std::runtime_error("power series: Poisson weight underflow");
    }

    T w_plus = param_exp(-y_plus);
    T w_minus = param_exp(-y_minus);

    T P_plus(0.0);
    T P_minus(0.0);

    constexpr double eps_tiny = 1e-300;
    T last_diff_change(0.0);
    T last_term_mag(0.0);
    int terms_used = 0;

    T ehat_plus = ehat(1, x_plus);
    T ehat_minus = ehat(1, x_minus);
    T amp_plus(1.0);
    T amp_minus(1.0);

    T const two_over_a = T(2.0) / p.a;
    T coeff_inc_plus = p.A_plus;
    T coeff_inc_minus = p.A_minus;

    for (int n = 0; n <= n_max_; ++n) {
        T const t_plus = w_plus * coeff_inc_plus * ehat_plus;
        T const t_minus = w_minus * coeff_inc_minus * ehat_minus;

        P_plus = P_plus + t_plus;
        P_minus = P_minus + t_minus;
        last_diff_change = param_abs(t_plus - t_minus);

        last_term_mag = param_abs(t_plus) + param_abs(t_minus);
        terms_used = n + 1;

        if (n >= n_min_ &&
            last_diff_change /
                    (param_abs(p.Psi + (P_plus - P_minus)) + eps_tiny) <
                eps_rel_)
            break;

        if (n < n_max_) {
            T const inv_np1 = T(1.0) / T(n + 1.0);

            w_plus = w_plus * y_plus * inv_np1;
            w_minus = w_minus * y_minus * inv_np1;

            amp_plus = amp_plus * x_plus * inv_np1;
            if (amp_plus < EhatAmpBudget<T>::value) {
                ehat_plus = (T(1.0) - x_plus * ehat_plus) * inv_np1;
            } else {
                ehat_plus = ehat(n + 2, x_plus);
                amp_plus = T(1.0);
            }

            amp_minus = amp_minus * x_minus * inv_np1;
            if (amp_minus < EhatAmpBudget<T>::value) {
                ehat_minus = (T(1.0) - x_minus * ehat_minus) * inv_np1;
            } else {
                ehat_minus = ehat(n + 2, x_minus);
                amp_minus = T(1.0);
            }

            coeff_inc_plus = coeff_inc_plus + two_over_a;
            coeff_inc_minus = coeff_inc_minus + two_over_a;
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

    T const cancel_P = (param_abs(P_plus) + param_abs(P_minus)) /
                       (param_abs(diff) + T(eps_tiny));
    T const cancel_Psi = (param_abs(p.Psi) + param_abs(diff)) /
                         (param_abs(normalized) + T(eps_tiny));

    T const trunc_rel = last_term_mag / norm_abs;
    T const round_rel = T(terms_used) * T_EPS * cancel_P * cancel_Psi;
    T const rel_error = std::max(trunc_rel, round_rel);
    T const abs_error = rel_error * param_abs(value);

    return ComptonResult{
        to_double(value),
        to_double(abs_error),
        to_double(rel_error),
        terms_used};
}

template ComptonResult ComptonPowerSeries::power_series<double>(
    double,
    double,
    double,
    double,
    double,
    double) const;
template ComptonResult ComptonPowerSeries::power_series<DD>(
    double,
    double,
    double,
    double,
    double,
    double) const;

template <typename T>
ComptonResult ComptonPowerSeries::power_series_derivative(
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
    double const kappa_val = kappa_ratio(tau);

    T const omega = param_sqrt(p.omega2);
    T const tau_t(tau);
    T const b = omega / (tau_t * 2.0);

    T const z_plus = p.rho_plus / omega;
    T const z_minus = p.rho_minus / omega;
    T const r_plus = param_sqrt(z_plus * z_plus + T(1.0));
    T const r_minus = param_sqrt(z_minus * z_minus + T(1.0));
    T const x_plus = b * (z_plus + r_plus);
    T const y_plus = b / (z_plus + r_plus);
    T const x_minus = b * (z_minus + r_minus);
    T const y_minus = b / (z_minus + r_minus);

    if (y_plus > POISSON_Y_MAX || y_minus > POISSON_Y_MAX) {
        throw std::runtime_error(
            "power series derivative: Poisson weight underflow");
    }

    T w_plus = param_exp(-y_plus);
    T w_minus = param_exp(-y_minus);

    T P_plus(0.0);
    T P_minus(0.0);
    T dP_plus(0.0);
    T dP_minus(0.0);

    constexpr double eps_tiny = 1e-300;
    T last_deriv_diff_change(0.0);
    T last_deriv_term_mag(0.0);
    T last_P_diff_change(0.0);
    T last_P_term_mag(0.0);
    int terms_used = 0;

    T ehat_plus = ehat(1, x_plus);
    T ehat_minus = ehat(1, x_minus);

    // Ê₀(x) = 1/x
    T ehat_prev_plus = T(1.0) / x_plus;
    T ehat_prev_minus = T(1.0) / x_minus;

    T amp_plus(1.0);
    T amp_minus(1.0);

    T const lk = T(to_double(p.lambda_plus)) - T(kappa_val);
    T const s_over_tau2_a2 = p.s / (tau_t * tau_t * p.a * p.a);

    T const two_over_a = T(2.0) / p.a;
    T coeff_inc_plus = p.A_plus;
    T coeff_inc_minus = p.A_minus;
    T const tau2 = tau_t * tau_t;
    T const rho_p_over_tau2 = p.rho_plus / tau2;
    T const rho_m_over_tau2 = p.rho_minus / tau2;
    T const inv_tau = T(1.0) / tau_t;
    T const x_plus_over_tau = x_plus * inv_tau;
    T const x_minus_over_tau = x_minus * inv_tau;
    T const dPsi_const = T(2.0 * gamma * gamma_p) / p.q;
    T const dlnSig0 = lk / tau2 - T(3.0) * inv_tau;
    T const abs_dlnSig0 = param_abs(dlnSig0);

    for (int n = 0; n <= n_max_; ++n) {
        T const t_plus = w_plus * coeff_inc_plus * ehat_plus;
        T const t_minus = w_minus * coeff_inc_minus * ehat_minus;

        P_plus = P_plus + t_plus;
        P_minus = P_minus + t_minus;
        last_P_diff_change = param_abs(t_plus - t_minus);
        last_P_term_mag = param_abs(t_plus) + param_abs(t_minus);

        T const n_over_tau = T(static_cast<double>(n)) * inv_tau;

        T const dcoeff_plus_ehat =
            (s_over_tau2_a2 - (rho_p_over_tau2 + n_over_tau) * coeff_inc_plus) *
                ehat_plus +
            x_plus_over_tau * coeff_inc_plus * ehat_prev_plus;
        T const dcoeff_minus_ehat =
            (-s_over_tau2_a2 -
             (rho_m_over_tau2 + n_over_tau) * coeff_inc_minus) *
                ehat_minus +
            x_minus_over_tau * coeff_inc_minus * ehat_prev_minus;

        T const dt_plus = w_plus * dcoeff_plus_ehat;
        T const dt_minus = w_minus * dcoeff_minus_ehat;

        dP_plus = dP_plus + dt_plus;
        dP_minus = dP_minus + dt_minus;
        last_deriv_diff_change = param_abs(dt_plus - dt_minus);

        last_deriv_term_mag = param_abs(dt_plus) + param_abs(dt_minus);
        terms_used = n + 1;

        T const deriv_normalized = dPsi_const + (dP_plus - dP_minus) +
                                   dlnSig0 * (p.Psi + (P_plus - P_minus));

        T const total_trunc_est =
            last_deriv_diff_change + abs_dlnSig0 * last_P_diff_change;
        if (n >= n_min_ &&
            total_trunc_est / (param_abs(deriv_normalized) + eps_tiny) <
                eps_rel_)
            break;

        if (n < n_max_) {
            T const inv_np1 = T(1.0) / T(n + 1.0);

            w_plus = w_plus * y_plus * inv_np1;
            w_minus = w_minus * y_minus * inv_np1;

            ehat_prev_plus = ehat_plus;
            ehat_prev_minus = ehat_minus;

            amp_plus = amp_plus * x_plus * inv_np1;
            if (amp_plus < EhatAmpBudget<T>::value) {
                ehat_plus = (T(1.0) - x_plus * ehat_plus) * inv_np1;
            } else {
                ehat_plus = ehat(n + 2, x_plus);
                amp_plus = T(1.0);
            }

            amp_minus = amp_minus * x_minus * inv_np1;
            if (amp_minus < EhatAmpBudget<T>::value) {
                ehat_minus = (T(1.0) - x_minus * ehat_minus) * inv_np1;
            } else {
                ehat_minus = ehat(n + 2, x_minus);
                amp_minus = T(1.0);
            }

            coeff_inc_plus = coeff_inc_plus + two_over_a;
            coeff_inc_minus = coeff_inc_minus + two_over_a;
        }
    }

    if (terms_used > n_max_) {
        throw std::runtime_error("power series derivative failed to converge");
    }

    T const diff = P_plus - P_minus;
    T const ddiff = dP_plus - dP_minus;
    T const sigma_term = dlnSig0 * (p.Psi + diff);
    T const deriv_normalized = dPsi_const + ddiff + sigma_term;
    T const value = sigma0 * deriv_normalized;

    constexpr double T_EPS = details::MachineEps<T>::value;
    T const norm_abs = param_abs(deriv_normalized) + eps_tiny;

    T const dPsi_plus_ddiff = dPsi_const + ddiff;

    T const cancel_dP = (param_abs(dP_plus) + param_abs(dP_minus)) /
                        (param_abs(ddiff) + T(eps_tiny));
    T const cancel_dPsi = (param_abs(dPsi_const) + param_abs(ddiff)) /
                          (param_abs(dPsi_plus_ddiff) + T(eps_tiny));

    T const cancel_P = (param_abs(P_plus) + param_abs(P_minus)) /
                       (param_abs(diff) + T(eps_tiny));
    T const cancel_Psi = (param_abs(p.Psi) + param_abs(diff)) /
                         (param_abs(p.Psi + diff) + T(eps_tiny));

    T const abs_err_dP_path =
        param_abs(dPsi_plus_ddiff) * cancel_dP * cancel_dPsi;
    T const abs_err_P_path = param_abs(sigma_term) * cancel_P * cancel_Psi;

    T const trunc_rel =
        (last_deriv_term_mag + abs_dlnSig0 * last_P_term_mag) / norm_abs;
    T const round_rel = T(terms_used) * T_EPS *
                        (abs_err_dP_path + abs_err_P_path) /
                        (param_abs(deriv_normalized) + T(eps_tiny));
    T const rel_error = std::max(trunc_rel, round_rel);
    T const abs_error = rel_error * param_abs(value);

    return ComptonResult{
        to_double(value),
        to_double(abs_error),
        to_double(rel_error),
        terms_used};
}

template ComptonResult ComptonPowerSeries::power_series_derivative<double>(
    double,
    double,
    double,
    double,
    double,
    double) const;
template ComptonResult ComptonPowerSeries::power_series_derivative<DD>(
    double,
    double,
    double,
    double,
    double,
    double) const;

ComptonResult ComptonPowerSeries::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau = T * units::k_boltz / units::me_c2;
    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    if (high_precision_)
        return power_series<DD>(gamma, gamma_p, xi, tau, E, Ne);
    else
        return power_series<double>(gamma, gamma_p, xi, tau, E, Ne);
}

double ComptonPowerSeries::sigma_E_precision_check(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau = T * units::k_boltz / units::me_c2;
    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    ComptonResult const dd_res =
        power_series<DD>(gamma, gamma_p, xi, tau, E, Ne);
    ComptonResult const dbl_res =
        power_series<double>(gamma, gamma_p, xi, tau, E, Ne);

    return std::abs(dd_res.value - dbl_res.value) /
           (std::abs(dd_res.value) + constants::REL_ERROR_TINY_SCALE);
}

ComptonResult ComptonPowerSeries::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau = T * units::k_boltz / units::me_c2;
    double const dtau_dT = units::k_boltz / units::me_c2;
    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    ComptonResult dtau_result;
    if (high_precision_)
        dtau_result =
            power_series_derivative<DD>(gamma, gamma_p, xi, tau, E, Ne);
    else
        dtau_result =
            power_series_derivative<double>(gamma, gamma_p, xi, tau, E, Ne);

    return ComptonResult{
        dtau_result.value * dtau_dT,
        dtau_result.estimated_abs_error * dtau_dT,
        dtau_result.estimated_rel_error,
        dtau_result.terms_used};
}

double ComptonPowerSeries::dsigma_E_dT_precision_check(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau = T * units::k_boltz / units::me_c2;
    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    ComptonResult const dd_res =
        power_series_derivative<DD>(gamma, gamma_p, xi, tau, E, Ne);
    ComptonResult const dbl_res =
        power_series_derivative<double>(gamma, gamma_p, xi, tau, E, Ne);

    return std::abs(dd_res.value - dbl_res.value) /
           (std::abs(dd_res.value) + constants::REL_ERROR_TINY_SCALE);
}

} // namespace compton
