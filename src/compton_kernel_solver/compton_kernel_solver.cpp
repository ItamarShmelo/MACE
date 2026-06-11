#include "compton_kernel_solver/compton_kernel_solver.hpp"

namespace compton {

ComptonKernelSolver::ComptonKernelSolver(
    double const asymp_tau_alpha_threshold,
    double const gamma_double_precision_safe,
    double const quadrature_self_tol,
    double const asymp_gamma_dd_threshold)
    : asymp_tau_alpha_threshold_(asymp_tau_alpha_threshold)
    , gamma_double_precision_safe_(gamma_double_precision_safe)
    , quadrature_self_tol_(quadrature_self_tol)
    , asymp_gamma_dd_threshold_(asymp_gamma_dd_threshold)
    , asymp_series_(false)
    , asymp_series_dd_(true)
    , power_series_(false)
    , power_series_dd_(true)
    , quadrature_(64)
{}

ComptonResult ComptonKernelSolver::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    double const tau     = T * units::k_boltz / units::me_c2;
    double const gamma   = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    auto const p = compute_params<double>(gamma, gamma_p, xi, tau);

    double const tau_alpha_max = std::max(tau * p.alpha_plus,
                                          tau * p.alpha_minus);

    if (tau_alpha_max < asymp_tau_alpha_threshold_) {
        if (std::min(gamma, gamma_p) < asymp_gamma_dd_threshold_)
            return asymp_series_dd_.sigma_E(E, E_prime, xi, T, Ne);
        return asymp_series_.sigma_E(E, E_prime, xi, T, Ne);
    }

    if (std::min(gamma, gamma_p) >= gamma_double_precision_safe_)
        return power_series_.sigma_E(E, E_prime, xi, T, Ne);

    auto const q_result = quadrature_.sigma_E(E, E_prime, xi, T, Ne);
    if (q_result.estimated_rel_error < quadrature_self_tol_)
        return q_result;

    return power_series_dd_.sigma_E(E, E_prime, xi, T, Ne);
}

ComptonResult ComptonKernelSolver::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    double const tau     = T * units::k_boltz / units::me_c2;
    double const gamma   = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    auto const p = compute_params<double>(gamma, gamma_p, xi, tau);

    double const tau_alpha_max = std::max(tau * p.alpha_plus,
                                          tau * p.alpha_minus);

    if (tau_alpha_max < asymp_tau_alpha_threshold_) {
        if (std::min(gamma, gamma_p) < asymp_gamma_dd_threshold_)
            return asymp_series_dd_.dsigma_E_dT(E, E_prime, xi, T, Ne);
        return asymp_series_.dsigma_E_dT(E, E_prime, xi, T, Ne);
    }

    if (std::min(gamma, gamma_p) >= gamma_double_precision_safe_)
        return power_series_.dsigma_E_dT(E, E_prime, xi, T, Ne);

    auto const q_result = quadrature_.dsigma_E_dT(E, E_prime, xi, T, Ne);
    if (q_result.estimated_rel_error < quadrature_self_tol_)
        return q_result;

    return power_series_dd_.dsigma_E_dT(E, E_prime, xi, T, Ne);
}

} // namespace compton
