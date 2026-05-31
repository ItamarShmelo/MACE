/**
 * @file compton_kernel_solver.cpp
 * @brief Adaptive solver: asymptotic or power series dispatch.
 */

#include "compton_kernel_solver.hpp"

#include <algorithm>
#include <cmath>

namespace compton {

ComptonKernelSolver::ComptonKernelSolver()
    : asymptotic_series_(SeriesMethod::Asymptotic, SERIES_EPS_REL, SERIES_N_MIN, SERIES_N_MAX),
      power_series_(SeriesMethod::Auto, SERIES_EPS_REL, SERIES_N_MIN, SERIES_N_MAX)
{}

SigmaResult ComptonKernelSolver::sigma_E(
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
    double const tau_alpha_max = std::max(tau * p.alpha_plus, tau * p.alpha_minus);

    if (tau_alpha_max < constants::ASYMP_TAU_ALPHA_THRESHOLD) {
        return asymptotic_series_.sigma_E(E, E_prime, xi, tau, Ne);
    }

    return power_series_.sigma_E(E, E_prime, xi, tau, Ne);
}

} // namespace compton
