#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"

#include "utilities/units.hpp"

namespace compton {

ComptonKernelApproximateSolver::ComptonKernelApproximateSolver(
    double const gamma_tau_ratio,
    double const tau_max,
    double const asymp_tau_alpha_threshold,
    double const power_series_self_tol,
    double const asymp_self_tol,
    double const dd_power_series_self_tol,
    double const dd_asymp_self_tol,
    bool const verbose)
    : gamma_tau_ratio_(gamma_tau_ratio),
      tau_max_(tau_max),
      approximate_(),
      solver_(
          asymp_tau_alpha_threshold,
          power_series_self_tol,
          asymp_self_tol,
          dd_power_series_self_tol,
          dd_asymp_self_tol,
          verbose)
{}

ComptonResult
ComptonKernelApproximateSolver::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    double const gamma = E / units::me_c2;
    double const tau = units::k_boltz * T / units::me_c2;

    if (gamma >= gamma_tau_ratio_ * tau && tau <= tau_max_) {
        ComptonResult const r = approximate_.sigma_E(E, E_prime, xi, T);
        if (r.estimated_abs_error < 1.0) {
            return r;
        }
    }

    return solver_.sigma_E(E, E_prime, xi, T);
}

ComptonResult
ComptonKernelApproximateSolver::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    return solver_.dsigma_E_dT(E, E_prime, xi, T);
}

} // namespace compton
