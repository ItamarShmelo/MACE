#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Adaptive dispatch kernel for Compton scattering.
 *
 * ComptonKernelSolver selects the fastest accurate method at each
 * phase-space point:
 *
 *   1a. Asymptotic series (double) -- when tau * max(alpha+, alpha-) is
 *       small and min(gamma, gamma') is above the DD threshold
 *   1b. Asymptotic series (DD)     -- same tau criterion but ultra-low
 *       gamma where double loses precision
 *   2.  Double power series  -- when min(gamma, gamma') is large enough
 *                               for double precision
 *   3.  Gauss-Laguerre Q64   -- in the DD regime, accepted when its
 *                               self-reported error is below a tolerance
 *   4.  DD power series      -- fallback when quadrature is unconverged
 *
 * All dispatch thresholds are configurable at construction time.
 */

#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"
#include "compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.hpp"

namespace compton {

namespace constants {
constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.025;
constexpr double GAMMA_DOUBLE_PRECISION_SAFE = 0.02;
constexpr double ASYMP_GAMMA_DD_THRESHOLD = 0.002;
} // namespace constants

class ComptonKernelSolver {
public:
    /**
     * @param asymp_tau_alpha_threshold  Dispatch to asymptotic series when
     *        tau * max(alpha+, alpha-) falls below this value.
     * @param gamma_double_precision_safe  Use double power series when
     *        min(gamma, gamma') is at or above this value.
     * @param quadrature_self_tol  Accept the quadrature result when its
     *        self-reported relative error is below this tolerance.
     * @param asymp_gamma_dd_threshold  Within the asymptotic regime, use
     *        DD arithmetic when min(gamma, gamma') falls below this value.
     *        Set to 0 to disable DD in the asymptotic path.
     */
    ComptonKernelSolver(
        double asymp_tau_alpha_threshold   = constants::ASYMP_TAU_ALPHA_THRESHOLD,
        double gamma_double_precision_safe = constants::GAMMA_DOUBLE_PRECISION_SAFE,
        double quadrature_self_tol         = 1e-6,
        double asymp_gamma_dd_threshold    = constants::ASYMP_GAMMA_DD_THRESHOLD);

    ComptonResult sigma_E(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

    ComptonResult dsigma_E_dT(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

private:
    double asymp_tau_alpha_threshold_;
    double gamma_double_precision_safe_;
    double quadrature_self_tol_;
    double asymp_gamma_dd_threshold_;

    ComptonKernelAsymptoticSeries asymp_series_;
    ComptonKernelAsymptoticSeries asymp_series_dd_;
    ComptonPowerSeries            power_series_;
    ComptonPowerSeries            power_series_dd_;
    ComptonKernelQuadrature       quadrature_;
};

} // namespace compton

#endif
