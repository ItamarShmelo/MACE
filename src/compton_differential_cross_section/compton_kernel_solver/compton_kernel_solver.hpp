#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Adaptive dispatch kernel for Compton scattering.
 *
 * ComptonKernelSolver selects the fastest accurate method at each
 * phase-space point via mutually exclusive regime dispatch:
 *
 *   Asymptotic regime (tau_alpha_max < threshold):
 *     A1. Asymptotic series (double) -- accepted if self-error < asymp_self_tol.
 *         The roundoff-aware error estimator naturally reports large errors
 *         at ultra-low gamma, triggering escalation to DD.
 *     A2. Asymptotic series (DD) -- accepted if self-error < dd_asymp_self_tol.
 *
 *   Power series regime (tau_alpha_max >= threshold):
 *     P1. Power series (double) -- accepted if self-error < power_series_self_tol
 *         and (for sigma_E) non-negative.
 *     P2. Power series (DD, n_max=500) -- accepted if self-error <
 *         dd_power_series_self_tol and (for sigma_E) non-negative.
 *
 *   Throws if no backend passes its tolerance.
 *
 * All dispatch thresholds are configurable at construction time.
 */

#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"

namespace compton {

namespace constants {
constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.035;
} // namespace constants

class ComptonKernelSolver {
public:
    /**
     * @param asymp_tau_alpha_threshold  Dispatch to asymptotic series when
     *        tau * max(alpha+, alpha-) falls below this value.
     * @param power_series_self_tol  Accept double power-series result when
     *        its self-reported relative error is below this tolerance.
     * @param asymp_self_tol  Accept a double asymptotic result only when
     *        its self-reported relative error is below this tolerance.
     *        When exceeded, the solver escalates to DD within the regime.
     * @param dd_power_series_self_tol  Accept DD power-series result when
     *        its self-reported relative error is below this (looser)
     *        tolerance.
     * @param dd_asymp_self_tol  Accept DD asymptotic result when its
     *        self-reported relative error is below this (looser) tolerance.
     */
    enum class KernelOp { sigma, dsigma_dT };

    ComptonKernelSolver(
        double asymp_tau_alpha_threshold  = constants::ASYMP_TAU_ALPHA_THRESHOLD,
        double power_series_self_tol      = 1e-7,
        double asymp_self_tol             = 1e-7,
        double dd_power_series_self_tol   = 0.5,
        double dd_asymp_self_tol          = 0.5);

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
    double power_series_self_tol_;
    double asymp_self_tol_;
    double dd_power_series_self_tol_;
    double dd_asymp_self_tol_;

    ComptonKernelAsymptoticSeries asymp_series_;
    ComptonKernelAsymptoticSeries asymp_series_dd_;
    ComptonPowerSeries            power_series_;
    ComptonPowerSeries            power_series_dd_;

    template <KernelOp Op>
    ComptonResult dispatch(double E, double E_prime, double xi,
                           double T, double Ne) const;
};

} // namespace compton

#endif
