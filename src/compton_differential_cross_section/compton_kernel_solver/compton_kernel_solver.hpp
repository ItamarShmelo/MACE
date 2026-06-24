#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Adaptive dispatch kernel for Compton scattering.
 *
 * ComptonKernelSolver selects the fastest accurate method at each
 * phase-space point via a purely error-driven cascade:
 *
 *   Asymptotic regime (tau_alpha_max < threshold):
 *     A1. Asymptotic series (double) -- accepted if self-error < asymp_self_tol.
 *         The roundoff-aware error estimator naturally reports large errors
 *         at ultra-low gamma, triggering escalation to DD.
 *     A2. Asymptotic series (DD) -- accepted if self-error < asymp_self_tol.
 *     Falls through to power series on failure.
 *
 *   Power series regime (tau_alpha_max >= threshold, or fallthrough):
 *     P1. Power series (double) -- accepted if self-error < power_series_self_tol
 *         and (for sigma_E) non-negative.  Only outside asymptotic regime.
 *     P2. Power series (DD) -- accepted if self-error < power_series_self_tol
 *         and (for sigma_E) non-negative.
 *     P3. Asymptotic DD (last resort) -- skipped if already tried in A2.
 *
 *   Returns best-seen result if error < 1e-3; throws otherwise.
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
     * @param power_series_self_tol  Accept power-series result when its
     *        self-reported relative error is below this tolerance.
     * @param asymp_self_tol  Accept an asymptotic result only when its
     *        self-reported relative error is below this tolerance.
     *        When exceeded, the solver escalates or falls through.
     */
    enum class KernelOp { sigma, dsigma_dT };

    ComptonKernelSolver(
        double asymp_tau_alpha_threshold = constants::ASYMP_TAU_ALPHA_THRESHOLD,
        double power_series_self_tol      = 1e-7,
        double asymp_self_tol            = 1e-7);

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
