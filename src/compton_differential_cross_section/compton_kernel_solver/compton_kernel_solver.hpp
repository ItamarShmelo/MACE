#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Adaptive dispatch kernel for Compton scattering.
 *
 * ComptonKernelSolver selects the fastest accurate method at each
 * phase-space point:
 *
 *   1.  Asymptotic series (double or DD) -- when tau * max(alpha+, alpha-)
 *       is small.  DD is used when min(gamma, gamma') falls below
 *       asymp_gamma_dd_threshold.  Near the dispatch boundary at ultra-low
 *       gamma, the result is cross-validated against DD power series.
 *   2.  Double power series (speculative) -- attempted outside the
 *       asymptotic regime; accepted when self-error < quadrature_self_tol
 *       and (for sigma_E) non-negative.
 *   3.  DD power series + conditional Asymp-DD cross-validation --
 *       fallback when both asymptotic and double PS fail.  Asymp-DD is
 *       skipped when PS-DD reports clearly trustworthy results.
 *
 * All dispatch thresholds are configurable at construction time.
 */

#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"

namespace compton {

namespace constants {
constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.04;
constexpr double ASYMP_GAMMA_DD_THRESHOLD = 0.002;
/// Cross-validate DD asymptotic against DD power series when gamma is
/// below this value.  At ultra-low gamma the asymptotic series can
/// silently return wrong values while reporting tiny self-error.
constexpr double ASYMP_GAMMA_DD_CROSS_VAL_THRESHOLD = 1e-4;
} // namespace constants

class ComptonKernelSolver {
public:
    /**
     * @param asymp_tau_alpha_threshold  Dispatch to asymptotic series when
     *        tau * max(alpha+, alpha-) falls below this value.
     * @param quadrature_self_tol  Accept speculative double PS result when
     *        its self-reported relative error is below this tolerance.
     * @param asymp_gamma_dd_threshold  Within the asymptotic regime, use
     *        DD arithmetic when min(gamma, gamma') falls below this value.
     *        Set to 0 to disable DD in the asymptotic path.
     * @param asymp_self_tol  Accept the asymptotic result only when its
     *        self-reported relative error is below this tolerance.
     *        When exceeded, the solver falls through to DD power series.
     * @param asymp_gamma_dd_cross_val_threshold  When gamma_min falls
     *        below this value, the DD asymptotic result is
     *        cross-validated against DD power series even when the
     *        asymptotic self-tolerance passes.
     */
    enum class KernelOp { sigma, dsigma_dT };

    ComptonKernelSolver(
        double asymp_tau_alpha_threshold   = constants::ASYMP_TAU_ALPHA_THRESHOLD,
        double quadrature_self_tol         = 1e-6,
        double asymp_gamma_dd_threshold    = constants::ASYMP_GAMMA_DD_THRESHOLD,
        double asymp_self_tol              = 1e-3,
        double asymp_gamma_dd_cross_val_threshold
            = constants::ASYMP_GAMMA_DD_CROSS_VAL_THRESHOLD);

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
    double quadrature_self_tol_;
    double asymp_gamma_dd_threshold_;
    double asymp_self_tol_;
    double asymp_gamma_dd_cross_val_threshold_;

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
