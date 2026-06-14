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
 *       small and min(gamma, gamma') is above the DD threshold.
 *       Accepted only when the series' self-reported relative error is
 *       below asymp_self_tol; otherwise falls through to steps 2-4.
 *   1b. Asymptotic series (DD)     -- same tau criterion but ultra-low
 *       gamma where double loses precision.  Same self-tolerance gate.
 *       Near the dispatch boundary (tau*max(alpha) > 40% of threshold)
 *       the DD asymptotic error estimate can silently underestimate the
 *       true error, so a cross-validation against DD power series is
 *       performed; if DD power series succeeds with tight self-error,
 *       its result is preferred.
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
constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.04;
constexpr double GAMMA_DOUBLE_PRECISION_SAFE = 0.02;
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
     * @param gamma_double_precision_safe  Use double power series when
     *        min(gamma, gamma') is at or above this value.
     * @param quadrature_self_tol  Accept the quadrature result when its
     *        self-reported relative error is below this tolerance.
     * @param asymp_gamma_dd_threshold  Within the asymptotic regime, use
     *        DD arithmetic when min(gamma, gamma') falls below this value.
     *        Set to 0 to disable DD in the asymptotic path.
     * @param asymp_self_tol  Accept the asymptotic result only when its
     *        self-reported relative error is below this tolerance.
     *        When exceeded, the solver falls through to Q64/DD power
     *        series.  Guards against the narrow mu band near the
     *        dispatch boundary where the asymptotic series reports
     *        very large errors at ultra-low gamma.
     * @param asymp_gamma_dd_cross_val_threshold  When gamma_min falls
     *        below this value, the DD asymptotic result is
     *        cross-validated against DD power series even when the
     *        asymptotic self-tolerance passes.  This guards against
     *        the asymptotic error estimate silently underreporting the
     *        true error at extreme forward scattering with ultra-low
     *        photon energy.
     */
    enum class KernelOp { sigma, dsigma_dT };

    ComptonKernelSolver(
        double asymp_tau_alpha_threshold   = constants::ASYMP_TAU_ALPHA_THRESHOLD,
        double gamma_double_precision_safe = constants::GAMMA_DOUBLE_PRECISION_SAFE,
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
    double gamma_double_precision_safe_;
    double quadrature_self_tol_;
    double asymp_gamma_dd_threshold_;
    double asymp_self_tol_;
    double asymp_gamma_dd_cross_val_threshold_;

    ComptonKernelAsymptoticSeries asymp_series_;
    ComptonKernelAsymptoticSeries asymp_series_dd_;
    ComptonPowerSeries            power_series_;
    ComptonPowerSeries            power_series_dd_;
    ComptonKernelQuadrature       quadrature_;

    template <KernelOp Op>
    ComptonResult dispatch(double E, double E_prime, double xi,
                           double T, double Ne) const;
};

} // namespace compton

#endif
