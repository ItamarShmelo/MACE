#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Adaptive dispatch kernel for Compton scattering.
 *
 * ComptonKernelSolver selects the fastest accurate method at each
 * phase-space point:
 *
 *   1. Asymptotic series    -- when tau * max(alpha+, alpha-) is small
 *   2. Double power series  -- when min(gamma, gamma') is large enough
 *                              for double precision
 *   3. Gauss-Laguerre Q64   -- in the DD regime, accepted when its
 *                              self-reported error is below a tolerance
 *   4. DD power series      -- fallback when quadrature is unconverged
 *
 * All dispatch thresholds are configurable at construction time.
 */

#include "compton_kernel_series/compton_kernel_series.hpp"
#include "compton_kernel_quadrature/compton_kernel_quadrature.hpp"

namespace compton {

class ComptonKernelSolver {
public:
    /**
     * @param asymp_tau_alpha_threshold  Dispatch to asymptotic series when
     *        tau * max(alpha+, alpha-) falls below this value.
     * @param gamma_double_precision_safe  Use double power series when
     *        min(gamma, gamma') is at or above this value.
     * @param quadrature_self_tol  Accept the quadrature result when its
     *        self-reported relative error is below this tolerance.
     */
    ComptonKernelSolver(
        double asymp_tau_alpha_threshold   = 0.025,
        double gamma_double_precision_safe = 0.02,
        double quadrature_self_tol         = 1e-6);

    SigmaResult sigma_E(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

    SigmaResult dsigma_E_dT(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

private:
    double asymp_tau_alpha_threshold_;
    double gamma_double_precision_safe_;
    double quadrature_self_tol_;

    ComptonKernelSeries   series_asymp_;
    ComptonKernelSeries   series_double_;
    ComptonKernelSeries   series_dd_;
    ComptonKernelQuadrature quadrature_;
};

} // namespace compton

#endif
