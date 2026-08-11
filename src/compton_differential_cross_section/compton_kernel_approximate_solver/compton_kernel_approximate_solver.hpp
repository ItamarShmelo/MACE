#ifndef COMPTON_KERNEL_APPROXIMATE_SOLVER_HPP
#define COMPTON_KERNEL_APPROXIMATE_SOLVER_HPP

#include "compton_common/compton_common.hpp"
#include "compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"

namespace compton {

/**
 * @brief Adaptive Compton kernel that tries the fast KG5 approximation
 * first, falling back to the full solver when outside the approximation's
 * validity domain.
 *
 * Dispatch logic for sigma_E:
 *   1. If gamma >= gamma_tau_ratio * tau AND tau <= tau_max, call KG5.
 *      Accept the result unless it is the failure sentinel.
 *   2. Otherwise (or on KG5 failure), delegate to ComptonKernelSolver.
 *
 * For dsigma_E_dT, always delegates to ComptonKernelSolver (KG5 does not
 * provide a temperature derivative).
 *
 * Typical performance: ~400 ns in the KG5 domain (vs 2000-10000 ns for
 * the full solver in power-series regimes). Falls back at ~0 extra cost
 * when KG5 is not applicable.
 */
class ComptonKernelApproximateSolver {
  public:
    /**
     * @param gamma_tau_ratio  Minimum E/(kT) ratio for KG5 dispatch.
     *        Default 3.0 guarantees < 1% error for all xi.
     * @param tau_max  Maximum dimensionless temperature (kT/me_c2) for
     *        KG5 dispatch. Default 0.098 corresponds to T ≈ 50 keV.
     * @param solver_args  Arguments forwarded to ComptonKernelSolver.
     */
    ComptonKernelApproximateSolver(
        double gamma_tau_ratio = 3.0,
        double tau_max = 0.098,
        double asymp_tau_alpha_threshold = constants::ASYMP_TAU_ALPHA_THRESHOLD,
        double power_series_self_tol = 1e-7,
        double asymp_self_tol = 1e-7,
        double dd_power_series_self_tol = 0.5,
        double dd_asymp_self_tol = 0.5,
        bool verbose = false);

    ComptonResult
    sigma_E(double E, double E_prime, double xi, double T) const;

    ComptonResult
    dsigma_E_dT(double E, double E_prime, double xi, double T) const;

  private:
    double gamma_tau_ratio_;
    double tau_max_;

    ComptonKernelApproximate approximate_;
    ComptonKernelSolver solver_;
};

} // namespace compton

#endif
