#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Robust adaptive solver for the Compton scattering kernel.
 *
 * Cascades through asymptotic series, power series, and Gauss-Laguerre
 * quadrature to achieve empirically validated 1e-8 relative error on the
 * calibrated domain while minimizing use of the expensive quadrature method.
 *
 * Dispatch threshold (tau_alpha_max < 0.2) decides what to try first.
 * Validity threshold (estimated_rel_error < target) decides acceptance.
 */

#include "compton_common/compton_common.hpp"
#include "compton_kernel_series/compton_kernel_series.hpp"
#include "compton_kernel_quadrature/compton_kernel_quadrature.hpp"

namespace compton {

enum class SolverMethod {
    Asymptotic,
    PowerSeries,
    Quadrature
};

struct SolverResult {
    double value;
    double estimated_abs_error;
    double estimated_rel_error;
    int terms_used;
    SolverMethod method_used;
    bool used_fallback;      ///< reserved (always false; asymptotic fallback removed)
    bool target_met;         ///< true if accepted path achieved rel_error < target (or negligible)
    bool clamped;            ///< true if negative value was clamped to zero
    double tau_alpha_max;    ///< diagnostic: max(tau*alpha_plus, tau*alpha_minus)
    double conditioning;     ///< diagnostic: power-series conditioning number (1.0 if not applicable)
};

class ComptonKernelSolver {
public:
    /**
     * @param target_rel_tol  Relative error target (default 1e-8)
     * @param target_abs_tol  Absolute value below which kernel is considered negligible (default 1e-300)
     */
    ComptonKernelSolver(double target_rel_tol = 1e-8,
                        double target_abs_tol = 1e-300);

    SolverResult sigma_E(double E, double E_prime, double xi, double tau, double Ne) const;

private:
    double target_rel_tol_;
    double target_abs_tol_;
    ComptonKernelSeries series_;
    ComptonKernelQuadrature quad256_;

    static constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.2;
    static constexpr double REL_ERROR_FLOOR = 1e-300;
};

} // namespace compton

#endif
