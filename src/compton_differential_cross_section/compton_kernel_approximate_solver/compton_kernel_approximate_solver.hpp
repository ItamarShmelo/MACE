#ifndef COMPTON_KERNEL_APPROXIMATE_SOLVER_HPP
#define COMPTON_KERNEL_APPROXIMATE_SOLVER_HPP

/**
 * @file compton_kernel_approximate_solver.hpp
 * @brief Fast three-regime Compton kernel dispatcher.
 *
 * The dispatch is deliberately limited to three cases:
 *
 *   1. asymptotic when the cold series is fastest, or when a cold point fails
 *      the explicit approximation gate
 *   2. approximate when the explicit-coefficient and Padé gates both pass
 *   3. power series otherwise
 *
 * In boolean form:
 *
 *   asymptotic = cold && (fast_asymptotic || !approximate_accurate)
 *   approximate = approximate_accurate
 *   power = !cold && !approximate_accurate
 *
 * Both series cases retain double-double fallbacks when binary64 does not pass
 * its numerical quality gate.
 */

#include "compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.hpp"
#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"
#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"

namespace compton {

namespace approximate_solver_constants {
constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.035;
constexpr double FAST_ASYMP_TAU_THRESHOLD = 0.02;
constexpr double FAST_ASYMP_MIN_GAMMA = 1e-4;
constexpr double APPROXIMATE_PADE_DISAGREEMENT_THRESHOLD = 3e-4;
constexpr double APPROXIMATE_DERIVATIVE_PADE_DISAGREEMENT_THRESHOLD = 1e-6;
constexpr double DOUBLE_SERIES_SELF_TOL = 5e-3;
constexpr double DOUBLE_DERIVATIVE_SERIES_SELF_TOL = 1e-7;
constexpr double DD_SERIES_SELF_TOL = 5e-3;
} // namespace approximate_solver_constants

class ComptonKernelApproximateSolver : public ComptonKernelSolver {
  public:
    enum class KernelOp { sigma, dsigma_dT };

    explicit ComptonKernelApproximateSolver(bool verbose = false);

    [[nodiscard]] ComptonResult
    sigma_E(double E, double E_prime, double xi, double T) const override;

    [[nodiscard]] ComptonResult
    dsigma_E_dT(double E, double E_prime, double xi, double T) const override;

  private:
    bool verbose_;
    ComptonKernelApproximate approximate_;
    ComptonKernelAsymptoticSeries asymptotic_;
    ComptonKernelAsymptoticSeries asymptotic_dd_;
    ComptonPowerSeries power_;
    ComptonPowerSeries power_dd_;

    template <KernelOp Op>
    [[nodiscard]] ComptonResult
    dispatch(double E, double E_prime, double xi, double T) const;
};

} // namespace compton

#endif
