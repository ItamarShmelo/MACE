#ifndef COMPTON_KERNEL_SOLVER_HPP
#define COMPTON_KERNEL_SOLVER_HPP
/**
 * @file compton_kernel_solver.hpp
 * @brief Adaptive solver for the Compton scattering kernel.
 *
 * Dispatches between asymptotic series and power series based on the
 * dimensionless product tau * alpha_max.  When tau * alpha_max is below
 * constants::ASYMP_TAU_ALPHA_THRESHOLD the asymptotic expansion is used;
 * otherwise the power series (which internally selects double or
 * double-double precision) handles the evaluation.
 */

#include "compton_common/compton_common.hpp"
#include "compton_kernel_series/compton_kernel_series.hpp"

namespace compton {

class ComptonKernelSolver {
public:
    ComptonKernelSolver();

    /**
     * @brief Evaluate the Compton scattering kernel Sigma_E(E -> E', xi; tau, Ne).
     *
     * @param E        Incident photon energy [erg]
     * @param E_prime  Scattered photon energy [erg]
     * @param xi       cos(scattering angle), strictly in (-1, 1)
     * @param tau      Dimensionless electron temperature kT/(m_e c^2)
     * @param Ne       Electron number density [cm^-3] (use 1.0 for microscopic)
     * @return SigmaResult with value and error estimates
     */
    SigmaResult sigma_E(
        double const E,
        double const E_prime,
        double const xi,
        double const tau,
        double const Ne) const;

private:
    /// Relative convergence tolerance for series term-decay checks.
    static constexpr double SERIES_EPS_REL = 1e-12;
    /// Minimum number of terms before convergence is tested.
    static constexpr int SERIES_N_MIN = 4;
    /// Maximum number of series terms.
    static constexpr int SERIES_N_MAX = 200;

    /// Asymptotic series for the low tau*alpha regime.
    ComptonKernelSeries asymptotic_series_;
    /// Power series with automatic precision selection for the general regime.
    ComptonKernelSeries power_series_;
};

} // namespace compton

#endif
