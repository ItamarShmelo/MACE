#ifndef COMPTON_MULTIGROUP_MONTE_CARLO_HPP
#define COMPTON_MULTIGROUP_MONTE_CARLO_HPP
/**
 * @file compton_monte_carlo.hpp
 * @brief Monte Carlo multigroup-multiangle Compton scattering matrix.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * PHYSICS
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Evaluates the same multigroup-multiangle scattering matrix as
 * ComptonMultigroupKernel, but via Monte Carlo integration of the
 * relativistic Compton kernel:
 *
 *     σ(g→g', [ξᵢ,ξᵢ₊₁]; T) =
 *         ∫_{ΔEg} w(E,T) Σ_KN(E→E',ξ; T) dE
 *         ───────────────────────────────────────
 *                  ∫_{ΔEg} w(E,T) dE
 *
 * The kernel is evaluated by direct MC sampling of the Klein-Nishina
 * differential cross section over thermal (Maxwell-Jüttner) electrons
 * with full relativistic Lorentz transforms.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * UNITS AND API
 * ─────────────────────────────────────────────────────────────────────────
 *
 * - Energy group boundaries are in [erg].
 * - Temperature T is in [K], electron density Nₑ in [cm⁻³].
 * - The returned matrix is *microscopic* [cm²] (per free electron).
 *   Nₑ is forwarded to the KernelMultiplier only; it does not scale
 *   the output.
 * - Angle-integrated overloads (no num_angle_bins) integrate ξ over
 *   [−1,1].
 */

#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/weight_function.hpp"

#include <boost/random.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace compton {

/**
 * @brief Configuration for Monte Carlo integration.
 *
 * @param num_samples         Number of MC samples per temperature evaluation.
 * @param seed                RNG seed; -1 for time-based.
 * @param discard_out_of_grid When true, scattered photons falling outside
 *                            the energy grid are discarded instead of
 *                            clamped to the nearest group.
 */
struct MCIntegrationConfig {
    std::size_t num_samples;
    int seed;
    bool discard_out_of_grid;

    MCIntegrationConfig(
        std::size_t num_samples = 1'000'000,
        int seed = -1,
        bool discard_out_of_grid = true);
};

/**
 * @brief Computes the multigroup-multiangle Compton scattering matrix
 *        by Monte Carlo integration of the Klein-Nishina kernel over
 *        thermal Maxwell-Jüttner electrons.
 *
 * Construct once with the energy group structure, weight function,
 * and MC config; then call compute_sigma_matrix at any temperature.
 *
 * Thread safety: compute methods are safe to call from one thread at a time.
 * With OpenMP enabled, the internal MC loop is parallelized automatically.
 * Same (seed, OMP_NUM_THREADS) pair produces statistically identical results;
 * bitwise identity is not guaranteed due to implementation-defined reduction
 * merge order.
 */
class ComptonMonteCarloKernel {
  public:
    /**
     * @brief Construct from energy group boundaries and a weight function.
     *
     * @param energy_group_boundaries  G+1 strictly increasing values [erg], all
     * > 0.
     * @param weight_function          Shared pointer to a WeightFunction
     * subclass.
     * @param config                   MC integration configuration.
     * @throws std::invalid_argument on invalid boundaries or config.
     */
    ComptonMonteCarloKernel(
        std::vector<double> const& energy_group_boundaries,
        std::shared_ptr<WeightFunction const> weight_function,
        MCIntegrationConfig const& config = MCIntegrationConfig{});

    /** @brief Number of energy groups G. */
    int num_groups() const { return static_cast<int>(group_centers_.size()); }

    /** @brief Geometric-mean group centers √(E_lo · E_hi) [erg]. */
    std::vector<double> const& group_centers() const { return group_centers_; }

    /** @brief Energy group boundaries [erg], length G+1. */
    std::vector<double> const& group_boundaries() const
    {
        return group_boundaries_;
    }

    /**
     * @brief Compute the multigroup-multiangle σ matrix via MC.
     *
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³] (forwarded to multiplier
     * only).
     * @param multiplier      Pointwise kernel multiplier applied before
     * accumulation.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_sigma_matrix(
        int num_angle_bins,
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the angle-integrated σ matrix via MC.
     *
     * Equivalent to compute_sigma_matrix(1, T, Ne, multiplier) reshaped to G×G.
     */
    std::vector<double> compute_sigma_matrix(
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the multigroup-multiangle ∂σ/∂T matrix via MC.
     *
     * Each MC sample's Klein-Nishina contribution is scaled by the
     * derivative weight (λ − κ)/τ² − 3/τ before accumulation, and the
     * result is multiplied by dτ/dT.  This is NOT the full d/dT of
     * compute_sigma_matrix (which would need quotient-rule terms); it
     * matches ComptonMultigroupKernel::compute_kernel_derivative_contribution.  Use
     * compute_full_dsigma_dT_matrix for the complete derivative including
     * weight-function and denominator temperature dependence.
     *
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³] (forwarded to multiplier
     * only).
     * @param multiplier      Pointwise kernel multiplier applied before
     * accumulation.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_kernel_derivative_contribution(
        int num_angle_bins,
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the angle-integrated ∂σ/∂T matrix via MC.
     *
     * Equivalent to compute_kernel_derivative_contribution(1, T, Ne, multiplier) reshaped to
     * G×G.
     */
    std::vector<double> compute_kernel_derivative_contribution(
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the full d/dT of the multigroup cross section via MC.
     *
     * Applies the quotient rule to account for weight-function and
     * denominator temperature dependence.  Three independent mc_integrate
     * calls are composed; the denominator correction uses the analytic
     * dD/dT / D ratio.
     *
     * The user-provided multiplier is NOT differentiated with respect to T.
     */
    std::vector<double> compute_full_dsigma_dT_matrix(
        int num_angle_bins,
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    std::vector<double> compute_full_dsigma_dT_matrix(
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

  private:
    /**
     * @brief Core MC integration loop parameterized by a multiplier callable.
     *
     * @tparam MultiplierFn  Callable (E0, E, xi, T, Ne, lam) -> double.
     *         For plain sigma this wraps KernelMultiplier (ignoring lam).
     *         For dsigma/dT it wraps KernelMultiplier times the derivative
     *         weight ((λ − κ)/τ² − 3/τ) · dτ/dT.
     */
    template <typename MultiplierFn>
    std::vector<double> mc_integrate(
        int num_angle_bins,
        double T,
        double Ne,
        MultiplierFn const& multiplier_fn) const;

    std::vector<double> group_boundaries_;
    std::vector<double> group_centers_;
    std::vector<double> group_widths_;

    std::shared_ptr<WeightFunction const> weight_func_;

    std::size_t num_samples_;
    bool discard_out_of_grid_;

    mutable boost::random::mt19937_64 rng_;
    mutable boost::random::uniform_01<> uniform_dist_;
};

} // namespace compton

#endif
