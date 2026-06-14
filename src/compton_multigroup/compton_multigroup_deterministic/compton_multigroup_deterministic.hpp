#ifndef COMPTON_MULTIGROUP_DETERMINISTIC_HPP
#define COMPTON_MULTIGROUP_DETERMINISTIC_HPP
/**
 * @file compton_multigroup.hpp
 * @brief Planck-weighted multigroup-multiangle Compton scattering matrix.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * PHYSICS
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Given the point-wise differential scattering kernel Σ_E(E→E', ξ; T, Nₑ),
 * this module computes the multigroup-multiangle cross section:
 *
 *     σ(g→g', [μᵢ,μᵢ₊₁]; T) =
 *         2π ∫_{ΔEg} ∫_{ΔEg'} ∫_{μᵢ}^{μᵢ₊₁} w(E,T) Σ_E dμ dE' dE
 *         ─────────────────────────────────────────────────────────────
 *                        ∫_{ΔEg} w(E,T) dE
 *
 * where the weight function w(E, T),
 *
 * The 2π factor accounts for azimuthal symmetry (dΩ = 2π dμ), ensuring
 * that summing over all angle bins gives the total group-to-group cross
 * section, consistent with the CMMC Monte Carlo convention.
 *
 * Energy groups are defined by boundaries E_{1/2} < E_{3/2} < … < E_{G+1/2}
 * with centers at the geometric mean √(E_{g−1/2} E_{g+1/2}).
 * Angle bins divide [−1, 1] into N equal segments of width 2/N.
 *
 * The numerator is evaluated by tensor-product Gauss-Legendre quadrature
 * over the three finite intervals (E, E', μ).  The denominator is computed
 * by the weight function.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * UNITS AND API
 * ─────────────────────────────────────────────────────────────────────────
 *
 * - Energy group boundaries are in [erg].
 * - Temperature T is in [K], electron density Nₑ in [cm⁻³].
 * - The returned matrix entries have units [cm²] (Nₑ=1) or [1/cm].
 * - Angle-integrated overloads (no num_angle_bins) integrate μ over [−1,1].
 *
 * The dsigma_dT variants plug the derivative kernel ∂Σ_E/∂T into the same
 * weighted-integral formula.  They are NOT the full ∂σ/∂T of the multigroup
 * cross section (which would need quotient-rule terms).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REFERENCE
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   B. A. Clark, "Computing multigroup radiation integrals using
 *   polylogarithm-based methods," JCP 70(2):311–329, 1987.
 */

#include "utilities/gauss_legendre.hpp"
#include "utilities/units.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace compton {

namespace constants {
/// Temperature [K] below which the multigroup integrator switches to
/// cold_temperature_order for the E and mu axes (0.005 keV).
constexpr double COLD_TEMPERATURE_THRESHOLD = 0.005 * units::kev_kelvin;
} // namespace constants

/**
 * @brief Consolidated configuration for multigroup integration.
 *
 * Controls the GL order for each integration axis and E' sub-region,
 * adaptive refinement depth, overall tolerance, and the outward-from-peak
 * group cutoff ratio.  All parameter validation is performed by the
 * constructor so that invalid configurations are rejected early.
 */
struct MGIntegrationConfig {
    int base_order;
    int cold_temperature_order;
    int peak_max_depth;
    std::optional<int> tail_order;
    std::optional<int> far_order;
    std::optional<int> mu_order;
    double integration_tolerance;
    double cutoff_ratio;

    /**
     * @brief Construct with validated defaults.
     *
     * @param base_order              GL panel order for E and E'-peak axes.
     * @param integration_tolerance   Overall relative tolerance for the outer integral.
     * @param cutoff_ratio            Outward-from-peak early-termination ratio.
     * @param peak_max_depth          Maximum recursion depth for adaptive E' peak.
     * @param cold_temperature_order  GL order for E/mu when T < COLD_TEMPERATURE_THRESHOLD.
     * @param tail_order              GL order for E' tail regions (defaults to base_order).
     * @param far_order               GL order for E' far-from-peak regions (defaults to base_order).
     * @param mu_order                GL order for the μ axis (defaults to base_order).
     * @throws std::invalid_argument on invalid parameters.
     */
    MGIntegrationConfig(
        int base_order = 24,
        double integration_tolerance = 1e-3,
        double cutoff_ratio = 1e-8,
        int peak_max_depth = 5,
        int cold_temperature_order = 48,
        std::optional<int> tail_order = std::nullopt,
        std::optional<int> far_order = std::nullopt,
        std::optional<int> mu_order = std::nullopt);

    /** @brief Effective tail GL order (tail_order if set, otherwise base_order). */
    int effective_tail_order() const { return tail_order.value_or(base_order); }

    /** @brief Effective far GL order (far_order if set, otherwise base_order). */
    int effective_far_order() const { return far_order.value_or(base_order); }

    /** @brief Effective μ GL order (mu_order if set, otherwise base_order). */
    int effective_mu_order() const { return mu_order.value_or(base_order); }
};

/**
 * @brief Abstract base for kernel multipliers.
 *
 * A kernel multiplier f(E, E', mu, T, Ne) is an extra factor that multiplies
 * the differential scattering kernel pointwise inside the multigroup integral.
 * The result is *not* normalised by the integral of f itself, so it behaves
 * like an observable averaged against the scattering distribution.
 */
class KernelMultiplier {
public:
    virtual ~KernelMultiplier() = default;
    virtual double operator()(double E, double Ep, double mu, double T, double Ne) const = 0;
};

/**
 * @brief Identity multiplier: always returns 1.
 */
class ConstantMultiplier : public KernelMultiplier {
public:
    double operator()(double, double, double, double, double) const override {
        return 1.0;
    }
};

/**
 * @brief Computes the weighted multigroup-multiangle Compton scattering
 *        matrix by adaptive recursive Gauss-Legendre quadrature.
 *
 * Construct once with the energy group structure, weight function,
 * tolerance, and base quadrature order; then call compute_sigma_matrix /
 * compute_dsigma_dT_matrix at any temperature and angular resolution.
 *
 * The 3D integral is evaluated adaptively: each axis recursively bisects
 * until the relative error estimate is below its tolerance.  Inner axes
 * use progressively tighter tolerances (E: tol, E': tol*0.1, mu: tol*0.01).
 */
class ComptonMultigroupKernel {
public:
    /**
     * @brief Construct from energy group boundaries and a weight function.
     *
     * @param energy_group_boundaries  G+1 strictly increasing values [erg], all > 0.
     * @param weight_function          Shared pointer to a WeightFunction subclass.
     * @param config                   Integration configuration (tolerance, orders, cutoff).
     * @throws std::invalid_argument on invalid boundaries.
     */
    ComptonMultigroupKernel(
        std::vector<double> const& energy_group_boundaries,
        std::shared_ptr<WeightFunction const> weight_function,
        MGIntegrationConfig const& config);

    /** @brief Number of energy groups G. */
    int num_groups() const { return static_cast<int>(group_centers_.size()); }

    /** @brief Geometric-mean group centers √(E_lo · E_hi) [erg]. */
    std::vector<double> const& group_centers() const { return group_centers_; }

    /** @brief Energy group boundaries [erg], length G+1. */
    std::vector<double> const& group_boundaries() const { return group_boundaries_; }

    // ── Multigroup-multiangle (3D: G × G × N_angles) ────────────────────

    /**
     * @brief Compute the multigroup-multiangle σ matrix.
     *
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³].
     * @param multiplier      Pointwise kernel multiplier applied before integration.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_sigma_matrix(
        ComptonKernelSolver const& kernel,
        int num_angle_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the multigroup-multiangle ∂σ/∂T matrix.
     *
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³].
     * @param multiplier      Pointwise kernel multiplier applied before integration.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_dsigma_dT_matrix(
        ComptonKernelSolver const& kernel,
        int num_angle_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

private:
    /**
     * @brief Core driver: assemble the full G×G×num_angle_bins scattering matrix.
     *
     * Orchestrates the multigroup integration for every incoming group g.
     * For each g the method starts at the peak target group (the one
     * containing the geometric-mean energy of g) and expands outward,
     * stopping in each direction once the angle-summed magnitude drops
     * below group_cutoff_ratio_ × peak_value.
     *
     * Each (g, gp) pair is evaluated by compute_group_entry().
     *
     * **Tolerance hierarchy** (set once here and forwarded):
     *   - E  axis:  integration_tolerance_
     *   - E' axis:  integration_tolerance_ × 0.1
     *   - μ  axis:  integration_tolerance_ × 0.01
     *
     * @param kernel         Point-wise kernel evaluator.
     * @param eval           Pointer-to-member: sigma_E or dsigma_E_dT.
     * @param num_angle_bins Number of equal-width μ bins on [−1, 1].
     * @param T              Electron temperature [K].
     * @param Ne             Electron density [cm⁻³].
     * @param multiplier     Optional pointwise factor applied inside the integrand.
     * @return Flat row-major vector of size G×G×num_angle_bins,
     *         indexed as result[g * G * num_angle_bins + gp * num_angle_bins + a].
     */
    std::vector<double> compute_matrix_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        int num_angle_bins,
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Evaluate one (g → gp) block of the scattering matrix.
     *
     * For a fixed incoming energy group g and target energy group gp this
     * method evaluates the weighted 3D integral
     *
     *     S(g, gp, a) = 2π / D(g) ∫_{E_lo}^{E_hi} w(E,T)
     *                   ∫_{Ep_lo}^{Ep_hi} ∫_{mu_lo}^{mu_hi}
     *                   f(E,E',μ) · Σ_E(E,E',μ,T,Ne) dμ dE' dE
     *
     * for each angle bin a ∈ [0, num_angle_bins).  Here D(g) is the weight
     * function denominator for group g, and f is the optional multiplier.
     *
     * **Integration strategy (inside-out):**
     *   1. μ axis  -- adaptive Gauss-Legendre on [mu_lo, mu_hi].
     *   2. E' axis -- peak-aware three-region quadrature (peak / tail / far),
     *                 split around the thermally broadened cold-recoil band
     *                 returned by peak_limits().
     *   3. E axis  -- adaptive quadrature on [E_lo, E_hi] with mapping chosen
     *                 by the weight-function contrast:
     *                   • linear  when the group is narrow and the weight is
     *                             smooth across it,
     *                   • log     (clusters nodes near E_lo) when w(E_lo) ≫ w(E_hi),
     *                   • rlog    (clusters nodes near E_hi) when w(E_hi) ≫ w(E_lo).
     *                 The switch is governed by constants::LOG_E_RATIO_THRESHOLD.
     *
     * @return Sum of |S(g, gp, a)| over angle bins -- used by the
     *         outward-from-peak cutoff in compute_matrix_impl().
     *
     * @param kernel         Point-wise kernel evaluator.
     * @param eval           Pointer-to-member returning ComptonResult.
     * @param g              Incoming energy group index.
     * @param gp             Target (scattered) energy group index.
     * @param num_angle_bins Number of equal-width μ bins.
     * @param dmu            Bin width: 2 / num_angle_bins.
     * @param T              Electron temperature [K].
     * @param Ne             Electron density [cm⁻³].
     * @param peak_tol       Tolerance for adaptive E' integration inside the recoil band.
     * @param inv_denom      1 / D(g), precomputed weight-function denominator.
     * @param multiplier     Pointwise factor f(E, E', μ, T, Ne).
     * @param[in,out] result Flat output vector; entries at the (g, gp, a)
     *                       positions are written (row-major layout).
     */
    double compute_group_entry(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        int g,
        int gp,
        int num_angle_bins,
        double dmu,
        double T,
        double Ne,
        double peak_tol,
        double inv_denom,
        KernelMultiplier const& multiplier,
        GaussLegendreRule const& active_rule,
        GaussLegendreRule const& active_mu_rule,
        std::vector<double>& result) const;

    std::vector<double> group_boundaries_;
    std::vector<double> group_centers_;

    /// Shared weight function for the Planck/Wien/Uniform numerator and denominator.
    std::shared_ptr<WeightFunction const> weight_func_;

    /// GL rule for E and E'-peak axes.
    GaussLegendreRule base_rule_;
    /// GL rule for E axes when T < COLD_TEMPERATURE_THRESHOLD.
    GaussLegendreRule cold_rule_;
    /// GL rule for E' tail (log/rlog) sub-regions.
    GaussLegendreRule tail_rule_;
    /// GL rule for E' far-from-peak sub-regions.
    GaussLegendreRule far_rule_;
    /// GL rule for the μ (scattering-angle) axis.
    GaussLegendreRule mu_rule_;
    /// GL rule for μ when T < COLD_TEMPERATURE_THRESHOLD.
    GaussLegendreRule mu_cold_rule_;

    double integration_tolerance_;
    int peak_max_depth_;
    double group_cutoff_ratio_;
};

/**
 * @brief E' limits for peak-aware quadrature in an angle bin [mu_lo, mu_hi].
 *
 * Starts from the cold-electron recoil band:
 *
 *     E'(mu) = E / (1 + gamma * (1 - mu)),    gamma = E / (m_e c^2)
 *
 * and extends each edge by the thermal Doppler width
 *
 *     dE = E * sqrt(2 k_B T / m_e c^2)
 *
 * so that the peak-aware E' quadrature captures the kernel peak even when
 * E sits right at a group boundary.
 *
 * @param E     Incoming photon energy [erg].
 * @param mu_lo Lower edge of the mu bin.
 * @param mu_hi Upper edge of the mu bin.
 * @param T     Electron temperature [K].
 * @return      {lo, hi} in [erg], thermally broadened.
 */
inline std::pair<double, double> peak_limits(
    double const E,
    double const mu_lo,
    double const mu_hi,
    double const T)
{
    double const gamma = E / units::me_c2;
    double const tau = T * units::k_boltz / units::me_c2;
    double thermal_dE = E * std::sqrt(2.0 * tau);
    if (T < constants::COLD_TEMPERATURE_THRESHOLD)
        thermal_dE *= 5.0;
    return {E / (1.0 + gamma * (1.0 - mu_lo)) - thermal_dE,
            E / (1.0 + gamma * (1.0 - mu_hi)) + thermal_dE};
}

} // namespace compton

#endif
