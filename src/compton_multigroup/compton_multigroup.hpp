#ifndef COMPTON_MULTIGROUP_HPP
#define COMPTON_MULTIGROUP_HPP
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

#include "compton_multigroup/gauss_legendre.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_kernel_solver/compton_kernel_solver.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>
#include <numbers>

namespace compton {

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
    double integration_tolerance;
    double cutoff_ratio;

    /**
     * @brief Construct with validated defaults.
     *
     * @param base_order              GL panel order for E, mu, and E'-peak axes.
     * @param integration_tolerance   Overall relative tolerance for the outer integral.
     * @param cutoff_ratio            Outward-from-peak early-termination ratio.
     * @param peak_max_depth          Maximum recursion depth for adaptive E' peak.
     * @param cold_temperature_order  GL order for E/mu when T < COLD_TEMPERATURE_THRESHOLD.
     * @param tail_order              GL order for E' tail regions (defaults to base_order).
     * @param far_order               GL order for E' far-from-peak regions (defaults to base_order).
     * @throws std::invalid_argument on invalid parameters.
     */
    MGIntegrationConfig(
        int base_order = 24,
        double integration_tolerance = 1e-3,
        double cutoff_ratio = 1e-8,
        int peak_max_depth = 5,
        int cold_temperature_order = 48,
        std::optional<int> tail_order = std::nullopt,
        std::optional<int> far_order = std::nullopt);

    /** @brief Effective tail GL order (tail_order if set, otherwise base_order). */
    int effective_tail_order() const { return tail_order.value_or(base_order); }

    /** @brief Effective far GL order (far_order if set, otherwise base_order). */
    int effective_far_order() const { return far_order.value_or(base_order); }
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
        MGIntegrationConfig const& config = MGIntegrationConfig{});

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
     * @tparam KernelT        Kernel type (e.g. ComptonKernelSolver, ComptonKernelQuadrature).
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³].
     * @param multiplier      Pointwise kernel multiplier applied before integration.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    template<typename KernelT>
    std::vector<double> compute_sigma_matrix(
        KernelT const& kernel,
        int num_angle_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the multigroup-multiangle ∂σ/∂T matrix.
     *
     * @tparam KernelT        Kernel type (e.g. ComptonKernelSolver, ComptonKernelQuadrature).
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³].
     * @param multiplier      Pointwise kernel multiplier applied before integration.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    template<typename KernelT>
    std::vector<double> compute_dsigma_dT_matrix(
        KernelT const& kernel,
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
     * @tparam KernelT  Kernel class whose @p eval member returns SigmaResult.
     * @param kernel         Point-wise kernel evaluator.
     * @param eval           Pointer-to-member: sigma_E or dsigma_E_dT.
     * @param num_angle_bins Number of equal-width μ bins on [−1, 1].
     * @param T              Electron temperature [K].
     * @param Ne             Electron density [cm⁻³].
     * @param multiplier     Optional pointwise factor applied inside the integrand.
     * @return Flat row-major vector of size G×G×num_angle_bins,
     *         indexed as result[g * G * num_angle_bins + gp * num_angle_bins + a].
     */
    template<typename KernelT>
    std::vector<double> compute_matrix_impl(
        KernelT const& kernel,
        SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
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
     * @param eval           Pointer-to-member returning SigmaResult.
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
    template<typename KernelT>
    double compute_group_entry(
        KernelT const& kernel,
        SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
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
        std::vector<double>& result) const;

    std::vector<double> group_boundaries_;
    std::vector<double> group_centers_;

    /// Shared weight function for the Planck/Wien/Uniform numerator and denominator.
    std::shared_ptr<WeightFunction const> weight_func_;

    /// GL rule for E, mu, and E'-peak axes.
    GaussLegendreRule base_rule_;
    /// GL rule for E/mu axes when T < COLD_TEMPERATURE_THRESHOLD.
    GaussLegendreRule cold_rule_;
    /// GL rule for E' tail (log/rlog) sub-regions.
    GaussLegendreRule tail_rule_;
    /// GL rule for E' far-from-peak sub-regions.
    GaussLegendreRule far_rule_;

    double integration_tolerance_;
    int peak_max_depth_;
    double group_cutoff_ratio_;
};

} // namespace compton

#endif
