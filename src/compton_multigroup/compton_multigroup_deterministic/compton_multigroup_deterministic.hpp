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
 *     σ(g→g', [ξᵢ,ξᵢ₊₁]; T) =
 *         2π ∫_{ΔEg} ∫_{ΔEg'} ∫_{ξᵢ}^{ξᵢ₊₁} w(E,T) Σ_E dξ dE' dE
 *         ─────────────────────────────────────────────────────────────
 *                        ∫_{ΔEg} w(E,T) dE
 *
 * where the weight function w(E, T),
 *
 * The 2π factor accounts for azimuthal symmetry (dΩ = 2π dξ), ensuring
 * that summing over all angle bins gives the total group-to-group cross
 * section, consistent with the CMMC Monte Carlo convention.
 *
 * Energy groups are defined by boundaries E_{1/2} < E_{3/2} < … < E_{G+1/2}
 * with centers at the geometric mean √(E_{g−1/2} E_{g+1/2}).
 * Angle bins divide [−1, 1] into N equal segments of width 2/N.
 *
 * The numerator is evaluated by tensor-product Gauss-Legendre quadrature
 * over the three finite intervals (E, E', ξ).  The denominator is computed
 * by the weight function.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * UNITS AND API
 * ─────────────────────────────────────────────────────────────────────────
 *
 * - Energy group boundaries are in [erg].
 * - Temperature T is in [K], electron density Nₑ in [cm⁻³].
 * - The returned matrix entries have units [cm²] (Nₑ=1) or [1/cm].
 * - Angle-integrated overloads (no num_angle_bins) integrate ξ over [−1,1].
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
/// cold_temperature_order for the E and ξ axes (0.005 keV).
constexpr double COLD_TEMPERATURE_THRESHOLD = 0.005 * units::kev_kelvin;
} // namespace constants

/// Density scaling strategy for the flat E' integration mode.
enum class FlatEpDensityMode { log_proportional, linear_proportional, points_per_decade };

/**
 * @brief Configuration for the density-based flat E' integration mode.
 *
 * When attached to MGIntegrationConfig, replaces the adaptive peak/tail/far
 * E' recursion with a single GL rule per target group whose order is
 * proportional to the group's width.
 */
struct FlatEpConfig {
    double density;
    int min_points;
    int max_points;
    FlatEpDensityMode mode;
    bool flat_E;

    FlatEpConfig(double density = 64.0,
                 int min_points = 8,
                 int max_points = 1024,
                 FlatEpDensityMode mode = FlatEpDensityMode::points_per_decade,
                 bool flat_E = true)
        : density(density), min_points(min_points),
          max_points(max_points), mode(mode),
          flat_E(flat_E) {}
};

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
    std::optional<int> xi_order;
    std::optional<int> xi_tail_order;
    double integration_tolerance;
    double cutoff_ratio;
    double xi_peak_k;
    std::optional<FlatEpConfig> flat_ep;

    /**
     * @brief Construct with validated defaults.
     *
     * @param base_order              GL panel order for E and E'-peak axes.
     * @param integration_tolerance   Overall relative tolerance for the outer integral.
     * @param cutoff_ratio            Outward-from-peak early-termination ratio.
     * @param peak_max_depth          Maximum recursion depth for adaptive E' peak.
     * @param cold_temperature_order  GL order for E/ξ when T < COLD_TEMPERATURE_THRESHOLD.
     * @param tail_order              GL order for E' tail regions (defaults to base_order).
     * @param far_order               GL order for E' far-from-peak regions (defaults to base_order).
     * @param xi_order                GL order for the ξ peak core (defaults to base_order).
     * @param xi_peak_k               Half-width of the ξ peak window in sigma units.
     * @param xi_tail_order           GL order for ξ tail sub-intervals (defaults to 16).
     * @param flat_ep                 Optional flat E' density config (disables adaptive E' recursion).
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
        std::optional<int> xi_order = std::nullopt,
        double xi_peak_k = 5.0,
        std::optional<int> xi_tail_order = std::nullopt,
        std::optional<FlatEpConfig> flat_ep = std::nullopt);

    /** @brief Effective tail GL order (tail_order if set, otherwise base_order). */
    int effective_tail_order() const { return tail_order.value_or(base_order); }

    /** @brief Effective far GL order (far_order if set, otherwise base_order). */
    int effective_far_order() const { return far_order.value_or(base_order); }

    /** @brief Effective ξ GL order (xi_order if set, otherwise base_order). */
    int effective_xi_order() const { return xi_order.value_or(base_order); }

    /** @brief Effective ξ tail GL order (xi_tail_order if set, otherwise 16). */
    int effective_xi_tail_order() const { return xi_tail_order.value_or(16); }

    /**
     * @brief High-accuracy adaptive config for cold temperatures (T < 0.1 keV).
     *
     * bo=192, pd=9, xi_order=512, xi_peak_k=5, xi_tail_order=24, tol=1e-8.
     * Achieves < 1e-4 row-sum accuracy (MC-noise limited at N=1e9).
     * Runtime: ~600-1300s per matrix depending on temperature.
     */
    static MGIntegrationConfig cold_adaptive() {
        return MGIntegrationConfig(
            192, 1e-8, 1e-12, 9, 192,
            192, 192, 512, 5.0, 24, std::nullopt);
    }

    /**
     * @brief High-accuracy flat E' config for warm temperatures (T >= 0.1 keV).
     *
     * bo=96, xi_order=96, xi_peak_k=10, flat_ep(d=512, ppd, max=8192).
     * flat_E=false (keeps boundary layers and peak-focused ξ).
     * Achieves < 1e-4 row-sum accuracy at T >= 1 keV.
     * Runtime: ~30-120s per matrix depending on temperature.
     */
    static MGIntegrationConfig warm_flat() {
        FlatEpConfig flat{512.0, 8, 8192, FlatEpDensityMode::points_per_decade, false};
        return MGIntegrationConfig(
            96, 1e-6, 1e-12, 7, 96,
            96, 96, 96, 5.0, 16, flat);
    }
};

/**
 * @brief Abstract base for kernel multipliers.
 *
 * A kernel multiplier f(E, E', ξ, T, Ne) is an extra factor that multiplies
 * the differential scattering kernel pointwise inside the multigroup integral.
 * The result is *not* normalised by the integral of f itself, so it behaves
 * like an observable averaged against the scattering distribution.
 */
class KernelMultiplier {
public:
    virtual ~KernelMultiplier() = default;
    virtual double operator()(double E, double Ep, double xi, double T, double Ne) const = 0;
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
 * use progressively tighter tolerances (E: tol, E': tol*0.1, ξ: tol*0.01).
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

    // ── ξ-bin integral for fixed (E, E') ────────────────────────────────
    //
    // Returns the raw kernel integral over each ξ bin:
    //   result[a] = ∫_{ξ_a}^{ξ_{a+1}} multiplier(E,E',ξ,T,Ne) · Σ_E dξ
    //
    // No 2π, no weight function, no denominator normalisation.

    /**
     * @brief Integrate σ_E over ξ bins for fixed (E, E').
     *
     * @param kernel        Point-wise kernel evaluator.
     * @param E             Incoming photon energy [erg], must be > 0.
     * @param Ep            Scattered photon energy [erg], must be > 0.
     * @param num_xi_bins   Number of equal-width bins on [−1, 1], must be >= 1.
     * @param T             Electron temperature [K].
     * @param Ne            Electron density [cm⁻³].
     * @param multiplier    Pointwise kernel multiplier.
     * @return Vector of length num_xi_bins.
     */
    std::vector<double> compute_xi_integral_sigma(
        ComptonKernelSolver const& kernel,
        double E, double Ep, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /** @brief Integrate ∂σ_E/∂T over ξ bins for fixed (E, E'). */
    std::vector<double> compute_xi_integral_dsigma_dT(
        ComptonKernelSolver const& kernel,
        double E, double Ep, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    // ── E' + ξ-bin integral for fixed E ─────────────────────────────────
    //
    // Returns the raw kernel integral over E' and each ξ bin:
    //   result[a] = ∫_{Ep_lo}^{Ep_hi} [∫_{ξ_a}^{ξ_{a+1}} multiplier · Σ_E dξ] dE'
    //
    // Uses adaptive peak-aware E' quadrature (no flat_ep mode).
    // No 2π, no weight function, no denominator normalisation.

    /**
     * @brief Integrate σ_E over E' and ξ bins for fixed E.
     *
     * @param kernel        Point-wise kernel evaluator.
     * @param E             Incoming photon energy [erg], must be > 0.
     * @param Ep_lo         Lower E' bound [erg], must be > 0.
     * @param Ep_hi         Upper E' bound [erg], must be > Ep_lo.
     * @param num_xi_bins   Number of equal-width bins on [−1, 1], must be >= 1.
     * @param T             Electron temperature [K].
     * @param Ne            Electron density [cm⁻³].
     * @param multiplier    Pointwise kernel multiplier.
     * @return Vector of length num_xi_bins.
     */
    std::vector<double> compute_Ep_xi_integral_sigma(
        ComptonKernelSolver const& kernel,
        double E, double Ep_lo, double Ep_hi, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /** @brief Integrate ∂σ_E/∂T over E' and ξ bins for fixed E. */
    std::vector<double> compute_Ep_xi_integral_dsigma_dT(
        ComptonKernelSolver const& kernel,
        double E, double Ep_lo, double Ep_hi, int num_xi_bins,
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
     *   - ξ  axis:  integration_tolerance_ × 0.01
     *
     * @param kernel         Point-wise kernel evaluator.
     * @param eval           Pointer-to-member: sigma_E or dsigma_E_dT.
     * @param num_angle_bins Number of equal-width ξ bins on [−1, 1].
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
     * @brief Integrate the kernel over a single ξ bin for fixed (E, E').
     *
     * Performs peak-focused GL quadrature with five branches:
     * elastic-like, peak-left, peak-right, three-region split, full-bin.
     *
     * @return ∫_{xi_lo}^{xi_hi} multiplier · kernel dξ
     */
    double integrate_xi_bin(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E, double Ep,
        double xi_lo, double xi_hi,
        double T, double Ne,
        KernelMultiplier const& multiplier,
        GaussLegendreRule const& active_xi_rule) const;

    /**
     * @brief Integrate the kernel over E' and a single ξ bin for fixed E.
     *
     * Computes peak_limits then dispatches to adaptive or flat E' quadrature,
     * calling integrate_xi_bin at each E' node.
     *
     * @param flat_ep_rule  If non-null, uses this GL rule for flat E' mode;
     *                      otherwise uses adaptive peak/tail/far splitting.
     * @return ∫_{Ep_lo}^{Ep_hi} [∫_{xi_lo}^{xi_hi} multiplier · kernel dξ] dE'
     */
    double integrate_Ep_xi_bin(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E,
        double Ep_lo, double Ep_hi,
        double xi_lo, double xi_hi,
        double T, double Ne,
        KernelMultiplier const& multiplier,
        GaussLegendreRule const& active_rule,
        GaussLegendreRule const& active_xi_rule,
        double peak_tol,
        GaussLegendreRule const* flat_ep_rule = nullptr) const;

    /**
     * @brief Evaluate one (g -> gp) block of the scattering matrix.
     *
     * Loops over angle bins, integrating w(E,T) * integrate_Ep_xi_bin
     * over the incoming E group with boundary-layer sub-panels.
     *
     * @return Sum of |S(g, gp, a)| over angle bins (for cutoff).
     */
    double compute_group_entry(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        int g,
        int gp,
        int num_angle_bins,
        double dxi,
        double T,
        double Ne,
        double peak_tol,
        double inv_denom,
        KernelMultiplier const& multiplier,
        GaussLegendreRule const& active_rule,
        GaussLegendreRule const& active_xi_rule,
        std::vector<double>& result) const;

    /** @brief Dispatch impl for compute_xi_integral_sigma / _dsigma_dT. */
    std::vector<double> compute_xi_integral_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E, double Ep, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /** @brief Dispatch impl for compute_Ep_xi_integral_sigma / _dsigma_dT. */
    std::vector<double> compute_Ep_xi_integral_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E, double Ep_lo, double Ep_hi, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

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
    /// GL rule for the ξ (scattering-angle) axis.
    GaussLegendreRule xi_rule_;
    /// GL rule for ξ when T < COLD_TEMPERATURE_THRESHOLD.
    GaussLegendreRule xi_cold_rule_;
    /// GL rule for ξ tails in peak-focused splitting (low order, tails are exponentially small).
    GaussLegendreRule xi_tail_rule_;

    /// Per-group GL rules for flat E' mode (empty when adaptive mode is active).
    std::vector<GaussLegendreRule> flat_ep_rules_;
    bool flat_E_ = false;

    double integration_tolerance_;
    double xi_peak_k_;
    int peak_max_depth_;
    double group_cutoff_ratio_;
};

/**
 * @brief Thermal Doppler half-width of the Compton kernel at energy E.
 *
 *     dE = E * sqrt(2 k_B T / m_e c^2)
 *
 * Below COLD_TEMPERATURE_THRESHOLD a 5x multiplier is applied to ensure
 * the extremely narrow recoil band is fully captured.
 *
 * @param E  Photon energy [erg].
 * @param T  Electron temperature [K].
 * @return   Thermal half-width [erg].
 */
inline double thermal_half_width(double const E, double const T)
{
    double const tau = T * units::k_boltz / units::me_c2;
    double dE = E * std::sqrt(2.0 * tau);
    if (T < constants::COLD_TEMPERATURE_THRESHOLD)
        dE *= 5.0;
    return dE;
}

/**
 * @brief E' limits for peak-aware quadrature in an angle bin [xi_lo, xi_hi].
 *
 * Starts from the cold-electron recoil band:
 *
 *     E'(ξ) = E / (1 + gamma * (1 - ξ)),    gamma = E / (m_e c^2)
 *
 * and extends each edge by thermal_half_width(E, T).
 *
 * @param E     Incoming photon energy [erg].
 * @param xi_lo Lower edge of the ξ bin.
 * @param xi_hi Upper edge of the ξ bin.
 * @param T     Electron temperature [K].
 * @return      {lo, hi} in [erg], thermally broadened.
 */
inline std::pair<double, double> peak_limits(
    double const E,
    double const xi_lo,
    double const xi_hi,
    double const T)
{
    double const gamma = E / units::me_c2;
    double const dE = thermal_half_width(E, T);
    return {E / (1.0 + gamma * (1.0 - xi_lo)) - dE,
            E / (1.0 + gamma * (1.0 - xi_hi)) + dE};
}

} // namespace compton

#endif
