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
#include <cassert>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace compton {

/// Coordinate mapping for an E-axis integration panel.
enum class EPanelMap {
    Linear,     ///< legendre_integrate (uniform node spacing)
    LogLower,   ///< log_legendre_integrate (nodes clustered near lower end)
    LogUpper    ///< rlog_legendre_integrate (nodes clustered near upper end)
};

/// Descriptor for one E-axis integration sub-panel.
struct EPanel {
    double lo;       ///< Lower panel boundary [erg].
    double hi;       ///< Upper panel boundary [erg].
    EPanelMap map;   ///< Coordinate mapping for this panel.
};

/**
 * @brief Consolidated configuration for multigroup integration.
 *
 * Controls the GL order for each integration axis and E' sub-region,
 * the ridge-based E' truncation parameters, and
 * the outward-from-peak group cutoff ratio.  All parameter validation
 * is performed by the constructor so that invalid configurations are
 * rejected early.
 */
struct MGIntegrationConfig {
    std::optional<int> xi_order;
    std::optional<int> xi_tail_order;
    double cutoff_ratio;
    double xi_peak_k;

    double ep_k_cut;
    double ep_k_in;
    std::optional<int> ep_edge_order;
    std::optional<int> ep_interior_order;

    std::optional<int> e_panel_order;
    double log_e_panel_ratio;
    double e_boundary_k;

    /**
     * @brief Construct with validated defaults.
     *
     * @param cutoff_ratio            Outward-from-peak early-termination ratio.
     * @param xi_order                GL order for the ξ peak core (defaults to 48).
     * @param xi_peak_k               Half-width of the ξ peak window in sigma units.
     * @param xi_tail_order           GL order for ξ tail sub-intervals (defaults to 16).
     * @param ep_k_cut                E' truncation width in sigma units (must be > 0).
     * @param ep_k_in                 E' interior-edge separator in sigma units (must be >= 0, < ep_k_cut).
     * @param ep_edge_order           GL order for E' edge regions (defaults to 24).
     * @param ep_interior_order       GL order for E' ridge interior (defaults to 24).
     * @param e_panel_order           GL order for E-axis sub-panels (defaults to 12).
     * @param log_e_panel_ratio       Panel width ratio threshold for log/rlog mapping (must be > 1).
     * @param e_boundary_k            E-panel boundary-layer width in sigma units (must be > 0).
     * @throws std::invalid_argument on invalid parameters.
     */
    MGIntegrationConfig(
        double cutoff_ratio = 1e-8,
        std::optional<int> xi_order = std::nullopt,
        double xi_peak_k = 5.0,
        std::optional<int> xi_tail_order = std::nullopt,
        double ep_k_cut = 5.0,
        double ep_k_in = 2.0,
        std::optional<int> ep_edge_order = std::nullopt,
        std::optional<int> ep_interior_order = std::nullopt,
        std::optional<int> e_panel_order = std::nullopt,
        double log_e_panel_ratio = 2.0,
        double e_boundary_k = 5.0);

    /** @brief Effective ξ GL order (xi_order if set, otherwise 48). */
    int effective_xi_order() const { return xi_order.value_or(48); }

    /** @brief Effective ξ tail GL order (xi_tail_order if set, otherwise 16). */
    int effective_xi_tail_order() const { return xi_tail_order.value_or(16); }

    /** @brief Effective E' edge GL order (ep_edge_order if set, otherwise 24). */
    int effective_ep_edge_order() const { return ep_edge_order.value_or(24); }

    /** @brief Effective E' interior GL order (ep_interior_order if set, otherwise 24). */
    int effective_ep_interior_order() const { return ep_interior_order.value_or(24); }

    /** @brief Effective E-axis per-panel GL order (e_panel_order if set, otherwise 12). */
    int effective_e_panel_order() const { return e_panel_order.value_or(12); }
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
 *        matrix by fixed-order Gauss-Legendre quadrature.
 *
 * Construct once with the energy group structure, weight function,
 * and quadrature configuration; then call compute_sigma_matrix /
 * compute_dsigma_dT_matrix at any temperature and angular resolution.
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
    int num_groups() const { return static_cast<int>(group_boundaries_.size()) - 1; }

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
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Integrate the kernel over E' and a single ξ bin for fixed E.
     *
     * Computes ridge bounds then dispatches to integrate_Ep_ridge,
     * calling integrate_xi_bin at each E' node.
     *
     * @return ∫_{Ep_lo}^{Ep_hi} [∫_{xi_lo}^{xi_hi} multiplier · kernel dξ] dE'
     */
    double integrate_Ep_xi_bin(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E,
        double Ep_lo, double Ep_hi,
        double xi_lo, double xi_hi,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Evaluate one (g -> gp) block of the scattering matrix.
     *
     * Loops over angle bins, integrating w(E,T) * integrate_Ep_xi_bin
     * over the incoming E group using feature-aware E-axis panels.
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
        double inv_denom,
        KernelMultiplier const& multiplier,
        std::vector<EPanel> const& panels,
        std::vector<double>& result) const;

    /// Build E-axis panel descriptors for incoming group g.
    /// Splits at the weight-function peak (if inside the group),
    /// then assigns per-panel coordinate mappings
    /// (Linear, LogLower, or LogUpper).
    std::vector<EPanel> compute_E_panels(int g, double T) const;

public:
    /** @brief Integrate the kernel over ξ bins for fixed (E, E'). */
    std::vector<double> compute_xi_integral_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E, double Ep, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

    /** @brief Integrate the kernel over E' and ξ bins for fixed E. */
    std::vector<double> compute_Ep_xi_integral_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
        double E, double Ep_lo, double Ep_hi, int num_xi_bins,
        double T, double Ne,
        KernelMultiplier const& multiplier) const;

private:
    std::vector<double> group_boundaries_;

    /// Shared weight function for the Planck/Wien/Uniform numerator and denominator.
    std::shared_ptr<WeightFunction const> weight_func_;

    /// GL rule for the ξ (scattering-angle) axis.
    GaussLegendreRule xi_rule_;
    /// GL rule for ξ tails in peak-focused splitting (low order, tails are exponentially small).
    GaussLegendreRule xi_tail_rule_;

    /// GL rule for E' left and right edge regions.
    GaussLegendreRule ep_edge_rule_;
    /// GL rule for the E' ridge interior region.
    GaussLegendreRule ep_interior_rule_;
    /// Fixed low-order GL rule for the double-peak elastic-core panel.
    GaussLegendreRule ep_elastic_core_rule_;

    /// GL rule for E-axis sub-panels (per panel, typically 12 points).
    GaussLegendreRule e_panel_rule_;

    double xi_peak_k_;
    double ep_k_cut_;
    double ep_k_in_;
    double group_cutoff_ratio_;

    /// Panel width ratio threshold for switching to log/rlog-E quadrature.
    double log_e_panel_ratio_;

    /// Boundary-layer width multiplier for E-panel edge splitting.
    double e_boundary_k_;
};

/**
 * @brief Local thermal width of the Compton ridge in E' at scattering angle xi.
 *
 * Derived from the curvature of lambda_+ at the cold-Compton saddle:
 *
 *     sigma_gamma'(xi) = gamma / [1+gamma(1-xi)]^2
 *                        * sqrt(tau(1-xi) [2 + 2gamma(1-xi) + gamma^2(1-xi)])
 *
 * Converted to energy units: sigma_E'(xi) = sigma_gamma'(xi) * m_e c^2.
 *
 * @param E   Incoming photon energy [erg].
 * @param xi  Cosine of the scattering angle, in [-1, 1).
 * @param T   Electron temperature [K].
 * @return    Thermal width in E' [erg]. Zero when T <= 0 or xi >= 1.
 */
inline double ridge_thermal_width(double const E, double const xi, double const T)
{
    double const gamma = E / units::me_c2;
    double const tau   = T * units::k_boltz / units::me_c2;
    double const u     = std::max(0.0, 1.0 - xi);
    if (tau <= 0.0 || u <= 0.0) return 0.0;
    double const d     = 1.0 + gamma * u;
    return (E / (d * d))
         * std::sqrt(tau * u * (2.0 + 2.0 * gamma * u + gamma * gamma * u));
}

/**
 * @brief Cold-Compton ridge endpoints and local thermal widths for a xi bin.
 *
 * All values are in energy units [erg].
 */
struct RidgeBounds {
    double cold_lo;    ///< E'_cold(xi_lo) = E / (1 + gamma*(1-xi_lo)) [erg]
    double cold_hi;    ///< E'_cold(xi_hi) = E / (1 + gamma*(1-xi_hi)) [erg]
    double sigma_lo;   ///< ridge_thermal_width(E, xi_lo, T) [erg]
    double sigma_hi;   ///< ridge_thermal_width(E, xi_hi, T) [erg]
};

/**
 * @brief Compute ridge bounds for peak-aware E' quadrature.
 *
 * All callers guarantee E > 0 (enforced by group boundary validation).
 *
 * @param E     Incoming photon energy [erg].
 * @param xi_lo Lower edge of the xi bin (must be <= xi_hi, in [-1, 1]).
 * @param xi_hi Upper edge of the xi bin (in [-1, 1]).
 * @param T     Electron temperature [K].
 * @return      RidgeBounds with cold endpoints and thermal widths.
 */
inline RidgeBounds compute_ridge_bounds(
    double const E, double const xi_lo, double const xi_hi, double const T)
{
    assert(xi_lo <= xi_hi);
    assert(xi_lo >= -1.0);
    assert(xi_hi <= 1.0);
    double const gamma = E / units::me_c2;
    return { E / (1.0 + gamma * (1.0 - xi_lo)),
             E / (1.0 + gamma * (1.0 - xi_hi)),
             ridge_thermal_width(E, xi_lo, T),
             ridge_thermal_width(E, xi_hi, T) };
}

} // namespace compton

#endif
