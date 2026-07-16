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
 * Given the point-wise differential scattering kernel Σ_E(E→E', ξ; T),
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
 * - Temperature T is in [K].
 * - The returned matrix entries have units [cm²] (microscopic, per free electron).
 * - Angle-integrated overloads (no num_angle_bins) integrate ξ over [−1,1].
 *
 * The dsigma_dT variants plug the derivative kernel ∂Σ_E/∂T into the same
 * weighted-integral formula.  They are NOT the full ∂σ/∂T of the multigroup
 * cross section (which would need quotient-rule terms).  Use
 * compute_dsigma_dT_matrix for the complete derivative including
 * weight-function and denominator temperature dependence.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REFERENCE
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   B. A. Clark, "Computing multigroup radiation integrals using
 *   polylogarithm-based methods," JCP 70(2):311–329, 1987.
 */

#include "compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "utilities/gauss_legendre.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace compton {

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
    std::optional<double> cutoff_ratio;
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
     * @param cutoff_ratio            Outward-from-peak early-termination ratio
     *                                 (must be > 0 when provided; nullopt
     *                                 disables cutoff).
     * @param xi_order                GL order for the ξ peak core (defaults to
     * 48).
     * @param xi_peak_k               Half-width of the ξ peak window in sigma
     * units.
     * @param xi_tail_order           GL order for ξ tail sub-intervals
     * (defaults to 16).
     * @param ep_k_cut                E' truncation width in sigma units (must
     * be > 0).
     * @param ep_k_in                 E' interior-edge separator in sigma units
     * (must be >= 0, < ep_k_cut).
     * @param ep_edge_order           GL order for E' edge regions (defaults to
     * 24).
     * @param ep_interior_order       GL order for E' ridge interior (defaults
     * to 24).
     * @param e_panel_order           GL order for E-axis sub-panels (defaults
     * to 12).
     * @param log_e_panel_ratio       Panel width ratio threshold for log/rlog
     * mapping (must be > 1).
     * @param e_boundary_k            E-panel boundary-layer width in sigma
     * units (must be > 0).
     * @throws std::invalid_argument on invalid parameters.
     */
    MGIntegrationConfig(
        std::optional<double> cutoff_ratio = 1e-8,
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

    /** @brief Effective ξ tail GL order (xi_tail_order if set, otherwise 16).
     */
    int effective_xi_tail_order() const { return xi_tail_order.value_or(16); }

    /** @brief Effective E' edge GL order (ep_edge_order if set, otherwise 24).
     */
    int effective_ep_edge_order() const { return ep_edge_order.value_or(24); }

    /** @brief Effective E' interior GL order (ep_interior_order if set,
     * otherwise 24). */
    int effective_ep_interior_order() const
    {
        return ep_interior_order.value_or(24);
    }

    /** @brief Effective E-axis per-panel GL order (e_panel_order if set,
     * otherwise 12). */
    int effective_e_panel_order() const { return e_panel_order.value_or(12); }
};

/**
 * @brief Abstract base for kernel multipliers.
 *
 * A kernel multiplier f(E, E', ξ) is an extra factor that multiplies
 * the differential scattering kernel pointwise inside the multigroup integral.
 * The result is *not* normalised by the integral of f itself, so it behaves
 * like an observable averaged against the scattering distribution.
 */
class KernelMultiplier {
  public:
    virtual ~KernelMultiplier() = default;
    KernelMultiplier() = default;
    KernelMultiplier(KernelMultiplier const&) = default;
    KernelMultiplier& operator=(KernelMultiplier const&) = default;
    KernelMultiplier(KernelMultiplier&&) = default;
    KernelMultiplier& operator=(KernelMultiplier&&) = default;
    virtual double
    operator()(double E, double Ep, double xi) const = 0;
};

/**
 * @brief Identity multiplier: always returns 1.
 */
class ConstantMultiplier : public KernelMultiplier {
  public:
    double operator()(
        double /*E*/,
        double /*E_prime*/,
        double /*xi*/) const override
    {
        return 1.0;
    }
};

/**
 * @brief Computes the weighted multigroup-multiangle Compton scattering
 *        matrix by fixed-order Gauss-Legendre quadrature.
 *
 * Construct once with the energy group structure, weight function,
 * and quadrature configuration; then call compute_sigma_matrix /
 * compute_kernel_derivative_contribution at any temperature and angular resolution.
 */
class ComptonMultigroupKernel {
  public:
    /**
     * @brief Construct from energy group boundaries and a weight function.
     *
     * @param energy_group_boundaries  G+1 strictly increasing values [erg], all
     * > 0.
     * @param weight_function          Shared pointer to a WeightFunction
     * subclass.
     * @param config                   Integration configuration (tolerance,
     * orders, cutoff).
     * @throws std::invalid_argument on invalid boundaries.
     */
    ComptonMultigroupKernel(
        std::vector<double> const& energy_group_boundaries,
        std::shared_ptr<WeightFunction const> weight_function,
        MGIntegrationConfig const& config);

    /** @brief Number of energy groups G. */
    int num_groups() const
    {
        return static_cast<int>(group_boundaries_.size()) - 1;
    }

    /** @brief Energy group boundaries [erg], length G+1. */
    std::vector<double> const& group_boundaries() const
    {
        return group_boundaries_;
    }

    // ── Multigroup-multiangle (3D: G × G × N_angles) ────────────────────

    /**
     * @brief Compute the multigroup-multiangle σ matrix.
     *
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param multiplier      Pointwise kernel multiplier applied before
     * integration.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_sigma_matrix(
        ComptonKernelSolver const& kernel,
        int num_angle_bins,
        double T,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the multigroup-multiangle ∂σ/∂T matrix.
     *
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param multiplier      Pointwise kernel multiplier applied before
     * integration.
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_kernel_derivative_contribution(
        ComptonKernelSolver const& kernel,
        int num_angle_bins,
        double T,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Compute the full d/dT of the multigroup cross section.
     *
     * Unlike compute_kernel_derivative_contribution (which only differentiates the kernel),
     * this method applies the quotient rule to account for the temperature
     * dependence of the weight function and its denominator:
     *
     *     d sigma / dT = (2pi/D)(dN_kd/dT + dN_wd/dT) - sigma * (dD/dT) / D
     *
     * The user-provided multiplier is NOT differentiated with respect to T.
     * Group cutoff is disabled (all G^2 pairs evaluated).
     *
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [-1, 1].
     * @param T               Electron temperature [K].
     * @param multiplier      Pointwise kernel multiplier.
     * @return Flat vector of size G*G*N_angles, row-major [g][g'][angle].
     */
    std::vector<double> compute_dsigma_dT_matrix(
        ComptonKernelSolver const& kernel,
        int num_angle_bins,
        double T,
        KernelMultiplier const& multiplier) const;

  private:
    /**
     * @brief Core driver: assemble the full G×G×num_angle_bins scattering
     * matrix.
     *
     * Orchestrates the multigroup integration for every incoming group g.
     * For each g the method starts at the peak target group (the one
     * containing the geometric-mean energy of g) and expands outward.
     * When group_cutoff_ratio_ is set, expansion stops in each direction
     * once the angle-summed magnitude drops below its value × peak_value;
     * otherwise all target groups are evaluated.
     *
     * Each selected (g, gp, angle) bin is evaluated by integrate_E_Ep_xi_bin().
     *
     * @param kernel         Point-wise kernel evaluator.
     * @param eval           Pointer-to-member: sigma_E or dsigma_E_dT.
     * @param num_angle_bins Number of equal-width ξ bins on [−1, 1].
     * @param T              Electron temperature [K].
     * @param multiplier     Optional pointwise factor applied inside the
     * integrand.
     * @return Flat row-major vector of size G×G×num_angle_bins,
     *         indexed as result[g * G * num_angle_bins + gp * num_angle_bins +
     * a].
     */
    std::vector<double> compute_matrix_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (
            ComptonKernelSolver::*eval)(double, double, double, double)
            const,
        int num_angle_bins,
        double T,
        KernelMultiplier const& multiplier,
        std::optional<double> effective_cutoff) const;

    /**
     * @brief Integrate the kernel over a single ξ bin for fixed (E, E').
     *
     * Performs peak-focused GL quadrature with four branches:
     * endpoint-localized rlog, peak-left, peak-right, three-region split.
     *
     * @return ∫_{xi_lo}^{xi_hi} multiplier · kernel dξ
     */
    double integrate_xi_bin(
        ComptonKernelSolver const& kernel,
        ComptonResult (
            ComptonKernelSolver::*eval)(double, double, double, double)
            const,
        double E,
        double Ep,
        double xi_lo,
        double xi_hi,
        double T,
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
        ComptonResult (
            ComptonKernelSolver::*eval)(double, double, double, double)
            const,
        double E,
        double Ep_lo,
        double Ep_hi,
        double xi_lo,
        double xi_hi,
        double T,
        KernelMultiplier const& multiplier) const;

    /**
     * @brief Integrate over E, E', and a single ξ bin for one group pair.
     *
     * Integrates w(E,T) * integrate_Ep_xi_bin over the incoming E group
     * using feature-aware E-axis panels computed on demand for g.
     *
     * @return ∫_{E_lo}^{E_hi} w(E,T) ∫_{Ep_lo}^{Ep_hi} ∫_{xi_lo}^{xi_hi}
     *         multiplier · kernel dξ dE' dE
     */
    double integrate_E_Ep_xi_bin(
        ComptonKernelSolver const& kernel,
        ComptonResult (
            ComptonKernelSolver::*eval)(double, double, double, double)
            const,
        int g,
        int gp,
        double xi_lo,
        double xi_hi,
        double T,
        KernelMultiplier const& multiplier) const;

  public:
    /** @brief Integrate the kernel over ξ bins for fixed (E, E'). */
    std::vector<double> compute_xi_integral_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (
            ComptonKernelSolver::*eval)(double, double, double, double)
            const,
        double E,
        double Ep,
        int num_xi_bins,
        double T,
        KernelMultiplier const& multiplier) const;

    /** @brief Integrate the kernel over E' and ξ bins for fixed E. */
    std::vector<double> compute_Ep_xi_integral_impl(
        ComptonKernelSolver const& kernel,
        ComptonResult (
            ComptonKernelSolver::*eval)(double, double, double, double)
            const,
        double E,
        double Ep_lo,
        double Ep_hi,
        int num_xi_bins,
        double T,
        KernelMultiplier const& multiplier) const;

  private:
    std::vector<double> group_boundaries_;

    /// Shared weight function for the Planck/Wien/Uniform numerator and
    /// denominator.
    std::shared_ptr<WeightFunction const> weight_func_;

    /// GL rule for the ξ (scattering-angle) axis.
    GaussLegendreRule xi_rule_;
    /// GL rule for ξ tails in peak-focused splitting (low order, tails are
    /// exponentially small).
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
    std::optional<double> group_cutoff_ratio_;

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
inline double
ridge_thermal_width(double const E, double const xi, double const T)
{
    double const gamma = E / units::me_c2;
    double const tau = T * units::k_boltz / units::me_c2;
    double const u = std::max(0.0, 1.0 - xi);
    if (tau <= 0.0 || u <= 0.0) {
        return 0.0;
    }
    double const d = 1.0 + gamma * u;
    return (E / (d * d)) *
           std::sqrt(tau * u * (2.0 + 2.0 * gamma * u + gamma * gamma * u));
}

/**
 * @brief Cold-Compton ridge endpoints and local thermal widths for a xi bin.
 *
 * All values are in energy units [erg].
 */
struct RidgeBounds {
    double cold_lo;  ///< E'_cold(xi_lo) = E / (1 + gamma*(1-xi_lo)) [erg]
    double cold_hi;  ///< E'_cold(xi_hi) = E / (1 + gamma*(1-xi_hi)) [erg]
    double sigma_lo; ///< ridge_thermal_width(E, xi_lo, T) [erg]
    double sigma_hi; ///< ridge_thermal_width(E, xi_hi, T) [erg]
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
    double const E,
    double const xi_lo,
    double const xi_hi,
    double const T)
{
    assert(xi_lo <= xi_hi);
    assert(xi_lo >= -1.0);
    assert(xi_hi <= 1.0);
    double const gamma = E / units::me_c2;
    return {
        E / (1.0 + gamma * (1.0 - xi_lo)),
        E / (1.0 + gamma * (1.0 - xi_hi)),
        ridge_thermal_width(E, xi_lo, T),
        ridge_thermal_width(E, xi_hi, T)};
}

/// Angular-localization threshold for the endpoint-localized rlog condition.
/// Calibrated against 50 log-spaced temperatures from 1e-5 to 1e3 keV.
inline constexpr double XI_ENDPOINT_EPS = 0.1;

/// Minimum dimensionless temperature τ = kT/m_e c² for the near-elastic
/// Klein–Nishina cusp path.  Below this τ, condition (A) alone is sufficient.
inline constexpr double XI_CUSP_TAU = 0.001;

/**
 * @brief Test whether the endpoint-localized reflected-log condition is met.
 *
 * The reflected-log ξ quadrature (clustering nodes near ξ = 1) activates
 * when either of two conditions holds:
 *
 *   A. **Thermal endpoint-localisation:** the peak is close to ξ = 1 AND
 *      narrow:
 *          Δγ / (γ γ') ≤ σ_ξ   AND   σ_ξ ≤ ε_ξ
 *
 *   B. **Near-elastic kinematic cusp:** at warm/hot temperatures the
 *      fractional energy transfer is small enough that the Klein–Nishina
 *      forward peak at ξ = 1 still dominates the last angular bin:
 *          |Δγ| / γ ≤ ε_ξ  AND  τ > τ_cusp
 *
 * Condition (A) handles the narrow-peak regime (cold/moderate T).
 * Condition (B) handles same-group scattering at hot T where σ_ξ ≫ 1
 * but the KN forward cusp still needs rlog resolution.
 *
 * The constants ε_ξ = XI_ENDPOINT_EPS and τ_cusp = XI_CUSP_TAU were
 * calibrated against a 50-point temperature sweep (1e-5 to 1e3 keV)
 * measuring last-angular-bin convergence at xi_order=48 vs 512.
 *
 * @param gamma   Dimensionless incoming photon energy  E  / m_e c².
 * @param gamma_p Dimensionless outgoing photon energy E' / m_e c².
 * @param tau     Dimensionless temperature  k_B T / m_e c².
 * @return true if the reflected-log path should be used.
 */
inline bool endpoint_localized_xi(
    double const gamma,
    double const gamma_p,
    double const tau)
{
    double const abs_dg = std::abs(gamma - gamma_p);
    double const gg = gamma * gamma_p;
    double const peak_distance_from_one = abs_dg / gg;
    double const sigma_xi =
        std::sqrt(tau * abs_dg * (2.0 + abs_dg)) / gg;

    bool const thermal_endpoint =
        peak_distance_from_one <= sigma_xi &&
        sigma_xi <= XI_ENDPOINT_EPS;
    bool const near_elastic_cusp =
        abs_dg <= gamma * XI_ENDPOINT_EPS &&
        tau > XI_CUSP_TAU;

    return thermal_endpoint || near_elastic_cusp;
}

} // namespace compton

#endif
