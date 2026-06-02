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
 * where the weight function is a capped Planck spectrum:
 *
 *     w(E,T) = x³/(eˣ−1)   for x = E/(kT) < N
 *            = N³/(eᴺ−1)    for x ≥ N          (default N = 25)
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
 * over the three finite intervals (E, E', μ).  The denominator uses the
 * analytic Clark (1987) polylogarithm method from planck_integral.hpp,
 * stitching at x = N when a group straddles the cap threshold.
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
#include "compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_kernel_series/compton_kernel_series.hpp"

#include <vector>
#include <numbers>

namespace compton {

/**
 * @brief Computes the Planck-weighted multigroup-multiangle Compton
 *        scattering matrix by tensor-product Gauss-Legendre quadrature.
 *
 * Construct once with the energy group structure and quadrature orders;
 * then call compute_sigma_matrix / compute_dsigma_dT_matrix at any
 * temperature and angular resolution.
 */
class ComptonMultigroupKernel {
public:
    /**
     * @brief Construct from energy group boundaries.
     *
     * @param energy_group_boundaries  G+1 strictly increasing values [erg], all > 0.
     * @param quad_order_E             Gauss-Legendre order for incident energy.
     * @param quad_order_Ep            Gauss-Legendre order for scattered energy.
     * @param quad_order_mu            Gauss-Legendre order for scattering angle.
     * @param planck_cap_x             Dimensionless cutoff x = E/(kT) for the
     *                                 Planck weight (default 25).
     * @throws std::invalid_argument on invalid boundaries or orders.
     */
    ComptonMultigroupKernel(
        std::vector<double> const& energy_group_boundaries,
        int quad_order_E  = 8,
        int quad_order_Ep = 8,
        int quad_order_mu = 8,
        double planck_cap_x = 25.0);

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
     * @tparam KernelT        Kernel type (ComptonKernelQuadrature or ComptonKernelSeries).
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³].
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    template<typename KernelT>
    std::vector<double> compute_sigma_matrix(
        KernelT const& kernel,
        int num_angle_bins,
        double T, double Ne) const;

    /**
     * @brief Compute the multigroup-multiangle ∂σ/∂T matrix.
     *
     * @tparam KernelT        Kernel type (ComptonKernelQuadrature or ComptonKernelSeries).
     * @param kernel          Point-wise kernel evaluator.
     * @param num_angle_bins  Number of equal-width bins on [−1, 1].
     * @param T               Electron temperature [K].
     * @param Ne              Electron density [cm⁻³].
     * @return Flat vector of size G×G×N_angles, row-major [g][g'][angle].
     */
    template<typename KernelT>
    std::vector<double> compute_dsigma_dT_matrix(
        KernelT const& kernel,
        int num_angle_bins,
        double T, double Ne) const;

    /**
     * @brief Analytic denominator ∫_{ΔEg} w(E,T) dE for group g.
     *
     * Uses the Clark polylogarithm series from planck_integral.hpp for the
     * below-cap portion and a constant for the above-cap portion, stitching
     * at E = cap_x · kT when the group straddles the threshold.
     *
     * @param g  Group index (0-based).
     * @param T  Electron temperature [K].
     * @return   Denominator integral value.
     */
    double compute_denominator(int g, double T) const;

private:
    /**
     * @brief Core implementation: 3D tensor-product quadrature.
     *
     * @tparam KernelT  Kernel class with a member returning SigmaResult.
     * @param kernel         Kernel instance.
     * @param eval           Pointer-to-member evaluating (E, E', xi, T, Ne) → SigmaResult.
     * @param num_angle_bins Number of equal-width bins on [−1, 1].
     * @param T              Electron temperature [K].
     * @param Ne             Electron density [cm⁻³].
     * @return Flat vector of size G×G×num_angle_bins.
     */
    template<typename KernelT>
    std::vector<double> compute_matrix_impl(
        KernelT const& kernel,
        SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
        int num_angle_bins,
        double T,
        double Ne) const;

    /**
     * @brief Capped Planck weight w(E, T).
     *
     * Returns x³/(eˣ−1) for x = E/(kT) < cap_x_, else cap constant w₀.
     *
     * @param E  Photon energy [erg].
     * @param T  Electron temperature [K].
     * @return   Weight value (dimensionless).
     */
    double planck_weight(double E, double T) const;

    std::vector<double> group_boundaries_;
    std::vector<double> group_centers_;
    double cap_x_;
    double w0_;

    GaussLegendreRule rule_E_;
    GaussLegendreRule rule_Ep_;
    GaussLegendreRule rule_mu_;
};

} // namespace compton

#endif
