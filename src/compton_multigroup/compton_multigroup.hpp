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

#include <memory>
#include <vector>
#include <numbers>

namespace compton {

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
 *        matrix by tensor-product Gauss-Legendre quadrature.
 *
 * Construct once with the energy group structure, weight function, and
 * quadrature orders; then call compute_sigma_matrix /
 * compute_dsigma_dT_matrix at any temperature and angular resolution.
 */
class ComptonMultigroupKernel {
public:
    /**
     * @brief Construct from energy group boundaries and a weight function.
     *
     * @param energy_group_boundaries  G+1 strictly increasing values [erg], all > 0.
     * @param weight_function          Shared pointer to a WeightFunction subclass.
     * @param quad_order_E             Gauss-Legendre order for incident energy.
     * @param quad_order_Ep            Gauss-Legendre order for scattered energy.
     * @param quad_order_mu            Gauss-Legendre order for scattering angle.
     * @throws std::invalid_argument on invalid boundaries or orders.
     */
    ComptonMultigroupKernel(
        std::vector<double> const& energy_group_boundaries,
        std::shared_ptr<WeightFunction const> weight_function,
        int quad_order_E  = 8,
        int quad_order_Ep = 8,
        int quad_order_mu = 8);

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
     * @brief Core implementation: 3D tensor-product quadrature.
     *
     * @tparam KernelT  Kernel class with a member returning SigmaResult.
     * @param kernel         Kernel instance.
     * @param eval           Pointer-to-member evaluating (E, E', xi, T, Ne) → SigmaResult.
     * @param num_angle_bins Number of equal-width bins on [−1, 1].
     * @param T              Electron temperature [K].
     * @param Ne             Electron density [cm⁻³].
     * @param multiplier     Pointwise kernel multiplier applied before integration.
     * @return Flat vector of size G×G×num_angle_bins.
     */
    template<typename KernelT>
    std::vector<double> compute_matrix_impl(
        KernelT const& kernel,
        SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
        int num_angle_bins,
        double T,
        double Ne,
        KernelMultiplier const& multiplier) const;

    std::vector<double> group_boundaries_;
    std::vector<double> group_centers_;
    std::shared_ptr<WeightFunction const> weight_func_;

    GaussLegendreRule rule_E_;
    GaussLegendreRule rule_Ep_;
    GaussLegendreRule rule_mu_;
};

} // namespace compton

#endif
