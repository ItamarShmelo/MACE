#ifndef COMPTON_KERNEL_QUADRATURE_HPP
#define COMPTON_KERNEL_QUADRATURE_HPP
/**
 * @file compton_kernel_quadrature.hpp
 * @brief Kershaw-Prasad-Beason thermal Compton frequency kernel via direct
 *        Gauss-Laguerre quadrature.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * PHYSICS
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Implements the differential scattering kernel Σ_E(E→E', ξ; τ) describing
 * photon scattering off a thermal Maxwell-Jüttner electron distribution at
 * dimensionless temperature τ = kT / m_e c².  This is the energy-domain form
 * of the Kershaw, Prasad & Beason (1986) "frequency kernel" originally
 * expressed in terms of photon frequency ν.
 *
 * The kernel factorizes as:
 *
 *     Σ_E = σ₀(E, τ, λ₊, Nₑ)  ×  I_Q(γ, γ', ξ, τ)
 *
 * where:
 *   - γ = E / m_e c²,  γ' = E' / m_e c²   (dimensionless photon energies)
 *   - ξ = cos(θ)                            (scattering angle cosine)
 *   - σ₀ contains the exponential suppression factor exp(−(λ₊−1)/τ) and
 *     normalization by the scaled Bessel function K̃₂(1/τ)
 *   - I_Q is the semi-infinite integral over electron momentum, evaluated by
 *     Gauss-Laguerre quadrature after a change of variable ρ = τ·x + ρ_offset
 *
 * ─────────────────────────────────────────────────────────────────────────
 * QUADRATURE FORMS
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Two mathematically equivalent forms of I_Q are provided:
 *
 * 1. PostIntegrationByParts (default):
 * 
 *
 *
 * 2. PreIntegrationByParts. 
 *
 *
 *
 * ─────────────────────────────────────────────────────────────────────────
 * UNITS AND API
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The function sigma_E returns the macroscopic kernel in
 * [1/(cm·erg)] when Nₑ = electron density [cm⁻³].
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REFERENCE
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   D. Kershaw, M. Prasad, and J. Beason, "Photon Transport in a
 *   Compton Scattering Medium," Technical Report UCRL-94345, 1986.
 */

#include "compton_common/compton_common.hpp"

namespace compton {

/// Selects which integral form to use for I_Q.
enum class QuadratureForm {
    PostIntegrationByParts,  ///< Default: IBP-transformed, O(1/√R) integrand
    PreIntegrationByParts    ///< Original O(1/R^{3/2}) integrand
};

/**
 * @brief Main class: evaluates the Compton frequency kernel at a single
 *        phase-space point (E, E', ξ) for temperature τ.
 *
 * Construction pre-selects the quadrature order NL and form.  Each call to
 * sigma_E evaluates the kinematic parameters and performs two Gauss-Laguerre
 * quadratures (order NL and NL/2) for the Richardson-style error estimate.
 */
class ComptonKernelQuadrature {
public:
    /**
     * @param NL    Number of Gauss-Laguerre nodes (64, 128, or 256).
     * @param form  Which integral representation to evaluate.
     */
    ComptonKernelQuadrature(int NL = 64,
                            QuadratureForm form = QuadratureForm::PostIntegrationByParts);

    /**
     * @brief Evaluate Σ_E(E → E', ξ; τ, Nₑ).
     *
     * @param E        Incident photon energy [erg]
     * @param E_prime  Scattered photon energy [erg]
     * @param xi       cos(scattering angle), strictly in (−1, 1)
     * @param tau      Dimensionless electron temperature kT/(m_e c²)
     * @param Ne       Electron number density [cm⁻³] (use 1.0 for microscopic)
     * @return SigmaResult with value and error estimates
     */
    SigmaResult sigma_E(double E, double E_prime, double xi, double tau, double Ne) const;

private:
    double compute_IQ_post_ibp(const KershawParams<double>& p, double tau, int NL) const;
    double compute_IQ_pre_ibp(const KershawParams<double>& p, double tau, int NL) const;

    int NL_;
    QuadratureForm form_;
};

} // namespace compton

#endif
