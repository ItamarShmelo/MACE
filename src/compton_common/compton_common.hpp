#ifndef COMPTON_COMMON_HPP
#define COMPTON_COMMON_HPP
/**
 * @file compton_common.hpp
 * @brief Shared kinematics and normalization for the Kershaw-Prasad-Beason
 *        Compton scattering kernel.
 *
 * Contains the kinematic parameter struct, prefactor computation, and scaled
 * Bessel function used by both the direct quadrature and series evaluation
 * modules.
 */

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "units/units.hpp"
#include "compton_kernel_series/double_double.hpp"

namespace compton {

/**
 * @brief Scaled modified Bessel function: K̃₂(x) = exp(x) · K₂(x).
 *
 * Uses Boost cyl_bessel_k for x < 50 (numerically stable after multiplying
 * by exp(x)), and a 5-term Hankel asymptotic expansion for x ≥ 50 where
 * the direct computation would overflow/underflow.
 */
double scaled_K2(double x);

/**
 * @brief Pre-computed kinematic parameters for a given (γ, γ', ξ, τ).
 *
 * Derived quantities used by both quadrature and series evaluations:
 *   a  = 1 − ξ                         (related to momentum transfer)
 *   s  = 1/γ + 1/γ'                    (sum of inverse energies)
 *   q  = √[(γ'−γ)² + 2γγ'a]           (momentum transfer magnitude)
 *   Δ  = √[(1 + γγ'a/2)(1 + (γ'−γ)²/(2γγ'a))]
 *   λ₊ = (γ'−γ)/2 + Δ                  (min electron Lorentz factor)
 *   ρ₊ = λ₊ + γ,  ρ₋ = λ₊ − γ'        (shifted momentum parameters)
 *   α± = 1/√(ρ±² + ω²)                 (appear in boundary terms)
 *   G, A±, Ψ                            (combined constants for the integrand)
 */
struct KershawParams {
    double a, s, q, omega2;
    double Delta, lambda_plus, rho_plus, rho_minus;
    double alpha_plus, alpha_minus;
    double G, A_plus, A_minus, Psi;
};

/// Result of a kernel evaluation: value plus heuristic error estimates.
struct SigmaResult {
    double value;               ///< Σ_E in [cm²/erg] (Nₑ=1) or [1/(cm·erg)]
    double estimated_abs_error; ///< |σ₀| · |IQ(N) − IQ(N/2)|
    double estimated_rel_error; ///< abs_error / |value|
};

/**
 * @brief Compute all kinematic parameters from dimensionless energies.
 *
 * This is a pure function (no state); it derives (a, s, q, Δ, λ₊, ρ±, α±,
 * G, A±, Ψ) from the inputs.  Used by both quadrature and series modules.
 */
KershawParams compute_params(double gamma, double gamma_p, double xi, double tau);

/**
 * @brief DD-precision kinematic parameters (mirrors KershawParams).
 *
 * Used by the power series to achieve eps^2 conditioning error.
 */
struct KershawParamsDD {
    dd a, s, q, omega2;
    dd Delta, lambda_plus, rho_plus, rho_minus;
    dd alpha_plus, alpha_minus;
    dd G, A_plus, A_minus, Psi;
};

/**
 * @brief DD-precision version of compute_params.
 *
 * Mirrors compute_params() exactly but performs all arithmetic in
 * double-double precision, producing dd-accurate kinematic parameters.
 */
KershawParamsDD compute_params_dd(double gamma, double gamma_p, double xi, double tau);

/**
 * @brief Compute the prefactor σ₀ = Nₑ r_e² m_e c² / (4E²τ)
 *                                    × exp(−(λ₊−1)/τ) / K̃₂(1/τ).
 *
 * The exponential suppression factor exp(−(λ₊−1)/τ) controls the kernel
 * magnitude: elastic scattering (λ₊→1) has no suppression, while large
 * energy transfers (λ₊≫1) are exponentially suppressed.
 */
double stable_sigma0_E(double E, double tau, double lambda_plus, double Ne);

} // namespace compton

#endif
