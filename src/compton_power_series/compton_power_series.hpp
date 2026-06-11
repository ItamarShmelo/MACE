#ifndef COMPTON_POWER_SERIES_HPP
#define COMPTON_POWER_SERIES_HPP
/**
 * @file compton_power_series.hpp
 * @brief Kershaw-Prasad-Beason thermal Compton frequency kernel via
 *        convergent power series (Poisson-weighted Ê sums).
 *
 * Implements the power-series evaluation of the differential scattering
 * kernel Σ_E(E→E', ξ; τ):
 *
 *   Σ_E / Σ₀ = Ψ + P₊ − P₋
 *   P± = Σₙ w_n± · c_n± · Ê_{n+1}(x±)
 *
 * with Poisson weights w_n±, coefficients c_n± = A± + 2n/a, and scaled
 * exponential integrals Ê_m(x) = eˣ E_m(x).
 *
 * The constructor flag `high_precision` selects double (~15 digits) or
 * double-double (~31 digits) arithmetic for the internal summation.
 */

#include "compton_common/compton_common.hpp"

namespace compton {

class ComptonPowerSeries {
public:
    /**
     * @param high_precision  When true, use double-double (~31 digits)
     *                        arithmetic; when false, use double (~15 digits).
     * @param eps_rel  Relative convergence tolerance for both term-decay
     *                 and difference-stability checks.
     * @param n_min    Minimum number of terms before convergence is tested.
     * @param n_max    Maximum number of terms; if reached without convergence
     *                 a std::runtime_error is thrown.
     */
    ComptonPowerSeries(
        bool high_precision = false,
        double eps_rel = 1e-12,
        int n_min = 4,
        int n_max = 200);

    /**
     * @brief Evaluate Σ_E(E → E', ξ; T, Nₑ) using the power series.
     *
     * @param E        Incident photon energy [erg]
     * @param E_prime  Scattered photon energy [erg]
     * @param xi       cos(scattering angle), strictly in (−1, 1)
     * @param T        Electron temperature [K]
     * @param Ne       Electron number density [cm⁻³] (use 1.0 for microscopic)
     * @return ComptonResult with value and error estimates
     */
    ComptonResult sigma_E(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

    /**
     * @brief Evaluate ∂Σ_E/∂T at a single phase-space point.
     *
     * Temperature derivative of the Compton kernel with respect to T [K],
     * computed analytically via power series expansion and applying the
     * chain rule dτ/dT = k_B / (m_e c²).
     *
     * @param E        Incident photon energy [erg]
     * @param E_prime  Scattered photon energy [erg]
     * @param xi       cos(scattering angle), strictly in (−1, 1)
     * @param T        Electron temperature [K]
     * @param Ne       Electron number density [cm⁻³] (use 1.0 for microscopic)
     * @return ComptonResult with value and error estimates
     */
    ComptonResult dsigma_E_dT(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

    /**
     * @brief Run both double and DD power series, return the relative error.
     *
     * Computes |dd_value - dbl_value| / (|dd_value| + 1e-300).
     * Independent of the high_precision flag.  Useful for checking
     * whether double precision is sufficient for a given set of parameters.
     */
    double sigma_E_precision_check(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

    /**
     * @brief Run both double and DD power series derivatives, return the
     *        relative error.
     *
     * Computes |dd_value - dbl_value| / (|dd_value| + 1e-300) on the
     * derivative.  Useful for checking whether double precision is
     * sufficient for the temperature derivative at a given parameter point.
     */
    double dsigma_E_dT_precision_check(
        double E,
        double E_prime,
        double xi,
        double T,
        double Ne) const;

private:
    /**
     * @brief Power series evaluation of the normalized kernel.
     *
     * Hyperbolic substitution:
     *   b = ω / (2τ)
     *   θ± = arcsinh(ρ± / ω)
     *   x± = b · exp(θ±),   y± = b · exp(−θ±)
     *
     * Series:
     *   Σ_E / Σ₀ = Ψ + P₊ − P₋
     *
     *   P± = Σₙ w_n± · c_n± · Ê_{n+1}(x±)
     *
     * where:
     *   w_0± = exp(−y±),  w_{n+1}± = w_n± · y± / (n+1)  (Poisson weights)
     *   c_n± = A± + 2n/a                                  (kinematic coefficients)
     *   Ê_m(x) = eˣ · E_m(x)                              (scaled exponential integral)
     *
     * Template parameter T selects double (~15 digits) or
     * double-double (~31 digits) arithmetic.
     */
    template<typename T>
    ComptonResult power_series(
        double gamma,
        double gamma_p,
        double xi,
        double tau,
        double E,
        double Ne) const;

    /**
     * @brief Power series temperature derivative of the normalized kernel.
     *
     * Computes both the series value (P₊ − P₋) and its τ-derivative
     * (∂P₊/∂τ − ∂P₋/∂τ) simultaneously.
     *
     * Per-term derivative (n-th contribution to ∂P₊/∂τ):
     *   w_n · { [s/(τ²a²) − (ρ₊/τ² + n/τ) L_n⁺] Ê_{n+1}(x₊)
     *         + (x₊/τ) L_n⁺ Ê_n(x₊) }
     *
     * where L_n± = A± + 2n/a, and Ê_n is tracked alongside Ê_{n+1}
     * via Ê₀(x) = 1/x as the base case.
     *
     * Full result (returned as d/dτ, without dτ/dT factor):
     *   σ₀ · { 2γγ'/q + (∂P₊/∂τ − ∂P₋/∂τ)
     *        + [(λ₊−κ)/τ² − 3/τ] · (Ψ + P₊ − P₋) }
     */
    template<typename T>
    ComptonResult power_series_derivative(
        double gamma,
        double gamma_p,
        double xi,
        double tau,
        double E,
        double Ne) const;

    bool high_precision_;
    double eps_rel_;
    int n_min_, n_max_;
};

} // namespace compton

#endif
