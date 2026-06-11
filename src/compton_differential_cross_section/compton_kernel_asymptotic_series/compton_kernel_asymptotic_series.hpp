#ifndef COMPTON_KERNEL_ASYMPTOTIC_SERIES_HPP
#define COMPTON_KERNEL_ASYMPTOTIC_SERIES_HPP
/**
 * @file compton_kernel_asymptotic_series.hpp
 * @brief Low-temperature asymptotic expansion of the Compton frequency kernel
 *        using Legendre polynomials.
 *
 * Implements the asymptotic series evaluation of the differential scattering
 * kernel Σ_E(E→E', ξ; τ):
 *
 *   Σ_E / Σ₀ = 2τγγ'/q + S₊ + S₋
 *   S± = Σₙ T_n±
 *
 * with terms built from powers (−τα±)^{n+1}, factorials, and Legendre
 * polynomials P_n(ζ±).  This is a divergent asymptotic series truncated at
 * the smallest term; best suited for τ·max(α₊, α₋) < 0.05.
 *
 * The constructor flag `high_precision` selects double (~15 digits) or
 * double-double (~31 digits) arithmetic for the internal summation.
 */

#include "compton_common/compton_common.hpp"

namespace compton {

class ComptonKernelAsymptoticSeries {
public:
    /**
     * @param high_precision  When true, use double-double (~31 digits)
     *                        arithmetic; when false, use double (~15 digits).
     * @param eps_rel  Relative convergence tolerance.
     * @param n_min    Minimum number of terms before convergence is tested.
     * @param n_max    Maximum number of terms; if reached without convergence
     *                 a std::runtime_error is thrown.
     */
    ComptonKernelAsymptoticSeries(
        bool high_precision = false,
        double eps_rel = 1e-12,
        int n_min = 4,
        int n_max = 200);

    /**
     * @brief Evaluate Σ_E(E → E', ξ; T, Nₑ) using the asymptotic series.
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
     * computed analytically via asymptotic series expansion and applying
     * the chain rule dτ/dT = k_B / (m_e c²).
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
     * @brief Run both double and DD asymptotic series, return the relative error.
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
     * @brief Run both double and DD asymptotic series derivatives, return the
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
     * @brief Low-temperature asymptotic expansion using Legendre polynomials.
     *
     * Setup:
     *   ζ± = ρ± · α±
     *   η₊ = α₊ · (s/a² + ρ₊/a)
     *   η₋ = α₋ · (−s/a² + ρ₋/a)
     *
     * Term structure:
     *   T_n⁺ = (−τα₊)^{n+1}
     *          · [(−G·n! + (n+1)!/a) P_n(ζ₊) − η₊ (n+1)! P_{n+1}(ζ₊)]
     *   T_n⁻ = (−τα₋)^{n+1}
     *          · [(G·n! − (n+1)!/a) P_n(ζ₋) + η₋ (n+1)! P_{n+1}(ζ₋)]
     *
     * Result:
     *   Σ_E / Σ₀ = 2τγγ'/q + Σₙ (T_n⁺ + T_n⁻)
     *
     * This is a divergent asymptotic series; truncation at the smallest
     * term gives accuracy ~ exp(−1/(τα)).  Best suited for
     * τ · max(α₊, α₋) < 0.05.
     *
     * Template parameter T selects double (~15 digits) or
     * double-double (~31 digits) arithmetic.
     */
    template<typename T>
    ComptonResult asymptotic_series(
        double gamma,
        double gamma_p,
        double xi,
        double tau,
        double E,
        double Ne) const;

    /**
     * @brief Low-temperature asymptotic derivative using Legendre polynomials.
     *
     * Since α±, ζ±, η±, G, a are all τ-independent, the derivative of
     * each term is the term itself times a weight:
     *
     *   weight(n) = (λ₊ − κ)/τ² + (n − 2)/τ
     *
     * Base term derivative:
     *   2γγ'/q · [(λ₊ − κ)/τ − 2]
     *
     * Same smallest-term truncation as the value series.
     * Result returned as d/dτ (without dτ/dT factor).
     *
     * Template parameter T selects double (~15 digits) or
     * double-double (~31 digits) arithmetic.
     */
    template<typename T>
    ComptonResult asymptotic_series_derivative(
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
