#ifndef COMPTON_KERNEL_SERIES_HPP
#define COMPTON_KERNEL_SERIES_HPP
/**
 * @file compton_kernel_series.hpp
 * @brief Kershaw-Prasad-Beason thermal Compton frequency kernel via
 *        closed-form series expansions (Section 4).
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
 *     Σ_E = Σ₀(E, τ, λ₊, Nₑ)  ×  [normalized ratio]
 *
 * where:
 *   - γ = E / m_e c²,  γ' = E' / m_e c²   (dimensionless photon energies)
 *   - ξ = cos(θ)                            (scattering angle cosine)
 *   - Σ₀ contains the exponential suppression factor exp(−(λ₊−1)/τ) and
 *     normalization by the scaled Bessel function K̃₂(1/τ)
 *   - The normalized ratio is evaluated by one of the series methods below,
 *     rather than by Gauss-Laguerre quadrature
 *
 * ─────────────────────────────────────────────────────────────────────────
 * SERIES FORMS
 * ─────────────────────────────────────────────────────────────────────────
 *
 * 1. PowerSeries / PowerSeriesHighPrecision:
 *    Σ_E / Σ₀ = Ψ + P₊ − P₋
 *    P± = Σₙ w_n± · c_n± · Ê_{n+1}(x±)
 *    with Poisson weights w_n±, coefficients c_n± = A± + 2n/a,
 *    and scaled exponential integrals Ê_m(x) = eˣ E_m(x).
 *    PowerSeries uses double precision (~15 digits);
 *    PowerSeriesHighPrecision uses double-double (~31 digits).
 *
 * 2. Asymptotic:
 *    Σ_E / Σ₀ = 2τγγ'/q + S₊ + S₋
 *    S± = Σₙ T_n±, with terms built from powers (−τα±)^{n+1},
 *    factorials, and Legendre polynomials P_n(ζ±).
 *    This is a divergent asymptotic series truncated at the smallest
 *    term; best suited for τ·max(α₊, α₋) < 0.05.
 *
 * 3. Auto (default):
 *    Selects Asymptotic when τ·max(α₊, α₋) < 0.05, otherwise
 *    PowerSeries if min(γ,γ') ≥ 0.02 or PowerSeriesHighPrecision.
 *
 * The series module does NOT depend on the quadrature module.  If the chosen
 * series fails to converge, it throws std::runtime_error.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * UNITS AND API
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The function sigma_E returns the macroscopic kernel in
 * [1/(cm·erg)] when Nₑ = electron density [cm⁻³].
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REFERENCES
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   D. S. Kershaw, M. K. Prasad, and J. D. Beason, "A simple and fast method
 *   for computing the relativistic Compton scattering kernel for radiative
 *   transfer," Journal of Quantitative Spectroscopy and Radiative Transfer
 *   36(4):273-282, 1986. doi:10.1016/0022-4073(86)90050-6.
 */

#include "compton_common/compton_common.hpp"

namespace compton {

enum class SeriesMethod {
    PowerSeries,
    PowerSeriesHighPrecision,
    Asymptotic,
    Auto
};

class ComptonKernelSeries {
public:
    /**
     * @param method   Series expansion strategy (PowerSeries,
     *                 PowerSeriesHighPrecision, Asymptotic, or Auto which
     *                 selects based on tau*alpha < 0.05).
     * @param eps_rel  Relative convergence tolerance for both term-decay
     *                 and difference-stability checks.
     * @param n_min    Minimum number of terms before convergence is tested.
     * @param n_max    Maximum number of terms; if reached without convergence
     *                 a std::runtime_error is thrown.
     */
    ComptonKernelSeries(
        SeriesMethod method = SeriesMethod::Auto,
        double eps_rel = 1e-12,
        int n_min = 4,
        int n_max = 200
    );

    /**
     * @brief Evaluate Σ_E(E → E', ξ; τ, Nₑ) using the configured
     *        series method.
     *
     * @param E        Incident photon energy [erg]
     * @param E_prime  Scattered photon energy [erg]
     * @param xi       cos(scattering angle), strictly in (−1, 1)
     * @param tau      Dimensionless electron temperature kT/(m_e c²)
     * @param Ne       Electron number density [cm⁻³] (use 1.0 for microscopic)
     * @return SigmaResult with value and error estimates
     */
    SigmaResult sigma_E(
        double const E,
        double const E_prime,
        double const xi,
        double const tau,
        double const Ne
    ) const;

    /**
     * @brief Run both double and DD power series, return the relative error.
     *
     * Computes |dd_value - dbl_value| / (|dd_value| + 1e-300).
     * Useful for checking whether double precision is sufficient for a
     * given set of parameters.
     */
    double sigma_E_precision_check(
        double const E,
        double const E_prime,
        double const xi,
        double const tau,
        double const Ne
    ) const;

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
    SigmaResult power_series(
        double const gamma,
        double const gamma_p,
        double const xi,
        double const tau,
        double const E,
        double const Ne
    ) const;

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
     */
    SigmaResult asymptotic_series(
        double const gamma,
        double const gamma_p,
        double const xi,
        double const tau,
        double const E,
        double const Ne
    ) const;

    SeriesMethod method_;
    double eps_rel_;
    int n_min_, n_max_;

};

} // namespace compton

#endif
