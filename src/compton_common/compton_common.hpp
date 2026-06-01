#ifndef COMPTON_COMMON_HPP
#define COMPTON_COMMON_HPP
/**
 * @file compton_common.hpp
 * @brief Shared kinematics and normalization for the Kershaw-Prasad-Beason
 *        Compton scattering kernel.
 *
 * Header-only common utilities used by both direct quadrature and series
 * evaluation modules.
 */

#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "doubledouble.h"
#include "units/units.hpp"

namespace compton {

using DD = doubledouble::DoubleDouble;

inline std::ostream& operator<<(std::ostream& os, DD const& x) {
    os << "(" << x.upper << ", " << x.lower << ")";
    return os;
}

namespace details {

inline double dd_to_double(DD const& x) {
    return x.upper + x.lower;
}

inline double to_double(double const value) {
    return value;
}

inline double to_double(DD const& value) {
    return dd_to_double(value);
}

inline DD dd_abs(DD const& x) {
    if (x.upper < 0.0 || (x.upper == 0.0 && x.lower < 0.0)) {
        return -x;
    }
    return x;
}

/**
 * @brief Inverse hyperbolic sine in double-double precision.
 *
 * asinh(x) = log(x + sqrt(x^2 + 1))
 */
inline DD dd_asinh(DD const& x) {
    return (x + (x * x + 1.0).sqrt()).log();
}

inline double param_abs(double const value) {
    return std::abs(value);
}

inline DD param_abs(DD const& value) {
    return dd_abs(value);
}

inline double param_sqrt(double const value) {
    return std::sqrt(value);
}

inline DD param_sqrt(DD const& value) {
    return value.sqrt();
}

inline double param_exp(double const value) {
    return std::exp(value);
}

inline DD param_exp(DD const& value) {
    return value.exp();
}

inline double param_asinh(double const value) {
    return std::asinh(value);
}

inline DD param_asinh(DD const& value) {
    return dd_asinh(value);
}

template<typename T> struct MachineEps;

template<>
struct MachineEps<double> {
    static constexpr double value = std::numeric_limits<double>::epsilon();
};

template<>
struct MachineEps<DD> {
    static constexpr double value = std::numeric_limits<double>::epsilon()
                                  * std::numeric_limits<double>::epsilon();
};

template<typename T>
struct EhatCfConfig;

template<>
struct EhatCfConfig<double> {
    static constexpr double cf_tol = 1e-14;
    static constexpr int max_iter = 1000;
};

template<>
struct EhatCfConfig<DD> {
    static constexpr double cf_tol = 1e-31;
    static constexpr int max_iter = 2000;
};

} // namespace details

/**
 * @brief Scaled exponential integral: Ehat_m(x) = exp(x) * E_m(x).
 *
 * Evaluated via the modified Lentz continued fraction (DLMF 8.9.2):
 *   Ehat_m(x) = 1/(x+m - m*1/(x+m+2 - (m+1)*2/(x+m+4 - ...)))
 */
template<typename T>
inline T ehat(
    int const m,
    T const& x,
    double const cf_tol = details::EhatCfConfig<T>::cf_tol,
    int const max_iter = details::EhatCfConfig<T>::max_iter) {
    using namespace details;

    if (!(x > 0.0)) {
        throw std::invalid_argument("ehat requires x > 0");
    }
    if (m < 1) {
        throw std::invalid_argument("ehat requires m >= 1");
    }

    constexpr double TINY = 1e-300;
    T const tiny_t = TINY;
    T b = x + m;
    if (param_abs(b) < tiny_t) {
        b = tiny_t;
    }

    T const one = 1.0;
    T f = b;
    T C = b;
    T D = 0.0;
    bool converged = false;

    for (int j = 1; j <= max_iter; ++j) {
        double const aj = -static_cast<double>(m + j - 1) * static_cast<double>(j);
        T const aj_t = aj;
        T const bj = x + (m + 2 * j);

        D = bj + D * aj;
        if (param_abs(D) < tiny_t) {
            D = tiny_t;
        }
        D = one / D;

        C = bj + aj_t / C;
        if (param_abs(C) < tiny_t) {
            C = tiny_t;
        }

        T const delta = C * D;
        f = f * delta;

        if (param_abs(delta - one) < cf_tol) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        std::ostringstream message;
        message << "ehat failed to converge: m=" << m;
        message << ", x=" << x;
        message << ", max_iter=" << max_iter
                << ", tol=" << cf_tol;
        throw std::runtime_error(message.str());
    }

    return one / f;
}

/**
 * @brief Scaled modified Bessel function: K̃₂(x) = exp(x) · K₂(x).
 *
 * Uses Boost cyl_bessel_k for x < 50 (numerically stable after multiplying
 * by exp(x)), and a 5-term Hankel asymptotic expansion for x ≥ 50 where
 * the direct computation would overflow/underflow.
 */
inline double scaled_K2(double x) {
    if (!(x > 0.0) || !std::isfinite(x))
        throw std::invalid_argument("scaled_K2 requires finite x > 0");

    if (x < 50.0) {
        return std::exp(x) * boost::math::cyl_bessel_k(2, x);
    }

    const double inv8x = 1.0 / (8.0 * x);
    constexpr double mu = 16.0;

    double term = 1.0;
    double sum = 1.0;

    term *= (mu - 1.0) * inv8x;
    sum += term;

    term *= (mu - 9.0) * inv8x / 2.0;
    sum += term;

    term *= (mu - 25.0) * inv8x / 3.0;
    sum += term;

    term *= (mu - 49.0) * inv8x / 4.0;
    sum += term;

    return std::sqrt(std::numbers::pi / (2.0 * x)) * sum;
}

/**
 * @brief Scaled modified Bessel function: K̃₁(x) = exp(x) · K₁(x).
 *
 * Same strategy as scaled_K2: Boost cyl_bessel_k for x < 50, Hankel
 * asymptotic for x >= 50.  Hankel parameter mu = 4·1² = 4 (vs 16 for K₂).
 */
inline double scaled_K1(double const x) {
    if (!(x > 0.0) || !std::isfinite(x))
        throw std::invalid_argument("scaled_K1 requires finite x > 0");

    if (x < 50.0) {
        return std::exp(x) * boost::math::cyl_bessel_k(1, x);
    }

    double const inv8x = 1.0 / (8.0 * x);
    constexpr double mu = 4.0;

    double term = 1.0;
    double sum = 1.0;

    term *= (mu - 1.0) * inv8x;
    sum += term;

    term *= (mu - 9.0) * inv8x / 2.0;
    sum += term;

    term *= (mu - 25.0) * inv8x / 3.0;
    sum += term;

    term *= (mu - 49.0) * inv8x / 4.0;
    sum += term;

    return std::sqrt(std::numbers::pi / (2.0 * x)) * sum;
}

/**
 * @brief Bessel ratio κ(τ) = K₁(1/τ) / K₂(1/τ).
 *
 * Computed via scaled Bessel functions to avoid overflow for small τ.
 * Asymptotics: κ → 1 as τ → 0, κ → 1/(2τ) as τ → ∞.
 */
inline double kappa_ratio(double const tau) {
    double const x = 1.0 / tau;
    return scaled_K1(x) / scaled_K2(x);
}

namespace constants {

/// Floor added to relative-error denominators to avoid division by zero.
constexpr double REL_ERROR_TINY_SCALE = 1e-300;

/// Guard to keep xi away from the direct-quadrature endpoint singularity.
constexpr double XI_DIRECT_QUADRATURE_GUARD = 1e-14;

constexpr double POISSON_Y_MAX = 500.0;

// rel_tol / eps_machine / safety_factor = 1e-13 / 1e-16 / 10
constexpr double EHAT_AMPLIFICATION_BUDGET = 1e2;

/// Minimum dimensionless photon energy (gamma = E / m_e c^2) at which the
/// double-precision power series agrees with the DD power series to better
/// than 1e-6.  Empirically validated: worst-case relative error at gamma =
/// 0.02 (~10 keV) is 3.15e-7 across all tested (E'/E, xi, T) combinations.
constexpr double GAMMA_DOUBLE_PRECISION_SAFE = 0.02;

/// Dispatch threshold for asymptotic vs power series: use asymptotic series
/// when tau * max(alpha_plus, alpha_minus) is below this value.
constexpr double ASYMP_TAU_ALPHA_THRESHOLD = 0.025;

} // namespace constants

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
template<typename T>
struct KershawParams {
    T a, s, q, omega2;
    T Delta, lambda_plus, rho_plus, rho_minus;
    T alpha_plus, alpha_minus;
    T G, A_plus, A_minus, Psi;
};

/**
 * @brief Compute all kinematic parameters from dimensionless energies.
 *
 * This is a pure function (no state); it derives (a, s, q, Δ, λ₊, ρ±, α±,
 * G, A±, Ψ) from the inputs.  Used by both quadrature and series modules.
 */
template<typename T>
inline KershawParams<T> compute_params(
    double const gamma, 
    double const gamma_p, 
    double const xi, 
    double const tau) {
    
    KershawParams<T> p{};

    T const gamma_t = static_cast<T>(gamma);
    T const gamma_p_t = static_cast<T>(gamma_p);
    T const xi_t = static_cast<T>(xi);
    T const tau_t = static_cast<T>(tau);
    T const one = static_cast<T>(1.0);
    T const two = static_cast<T>(2.0);

    p.a = one - xi_t;
    p.s = one / gamma_t + one / gamma_p_t;

    T const dg = gamma_p_t - gamma_t;
    T const q2 = dg * dg + two * gamma_t * gamma_p_t * p.a;
    p.q = details::param_sqrt(q2);

    p.omega2 = (one + xi_t) / p.a;

    T const gg_a = gamma_t * gamma_p_t * p.a;
    T const factor1 = one + gg_a / two;
    T const factor2 = one + (dg * dg) / (two * gg_a);
    p.Delta = details::param_sqrt(factor1 * factor2);

    p.lambda_plus = dg / two + p.Delta;

    if (p.lambda_plus < one - 1e-12)
        throw std::runtime_error("lambda_plus significantly below 1");
    if (p.lambda_plus < one)
        p.lambda_plus = one;

    p.rho_plus = p.lambda_plus + gamma_t;
    p.rho_minus = p.lambda_plus - gamma_p_t;

    T const Rp0 = p.rho_plus * p.rho_plus + p.omega2;
    T const Rm0 = p.rho_minus * p.rho_minus + p.omega2;
    p.alpha_plus = one / details::param_sqrt(Rp0);
    p.alpha_minus = one / details::param_sqrt(Rm0);

    T const a2 = p.a * p.a;
    p.G = -gamma_t * gamma_p_t + two / p.a + two / (gamma_t * gamma_p_t * a2);

    T const s_over_tau_a2 = p.s / (tau_t * a2);
    p.A_plus = p.G - s_over_tau_a2;
    p.A_minus = p.G + s_over_tau_a2;

    p.Psi = two * tau_t * gamma_t * gamma_p_t / p.q
          + p.s / a2 * (p.alpha_plus + p.alpha_minus)
          + (p.rho_plus * p.alpha_plus - p.rho_minus * p.alpha_minus) / p.a;

    return p;
}

/**
 * @brief Compute the prefactor σ₀ = Nₑ r_e² m_e c² / (4E²τ)
 *                                    × exp(−(λ₊−1)/τ) / K̃₂(1/τ).
 *
 * The exponential suppression factor exp(−(λ₊−1)/τ) controls the kernel
 * magnitude: elastic scattering (λ₊→1) has no suppression, while large
 * energy transfers (λ₊≫1) are exponentially suppressed.
 */
inline double sigma0_E(
    double const E, 
    double const tau, 
    double const lambda_plus, 
    double const Ne) {
    return Ne * units::r_e2 * units::me_c2
           / (4.0 * E * E * tau)
           * std::exp(-(lambda_plus - 1.0) / tau)
           / scaled_K2(1.0 / tau);
}

/// Result of a kernel evaluation: value plus heuristic error estimates.
struct SigmaResult {
    double value;               /// Σ_E in [cm²/erg] (Nₑ=1) or [1/(cm·erg)]
    double estimated_abs_error; ///
    double estimated_rel_error; /// abs_error / |value|
};

/**
 * @brief Validate the five common kernel parameters.
 *
 * Throws std::invalid_argument when any precondition is violated.
 *
 * @param E        Incident photon energy [erg]
 * @param E_prime  Scattered photon energy [erg]
 * @param xi       cos(scattering angle), strictly in (-1, 1)
 * @param T        Electron temperature [K]
 * @param Ne       Electron number density [cm^-3]
 */
inline void assert_parameters(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne)
{
    if (!(E > 0.0) || !std::isfinite(E))
        throw std::invalid_argument("E must be finite and > 0");
    if (!(E_prime > 0.0) || !std::isfinite(E_prime))
        throw std::invalid_argument("E_prime must be finite and > 0");
    if (!(T > 0.0) || !std::isfinite(T))
        throw std::invalid_argument("T must be finite and > 0");
    if (!(xi > -1.0 && xi < 1.0) || !std::isfinite(xi))
        throw std::invalid_argument("xi must be finite and strictly inside (-1, 1)");
    if (!std::isfinite(Ne))
        throw std::invalid_argument("Ne must be finite");
    if (1.0 - xi < 1e-14)
        throw std::invalid_argument("xi too close to 1");
}

} // namespace compton

#endif
