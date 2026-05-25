/**
 * @file compton_common.cpp
 * @brief Shared kinematics and normalization for the Compton kernel.
 *
 * Implements scaled_K2, compute_params, and stable_sigma0_E as free
 * functions used by both the quadrature and series evaluation modules.
 */

#include "compton_common.hpp"

#include <boost/math/special_functions/bessel.hpp>

namespace compton {

// ═══════════════════════════════════════════════════════════════════════════
// scaled_K2:  K̃₂(x) = exp(x) · K₂(x)
// ═══════════════════════════════════════════════════════════════════════════
//
// For x < 50: compute K₂(x) via Boost and multiply by exp(x).
// For x ≥ 50: use the Hankel asymptotic expansion
//     K̃_ν(x) ~ √(π/(2x)) · Σ_{k=0}^{4} (μ−1²)(μ−3²)...(μ−(2k−1)²) / (k! (8x)^k)
// with μ = 4ν² = 16.  Five terms give relative error < 10⁻¹⁵ for x ≥ 50.

double scaled_K2(double x) {
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

// ═══════════════════════════════════════════════════════════════════════════
// compute_params: derive all kinematic intermediates from (γ, γ', ξ, τ).
// ═══════════════════════════════════════════════════════════════════════════

KershawParams compute_params(double gamma, double gamma_p, double xi, double tau) {
    KershawParams p{};

    p.a = 1.0 - xi;
    p.s = 1.0 / gamma + 1.0 / gamma_p;

    double dg = gamma_p - gamma;
    double q2 = dg * dg + 2.0 * gamma * gamma_p * p.a;
    p.q = std::sqrt(q2);

    p.omega2 = (1.0 + xi) / p.a;

    double gg_a = gamma * gamma_p * p.a;
    double factor1 = 1.0 + gg_a / 2.0;
    double factor2 = 1.0 + (dg * dg) / (2.0 * gg_a);
    p.Delta = std::sqrt(factor1 * factor2);

    p.lambda_plus = dg / 2.0 + p.Delta;

    if (p.lambda_plus < 1.0 - 1e-12)
        throw std::runtime_error("lambda_plus significantly below 1");
    if (p.lambda_plus < 1.0)
        p.lambda_plus = 1.0;

    p.rho_plus = p.lambda_plus + gamma;
    p.rho_minus = p.lambda_plus - gamma_p;

    double Rp0 = p.rho_plus * p.rho_plus + p.omega2;
    double Rm0 = p.rho_minus * p.rho_minus + p.omega2;
    p.alpha_plus = 1.0 / std::sqrt(Rp0);
    p.alpha_minus = 1.0 / std::sqrt(Rm0);

    double a2 = p.a * p.a;
    p.G = -gamma * gamma_p + 2.0 / p.a + 2.0 / (gamma * gamma_p * a2);

    double s_over_tau_a2 = p.s / (tau * a2);
    p.A_plus = p.G - s_over_tau_a2;
    p.A_minus = p.G + s_over_tau_a2;

    p.Psi = 2.0 * tau * gamma * gamma_p / p.q
           + p.s / a2 * (p.alpha_plus + p.alpha_minus)
           + (p.rho_plus * p.alpha_plus - p.rho_minus * p.alpha_minus) / p.a;

    return p;
}

// ═══════════════════════════════════════════════════════════════════════════
// stable_sigma0_E
// ═══════════════════════════════════════════════════════════════════════════

double stable_sigma0_E(double E, double tau, double lambda_plus, double Ne) {
    return Ne * units::r_e2 * units::me_c2
           / (4.0 * E * E * tau)
           * std::exp(-(lambda_plus - 1.0) / tau)
           / scaled_K2(1.0 / tau);
}

} // namespace compton
