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
// compute_params_dd: DD-precision kinematic parameters
// ═══════════════════════════════════════════════════════════════════════════

KershawParamsDD compute_params_dd(double gamma, double gamma_p, double xi, double tau) {
    KershawParamsDD p{};

    dd gamma_dd   = dd_from_double(gamma);
    dd gamma_p_dd = dd_from_double(gamma_p);
    dd xi_dd      = dd_from_double(xi);
    dd tau_dd     = dd_from_double(tau);
    dd one        = dd_from_double(1.0);
    dd two        = dd_from_double(2.0);

    p.a = dd_sub(one, xi_dd);
    p.s = dd_add(dd_div(one, gamma_dd), dd_div(one, gamma_p_dd));

    dd dg  = dd_sub(gamma_p_dd, gamma_dd);
    dd dg2 = dd_mul(dg, dg);
    dd gg  = dd_mul(gamma_dd, gamma_p_dd);
    dd q2  = dd_add(dg2, dd_mul(dd_mul(two, gg), p.a));
    p.q    = dd_sqrt(q2);

    p.omega2 = dd_div(dd_add(one, xi_dd), p.a);

    dd gg_a    = dd_mul(gg, p.a);
    dd factor1 = dd_add(one, dd_div(gg_a, two));
    dd factor2 = dd_add(one, dd_div(dg2, dd_mul(two, gg_a)));
    p.Delta    = dd_sqrt(dd_mul(factor1, factor2));

    p.lambda_plus = dd_add(dd_div(dg, two), p.Delta);

    if (p.lambda_plus.hi < 1.0 - 1e-12)
        throw std::runtime_error("lambda_plus significantly below 1");
    if (p.lambda_plus.hi < 1.0)
        p.lambda_plus = one;

    p.rho_plus  = dd_add(p.lambda_plus, gamma_dd);
    p.rho_minus = dd_sub(p.lambda_plus, gamma_p_dd);

    dd Rp0 = dd_add(dd_mul(p.rho_plus, p.rho_plus), p.omega2);
    dd Rm0 = dd_add(dd_mul(p.rho_minus, p.rho_minus), p.omega2);
    p.alpha_plus  = dd_div(one, dd_sqrt(Rp0));
    p.alpha_minus = dd_div(one, dd_sqrt(Rm0));

    dd a2 = dd_mul(p.a, p.a);
    p.G = dd_add(dd_sub(dd_from_double(0.0), gg),
                 dd_add(dd_div(two, p.a),
                        dd_div(two, dd_mul(gg, a2))));

    dd s_over_tau_a2 = dd_div(p.s, dd_mul(tau_dd, a2));
    p.A_plus  = dd_sub(p.G, s_over_tau_a2);
    p.A_minus = dd_add(p.G, s_over_tau_a2);

    // Psi = 2*tau*gamma*gamma_p/q + s/a^2*(alpha+ + alpha-) + (rho+*alpha+ - rho-*alpha-)/a
    dd term1 = dd_div(dd_mul(dd_mul(two, tau_dd), gg), p.q);
    dd term2 = dd_mul(dd_div(p.s, a2),
                      dd_add(p.alpha_plus, p.alpha_minus));
    dd term3 = dd_div(dd_sub(dd_mul(p.rho_plus, p.alpha_plus),
                             dd_mul(p.rho_minus, p.alpha_minus)),
                      p.a);
    p.Psi = dd_add(dd_add(term1, term2), term3);

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
