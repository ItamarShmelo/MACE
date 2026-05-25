/**
 * @file compton_kernel_quadrature.cpp
 * @brief Implementation of the Kershaw-Prasad-Beason Compton frequency kernel.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * OVERALL ALGORITHM
 * ─────────────────────────────────────────────────────────────────────────
 *
 * For a given (E, E', ξ, τ, Nₑ) the evaluation proceeds as:
 *
 *   1.  Convert to dimensionless energies: γ = E/m_e c², γ' = E'/m_e c².
 *
 *   2.  compute_params():  Derive all kinematic quantities (a, s, q, Δ, λ₊,
 *       ρ±, α±, G, A±, Ψ) from (γ, γ', ξ, τ).  Special care:
 *         - q² uses the "two-term" form  (γ'−γ)² + 2γγ'(1−ξ)  to avoid
 *           cancellation when γ ≈ γ'.
 *         - λ₊ is clamped to ≥ 1 (physical lower bound: min electron energy).
 *
 *   3.  stable_sigma0_E():  Compute the prefactor
 *           σ₀ = Nₑ r_e² m_e c² / (4 E² τ) · exp(−(λ₊−1)/τ) / K̃₂(1/τ)
 *       The exponential and K̃₂ are kept in "scaled" form to avoid overflow:
 *       the exp(−(λ₊−1)/τ) suppression is the dominant factor controlling
 *       the kernel magnitude (elastic: λ₊→1, no suppression; large energy
 *       transfer: λ₊≫1, exponentially small).
 *
 *   4.  Evaluate the semi-infinite integral I_Q via Gauss-Laguerre quadrature.
 *       The integral has the form:
 *
 *           I_Q = ∫₀^∞  f(τx + ρ_offset) · e^{−x} dx
 *
 *       after the substitution ρ = τx + ρ_offset that maps the domain
 *       [λ₊, ∞) → [0, ∞) and exposes the e^{−x} weight for Gauss-Laguerre.
 *       Two forms are available (see header).
 *
 *   5.  Return  σ₀ · (Ψ + I_Q)  [post-IBP]  or  σ₀ · I_Q  [pre-IBP],
 *       together with a Richardson-extrapolation error estimate:
 *           abs_error = |σ₀| · |IQ(N) − IQ(N/2)|
 *
 * ─────────────────────────────────────────────────────────────────────────
 * NUMERICAL STABILITY CONSIDERATIONS
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   - scaled_K2:  For large 1/τ (cold plasma), K₂(1/τ) overflows but
 *     exp(1/τ)·K₂(1/τ) remains O(√τ).  We use Boost for moderate arguments
 *     and an asymptotic Hankel series for x ≥ 50.
 *
 *   - The post-IBP integrand has a boundary term Ψ that captures the
 *     "elastic peak" contribution analytically, improving convergence of
 *     the numerical integral for the smooth remainder.  However, at very
 *     low τ the Ψ and I_Q terms nearly cancel (catastrophic cancellation),
 *     making pre-IBP preferable in that regime.
 *
 *   - Quadrature rules are computed once and cached (static) for the
 *     supported orders (32, 64, 128, 256).
 */

#include "compton_kernel_quadrature.hpp"
#include "gauss_laguerre.hpp"

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

    // Large-x asymptotic: kve(nu, x) ~ sqrt(pi/(2x)) * sum
    // mu = 4*nu^2 = 16 for nu=2
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
// Gauss-Laguerre integration helpers
// ═══════════════════════════════════════════════════════════════════════════

template<typename F>
static double laguerre_integrate(F&& integrand, const GaussLaguerreRule& rule) {
    double sum = 0.0;
    const int N = static_cast<int>(rule.nodes.size());
    for (int i = 0; i < N; ++i) {
        sum += rule.weights[i] * integrand(rule.nodes[i]);
    }
    return sum;
}

// Lazily-initialized static cache of quadrature rules.
// Computing a 256-point rule via tql2 takes ~ms; caching avoids repetition.
static const GaussLaguerreRule& get_rule(int N) {
    static const GaussLaguerreRule rule_32 = compute_gauss_laguerre(32);
    static const GaussLaguerreRule rule_64 = compute_gauss_laguerre(64);
    static const GaussLaguerreRule rule_128 = compute_gauss_laguerre(128);
    static const GaussLaguerreRule rule_256 = compute_gauss_laguerre(256);

    switch (N) {
        case 32:  return rule_32;
        case 64:  return rule_64;
        case 128: return rule_128;
        case 256: return rule_256;
        default:  throw std::invalid_argument("N must be one of: 32, 64, 128, 256");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ComptonKernelQuadrature implementation
// ═══════════════════════════════════════════════════════════════════════════

ComptonKernelQuadrature::ComptonKernelQuadrature(int NL, QuadratureForm form)
    : NL_(NL), form_(form)
{
    if (NL != 64 && NL != 128 && NL != 256)
        throw std::invalid_argument("NL must be one of: 64, 128, 256");
}

// ─────────────────────────────────────────────────────────────────────────
// compute_params: derive all kinematic intermediates from (γ, γ', ξ, τ).
//
// Key derived quantities and their physical meaning:
//   a = 1−ξ ∈ (0,2]        : proportional to squared momentum transfer
//   q = √(Δγ² + 2γγ'a)    : 3-momentum transfer in electron rest frame
//   λ₊                     : minimum electron Lorentz factor that can
//                             produce the (E→E',ξ) transition
//   Ψ                      : boundary-evaluation term arising from IBP
// ─────────────────────────────────────────────────────────────────────────
KershawParams ComptonKernelQuadrature::compute_params(
    double gamma, double gamma_p, double xi, double tau) const
{
    KershawParams p{};

    p.a = 1.0 - xi;
    p.s = 1.0 / gamma + 1.0 / gamma_p;

    // Stable q^2
    double dg = gamma_p - gamma;
    double q2 = dg * dg + 2.0 * gamma * gamma_p * p.a;
    p.q = std::sqrt(q2);

    // omega^2
    p.omega2 = (1.0 + xi) / p.a;

    // Delta
    double gg_a = gamma * gamma_p * p.a;
    double factor1 = 1.0 + gg_a / 2.0;
    double factor2 = 1.0 + (dg * dg) / (2.0 * gg_a);
    p.Delta = std::sqrt(factor1 * factor2);

    // lambda_+, rho_+, rho_-
    p.lambda_plus = dg / 2.0 + p.Delta;

    if (p.lambda_plus < 1.0 - 1e-12)
        throw std::runtime_error("lambda_plus significantly below 1");
    if (p.lambda_plus < 1.0)
        p.lambda_plus = 1.0;

    p.rho_plus = p.lambda_plus + gamma;
    p.rho_minus = p.lambda_plus - gamma_p;

    // alpha_+/-
    double Rp0 = p.rho_plus * p.rho_plus + p.omega2;
    double Rm0 = p.rho_minus * p.rho_minus + p.omega2;
    p.alpha_plus = 1.0 / std::sqrt(Rp0);
    p.alpha_minus = 1.0 / std::sqrt(Rm0);

    // G
    double a2 = p.a * p.a;
    p.G = -gamma * gamma_p + 2.0 / p.a + 2.0 / (gamma * gamma_p * a2);

    // A_+, A_-
    double s_over_tau_a2 = p.s / (tau * a2);
    p.A_plus = p.G - s_over_tau_a2;
    p.A_minus = p.G + s_over_tau_a2;

    // Psi
    p.Psi = 2.0 * tau * gamma * gamma_p / p.q
           + p.s / a2 * (p.alpha_plus + p.alpha_minus)
           + (p.rho_plus * p.alpha_plus - p.rho_minus * p.alpha_minus) / p.a;

    return p;
}

// ─────────────────────────────────────────────────────────────────────────
// stable_sigma0_E:  prefactor  σ₀ = Nₑ r_e² m_e c² / (4E²τ)
//                                    × exp(−(λ₊−1)/τ) / K̃₂(1/τ)
//
// The factor exp(−(λ₊−1)/τ) provides exponential suppression for
// inelastic transitions (λ₊ > 1).  For elastic scattering (E≈E', ξ≈1),
// λ₊ → 1 and the suppression vanishes.
// ─────────────────────────────────────────────────────────────────────────
double ComptonKernelQuadrature::stable_sigma0_E(
    double E, double tau, double lambda_plus, double Ne) const
{
    return Ne * units::r_e2 * units::me_c2
           / (4.0 * E * E * tau)
           * std::exp(-(lambda_plus - 1.0) / tau)
           / scaled_K2(1.0 / tau);
}

// ─────────────────────────────────────────────────────────────────────────
// Post-IBP quadrature form.
//
// After integration by parts the original 1/R^{3/2} integrand becomes
// a 1/√R integrand (less singular, faster Gauss-Laguerre convergence)
// plus a boundary term Ψ evaluated at ρ = λ₊ (already computed in params).
//
// The integrand (as a function of x, where ρ = τx + ρ_offset):
//
//    H(x) = [(A₊ − ρ₊/τa) / √R₊  +  (−A₋ + ρ₋/τa) / √R₋] · τ
//
// where R± = (τx + ρ±)² + ω²  are always > 0.
// ─────────────────────────────────────────────────────────────────────────
double ComptonKernelQuadrature::compute_IQ_post_ibp(
    const KershawParams& p, double tau, int NL) const
{
    auto integrand = [&](double x) -> double {
        double rho = tau * x;

        double rp = rho + p.rho_plus;
        double rm = rho + p.rho_minus;

        double Rp = rp * rp + p.omega2;
        double Rm = rm * rm + p.omega2;

        double inv_sqrt_Rp = 1.0 / std::sqrt(Rp);
        double inv_sqrt_Rm = 1.0 / std::sqrt(Rm);

        double tau_a = tau * p.a;
        double H = (p.A_plus - rp / tau_a) * inv_sqrt_Rp
                 + (-p.A_minus + rm / tau_a) * inv_sqrt_Rm;

        return tau * H;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

// ─────────────────────────────────────────────────────────────────────────
// Pre-IBP quadrature form.
//
// The original integrand before integration by parts, containing terms
// with 1/R±^{3/2} (from the derivative of 1/√R±).  More terms but no
// boundary contribution (no Ψ), and converges uniformly even at low τ
// where the post-IBP form suffers cancellation between Ψ and I_Q.
//
// The integrand structure:
//   F(x) = const_term + bracket_{3/2}(x) + bracket_{1/2}(x)
//
// where:
//   const_term = 2γγ'/q  (angle-dependent Klein-Nishina-like factor)
//   bracket_{3/2} contains numerator polynomials divided by R±^{3/2}
//   bracket_{1/2} contains G·(1/√R₊ − 1/√R₋)
// ─────────────────────────────────────────────────────────────────────────
double ComptonKernelQuadrature::compute_IQ_pre_ibp(
    const KershawParams& p, double tau, int NL) const
{
    double gamma_val = p.rho_plus - p.lambda_plus;
    double gamma_p_val = p.lambda_plus - p.rho_minus;
    double const_term = 2.0 * gamma_val * gamma_p_val / p.q;
    double a2 = p.a * p.a;
    double one_plus_xi = 2.0 - p.a;

    auto integrand = [&](double x) -> double {
        double t_plus = tau * x + p.rho_plus;
        double t_minus = tau * x + p.rho_minus;

        double Rp = t_plus * t_plus + p.omega2;
        double Rm = t_minus * t_minus + p.omega2;

        double inv_sqrt_Rp = 1.0 / std::sqrt(Rp);
        double inv_sqrt_Rm = 1.0 / std::sqrt(Rm);
        double inv_Rp_32 = inv_sqrt_Rp / Rp;
        double inv_Rm_32 = inv_sqrt_Rm / Rm;

        double num_plus = t_minus * p.s + one_plus_xi;
        double num_minus = t_plus * p.s - one_plus_xi;

        double bracket_32 = (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2;
        double bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm);

        double F = const_term + bracket_32 + bracket_12;
        return tau * F;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

// ─────────────────────────────────────────────────────────────────────────
// sigma_E:  top-level entry point combining prefactor and quadrature.
//
// Evaluates at a single phase-space point.  For bin-integrated quantities
// (as needed for multigroup transport), the caller must perform the outer
// integration over ξ ∈ (−1,1) and E' ∈ [E'_lo, E'_hi] externally
// (typically via scipy.integrate in Python).
// ─────────────────────────────────────────────────────────────────────────
SigmaResult ComptonKernelQuadrature::sigma_E(
    double E, double E_prime, double xi, double tau, double Ne) const
{
    if (!(E > 0.0) || !std::isfinite(E))
        throw std::invalid_argument("E must be finite and > 0");
    if (!(E_prime > 0.0) || !std::isfinite(E_prime))
        throw std::invalid_argument("E_prime must be finite and > 0");
    if (!(tau > 0.0) || !std::isfinite(tau))
        throw std::invalid_argument("tau must be finite and > 0");
    if (!(xi > -1.0 && xi < 1.0) || !std::isfinite(xi))
        throw std::invalid_argument("xi must be finite and strictly inside (-1, 1)");
    if (!std::isfinite(Ne))
        throw std::invalid_argument("Ne must be finite");
    if (1.0 - xi < 1e-14)
        throw std::invalid_argument("xi too close to 1 for direct quadrature");

    double gamma = E / units::me_c2;
    double gamma_p = E_prime / units::me_c2;

    KershawParams p = compute_params(gamma, gamma_p, xi, tau);
    double sigma0 = stable_sigma0_E(E, tau, p.lambda_plus, Ne);

    double IQ_hi, IQ_lo;

    if (form_ == QuadratureForm::PostIntegrationByParts) {
        IQ_hi = compute_IQ_post_ibp(p, tau, NL_);
        IQ_lo = compute_IQ_post_ibp(p, tau, NL_ / 2);
        double value = sigma0 * (p.Psi + IQ_hi);

        double abs_err = std::abs(sigma0) * std::abs(IQ_hi - IQ_lo);
        constexpr double tiny_scale = 1e-300;
        double rel_err = abs_err / (std::abs(value) + tiny_scale);

        return SigmaResult{value, abs_err, rel_err};
    } else {
        IQ_hi = compute_IQ_pre_ibp(p, tau, NL_);
        IQ_lo = compute_IQ_pre_ibp(p, tau, NL_ / 2);
        double value = sigma0 * IQ_hi;

        double abs_err = std::abs(sigma0) * std::abs(IQ_hi - IQ_lo);
        constexpr double tiny_scale = 1e-300;
        double rel_err = abs_err / (std::abs(value) + tiny_scale);

        return SigmaResult{value, abs_err, rel_err};
    }
}

} // namespace compton
