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
 *   2.  compute_params<T>():  Derive all kinematic quantities (a, s, q, Δ, λ₊,
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

namespace compton {

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

double ComptonKernelQuadrature::compute_IQ_post_ibp(
    const KershawParams<double>& p, double tau, int NL) const
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

double ComptonKernelQuadrature::compute_IQ_pre_ibp(
    const KershawParams<double>& p, double tau, int NL) const
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

    KershawParams<double> p = compute_params<double>(gamma, gamma_p, xi, tau);
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
