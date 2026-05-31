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
 *   3.  sigma0_E():  Compute the prefactor
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
 *   - scaled_K2:  For large 1/τ, K₂(1/τ) overflows but
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
// ComptonKernelQuadrature implementation
// ═══════════════════════════════════════════════════════════════════════════

ComptonKernelQuadrature::ComptonKernelQuadrature(int NL, QuadratureForm form)
    : NL_(NL), form_(form)
{
    if (NL != 64 && NL != 128 && NL != 256)
        throw std::invalid_argument("NL must be one of: 64, 128, 256");
}

double ComptonKernelQuadrature::compute_IQ_post_ibp(
    KershawParams<double> const& p,
    double const tau,
    int const NL) const
{
    auto integrand = [&](double const x) -> double {
        double const rho = tau * x;

        double const rp = rho + p.rho_plus;
        double const rm = rho + p.rho_minus;

        double const Rp = rp * rp + p.omega2;
        double const Rm = rm * rm + p.omega2;

        double const inv_sqrt_Rp = 1.0 / std::sqrt(Rp);
        double const inv_sqrt_Rm = 1.0 / std::sqrt(Rm);

        double const tau_a = tau * p.a;
        double const H = (p.A_plus - rp / tau_a) * inv_sqrt_Rp
                       + (-p.A_minus + rm / tau_a) * inv_sqrt_Rm;

        return tau * H;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

double ComptonKernelQuadrature::compute_IQ_pre_ibp(
    KershawParams<double> const& p,
    double const tau,
    int const NL) const
{
    double const gamma_val = p.rho_plus - p.lambda_plus;
    double const gamma_p_val = p.lambda_plus - p.rho_minus;
    double const const_term = 2.0 * gamma_val * gamma_p_val / p.q;
    double const a2 = p.a * p.a;
    double const one_plus_xi = 2.0 - p.a;

    auto integrand = [&](double const x) -> double {
        double const t_plus = tau * x + p.rho_plus;
        double const t_minus = tau * x + p.rho_minus;

        double const Rp = t_plus * t_plus + p.omega2;
        double const Rm = t_minus * t_minus + p.omega2;

        double const inv_sqrt_Rp = 1.0 / std::sqrt(Rp);
        double const inv_sqrt_Rm = 1.0 / std::sqrt(Rm);
        double const inv_Rp_32 = inv_sqrt_Rp / Rp;
        double const inv_Rm_32 = inv_sqrt_Rm / Rm;

        double const num_plus = t_minus * p.s + one_plus_xi;
        double const num_minus = t_plus * p.s - one_plus_xi;

        double const bracket_32 = (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2;
        double const bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm);

        double const F = const_term + bracket_32 + bracket_12;
        return tau * F;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

SigmaResult ComptonKernelQuadrature::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const tau,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, tau, Ne);

    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    KershawParams<double> const p = compute_params<double>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, p.lambda_plus, Ne);

    if (form_ == QuadratureForm::PostIntegrationByParts) {
        double const IQ_hi = compute_IQ_post_ibp(p, tau, NL_);
        double const IQ_lo = compute_IQ_post_ibp(p, tau, NL_ / 2);
        double const value = sigma0 * (p.Psi + IQ_hi);

        double const abs_err = std::abs(sigma0) * std::abs(IQ_hi - IQ_lo);
        double const rel_err = abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);

        return SigmaResult{value, abs_err, rel_err};
    } else {
        double const IQ_hi = compute_IQ_pre_ibp(p, tau, NL_);
        double const IQ_lo = compute_IQ_pre_ibp(p, tau, NL_ / 2);
        double const value = sigma0 * IQ_hi;

        double const abs_err = std::abs(sigma0) * std::abs(IQ_hi - IQ_lo);
        double const rel_err = abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);

        return SigmaResult{value, abs_err, rel_err};
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Temperature derivative: ∂Σ_E/∂τ
// ═══════════════════════════════════════════════════════════════════════════

double ComptonKernelQuadrature::compute_dIQ_post_ibp(
    KershawParams<double> const& p,
    double const tau,
    int const NL,
    double const kappa_val) const
{
    double const a2 = p.a * p.a;
    double const tau2 = tau * tau;

    auto integrand = [&](double const x) -> double {
        double const rho = tau * x;

        double const rp = rho + p.rho_plus;
        double const rm = rho + p.rho_minus;

        double const Rp = rp * rp + p.omega2;
        double const Rm = rm * rm + p.omega2;

        double const inv_sqrt_Rp = 1.0 / std::sqrt(Rp);
        double const inv_sqrt_Rm = 1.0 / std::sqrt(Rm);

        double const A_plus_rho = p.A_plus - rp / (tau * p.a);
        double const A_minus_rho = -p.A_minus + rm / (tau * p.a);

        double const B_plus = p.s / a2 + rp / p.a;
        double const B_minus = rm / p.a - p.s / a2;

        double const lam = p.lambda_plus + rho;
        double const dlnSig0 = (lam - kappa_val) / tau2 - 3.0 / tau;

        double const plus_term = (dlnSig0 * A_plus_rho + B_plus / tau2) * inv_sqrt_Rp;
        double const minus_term = (dlnSig0 * A_minus_rho - B_minus / tau2) * inv_sqrt_Rm;

        return tau * (plus_term + minus_term);
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

double ComptonKernelQuadrature::compute_dIQ_pre_ibp(
    KershawParams<double> const& p,
    double const tau,
    int const NL,
    double const kappa_val) const
{
    double const gamma_val = p.rho_plus - p.lambda_plus;
    double const gamma_p_val = p.lambda_plus - p.rho_minus;
    double const const_term = 2.0 * gamma_val * gamma_p_val / p.q;
    double const a2 = p.a * p.a;
    double const one_plus_xi = 2.0 - p.a;
    double const tau2 = tau * tau;

    auto integrand = [&](double const x) -> double {
        double const t_plus = tau * x + p.rho_plus;
        double const t_minus = tau * x + p.rho_minus;

        double const Rp = t_plus * t_plus + p.omega2;
        double const Rm = t_minus * t_minus + p.omega2;

        double const inv_sqrt_Rp = 1.0 / std::sqrt(Rp);
        double const inv_sqrt_Rm = 1.0 / std::sqrt(Rm);
        double const inv_Rp_32 = inv_sqrt_Rp / Rp;
        double const inv_Rm_32 = inv_sqrt_Rm / Rm;

        double const num_plus = t_minus * p.s + one_plus_xi;
        double const num_minus = t_plus * p.s - one_plus_xi;

        double const bracket_32 = (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2;
        double const bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm);

        double const F = const_term + bracket_32 + bracket_12;

        double const lam = p.lambda_plus + tau * x;
        double const weight = (lam - 3.0 * tau - kappa_val) / tau2;

        return tau * weight * F;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

SigmaResult ComptonKernelQuadrature::dsigma_E_dtau(
    double const E,
    double const E_prime,
    double const xi,
    double const tau,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, tau, Ne);

    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    KershawParams<double> const p = compute_params<double>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, p.lambda_plus, Ne);
    double const kappa_val = kappa_ratio(tau);

    if (form_ == QuadratureForm::PostIntegrationByParts) {
        double const dlnSig0 = (p.lambda_plus - kappa_val) / (tau * tau) - 3.0 / tau;
        double const dPsi_dtau = 2.0 * gamma * gamma_p / p.q;
        double const non_integral = dlnSig0 * p.Psi + dPsi_dtau;

        double const dIQ_hi = compute_dIQ_post_ibp(p, tau, NL_, kappa_val);
        double const dIQ_lo = compute_dIQ_post_ibp(p, tau, NL_ / 2, kappa_val);
        double const value = sigma0 * (non_integral + dIQ_hi);

        double const abs_err = std::abs(sigma0) * std::abs(dIQ_hi - dIQ_lo);
        double const rel_err = abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
        return SigmaResult{value, abs_err, rel_err};

    } else {
        double const dIQ_hi = compute_dIQ_pre_ibp(p, tau, NL_, kappa_val);
        double const dIQ_lo = compute_dIQ_pre_ibp(p, tau, NL_ / 2, kappa_val);
        double const value = sigma0 * dIQ_hi;

        double const abs_err = std::abs(sigma0) * std::abs(dIQ_hi - dIQ_lo);
        double const rel_err = abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
        return SigmaResult{value, abs_err, rel_err};
    }
}

} // namespace compton
