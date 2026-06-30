#include "compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_common/compton_common.hpp"
#include "utilities/gauss_laguerre.hpp"
#include "utilities/units.hpp"

#include <cmath>
#include <stdexcept>

namespace compton {

ComptonKernelQuadrature::ComptonKernelQuadrature(int NL, QuadratureForm form)
    : NL_(NL),
      form_(form)
{
    if (NL != 32 && NL != 64 && NL != 128 && NL != 256)
        throw std::invalid_argument("NL must be one of: 32, 64, 128, 256");
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
        double const H = (p.A_plus - rp / tau_a) * inv_sqrt_Rp +
                         (-p.A_minus + rm / tau_a) * inv_sqrt_Rm;

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

        double const bracket_32 =
            (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2;
        double const bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm);

        double const F = const_term + bracket_32 + bracket_12;
        return tau * F;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

ComptonResult ComptonKernelQuadrature::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau = T * units::k_boltz / units::me_c2;

    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    KershawParams<double> const p =
        compute_params<double>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, p.lambda_plus, Ne);

    if (form_ == QuadratureForm::PostIntegrationByParts) {
        double const IQ_hi = compute_IQ_post_ibp(p, tau, NL_);
        double const IQ_lo = compute_IQ_post_ibp(p, tau, NL_ / 2);
        double const value = sigma0 * (p.Psi + IQ_hi);

        double const abs_err = std::abs(sigma0) * std::abs(IQ_hi - IQ_lo);
        double const rel_err =
            abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);

        return ComptonResult{value, abs_err, rel_err};
    } else {
        double const IQ_hi = compute_IQ_pre_ibp(p, tau, NL_);
        double const IQ_lo = compute_IQ_pre_ibp(p, tau, NL_ / 2);
        double const value = sigma0 * IQ_hi;

        double const abs_err = std::abs(sigma0) * std::abs(IQ_hi - IQ_lo);
        double const rel_err =
            abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);

        return ComptonResult{value, abs_err, rel_err};
    }
}

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

        double const plus_term =
            (dlnSig0 * A_plus_rho + B_plus / tau2) * inv_sqrt_Rp;
        double const minus_term =
            (dlnSig0 * A_minus_rho - B_minus / tau2) * inv_sqrt_Rm;

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

        double const bracket_32 =
            (num_plus * inv_Rm_32 + num_minus * inv_Rp_32) / a2;
        double const bracket_12 = p.G * (inv_sqrt_Rp - inv_sqrt_Rm);

        double const F = const_term + bracket_32 + bracket_12;

        double const lam = p.lambda_plus + tau * x;
        double const weight = (lam - 3.0 * tau - kappa_val) / tau2;

        return tau * weight * F;
    };

    return laguerre_integrate(integrand, get_rule(NL));
}

ComptonResult ComptonKernelQuadrature::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    assert_parameters(E, E_prime, xi, T, Ne);
    double const tau = T * units::k_boltz / units::me_c2;
    double const dtau_dT = units::k_boltz / units::me_c2;

    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    KershawParams<double> const p =
        compute_params<double>(gamma, gamma_p, xi, tau);
    double const sigma0 = sigma0_E(E, tau, p.lambda_plus, Ne);
    double const kappa_val = kappa_ratio(tau);

    if (form_ == QuadratureForm::PostIntegrationByParts) {
        double const dlnSig0 =
            (p.lambda_plus - kappa_val) / (tau * tau) - 3.0 / tau;
        double const dPsi_dtau = 2.0 * gamma * gamma_p / p.q;
        double const non_integral = dlnSig0 * p.Psi + dPsi_dtau;

        double const dIQ_hi = compute_dIQ_post_ibp(p, tau, NL_, kappa_val);
        double const dIQ_lo = compute_dIQ_post_ibp(p, tau, NL_ / 2, kappa_val);
        double const value = sigma0 * (non_integral + dIQ_hi) * dtau_dT;

        double const abs_err =
            std::abs(sigma0) * std::abs(dIQ_hi - dIQ_lo) * dtau_dT;
        double const rel_err =
            abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
        return ComptonResult{value, abs_err, rel_err};

    } else {
        double const dIQ_hi = compute_dIQ_pre_ibp(p, tau, NL_, kappa_val);
        double const dIQ_lo = compute_dIQ_pre_ibp(p, tau, NL_ / 2, kappa_val);
        double const value = sigma0 * dIQ_hi * dtau_dT;

        double const abs_err =
            std::abs(sigma0) * std::abs(dIQ_hi - dIQ_lo) * dtau_dT;
        double const rel_err =
            abs_err / (std::abs(value) + constants::REL_ERROR_TINY_SCALE);
        return ComptonResult{value, abs_err, rel_err};
    }
}

} // namespace compton
