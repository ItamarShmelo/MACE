#include "compton_multigroup/compton_multigroup.hpp"
#include "planck_integral.hpp"
#include "units/units.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <sstream>

namespace compton {

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    int const quad_order_E,
    int const quad_order_Ep,
    int const quad_order_mu,
    double const planck_cap_x)
    : group_boundaries_(energy_group_boundaries)
    , cap_x_(planck_cap_x)
    , w0_(planck_cap_x * planck_cap_x * planck_cap_x / std::expm1(planck_cap_x))
    , rule_E_(compute_gauss_legendre(quad_order_E))
    , rule_Ep_(compute_gauss_legendre(quad_order_Ep))
    , rule_mu_(compute_gauss_legendre(quad_order_mu))
{
    if (energy_group_boundaries.size() < 2)
        throw std::invalid_argument("need at least 2 boundaries (1 group)");

    for (std::size_t i = 0; i < energy_group_boundaries.size(); ++i) {
        if (!(energy_group_boundaries[i] > 0.0) ||
            !std::isfinite(energy_group_boundaries[i]))
            throw std::invalid_argument("all boundaries must be finite and > 0");
    }

    for (std::size_t i = 0; i + 1 < energy_group_boundaries.size(); ++i) {
        if (energy_group_boundaries[i] >= energy_group_boundaries[i + 1])
            throw std::invalid_argument("boundaries must be strictly increasing");
    }

    if (quad_order_E < 1 || quad_order_Ep < 1 || quad_order_mu < 1)
        throw std::invalid_argument("quadrature orders must be >= 1");

    if (!(planck_cap_x > 0.0))
        throw std::invalid_argument("planck_cap_x must be > 0");

    int const G = static_cast<int>(energy_group_boundaries.size()) - 1;
    group_centers_.resize(G);
    for (int g = 0; g < G; ++g) {
        group_centers_[g] = std::sqrt(group_boundaries_[g] * group_boundaries_[g + 1]);
    }
}

double ComptonMultigroupKernel::planck_weight(double const E, double const T) const {
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_)
        return x * x * x / std::expm1(x);
    return w0_;
}

double ComptonMultigroupKernel::compute_denominator(int const g, double const T) const {
    double const kT = units::k_boltz * T;
    double const x_lo = group_boundaries_[g] / kT;
    double const x_hi = group_boundaries_[g + 1] / kT;

    static double constexpr pi4_over_15 =
        std::numbers::pi * std::numbers::pi * std::numbers::pi * std::numbers::pi / 15.0;

    if (x_hi <= cap_x_) {
        return kT * pi4_over_15 * planck_integral::planck_integral(x_lo, x_hi);
    }

    if (x_lo >= cap_x_) {
        return kT * w0_ * (x_hi - x_lo);
    }

    double const planck_part = pi4_over_15 * planck_integral::planck_integral(x_lo, cap_x_);
    double const const_part = w0_ * (x_hi - cap_x_);
    return kT * (planck_part + const_part);
}

template<typename KernelT>
std::vector<double> ComptonMultigroupKernel::compute_matrix_impl(
    KernelT const& kernel,
    SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
    int const num_angle_bins,
    double const T,
    double const Ne) const
{
    if (num_angle_bins < 1)
        throw std::invalid_argument("num_angle_bins must be >= 1");

    int const G = num_groups();
    std::vector<double> result(
        static_cast<std::size_t>(G) * G * num_angle_bins, 0.0);

    double const dmu = 2.0 / static_cast<double>(num_angle_bins);

    std::vector<double> denominators(G);
    for (int g = 0; g < G; ++g) {
        denominators[g] = compute_denominator(g, T);
    }

    for (int g = 0; g < G; ++g) {
        double const E_lo = group_boundaries_[g];
        double const E_hi = group_boundaries_[g + 1];
        double const inv_denom = 1.0 / denominators[g];

        for (int gp = 0; gp < G; ++gp) {
            double const Ep_lo = group_boundaries_[gp];
            double const Ep_hi = group_boundaries_[gp + 1];

            for (int a = 0; a < num_angle_bins; ++a) {
                double const mu_lo = -1.0 + a * dmu;
                double const mu_hi = -1.0 + (a + 1) * dmu;

                double const numerator = legendre_integrate(
                    [&](double const E) {
                        double const w = planck_weight(E, T);
                        double const inner = legendre_integrate(
                            [&](double const Ep) {
                                return legendre_integrate(
                                    [&](double const mu) {
                                        return (kernel.*eval)(E, Ep, mu, T, Ne).value;
                                    },
                                    rule_mu_, mu_lo, mu_hi);
                            },
                            rule_Ep_, Ep_lo, Ep_hi);
                        return w * inner;
                    },
                    rule_E_, E_lo, E_hi);

                std::size_t const idx =
                    static_cast<std::size_t>(g) * G * num_angle_bins +
                    static_cast<std::size_t>(gp) * num_angle_bins +
                    static_cast<std::size_t>(a);
                result[idx] = 2.0 * std::numbers::pi * numerator * inv_denom;
            }
        }
    }

    return result;
}

template<typename KernelT>
std::vector<double> ComptonMultigroupKernel::compute_sigma_matrix(
    KernelT const& kernel,
    int const num_angle_bins,
    double const T, double const Ne) const
{
    return compute_matrix_impl(kernel, &KernelT::sigma_E, num_angle_bins, T, Ne);
}

template<typename KernelT>
std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    KernelT const& kernel,
    int const num_angle_bins,
    double const T, double const Ne) const
{
    return compute_matrix_impl(kernel, &KernelT::dsigma_E_dT, num_angle_bins, T, Ne);
}

// ── Explicit instantiations ─────────────────────────────────────────────

template std::vector<double> ComptonMultigroupKernel::compute_sigma_matrix(
    ComptonKernelQuadrature const&, int, double, double) const;
template std::vector<double> ComptonMultigroupKernel::compute_sigma_matrix(
    ComptonKernelSeries const&, int, double, double) const;
template std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    ComptonKernelQuadrature const&, int, double, double) const;
template std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    ComptonKernelSeries const&, int, double, double) const;

} // namespace compton
