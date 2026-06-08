#include "compton_multigroup/compton_multigroup.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace compton {

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    double const tol,
    int const base_order)
    : group_boundaries_(energy_group_boundaries)
    , weight_func_(std::move(weight_function))
    , base_rule_(compute_gauss_legendre(base_order))
    , tol_(tol)
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

    if (base_order < 1)
        throw std::invalid_argument("base_order must be >= 1");
    if (!(tol > 0.0))
        throw std::invalid_argument("tol must be > 0");

    int const G = static_cast<int>(energy_group_boundaries.size()) - 1;
    group_centers_.resize(G);
    for (int g = 0; g < G; ++g) {
        group_centers_[g] = std::sqrt(group_boundaries_[g] * group_boundaries_[g + 1]);
    }
}

template<typename KernelT>
std::vector<double> ComptonMultigroupKernel::compute_matrix_impl(
    KernelT const& kernel,
    SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
    int const num_angle_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    if (num_angle_bins < 1)
        throw std::invalid_argument("num_angle_bins must be >= 1");

    int const G = num_groups();
    std::vector<double> result(
        static_cast<std::size_t>(G) * G * num_angle_bins, 0.0);

    double const dmu = 2.0 / static_cast<double>(num_angle_bins);

    std::vector<double> denominators(G);
    for (int g = 0; g < G; ++g) {
        denominators[g] = weight_func_->compute_denominator(
            group_boundaries_[g], group_boundaries_[g + 1], T);
    }

    double const tol_E  = tol_;
    double const tol_Ep = tol_ * 0.1;
    double const tol_mu = tol_ * 0.01;

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

                double const numerator = adaptive_legendre_integrate(
                    [&](double const E) {
                        double const w = weight_func_->weight(E, T);
                        double const inner = adaptive_legendre_integrate(
                            [&](double const Ep) {
                                return adaptive_legendre_integrate(
                                    [&](double const mu) {
                                        return multiplier(E, Ep, mu, T, Ne) *
                                               (kernel.*eval)(E, Ep, mu, T, Ne).value;
                                    },
                                    base_rule_, mu_lo, mu_hi, tol_mu);
                            },
                            base_rule_, Ep_lo, Ep_hi, tol_Ep);
                        return w * inner;
                    },
                    base_rule_, E_lo, E_hi, tol_E);

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
    double const T, double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_matrix_impl(kernel, &KernelT::sigma_E, num_angle_bins, T, Ne, multiplier);
}

template<typename KernelT>
std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    KernelT const& kernel,
    int const num_angle_bins,
    double const T, double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_matrix_impl(kernel, &KernelT::dsigma_E_dT, num_angle_bins, T, Ne, multiplier);
}

// ── Explicit instantiations ─────────────────────────────────────────────

template std::vector<double> ComptonMultigroupKernel::compute_sigma_matrix(
    ComptonKernelSolver const&, int, double, double, KernelMultiplier const&) const;

template std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    ComptonKernelSolver const&, int, double, double, KernelMultiplier const&) const;

} // namespace compton
