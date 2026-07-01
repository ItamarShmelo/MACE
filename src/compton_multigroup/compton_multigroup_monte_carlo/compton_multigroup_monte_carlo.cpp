#include "compton_multigroup/compton_multigroup_monte_carlo/compton_multigroup_monte_carlo.hpp"
#include "compton_common/compton_common.hpp"
#include "utilities/compute_logger.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>

namespace compton {

// ── MCIntegrationConfig ─────────────────────────────────────────────────

MCIntegrationConfig::MCIntegrationConfig(
    std::size_t const num_samples,
    int const seed,
    bool const discard_out_of_grid)
    : num_samples(num_samples),
      seed(seed),
      discard_out_of_grid(discard_out_of_grid)
{
    if (num_samples < 1) {
        throw std::invalid_argument("num_samples must be >= 1");
    }
}

// ── ComptonMonteCarloKernel ─────────────────────────────────────────────

ComptonMonteCarloKernel::ComptonMonteCarloKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    MCIntegrationConfig const& config)
    : group_boundaries_(energy_group_boundaries),
      weight_func_(std::move(weight_function)),
      num_samples_(config.num_samples),
      discard_out_of_grid_(config.discard_out_of_grid),
      rng_(
          config.seed >= 0 ? static_cast<std::uint64_t>(config.seed)
                           : static_cast<std::uint64_t>(std::time(nullptr))),
      uniform_dist_()
{
    int const G = static_cast<int>(group_boundaries_.size()) - 1;
    if (G < 1) {
        throw std::invalid_argument(
            "need at least 2 boundaries (1 energy group)");
    }

    for (int g = 0; g < G; ++g) {
        if (group_boundaries_[g] <= 0.0) {
            throw std::invalid_argument("energy group boundaries must be > 0");
        }
        if (group_boundaries_[g] >= group_boundaries_[g + 1]) {
            throw std::invalid_argument(
                "energy group boundaries must be strictly increasing");
        }
    }

    group_centers_.resize(G);
    group_widths_.resize(G);
    for (int g = 0; g < G; ++g) {
        group_centers_[g] =
            std::sqrt(group_boundaries_[g] * group_boundaries_[g + 1]);
        group_widths_[g] = group_boundaries_[g + 1] - group_boundaries_[g];
    }
}

// ── Maxwell-Jüttner sampling ────────────────────────────────────────────

namespace {

/** @brief Thread-safe Maxwell-Jüttner sampling accepting external RNG state. */
double sample_gamma(
    double const theta,
    boost::random::mt19937_64& rng,
    boost::random::uniform_01<>& dist)
{
    double const sum_1_bt = 1.0 + 1.0 / theta;
    double const Sb = sum_1_bt + 0.5 / (theta * theta);

    double const r0Sb = dist(rng) * Sb;
    double const r1 = dist(rng);

    if (r0Sb <= 1.0) {
        double const r2 = dist(rng);
        double const r3 = dist(rng);
        return 1.0 - theta * std::log(r1 * r2 * r3);
    }

    if (r0Sb <= sum_1_bt) {
        double const r2 = dist(rng);
        return 1.0 - theta * std::log(r1 * r2);
    }

    return 1.0 - theta * std::log(r1);
}

} // anonymous namespace

// ── Core MC integration ──────────────────────────────────────────────────

template <typename MultiplierFn>
std::vector<double> ComptonMonteCarloKernel::mc_integrate(
    int const num_angle_bins,
    double const T,
    double const Ne,
    MultiplierFn const& multiplier_fn) const
{
    if (num_angle_bins < 1) {
        throw std::invalid_argument("num_angle_bins must be >= 1");
    }
    if (T <= 0.0) {
        throw std::invalid_argument("temperature T must be > 0");
    }

    int const G = num_groups();
    std::size_t const total = static_cast<std::size_t>(G) * G * num_angle_bins;
    std::vector<double> result(total, 0.0);

    double const theta = units::k_boltz * T / units::me_c2;

    double sum_beta = 0.0;
    std::vector<double> weight_sum(G, 0.0);

    ComputeLogger logger(
        "monte_carlo",
        std::format(
            "N={}, G={}, angle_bins={}, T={:.4g} keV, theta={:.2e}",
            num_samples_,
            G,
            num_angle_bins,
            T / units::kev_kelvin,
            theta));

    std::uint64_t const base_seed = rng_();

    double* result_ptr = result.data();
    double* ws_ptr = weight_sum.data();

#ifdef _OPENMP
#pragma omp parallel reduction(+ : sum_beta)                                   \
    reduction(+ : result_ptr[ : total]) reduction(+ : ws_ptr[ : G])
    {
        int const tid = omp_get_thread_num();
        boost::random::mt19937_64 local_rng(
            base_seed +
            static_cast<std::uint64_t>(tid) * 6364136223846793005ULL);
        boost::random::uniform_01<> local_dist;
        auto& rng = local_rng;
        auto& dist = local_dist;
#else
    auto& rng = rng_;
    auto& dist = uniform_dist_;
    {
#endif

#pragma omp for schedule(static)
        for (std::size_t sample_i = 0; sample_i < num_samples_; ++sample_i) {

            // Step 1: sample electron Lorentz factor from Maxwell-Jüttner
            double const lam = sample_gamma(theta, rng, dist);
            double const beta = std::sqrt(1.0 - 1.0 / (lam * lam));
            sum_beta += beta;

            // Step 2: sample isotropic electron direction
            double const mu_e = 1.0 - 2.0 * dist(rng);
            double const sin_e = std::sqrt(1.0 - mu_e * mu_e);

            // Step 3: incoming-photon Doppler factor
            double const D0 = lam * (1.0 - beta * mu_e);

            // Step 4: incoming photon direction in electron rest frame
            double const mu_0_tag =
                (1.0 / D0) *
                (1.0 - lam / (1.0 + lam) * (D0 + 1.0) * beta * mu_e);
            double const sin_0_tag = std::sqrt(1.0 - mu_0_tag * mu_0_tag);

            // Step 5: sample isotropic scattering in electron rest frame
            double const mu_p_tag = 1.0 - 2.0 * dist(rng);
            double const sin_p_tag = std::sqrt(1.0 - mu_p_tag * mu_p_tag);
            double const psi_p_tag = dist(rng) * 2.0 * std::numbers::pi;
            double const cos_psi = std::cos(psi_p_tag);

            // Step 6: rotate scattered direction by -theta_0 into electron
            // frame
            double const Omega_tag_x =
                mu_0_tag * sin_p_tag * cos_psi - sin_0_tag * mu_p_tag;
            double const Omega_tag_z =
                sin_0_tag * sin_p_tag * cos_psi + mu_0_tag * mu_p_tag;

            // Step 7: outgoing Doppler factor and lab-frame scattering cosine
            double const dot_Omega_tag_e =
                Omega_tag_x * sin_e + Omega_tag_z * mu_e;
            double const D_tag = lam * (1.0 + beta * dot_Omega_tag_e);

            double xi_scat_lab =
                (Omega_tag_z +
                 ((lam - 1.0) * dot_Omega_tag_e + lam * beta) * mu_e) /
                D_tag;
            xi_scat_lab = std::clamp(xi_scat_lab, -1.0, 1.0);

            int const angle_bin = std::min(
                static_cast<int>((xi_scat_lab + 1.0) * 0.5 * num_angle_bins),
                num_angle_bins - 1);

            // Step 8: loop over incoming energy groups
            double const interp = dist(rng);

            for (int g0 = 0; g0 < G; ++g0) {
                double const E0 =
                    group_boundaries_[g0] + interp * group_widths_[g0];

                double const w_E0 = weight_func_->weight(E0, T);
                ws_ptr[g0] += w_E0;

                // Lorentz transform: lab → electron rest → scatter → lab
                double const E0_tag = D0 * E0;
                double const A =
                    1.0 / (1.0 + (1.0 - mu_p_tag) * E0_tag / units::me_c2);
                double const E = D_tag * A * E0_tag;

                // Find outgoing group
                auto const it = std::lower_bound(
                    group_boundaries_.begin(),
                    group_boundaries_.end(),
                    E);
                long g = std::distance(group_boundaries_.begin(), it) - 1;

                if (discard_out_of_grid_) {
                    if (g < 0 || g >= G) {
                        continue;
                    }
                } else {
                    g = std::clamp(g, 0L, static_cast<long>(G) - 1);
                }

                // Klein-Nishina contribution
                double const sigma = 0.75 * D0 / lam * A * A *
                                     (A + 1.0 / A - sin_p_tag * sin_p_tag) *
                                     w_E0 * beta;

                double const mult =
                    multiplier_fn(E0, E, xi_scat_lab, T, Ne, lam);

                std::size_t const idx =
                    static_cast<std::size_t>(g0) * G * num_angle_bins +
                    static_cast<std::size_t>(g) * num_angle_bins + angle_bin;
                result_ptr[idx] += sigma * mult;
            }
        }
    }

    // Normalization
    double const beta_avg = sum_beta / static_cast<double>(num_samples_);

    for (int g0 = 0; g0 < G; ++g0) {
        double const weight_avg =
            weight_sum[g0] / static_cast<double>(num_samples_);
        double const norm =
            units::sigma_thomson /
            (static_cast<double>(num_samples_) * beta_avg * weight_avg);

        for (int gp = 0; gp < G; ++gp) {
            for (int a = 0; a < num_angle_bins; ++a) {
                std::size_t const idx =
                    static_cast<std::size_t>(g0) * G * num_angle_bins +
                    static_cast<std::size_t>(gp) * num_angle_bins + a;
                result[idx] *= norm;
            }
        }
    }

    logger.done();

    return result;
}

// ── Public API ──────────────────────────────────────────────────────────

std::vector<double> ComptonMonteCarloKernel::compute_sigma_matrix(
    int const num_angle_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return mc_integrate(
        num_angle_bins,
        T,
        Ne,
        [&](double E0, double E, double xi, double Tv, double Nev, double) {
            return multiplier(E0, E, xi, Tv, Nev);
        });
}

std::vector<double> ComptonMonteCarloKernel::compute_sigma_matrix(
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_sigma_matrix(1, T, Ne, multiplier);
}

std::vector<double> ComptonMonteCarloKernel::compute_dsigma_dT_matrix(
    int const num_angle_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    double const tau = units::k_boltz * T / units::me_c2;
    double const tau2 = tau * tau;
    double const kappa_val = kappa_ratio(tau);
    double const dtau_dT = units::k_boltz / units::me_c2;

    return mc_integrate(
        num_angle_bins,
        T,
        Ne,
        [&, kappa_val, tau2, dtau_dT](
            double E0,
            double E,
            double xi,
            double Tv,
            double Nev,
            double lam) {
            return multiplier(E0, E, xi, Tv, Nev) *
                   ((lam - kappa_val) / tau2 - 3.0 / tau) * dtau_dT;
        });
}

std::vector<double> ComptonMonteCarloKernel::compute_dsigma_dT_matrix(
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_dsigma_dT_matrix(1, T, Ne, multiplier);
}

} // namespace compton
