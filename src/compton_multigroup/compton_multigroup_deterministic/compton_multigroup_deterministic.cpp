#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_common/compton_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace compton {

namespace constants {
static constexpr double LOG_E_RATIO_THRESHOLD = 10.0;
static constexpr double LOG_MU_EP_RATIO_THRESHOLD = 1.5;
static constexpr double E_BOUNDARY_LAYER_MULTIPLIER = 10.0;
static constexpr double MU_PEAK_RATIO_THRESHOLD = 0.05;
} // namespace constants

// ── MGIntegrationConfig ─────────────────────────────────────────────────

MGIntegrationConfig::MGIntegrationConfig(
    int const base_order,
    double const integration_tolerance,
    double const cutoff_ratio,
    int const peak_max_depth,
    int const cold_temperature_order,
    std::optional<int> const tail_order,
    std::optional<int> const far_order,
    std::optional<int> const mu_order,
    double const mu_peak_k,
    std::optional<FlatEpConfig> const flat_ep)
    : base_order(base_order)
    , cold_temperature_order(cold_temperature_order)
    , peak_max_depth(peak_max_depth)
    , tail_order(tail_order)
    , far_order(far_order)
    , mu_order(mu_order)
    , integration_tolerance(integration_tolerance)
    , cutoff_ratio(cutoff_ratio)
    , mu_peak_k(mu_peak_k)
    , flat_ep(flat_ep)
{
    if (base_order < 1)
        throw std::invalid_argument("base_order must be >= 1");
    if (cold_temperature_order < base_order)
        throw std::invalid_argument("cold_temperature_order must be >= base_order");
    if (!(integration_tolerance > 0.0))
        throw std::invalid_argument("integration_tolerance must be > 0");
    if (!(cutoff_ratio > 0.0))
        throw std::invalid_argument("cutoff_ratio must be > 0");
    if (peak_max_depth < 0)
        throw std::invalid_argument("peak_max_depth must be >= 0");
    if (tail_order.has_value() && tail_order.value() < 1)
        throw std::invalid_argument("tail_order must be >= 1");
    if (far_order.has_value() && far_order.value() < 1)
        throw std::invalid_argument("far_order must be >= 1");
    if (mu_order.has_value() && mu_order.value() < 1)
        throw std::invalid_argument("mu_order must be >= 1");
    if (!(mu_peak_k > 0.0))
        throw std::invalid_argument("mu_peak_k must be > 0");
    if (flat_ep.has_value()) {
        if (!(flat_ep->density > 0.0))
            throw std::invalid_argument("flat_ep.density must be > 0");
        if (flat_ep->min_points < 2)
            throw std::invalid_argument("flat_ep.min_points must be >= 2");
        if (flat_ep->max_points < flat_ep->min_points)
            throw std::invalid_argument("flat_ep.max_points must be >= min_points");
    }
}

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    MGIntegrationConfig const& config)
    : group_boundaries_(energy_group_boundaries)
    , weight_func_(std::move(weight_function))
    , base_rule_(compute_gauss_legendre(config.base_order))
    , cold_rule_(compute_gauss_legendre(config.cold_temperature_order))
    , tail_rule_(compute_gauss_legendre(config.effective_tail_order()))
    , far_rule_(compute_gauss_legendre(config.effective_far_order()))
    , mu_rule_(compute_gauss_legendre(config.effective_mu_order()))
    , mu_cold_rule_(compute_gauss_legendre(
          std::max(config.cold_temperature_order, config.effective_mu_order())))
    , mu_tail_rule_(compute_gauss_legendre(8))
    , integration_tolerance_(config.integration_tolerance)
    , mu_peak_k_(config.mu_peak_k)
    , peak_max_depth_(config.peak_max_depth)
    , group_cutoff_ratio_(config.cutoff_ratio)
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

    int const G = static_cast<int>(energy_group_boundaries.size()) - 1;
    group_centers_.resize(G);
    for (int g = 0; g < G; ++g) {
        group_centers_[g] = std::sqrt(group_boundaries_[g] * group_boundaries_[g + 1]);
    }

    // Build per-group GL rules for flat E' mode.
    if (config.flat_ep.has_value()) {
        auto const& cfg = *config.flat_ep;
        flat_E_ = cfg.flat_E;
        flat_mu_ = cfg.flat_mu;
        double const E_ref = std::sqrt(
            group_boundaries_.front() * group_boundaries_.back());

        std::unordered_map<int, GaussLegendreRule> rule_cache;
        flat_ep_rules_.resize(G);

        for (int gp = 0; gp < G; ++gp) {
            double const Ep_lo = group_boundaries_[gp];
            double const Ep_hi = group_boundaries_[gp + 1];

            double raw = 0.0;
            switch (cfg.mode) {
            case FlatEpDensityMode::log_proportional:
                raw = cfg.density * std::log(Ep_hi / Ep_lo);
                break;
            case FlatEpDensityMode::linear_proportional:
                raw = cfg.density * (Ep_hi - Ep_lo) / E_ref;
                break;
            case FlatEpDensityMode::points_per_decade:
                raw = cfg.density * std::log10(Ep_hi / Ep_lo);
                break;
            }
            int const N = std::clamp(static_cast<int>(std::round(raw)),
                                     cfg.min_points, cfg.max_points);

            auto it = rule_cache.find(N);
            if (it == rule_cache.end()) {
                it = rule_cache.emplace(N, compute_gauss_legendre(N)).first;
            }
            flat_ep_rules_[gp] = it->second;
        }
    }
}

// ── E' sub-interval integration helpers ─────────────────────────────────

namespace {

void log_ts(std::FILE* f) {
    auto const now = std::chrono::system_clock::now();
    auto const t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::fprintf(f, "%02d:%02d:%02d",
        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
}

template<typename F>
double integrate_Ep_peak(
    F&& f,
    GaussLegendreRule const& rule,
    double const Ep_lo,
    double const Ep_hi,
    double const tol,
    int const max_depth)
{
    return adaptive_legendre_integrate(f, rule, Ep_lo, Ep_hi, tol, max_depth);
}

template<typename F>
double integrate_Ep_left_tail(
    F&& f,
    GaussLegendreRule const& rule,
    double const Ep_lo,
    double const Ep_hi)
{
    return rlog_legendre_integrate(f, rule, Ep_lo, Ep_hi);
}

template<typename F>
double integrate_Ep_right_tail(
    F&& f,
    GaussLegendreRule const& rule,
    double const Ep_lo,
    double const Ep_hi)
{
    return log_legendre_integrate(f, rule, Ep_lo, Ep_hi);
}

template<typename F>
double integrate_Ep_group(
    F&& f,
    double const Ep_lo,
    double const Ep_hi,
    double const peak_lo,
    double const peak_hi,
    GaussLegendreRule const& peak_rule,
    double const peak_tol,
    int const peak_depth,
    GaussLegendreRule const& tail_rule,
    GaussLegendreRule const& far_rule)
{
    double const overlap_lo = std::clamp(peak_lo, Ep_lo, Ep_hi);
    double const overlap_hi = std::clamp(peak_hi, Ep_lo, Ep_hi);

    if (overlap_lo >= overlap_hi) {
        if (peak_hi <= Ep_lo) {
            return log_legendre_integrate(f, far_rule, Ep_lo, Ep_hi);
        }
        return rlog_legendre_integrate(f, far_rule, Ep_lo, Ep_hi);
    }

    double result = 0.0;

    if (overlap_lo > Ep_lo)
        result += integrate_Ep_left_tail(f, tail_rule, Ep_lo, overlap_lo);

    result += integrate_Ep_peak(f, peak_rule, overlap_lo, overlap_hi, peak_tol, peak_depth);

    if (overlap_hi < Ep_hi)
        result += integrate_Ep_right_tail(f, tail_rule, overlap_hi, Ep_hi);

    return result;
}

} // anonymous namespace

// ── Single (g, gp) entry ────────────────────────────────────────────────
//
// Evaluates the weighted multigroup scattering matrix element for one
// incoming group g scattering into target group gp, across all angle bins.
//
// The integral is three-dimensional (E × E' × μ) and evaluated inside-out:
//
//   1. Innermost: μ integral via single-panel Gauss-Legendre over one angle bin.
//   2. Middle:    E' integral via peak-aware three-region quadrature.
//                 The cold-electron recoil band (thermally broadened by
//                 peak_limits) splits E' into a peak region where the kernel
//                 is strongest, exponentially suppressed tail regions on
//                 either side, and far regions beyond.  Only the peak uses
//                 adaptive refinement; tails use single-panel log/rlog GL
//                 and far uses single-panel linear GL.
//   3. Outermost: E integral over the incoming group [E_lo, E_hi].
//                 Single-panel GL with mapping (linear / log / reflected-log)
//                 chosen per angle bin based on the weight-function contrast:
//                 when E_hi/E_lo exceeds LOG_E_RATIO_THRESHOLD,
//                 a logarithmic change of variable clusters quadrature nodes near
//                 the heavy-weight boundary.
//
// The final matrix element is:
//
//   result[g, gp, a] = 2π / D(g) · ∫ w(E) · [∫∫ f·Σ dμ dE'] dE
//
// Returns Σ_a |result[g, gp, a]| so the caller can apply the
// outward-from-peak cutoff.

double ComptonMultigroupKernel::compute_group_entry(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    int const g,
    int const gp,
    int const num_angle_bins,
    double const dmu,
    double const T,
    double const Ne,
    double const peak_tol,
    double const inv_denom,
    KernelMultiplier const& multiplier,
    GaussLegendreRule const& active_rule,
    GaussLegendreRule const& active_mu_rule,
    std::vector<double>& result) const
{
    int const G = num_groups();
    double const tau = T * units::k_boltz / units::me_c2;
    double const sqrt_tau = std::sqrt(tau);

    // Incoming group [E_lo, E_hi] and target group [Ep_lo, Ep_hi].
    double const E_lo = group_boundaries_[g];
    double const E_hi = group_boundaries_[g + 1];
    double const Ep_lo = group_boundaries_[gp];
    double const Ep_hi = group_boundaries_[gp + 1];

    // Accumulates Σ_a |S(g,gp,a)| across angle bins for cutoff decisions.
    double group_sum = 0.0;

    // --- Loop over angle bins ---
    // Each bin [mu_lo, mu_hi] is an equal-width slice of [-1, 1].
    // mu = 1 (xi = 1) is an integrable singularity where a = 1 - xi = 0,
    // causing division by zero in kinematic parameters.  Clamping the last
    // bin edge to 1 - MU_UPPER_EPS avoids this; the excluded sliver
    // [1-eps, 1] contributes O(eps / bin_width) ≈ 4e-10 relative error.
    constexpr double MU_UPPER_EPS = 1e-10;
    for (int a = 0; a < num_angle_bins; ++a) {
        double const mu_lo = -1.0 + a * dmu;
        double const mu_hi = std::min(-1.0 + (a + 1) * dmu,
                                      1.0 - MU_UPPER_EPS);

        // --- E integrand (outermost axis) ---
        // For a given incoming energy E, this lambda computes:
        //   w(E, T) · ∫_{Ep_lo}^{Ep_hi} ∫_{mu_lo}^{mu_hi} f·Σ dμ dE'
        auto E_integrand = [&](double const E) {
            double const w = weight_func_->weight(E, T);

            // Thermally broadened cold-recoil band: the E' interval where
            // the kernel peaks for a cold electron.  Outside this band
            // the kernel is exponentially suppressed ∝ exp(-(λ_min-1)/τ).
            auto const [band_lo, band_hi] = peak_limits(E, mu_lo, mu_hi, T);

            // --- E' integral (middle axis) ---
            // integrate_Ep_group splits [Ep_lo, Ep_hi] into three regions
            // relative to [band_lo, band_hi]:
            //   peak: adaptive GL
            //   tail: single-panel log/rlog GL
            //   far:  single-panel linear GL
            auto mu_integrand = [&](double const E_in, double const Ep) {
                auto f = [&](double const mu) {
                    return multiplier(E_in, Ep, mu, T, Ne) *
                           (kernel.*eval)(E_in, Ep, mu, T, Ne).value;
                };
                if (flat_mu_) {
                    return legendre_integrate(f, active_mu_rule, mu_lo, mu_hi);
                }

                double const r = Ep / E_in;

                // Peak-focused splitting: for non-elastic scatter, compute
                // the Compton peak location and FWHM. If the peak is narrow
                // relative to the mu interval, concentrate quadrature there.
                if (std::abs(r - 1.0) > constants::MU_PEAK_RATIO_THRESHOLD) {
                    double const gamma = E_in / units::me_c2;
                    double const mu_c_raw = 1.0 - (1.0 / gamma) * (1.0 / r - 1.0);
                    double const mu_c = std::clamp(mu_c_raw, mu_lo, mu_hi);
                    double const fwhm = 4.0 * std::sqrt(std::abs(1.0 - r) / r)
                                      * sqrt_tau / std::pow(gamma, 1.5);
                    double const half_w = mu_peak_k_ * fwhm;

                    double const peak_lo = std::max(mu_lo, mu_c - half_w);
                    double const peak_hi = std::min(mu_hi, mu_c + half_w);

                    if (peak_hi > peak_lo &&
                        (peak_hi - peak_lo) < 0.8 * (mu_hi - mu_lo)) {
                        double result = legendre_integrate(
                            f, active_mu_rule, peak_lo, peak_hi);
                        if (peak_lo > mu_lo)
                            result += legendre_integrate(
                                f, mu_tail_rule_, mu_lo, peak_lo);
                        if (peak_hi < mu_hi)
                            result += legendre_integrate(
                                f, mu_tail_rule_, peak_hi, mu_hi);
                        return result;
                    }
                }

                // Fallback: log/rlog mapping for boundary-peaked cases
                if (r > constants::LOG_MU_EP_RATIO_THRESHOLD) {
                    double const dmu_span = mu_hi - mu_lo;
                    double const eps = dmu_span * 1e-14;
                    return log_legendre_integrate(
                        [&](double const s) { return f(mu_lo + s); },
                        active_mu_rule, eps, dmu_span);
                }
                if (r < 1.0 / constants::LOG_MU_EP_RATIO_THRESHOLD) {
                    double const dmu_span = mu_hi - mu_lo;
                    double const eps = dmu_span * 1e-14;
                    return log_legendre_integrate(
                        [&](double const s) { return f(mu_hi - s); },
                        active_mu_rule, eps, dmu_span);
                }

                return legendre_integrate(f, active_mu_rule, mu_lo, mu_hi);
            };

            double inner;
            if (!flat_ep_rules_.empty()) {
                auto ep_integrand = [&](double const Ep) {
                    return mu_integrand(E, Ep);
                };
                GaussLegendreRule const& ep_rule = flat_ep_rules_[gp];
                if (band_hi <= Ep_lo) {
                    inner = log_legendre_integrate(ep_integrand, ep_rule, Ep_lo, Ep_hi);
                } else if (band_lo >= Ep_hi) {
                    inner = rlog_legendre_integrate(ep_integrand, ep_rule, Ep_lo, Ep_hi);
                } else {
                    inner = legendre_integrate(ep_integrand, ep_rule, Ep_lo, Ep_hi);
                }
            } else {
                inner = integrate_Ep_group(
                    [&](double const Ep) {
                        return mu_integrand(E, Ep);
                    },
                    Ep_lo, Ep_hi, band_lo, band_hi,
                    active_rule, peak_tol, peak_max_depth_,
                    tail_rule_,
                    far_rule_);
            }

            return w * inner;
        };

        double numerator = 0.0;

        if (flat_E_) {
            // Flat E mode: single GL pass over [E_lo, E_hi], no boundary layers.
            numerator = legendre_integrate(E_integrand, active_rule, E_lo, E_hi);
        } else {
            // --- E-axis boundary sub-panels ---
            //
            // The Compton kernel near a group boundary has a thermal peak of
            // width ~thermal_half_width(E, T).  At cold T this width can be
            // much smaller than the GL node spacing, causing the quadrature
            // to miss the boundary-straddling contribution entirely.
            //
            // Split the E integral into up to three sub-panels:
            //   [E_lo, E_lo + delta_lo]             -- left boundary  (linear GL)
            //   [E_lo + delta_lo, E_hi - delta_hi]  -- interior       (existing mapping)
            //   [E_hi - delta_hi, E_hi]             -- right boundary (linear GL)

            double const span = E_hi - E_lo;
            double const delta_lo = std::min(
                constants::E_BOUNDARY_LAYER_MULTIPLIER * thermal_half_width(E_lo, T),
                0.4 * span);
            double const delta_hi = std::min(
                constants::E_BOUNDARY_LAYER_MULTIPLIER * thermal_half_width(E_hi, T),
                0.4 * span);

            // Left boundary layer.
            if (delta_lo > 1e-14 * span) {
                numerator += legendre_integrate(
                    E_integrand, active_rule, E_lo, E_lo + delta_lo);
            }

            // Interior: existing mapping logic on the reduced interval.
            double const E_int_lo = E_lo + delta_lo;
            double const E_int_hi = E_hi - delta_hi;
            if (E_int_hi > E_int_lo) {
                if (E_int_hi / E_int_lo > constants::LOG_E_RATIO_THRESHOLD) {
                    double const w_lo = weight_func_->weight(E_int_lo, T);
                    double const w_hi = weight_func_->weight(E_int_hi, T);
                    if (w_lo >= w_hi) {
                        numerator += log_legendre_integrate(
                            E_integrand, active_rule, E_int_lo, E_int_hi);
                    } else {
                        numerator += rlog_legendre_integrate(
                            E_integrand, active_rule, E_int_lo, E_int_hi);
                    }
                } else {
                    numerator += legendre_integrate(
                        E_integrand, active_rule, E_int_lo, E_int_hi);
                }
            }

            // Right boundary layer.
            if (delta_hi > 1e-14 * span) {
                numerator += legendre_integrate(
                    E_integrand, active_rule, E_hi - delta_hi, E_hi);
            }
        }

        // Store the final matrix element:
        //   S(g, gp, a) = 2π · numerator / D(g)
        // The 2π factor comes from the azimuthal symmetry of Compton
        // scattering (∫_0^{2π} dφ = 2π).
        std::size_t const idx =
            static_cast<std::size_t>(g) * G * num_angle_bins +
            static_cast<std::size_t>(gp) * num_angle_bins +
            static_cast<std::size_t>(a);
        result[idx] = 2.0 * std::numbers::pi * numerator * inv_denom;
        group_sum += std::abs(result[idx]);
    }
    return group_sum;
}

// ── Core 3D integration ─────────────────────────────────────────────────
//
// Assembles the full G × G × num_angle_bins scattering matrix.
//
// For each incoming group g the method:
//
//   1. Precomputes the weight-function denominator D(g) = ∫ w(E,T) dE
//      over the group, which normalises the numerator integrals.
//
//   2. Iterates over target groups gp using outward-from-peak traversal:
//          - Identify gp_peak: the target group whose energy range
//            contains the geometric-mean energy of group g.
//          - Evaluate gp_peak first; its angle-summed magnitude is the
//            reference "peak_sum".
//          - Expand rightward (gp_peak+1, gp_peak+2, …) until the
//            angle-summed magnitude drops below cutoff_ratio × peak_sum.
//          - Expand leftward similarly.
//          - Unevaluated groups are left at zero.
//
//      This exploits the physical locality of Compton scattering:
//      most of the scattered energy stays near the incoming energy,
//      so distant target groups contribute negligibly.
//
//   3. Each (g, gp) pair is delegated to compute_group_entry().
//
// Adaptive refinement is used only for the E' peak region:
//   peak_tol = integration_tolerance_ * 0.1
// All other axes (E, mu, tail, far) use single-panel GL quadrature
// whose accuracy is controlled by increasing base_order.

std::vector<double> ComptonMultigroupKernel::compute_matrix_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    int const num_angle_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    if (num_angle_bins < 1)
        throw std::invalid_argument("num_angle_bins must be >= 1");

    int const G = num_groups();

    // Output buffer: flat row-major [g][gp][angle_bin], initialised to zero.
    // Unevaluated entries (skipped by the cutoff) remain at zero.
    std::vector<double> result(
        static_cast<std::size_t>(G) * G * num_angle_bins, 0.0);

    // Uniform angle-bin width: μ ∈ [-1, 1] split into num_angle_bins slices.
    double const dmu = 2.0 / static_cast<double>(num_angle_bins);

    // --- Weight-function denominators ---
    // D(g) = ∫_{E_lo}^{E_hi} w(E, T) dE normalises each row of the matrix
    // so that row sums represent physical scattering probabilities.
    std::vector<double> denominators(G);
    for (int g = 0; g < G; ++g) {
        denominators[g] = weight_func_->compute_denominator(
            group_boundaries_[g], group_boundaries_[g + 1], T);
    }

    // --- Peak tolerance ---
    // Only the E' peak region uses adaptive refinement; all other axes
    // (E, mu, tail, far) use single-panel GL quadrature.
    double const peak_tol = integration_tolerance_ * 0.1;

    // --- Cold-temperature rule selection ---
    // Below the threshold the kernel is extremely sharp (near-Thomson);
    // a higher-order rule is needed for the E and mu axes.
    bool const is_cold = T < constants::COLD_TEMPERATURE_THRESHOLD;
    GaussLegendreRule const& active_rule = is_cold ? cold_rule_ : base_rule_;
    GaussLegendreRule const& active_mu_rule = is_cold ? mu_cold_rule_ : mu_rule_;

    auto const wall_t0 = std::chrono::steady_clock::now();
    std::FILE* log_file = std::fopen("compton_multigroup.log", "a");
    if (log_file) {
        log_ts(log_file);
        std::fprintf(log_file,
            " [compton] deterministic: G=%d, angle_bins=%d, T=%.4g keV, mode=%s\n",
            G, num_angle_bins, T / units::kev_kelvin, is_cold ? "cold" : "hot");
        std::fflush(log_file);
    }

    // --- Main loop over incoming groups g ---
    #pragma omp parallel for schedule(dynamic)
    for (int g = 0; g < G; ++g) {
        double const inv_denom = 1.0 / denominators[g];

        auto do_group = [&](int const gp) {
            return compute_group_entry(
                kernel, eval, g, gp, num_angle_bins, dmu,
                T, Ne, peak_tol,
                inv_denom, multiplier, active_rule, active_mu_rule, result);
        };

        // --- Outward-from-peak target-group traversal ---
        // the target group which is the dominant elastic-scattering target.
        int const gp_peak = g;

        // Evaluate the peak group first to establish the reference magnitude.
        double const peak_sum = do_group(gp_peak);

        double const cutoff = group_cutoff_ratio_ * peak_sum;

        // Expand rightward (higher E' groups) until below cutoff.
        for (int gp = gp_peak + 1; gp < G; ++gp) {
            if (do_group(gp) < cutoff) break;
        }
        // Expand leftward (lower E' groups) until below cutoff.
        for (int gp = gp_peak - 1; gp >= 0; --gp) {
            if (do_group(gp) < cutoff) break;
        }
    }

    if (log_file) {
        auto const wall_t1 = std::chrono::steady_clock::now();
        double const elapsed =
            std::chrono::duration<double>(wall_t1 - wall_t0).count();
        log_ts(log_file);
        std::fprintf(log_file,
            " [compton] deterministic: done in %.1f s\n", elapsed);
        std::fclose(log_file);
    }

    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_sigma_matrix(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T, double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_matrix_impl(kernel, &ComptonKernelSolver::sigma_E, num_angle_bins, T, Ne, multiplier);
}

std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T, double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_matrix_impl(kernel, &ComptonKernelSolver::dsigma_E_dT, num_angle_bins, T, Ne, multiplier);
}

} // namespace compton
