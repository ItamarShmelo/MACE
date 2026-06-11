#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_common/compton_common.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace compton {

namespace constants {
static constexpr double LOG_E_RATIO_THRESHOLD = 10.0;
} // namespace constants

// ── MGIntegrationConfig ─────────────────────────────────────────────────

MGIntegrationConfig::MGIntegrationConfig(
    int const base_order,
    double const integration_tolerance,
    double const cutoff_ratio,
    int const peak_max_depth,
    int const cold_temperature_order,
    std::optional<int> const tail_order,
    std::optional<int> const far_order)
    : base_order(base_order)
    , cold_temperature_order(cold_temperature_order)
    , peak_max_depth(peak_max_depth)
    , tail_order(tail_order)
    , far_order(far_order)
    , integration_tolerance(integration_tolerance)
    , cutoff_ratio(cutoff_ratio)
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
    , integration_tolerance_(config.integration_tolerance)
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
}

// ── E' sub-interval integration helpers ─────────────────────────────────

namespace {

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
        return legendre_integrate(f, far_rule, Ep_lo, Ep_hi);
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
    std::vector<double>& result) const
{
    int const G = num_groups();

    // Incoming group [E_lo, E_hi] and target group [Ep_lo, Ep_hi].
    double const E_lo = group_boundaries_[g];
    double const E_hi = group_boundaries_[g + 1];
    double const Ep_lo = group_boundaries_[gp];
    double const Ep_hi = group_boundaries_[gp + 1];

    // Accumulates Σ_a |S(g,gp,a)| across angle bins for cutoff decisions.
    double group_sum = 0.0;

    // --- Loop over angle bins ---
    // Each bin [mu_lo, mu_hi] is an equal-width slice of [-1, 1].
    for (int a = 0; a < num_angle_bins; ++a) {
        double const mu_lo = -1.0 + a * dmu;
        double const mu_hi = -1.0 + (a + 1) * dmu;

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
            double const inner = integrate_Ep_group(
                [&](double const Ep) {
                    // --- μ integral (innermost axis) ---
                    // Single-panel GL of f(E,E',μ)·Σ_E(E,E',μ,T,Ne)
                    // over the angle bin [mu_lo, mu_hi].
                    return legendre_integrate(
                        [&](double const mu) {
                            return multiplier(E, Ep, mu, T, Ne) *
                                   (kernel.*eval)(E, Ep, mu, T, Ne).value;
                        },
                        active_rule, mu_lo, mu_hi);
                },
                Ep_lo, Ep_hi, band_lo, band_hi,
                active_rule, peak_tol, peak_max_depth_,
                tail_rule_,
                far_rule_);

            return w * inner;
        };

        // --- E-axis mapping selection ---
        // Wide groups (E_hi/E_lo > threshold) use a logarithmic change of
        // variable to cluster quadrature nodes where the integrand is
        // largest.  The weight function at the group edges determines which
        // end to cluster toward:
        //   log   (nodes near E_lo) when w(E_lo) >= w(E_hi)
        //   rlog  (nodes near E_hi) when w(E_hi) >  w(E_lo)
        double numerator = 0.0;
        if (E_hi / E_lo > constants::LOG_E_RATIO_THRESHOLD) {
            double const w_lo = weight_func_->weight(E_lo, T);
            double const w_hi = weight_func_->weight(E_hi, T);
            if (w_lo >= w_hi) {
                numerator = log_legendre_integrate(
                    E_integrand, active_rule, E_lo, E_hi);
            } else {
                numerator = rlog_legendre_integrate(
                    E_integrand, active_rule, E_lo, E_hi);
            }
        } else {
            numerator = legendre_integrate(
                E_integrand, active_rule, E_lo, E_hi);
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
    GaussLegendreRule const& active_rule =
        (T < constants::COLD_TEMPERATURE_THRESHOLD) ? cold_rule_ : base_rule_;

    // --- Main loop over incoming groups g ---
    for (int g = 0; g < G; ++g) {
        double const inv_denom = 1.0 / denominators[g];

        auto do_group = [&](int const gp) {
            return compute_group_entry(
                kernel, eval, g, gp, num_angle_bins, dmu,
                T, Ne, peak_tol,
                inv_denom, multiplier, active_rule, result);
        };

        // --- Outward-from-peak target-group traversal ---
        // Find gp_peak: the target group containing the geometric-mean
        // energy of group g.  For elastic / near-elastic scattering this
        // is always the dominant target.
        double const E_center = group_centers_[g];
        auto it = std::upper_bound(
            group_boundaries_.begin(), group_boundaries_.end(), E_center);
        int gp_peak = static_cast<int>(
            std::distance(group_boundaries_.begin(), it)) - 1;
        gp_peak = std::clamp(gp_peak, 0, G - 1);

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
