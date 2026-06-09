#include "compton_multigroup/compton_multigroup.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace compton {

// ── Constructor ──────────────────────────────────────────────────────────

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    double const tol,
    int const base_order,
    EpQuadratureConfig const& ep)
    : group_boundaries_(energy_group_boundaries)
    , weight_func_(std::move(weight_function))
    , base_rule_(compute_gauss_legendre(base_order))
    , tol_(tol)
    , peak_rule_(compute_gauss_legendre(ep.peak_base_order > 0 ? ep.peak_base_order : base_order))
    , peak_tol_factor_(ep.peak_tol_factor)
    , peak_max_depth_(ep.peak_max_depth)
    , tail_rule_(compute_gauss_legendre(ep.tail_base_order > 0 ? ep.tail_base_order : base_order))
    , tail_tol_factor_(ep.tail_tol_factor)
    , tail_max_depth_(ep.tail_max_depth)
    , far_rule_(compute_gauss_legendre(ep.far_base_order > 0 ? ep.far_base_order : std::max(base_order / 2, 4)))
    , far_tol_factor_(ep.far_tol_factor)
    , far_max_depth_(ep.far_max_depth)
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
    double const Ep_hi,
    double const tol,
    int const max_depth)
{
    return adaptive_rlog_legendre_integrate(f, rule, Ep_lo, Ep_hi, tol, max_depth);
}

template<typename F>
double integrate_Ep_right_tail(
    F&& f,
    GaussLegendreRule const& rule,
    double const Ep_lo,
    double const Ep_hi,
    double const tol,
    int const max_depth)
{
    return adaptive_log_legendre_integrate(f, rule, Ep_lo, Ep_hi, tol, max_depth);
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
    double const tail_tol,
    int const tail_depth,
    GaussLegendreRule const& far_rule,
    double const far_tol,
    int const far_depth)
{
    double const overlap_lo = std::clamp(peak_lo, Ep_lo, Ep_hi);
    double const overlap_hi = std::clamp(peak_hi, Ep_lo, Ep_hi);

    if (overlap_lo >= overlap_hi) {
        return adaptive_legendre_integrate(f, far_rule, Ep_lo, Ep_hi, far_tol, far_depth);
    }

    double result = 0.0;

    if (overlap_lo > Ep_lo)
        result += integrate_Ep_left_tail(f, tail_rule, Ep_lo, overlap_lo, tail_tol, tail_depth);

    result += integrate_Ep_peak(f, peak_rule, overlap_lo, overlap_hi, peak_tol, peak_depth);

    if (overlap_hi < Ep_hi)
        result += integrate_Ep_right_tail(f, tail_rule, overlap_hi, Ep_hi, tail_tol, tail_depth);

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
//   1. Innermost: μ integral via adaptive Gauss-Legendre over one angle bin.
//   2. Middle:    E' integral via peak-aware three-region quadrature.
//                 The cold-electron recoil band (thermally broadened by
//                 peak_limits) splits E' into a peak region where the kernel
//                 is strongest, exponentially suppressed tail regions on
//                 either side, and far regions beyond.  Each region uses its
//                 own GL rule, tolerance, and recursion depth.
//   3. Outermost: E integral over the incoming group [E_lo, E_hi].
//                 The mapping (linear / log / reflected-log) is chosen per
//                 angle bin based on the weight-function contrast: when
//                 w(E_lo)/w(E_hi) or E_hi/E_lo exceeds log_E_ratio_threshold_,
//                 a logarithmic change of variable clusters quadrature nodes
//                 near the heavy-weight boundary to resolve steep integrands.
//
// The final matrix element is:
//
//   result[g, gp, a] = 2π / D(g) · ∫ w(E) · [∫∫ f·Σ dμ dE'] dE
//
// Returns Σ_a |result[g, gp, a]| so the caller can apply the
// outward-from-peak cutoff.

template<typename KernelT>
double ComptonMultigroupKernel::compute_group_entry(
    KernelT const& kernel,
    SigmaResult (KernelT::*eval)(double, double, double, double, double) const,
    int const g,
    int const gp,
    int const num_angle_bins,
    double const dmu,
    double const T,
    double const Ne,
    double const tol_E,
    double const tol_mu,
    double const peak_tol,
    double const tail_tol,
    double const far_tol,
    double const inv_denom,
    KernelMultiplier const& multiplier,
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
            //   peak: [max(Ep_lo, band_lo), min(Ep_hi, band_hi)]
            //   tail: transition zones adjacent to the peak
            //   far:  remainder, where the kernel is negligible
            // Each region gets its own GL rule, tolerance, and max depth.
            double const inner = integrate_Ep_group(
                [&](double const Ep) {
                    // --- μ integral (innermost axis) ---
                    // Adaptive GL quadrature of f(E,E',μ)·Σ_E(E,E',μ,T,Ne)
                    // over the angle bin [mu_lo, mu_hi].
                    return adaptive_legendre_integrate(
                        [&](double const mu) {
                            return multiplier(E, Ep, mu, T, Ne) *
                                   (kernel.*eval)(E, Ep, mu, T, Ne).value;
                        },
                        base_rule_, mu_lo, mu_hi, tol_mu, max_depth_mu_);
                },
                Ep_lo, Ep_hi, band_lo, band_hi,
                peak_rule_, peak_tol, peak_max_depth_,
                tail_rule_, tail_tol, tail_max_depth_,
                far_rule_,  far_tol,  far_max_depth_);

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
        if (E_hi / E_lo > log_E_ratio_threshold_) {
            double const w_lo = weight_func_->weight(E_lo, T);
            double const w_hi = weight_func_->weight(E_hi, T);
            if (w_lo >= w_hi) {
                numerator = adaptive_log_legendre_integrate(
                    E_integrand, base_rule_, E_lo, E_hi,
                    tol_E, max_depth_E_);
            } else {
                numerator = adaptive_rlog_legendre_integrate(
                    E_integrand, base_rule_, E_lo, E_hi,
                    tol_E, max_depth_E_);
            }
        } else {
            numerator = adaptive_legendre_integrate(
                E_integrand, base_rule_, E_lo, E_hi,
                tol_E, max_depth_E_);
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
// The tolerance hierarchy is:
//   tol_E  = tol_           (outermost, loosest)
//   tol_E' = tol_ * 0.1 * {peak,tail,far}_tol_factor_
//   tol_mu = tol_ * 0.01   (innermost, tightest)

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

    // --- Tolerance hierarchy ---
    // Inner axes use progressively tighter tolerances so that quadrature
    // errors do not accumulate across the three nested integrals.
    double const tol_E  = tol_;               // outermost (E)
    double const tol_Ep_base = tol_ * 0.1;    // middle    (E')
    double const tol_mu = tol_ * 0.01;        // innermost (μ)

    // E' tolerances are further split by region (peak / tail / far).
    double const peak_tol = tol_Ep_base * peak_tol_factor_;
    double const tail_tol = tol_Ep_base * tail_tol_factor_;
    double const far_tol  = tol_Ep_base * far_tol_factor_;

    // --- Main loop over incoming groups g ---
    for (int g = 0; g < G; ++g) {
        double const inv_denom = 1.0 / denominators[g];

        // Thin forwarding lambda: captures the per-g state and delegates
        // the full 3D integration for a single (g, gp) to compute_group_entry.
        auto do_group = [&](int const gp) {
            return compute_group_entry(
                kernel, eval, g, gp, num_angle_bins, dmu,
                T, Ne, tol_E, tol_mu, peak_tol, tail_tol, far_tol,
                inv_denom, multiplier, result);
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
