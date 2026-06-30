#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_common/compton_common.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <numbers>
#include <stdexcept>

namespace compton {

namespace constants {
static constexpr double XI_UPPER_EPS = 1e-10;
static constexpr double DOUBLE_PEAK_RATIO_THRESHOLD = 10.0;

// Temperature thresholds (in tau = kT/me_c2) for the elastic-like rlog path.
// When |E'-E|/E < threshold, the xi integrand peaks steeply near xi=1
// and rlog quadrature (which clusters nodes at the upper endpoint) is
// far more effective than peak-splitting GL.
//
// The threshold must be tight enough that xi_pk is genuinely near xi_hi
// when rlog activates.  If too loose, E' values whose xi peak is mid-bin
// get routed to rlog, which wastes nodes at xi=1 instead of the peak.
// Conversely, if too tight at warm/hot T, the peak-splitting path has to
// resolve a steep forward spike that rlog handles effortlessly.
//
// Optimal thresholds (validated by sweeping at xi_order=48, measuring
// bin 3 error vs xi_order=512 reference):
//
//   Very cold (tau < 2e-8,  T <   ~10 eV): thr = 1e-5
//   Cold      (tau < 2e-6,  T <    ~1 keV): thr = 1e-4
//   Cool      (tau < 2e-4,  T <  ~100 eV ): thr = 1e-3
//   Warm      (tau < 0.02,  T <   ~10 keV): thr = 0.01
//   Moderate  (tau < 0.2,   T <  ~100 keV): thr = 0.05
//   Hot       (tau >= 0.2,  T >= ~100 keV): thr = 0.3
static constexpr double XI_ELASTIC_TAU_VCOLD = 2e-8;
static constexpr double XI_ELASTIC_TAU_COLD  = 2e-6;
static constexpr double XI_ELASTIC_TAU_COOL  = 2e-4;
static constexpr double XI_ELASTIC_TAU_WARM  = 0.02;
static constexpr double XI_ELASTIC_TAU_HOT   = 0.2;
static constexpr double XI_ELASTIC_VCOLD_THR = 1e-5;
static constexpr double XI_ELASTIC_COLD_THR  = 1e-4;
static constexpr double XI_ELASTIC_COOL_THR  = 1e-3;
static constexpr double XI_ELASTIC_WARM_THR  = 0.01;
static constexpr double XI_ELASTIC_MOD_THR   = 0.05;
static constexpr double XI_ELASTIC_HOT_THR   = 0.3;
} // namespace constants

// ── MGIntegrationConfig ─────────────────────────────────────────────────

MGIntegrationConfig::MGIntegrationConfig(
    double const cutoff_ratio,
    std::optional<int> const xi_order,
    double const xi_peak_k,
    std::optional<int> const xi_tail_order,
    double const ep_k_cut,
    double const ep_k_in,
    std::optional<int> const ep_edge_order,
    std::optional<int> const ep_interior_order,
    std::optional<int> const e_panel_order,
    double const log_e_panel_ratio,
    double const e_boundary_k)
    : xi_order(xi_order)
    , xi_tail_order(xi_tail_order)
    , cutoff_ratio(cutoff_ratio)
    , xi_peak_k(xi_peak_k)
    , ep_k_cut(ep_k_cut)
    , ep_k_in(ep_k_in)
    , ep_edge_order(ep_edge_order)
    , ep_interior_order(ep_interior_order)
    , e_panel_order(e_panel_order)
    , log_e_panel_ratio(log_e_panel_ratio)
    , e_boundary_k(e_boundary_k)
{
    if (cutoff_ratio < 0.0)
        throw std::invalid_argument("cutoff_ratio must be >= 0");
    if (xi_order.has_value() && xi_order.value() < 1)
        throw std::invalid_argument("xi_order must be >= 1");
    if (!(xi_peak_k > 0.0))
        throw std::invalid_argument("xi_peak_k must be > 0");
    if (!(ep_k_cut > 0.0))
        throw std::invalid_argument("ep_k_cut must be > 0");
    if (!(ep_k_in >= 0.0))
        throw std::invalid_argument("ep_k_in must be >= 0");
    if (!(ep_k_cut > ep_k_in))
        throw std::invalid_argument(
            "ep_k_cut must be > ep_k_in "
            "(semantic constraint: the edge region must be narrower than the retained interval)");
    if (ep_edge_order.has_value() && ep_edge_order.value() < 1)
        throw std::invalid_argument("ep_edge_order must be >= 1");
    if (ep_interior_order.has_value() && ep_interior_order.value() < 1)
        throw std::invalid_argument("ep_interior_order must be >= 1");
    if (e_panel_order.has_value() && e_panel_order.value() < 1)
        throw std::invalid_argument("e_panel_order must be >= 1");
    if (!(log_e_panel_ratio > 1.0))
        throw std::invalid_argument("log_e_panel_ratio must be > 1.0");
    if (!(e_boundary_k > 0.0))
        throw std::invalid_argument("e_boundary_k must be > 0");
}

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    MGIntegrationConfig const& config)
    : group_boundaries_(energy_group_boundaries)
    , weight_func_(std::move(weight_function))
    , xi_rule_(compute_gauss_legendre(config.effective_xi_order()))
    , xi_tail_rule_(compute_gauss_legendre(config.effective_xi_tail_order()))
    , ep_edge_rule_(compute_gauss_legendre(config.effective_ep_edge_order()))
    , ep_interior_rule_(compute_gauss_legendre(config.effective_ep_interior_order()))
    , ep_elastic_core_rule_(compute_gauss_legendre(8))
    , e_panel_rule_(compute_gauss_legendre(config.effective_e_panel_order()))
    , xi_peak_k_(config.xi_peak_k)
    , ep_k_cut_(config.ep_k_cut)
    , ep_k_in_(config.ep_k_in)
    , group_cutoff_ratio_(config.cutoff_ratio)
    , log_e_panel_ratio_(config.log_e_panel_ratio)
    , e_boundary_k_(config.e_boundary_k)
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
}


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
double integrate_Ep_ridge(
    F&& f,
    double Ep_lo, double Ep_hi,
    RidgeBounds const& rb,
    double k_cut, double k_in,
    GaussLegendreRule const& edge_rule,
    GaussLegendreRule const& interior_rule,
    GaussLegendreRule const& elastic_core_rule)
{
    double const keep_lo = std::max(Ep_lo, rb.cold_lo - k_cut * rb.sigma_lo);
    double const keep_hi = std::min(Ep_hi,
        std::max(rb.cold_lo + k_cut * rb.sigma_lo,
                 rb.cold_hi + k_cut * rb.sigma_hi));

    // ── Double-peak path: isolate the narrow elastic endpoint feature ────
    //
    // When sigma_lo >> sigma_hi the xi-integrated E' integrand has two
    // features at very different scales: a broad thermally-broadened
    // Compton ridge (width ~ sigma_lo) and a narrow near-forward elastic
    // endpoint feature (width ~ sigma_hi) centred at cold_hi ~ E.
    // The standard 3-region scheme collapses to a single GL panel that
    // cannot resolve the meV-scale elastic feature.  Split into 4 regions:
    //   left-tail | broad-left | elastic-core | broad-right | right-tail
    if (rb.sigma_lo / rb.sigma_hi > constants::DOUBLE_PEAK_RATIO_THRESHOLD) {
        double const ec_lo = std::max(keep_lo, rb.cold_hi - k_cut * rb.sigma_hi);
        double const ec_hi = std::min(keep_hi, rb.cold_hi + k_cut * rb.sigma_hi);

        if (ec_hi > ec_lo && ec_hi > Ep_lo && ec_lo < Ep_hi) {
            double dp_result = 0.0;

            if (keep_lo > Ep_lo) {
                assert(Ep_lo > 0.0);
                dp_result += rlog_legendre_integrate(f, edge_rule, Ep_lo, keep_lo);
            }

            if (ec_lo > keep_lo)
                dp_result += legendre_integrate(f, edge_rule, keep_lo, ec_lo);

            dp_result += legendre_integrate(f, elastic_core_rule, ec_lo, ec_hi);

            if (keep_hi > ec_hi)
                dp_result += legendre_integrate(f, interior_rule, ec_hi, keep_hi);

            if (Ep_hi > keep_hi) {
                assert(keep_hi > 0.0);
                if (Ep_hi / keep_hi > 2.0)
                    dp_result += log_legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
                else
                    dp_result += legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
            }

            return dp_result;
        }
    }

    // Ridge entirely outside [Ep_lo, Ep_hi]: integrate the full group range.
    if (keep_lo >= keep_hi) {
        assert(Ep_lo > 0.0);
        if (Ep_hi <= rb.cold_lo)
            return rlog_legendre_integrate(f, edge_rule, Ep_lo, Ep_hi);
        if (Ep_hi / Ep_lo > 2.0)
            return log_legendre_integrate(f, edge_rule, Ep_lo, Ep_hi);
        return legendre_integrate(f, edge_rule, Ep_lo, Ep_hi);
    }

    double result = 0.0;

    if (keep_lo > Ep_lo) {
        assert(Ep_lo > 0.0);
        result += rlog_legendre_integrate(f, edge_rule, Ep_lo, keep_lo);
    }

    double const edge_lo = rb.cold_lo + k_in * rb.sigma_lo;
    double const edge_hi = rb.cold_hi - k_in * rb.sigma_hi;

    if (edge_lo >= edge_hi) {
        result += legendre_integrate(f, interior_rule, keep_lo, keep_hi);
    } else {
        double const left_hi  = std::min(keep_hi, edge_lo);
        double const mid_lo   = std::max(keep_lo, edge_lo);
        double const mid_hi   = std::min(keep_hi, edge_hi);
        double const right_lo = std::max(keep_lo, edge_hi);

        if (left_hi > keep_lo)
            result += legendre_integrate(f, edge_rule, keep_lo, left_hi);
        if (mid_hi > mid_lo)
            result += legendre_integrate(f, interior_rule, mid_lo, mid_hi);
        if (keep_hi > right_lo)
            result += legendre_integrate(f, edge_rule, right_lo, keep_hi);
    }

    if (Ep_hi > keep_hi) {
        assert(keep_hi > 0.0);
        if (Ep_hi / keep_hi > 2.0)
            result += log_legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
        else
            result += legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
    }

    return result;
}

template<typename F>
double integrate_E_panels(
    F&& integrand,
    std::vector<EPanel> const& panels,
    GaussLegendreRule const& rule)
{
    double sum = 0.0;
    for (auto const& panel : panels) {
        switch (panel.map) {
        case EPanelMap::Linear:
            sum += legendre_integrate(integrand, rule, panel.lo, panel.hi);
            break;
        case EPanelMap::LogLower:
            sum += log_legendre_integrate(integrand, rule, panel.lo, panel.hi);
            break;
        case EPanelMap::LogUpper:
            sum += rlog_legendre_integrate(integrand, rule, panel.lo, panel.hi);
            break;
        }
    }
    return sum;
}

} // anonymous namespace

double ComptonMultigroupKernel::integrate_xi_bin(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    double const E,
    double const Ep,
    double const xi_lo,
    double const xi_hi,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    double const tau = T * units::k_boltz / units::me_c2;

    auto f = [&](double const xi) {
        return multiplier(E, Ep, xi, T, Ne) *
               (kernel.*eval)(E, Ep, xi, T, Ne).value;
    };

    double const gamma = E / units::me_c2;
    double const gamma_p = Ep / units::me_c2;
    double const abs_dg = std::abs(gamma - gamma_p);

    double const elastic_thr =
        tau < constants::XI_ELASTIC_TAU_VCOLD ? constants::XI_ELASTIC_VCOLD_THR :
        tau < constants::XI_ELASTIC_TAU_COLD  ? constants::XI_ELASTIC_COLD_THR :
        tau < constants::XI_ELASTIC_TAU_COOL  ? constants::XI_ELASTIC_COOL_THR :
        tau < constants::XI_ELASTIC_TAU_WARM  ? constants::XI_ELASTIC_WARM_THR :
        tau < constants::XI_ELASTIC_TAU_HOT   ? constants::XI_ELASTIC_MOD_THR :
                                                constants::XI_ELASTIC_HOT_THR;
    bool const elastic_like = abs_dg / gamma < elastic_thr;

    if (elastic_like) {
        double const span = xi_hi - xi_lo;
        double const eps = span * 1e-14;
        return rlog_legendre_integrate(
            [&](double const s) { return f(xi_lo + s); },
            xi_rule_, eps, span);
    }

    double const xi_pk = 1.0 - abs_dg / (gamma * gamma_p);
    double const sigma_xi =
        std::sqrt(tau * abs_dg * (2.0 + abs_dg))
        / (gamma * gamma_p);
    double const half_w = xi_peak_k_ * sigma_xi;
    double const bin_span = xi_hi - xi_lo;

    double const peak_lo = xi_pk - half_w;
    double const peak_hi = xi_pk + half_w;

    if (peak_hi <= xi_lo) {
        return legendre_integrate(f, xi_rule_, xi_lo, xi_hi);
    }

    if (peak_lo >= xi_hi) {
        return legendre_integrate(f, xi_rule_, xi_lo, xi_hi);
    }

    double const core_lo = std::max(xi_lo, peak_lo);
    double const core_hi = std::min(xi_hi, peak_hi);

    double result = 0.0;

    if (core_lo > xi_lo) {
        double const span = core_lo - xi_lo;
        if (span > 1e-14 * bin_span) {
            result += legendre_integrate(f, xi_tail_rule_, xi_lo, core_lo);
        }
    }

    if (core_hi > core_lo) {
        result += legendre_integrate(
            f, xi_rule_, core_lo, core_hi);
    }

    if (core_hi < xi_hi) {
        double const span = xi_hi - core_hi;
        if (span > 1e-14 * bin_span) {
            result += legendre_integrate(f, xi_tail_rule_, core_hi, xi_hi);
        }
    }

    return result;
}

double ComptonMultigroupKernel::integrate_Ep_xi_bin(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    double const xi_lo,
    double const xi_hi,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    auto const rb = compute_ridge_bounds(E, xi_lo, xi_hi, T);

    auto ep_integrand = [&](double const Ep) {
        return integrate_xi_bin(
            kernel, eval, E, Ep, xi_lo, xi_hi, T, Ne,
            multiplier);
    };

    return integrate_Ep_ridge(
        ep_integrand,
        Ep_lo, Ep_hi, rb,
        ep_k_cut_, ep_k_in_,
        ep_edge_rule_, ep_interior_rule_, ep_elastic_core_rule_);
}

// ── E-axis panel construction ───────────────────────────────────────────

std::vector<EPanel> ComptonMultigroupKernel::compute_E_panels(
    int const g,
    double const T) const
{
    double const E_lo = group_boundaries_[g];
    double const E_hi = group_boundaries_[g + 1];

    auto make_panel = [&](double const a, double const b) -> EPanel {
        EPanelMap map;
        if (b / a <= log_e_panel_ratio_) {
            map = EPanelMap::Linear;
        } else if (weight_func_->weight(a, T) >= weight_func_->weight(b, T)) {
            map = EPanelMap::LogLower;
        } else {
            map = EPanelMap::LogUpper;
        }
        return {a, b, map};
    };

    std::vector<EPanel> panels;

    auto add_panel = [&](double const a, double const b) {
        if (b > a)
            panels.push_back(make_panel(a, b));
    };

    auto add_middle_panels = [&](double const a, double const b) {
        auto const Epk = weight_func_->peak_energy(T);
        if (Epk && *Epk > a && *Epk < b) {
            add_panel(a, *Epk);
            add_panel(*Epk, b);
            return;
        }
        add_panel(a, b);
    };

    double const sigma_lo = ridge_thermal_width(E_lo, -1.0, T);
    double const sigma_hi = ridge_thermal_width(E_hi, -1.0, T);
    double const bl_lo = E_lo + e_boundary_k_ * sigma_lo;
    double const bl_hi = E_hi - e_boundary_k_ * sigma_hi;

    if (bl_lo < bl_hi) {
        add_panel(E_lo, bl_lo);
        add_middle_panels(bl_lo, bl_hi);
        add_panel(bl_hi, E_hi);
    } else {
        add_middle_panels(E_lo, E_hi);
    }

    return panels;
}

double ComptonMultigroupKernel::compute_group_entry(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    int const g,
    int const gp,
    int const num_angle_bins,
    double const dxi,
    double const T,
    double const Ne,
    double const inv_denom,
    KernelMultiplier const& multiplier,
    std::vector<EPanel> const& panels,
    std::vector<double>& result) const
{
    int const G = num_groups();

    double const Ep_lo = group_boundaries_[gp];
    double const Ep_hi = group_boundaries_[gp + 1];

    double group_sum = 0.0;

    for (int a = 0; a < num_angle_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi = std::min(-1.0 + (a + 1) * dxi,
                                      1.0 - constants::XI_UPPER_EPS);

        auto E_integrand = [&](double const E) {
            double const w = weight_func_->weight(E, T);
            double const inner = integrate_Ep_xi_bin(
                kernel, eval, E, Ep_lo, Ep_hi, xi_lo, xi_hi,
                T, Ne, multiplier);
            return w * inner;
        };

        double const numerator = integrate_E_panels(
            E_integrand, panels, e_panel_rule_);

        std::size_t const idx =
            static_cast<std::size_t>(g) * G * num_angle_bins +
            static_cast<std::size_t>(gp) * num_angle_bins +
            static_cast<std::size_t>(a);
        result[idx] = 2.0 * std::numbers::pi * numerator * inv_denom;
        group_sum += std::abs(result[idx]);
    }
    return group_sum;
}

std::vector<double> ComptonMultigroupKernel::compute_xi_integral_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    double const E,
    double const Ep,
    int const num_xi_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    if (num_xi_bins < 1)
        throw std::invalid_argument("num_xi_bins must be >= 1");
    if (!(E > 0.0) || !std::isfinite(E))
        throw std::invalid_argument("E must be finite and > 0");
    if (!(Ep > 0.0) || !std::isfinite(Ep))
        throw std::invalid_argument("Ep must be finite and > 0");

    double const dxi = 2.0 / static_cast<double>(num_xi_bins);

    std::vector<double> result(num_xi_bins);
    for (int a = 0; a < num_xi_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi = std::min(-1.0 + (a + 1) * dxi,
                                      1.0 - constants::XI_UPPER_EPS);
        result[a] = integrate_xi_bin(
            kernel, eval, E, Ep, xi_lo, xi_hi, T, Ne,
            multiplier);
    }
    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_Ep_xi_integral_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    int const num_xi_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    if (num_xi_bins < 1)
        throw std::invalid_argument("num_xi_bins must be >= 1");
    if (!(E > 0.0) || !std::isfinite(E))
        throw std::invalid_argument("E must be finite and > 0");
    if (!(Ep_lo > 0.0) || !std::isfinite(Ep_lo))
        throw std::invalid_argument("Ep_lo must be finite and > 0");
    if (!(Ep_hi > 0.0) || !std::isfinite(Ep_hi))
        throw std::invalid_argument("Ep_hi must be finite and > 0");
    if (Ep_lo >= Ep_hi)
        throw std::invalid_argument("Ep_lo must be < Ep_hi");

    double const dxi = 2.0 / static_cast<double>(num_xi_bins);

    std::vector<double> result(num_xi_bins);
    for (int a = 0; a < num_xi_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi = std::min(-1.0 + (a + 1) * dxi,
                                      1.0 - constants::XI_UPPER_EPS);
        result[a] = integrate_Ep_xi_bin(
            kernel, eval, E, Ep_lo, Ep_hi, xi_lo, xi_hi,
            T, Ne, multiplier);
    }
    return result;
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
// E' uses fixed-order GL via integrate_Ep_ridge (no adaptive refinement).
// Convergence is controlled by ep_edge_order and ep_interior_order.

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

    // Uniform angle-bin width: ξ ∈ [-1, 1] split into num_angle_bins slices.
    double const dxi = 2.0 / static_cast<double>(num_angle_bins);

    auto const wall_t0 = std::chrono::steady_clock::now();
    std::FILE* log_file = std::fopen("compton_multigroup.log", "a");
    if (log_file) {
        log_ts(log_file);
        std::fprintf(log_file,
            " [compton] deterministic: G=%d, angle_bins=%d, T=%.4g keV\n",
            G, num_angle_bins, T / units::kev_kelvin);
        std::fflush(log_file);
    }

    // --- Main loop over incoming groups g ---
    #pragma omp parallel for schedule(dynamic)
    for (int g = 0; g < G; ++g) {
        auto const panels = compute_E_panels(g, T);

        double const denom = weight_func_->compute_denominator(
            group_boundaries_[g], group_boundaries_[g + 1], T);
        double const inv_denom = 1.0 / denom;

        auto do_group = [&](int const gp) {
            return compute_group_entry(
                kernel, eval, g, gp, num_angle_bins, dxi,
                T, Ne,
                inv_denom, multiplier, panels, result);
        };

        // --- Outward-from-peak target-group traversal ---
        int const gp_peak = g;

        double const peak_sum = do_group(gp_peak);

        if (group_cutoff_ratio_ > 0.0) {
            double const cutoff = group_cutoff_ratio_ * peak_sum;

            // Expand rightward (higher E' groups) until below cutoff.
            for (int gp = gp_peak + 1; gp < G; ++gp) {
                if (do_group(gp) < cutoff) break;
            }
            // Expand leftward (lower E' groups) until below cutoff.
            for (int gp = gp_peak - 1; gp >= 0; --gp) {
                if (do_group(gp) < cutoff) break;
            }
        } else {
            for (int gp = gp_peak + 1; gp < G; ++gp) do_group(gp);
            for (int gp = gp_peak - 1; gp >= 0; --gp) do_group(gp);
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
