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
static constexpr double LOG_E_RATIO_THRESHOLD = 10.0;
static constexpr double E_BOUNDARY_LAYER_MULTIPLIER = 10.0;
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
    int const base_order,
    double const integration_tolerance,
    double const cutoff_ratio,
    int const cold_temperature_order,
    std::optional<int> const xi_order,
    double const xi_peak_k,
    std::optional<int> const xi_tail_order,
    double const ep_k_cut,
    double const ep_k_in,
    double const ep_k_tail,
    std::optional<int> const ep_edge_order,
    std::optional<int> const ep_interior_order,
    bool const ep_diagnostic_tails,
    std::optional<int> const ep_diagnostic_tail_order)
    : base_order(base_order)
    , cold_temperature_order(cold_temperature_order)
    , xi_order(xi_order)
    , xi_tail_order(xi_tail_order)
    , integration_tolerance(integration_tolerance)
    , cutoff_ratio(cutoff_ratio)
    , xi_peak_k(xi_peak_k)
    , ep_k_cut(ep_k_cut)
    , ep_k_in(ep_k_in)
    , ep_k_tail(ep_k_tail)
    , ep_edge_order(ep_edge_order)
    , ep_interior_order(ep_interior_order)
    , ep_diagnostic_tails(ep_diagnostic_tails)
    , ep_diagnostic_tail_order(ep_diagnostic_tail_order)
{
    if (base_order < 1)
        throw std::invalid_argument("base_order must be >= 1");
    if (cold_temperature_order < base_order)
        throw std::invalid_argument("cold_temperature_order must be >= base_order");
    if (!(integration_tolerance > 0.0))
        throw std::invalid_argument("integration_tolerance must be > 0");
    if (!(cutoff_ratio > 0.0))
        throw std::invalid_argument("cutoff_ratio must be > 0");
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
    if (!(ep_k_tail >= ep_k_cut))
        throw std::invalid_argument("ep_k_tail must be >= ep_k_cut");
    if (ep_edge_order.has_value() && ep_edge_order.value() < 1)
        throw std::invalid_argument("ep_edge_order must be >= 1");
    if (ep_interior_order.has_value() && ep_interior_order.value() < 1)
        throw std::invalid_argument("ep_interior_order must be >= 1");
    if (ep_diagnostic_tail_order.has_value() && ep_diagnostic_tail_order.value() < 1)
        throw std::invalid_argument("ep_diagnostic_tail_order must be >= 1");
}

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    MGIntegrationConfig const& config)
    : group_boundaries_(energy_group_boundaries)
    , weight_func_(std::move(weight_function))
    , base_rule_(compute_gauss_legendre(config.base_order))
    , cold_rule_(compute_gauss_legendre(config.cold_temperature_order))
    , xi_rule_(compute_gauss_legendre(config.effective_xi_order()))
    , xi_cold_rule_(compute_gauss_legendre(
          std::max(config.cold_temperature_order, config.effective_xi_order())))
    , xi_tail_rule_(compute_gauss_legendre(config.effective_xi_tail_order()))
    , ep_edge_rule_(compute_gauss_legendre(config.effective_ep_edge_order()))
    , ep_interior_rule_(compute_gauss_legendre(config.effective_ep_interior_order()))
    , ep_elastic_core_rule_(compute_gauss_legendre(8))
    , ep_diagnostic_tail_rule_(compute_gauss_legendre(config.effective_ep_diagnostic_tail_order()))
    , integration_tolerance_(config.integration_tolerance)
    , xi_peak_k_(config.xi_peak_k)
    , ep_k_cut_(config.ep_k_cut)
    , ep_k_in_(config.ep_k_in)
    , ep_k_tail_(config.ep_k_tail)
    , ep_diagnostic_tails_(config.ep_diagnostic_tails)
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

// ── E' ridge-based integration ──────────────────────────────────────────

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
    double k_cut, double k_in, double k_tail,
    GaussLegendreRule const& edge_rule,
    GaussLegendreRule const& interior_rule,
    GaussLegendreRule const& elastic_core_rule,
    bool diagnostic_tails,
    GaussLegendreRule const& diag_tail_rule)
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
    //   broad-left | elastic-core | broad-right | far-tail
    if (rb.sigma_lo / rb.sigma_hi > constants::DOUBLE_PEAK_RATIO_THRESHOLD) {
        double const ec_lo = std::max(keep_lo, rb.cold_hi - k_cut * rb.sigma_hi);
        double const ec_hi = std::min(keep_hi, rb.cold_hi + k_cut * rb.sigma_hi);

        if (ec_hi > ec_lo && ec_hi > Ep_lo && ec_lo < Ep_hi) {
            double dp_result = 0.0;

            if (diagnostic_tails && keep_lo > Ep_lo) {
                assert(Ep_lo > 0.0);
                dp_result += rlog_legendre_integrate(f, diag_tail_rule, Ep_lo, keep_lo);
            }

            if (ec_lo > keep_lo)
                dp_result += legendre_integrate(f, edge_rule, keep_lo, ec_lo);

            dp_result += legendre_integrate(f, elastic_core_rule, ec_lo, ec_hi);

            if (keep_hi > ec_hi)
                dp_result += legendre_integrate(f, interior_rule, ec_hi, keep_hi);

            double const tc = std::min(Ep_hi,
                std::max(rb.cold_lo + k_tail * rb.sigma_lo,
                         rb.cold_hi + k_tail * rb.sigma_hi));
            if (tc > keep_hi) {
                assert(keep_hi > 0.0);
                auto const& tail_rule = diagnostic_tails ? diag_tail_rule : edge_rule;
                if (tc / keep_hi > 2.0)
                    dp_result += log_legendre_integrate(f, tail_rule, keep_hi, tc);
                else
                    dp_result += legendre_integrate(f, tail_rule, keep_hi, tc);
            }

            return dp_result;
        }
    }

    if (keep_lo >= keep_hi) {
        if (!diagnostic_tails) {
            double const degen_cap = std::min(Ep_hi,
                std::max(rb.cold_lo + k_tail * rb.sigma_lo,
                         rb.cold_hi + k_tail * rb.sigma_hi));
            double const tail_lo = std::max(Ep_lo, keep_hi);
            if (degen_cap > tail_lo && tail_lo > 0.0)
                return log_legendre_integrate(f, edge_rule, tail_lo, degen_cap);
            return 0.0;
        }
        assert(Ep_lo > 0.0);
        if (Ep_hi <= rb.cold_lo)
            return rlog_legendre_integrate(f, diag_tail_rule, Ep_lo, Ep_hi);
        return log_legendre_integrate(f, diag_tail_rule, Ep_lo, Ep_hi);
    }

    double result = 0.0;

    if (diagnostic_tails && keep_lo > Ep_lo) {
        assert(Ep_lo > 0.0);
        result += rlog_legendre_integrate(f, diag_tail_rule, Ep_lo, keep_lo);
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

    double const tail_cap = std::min(Ep_hi,
        std::max(rb.cold_lo + k_tail * rb.sigma_lo,
                 rb.cold_hi + k_tail * rb.sigma_hi));
    if (tail_cap > keep_hi) {
        assert(keep_hi > 0.0);
        auto const& tail_rule = diagnostic_tails ? diag_tail_rule : edge_rule;
        result += log_legendre_integrate(f, tail_rule, keep_hi, tail_cap);
    }

    return result;
}

} // anonymous namespace

// ── Layer 1: single ξ-bin integration for fixed (E, E') ─────────────────

double ComptonMultigroupKernel::integrate_xi_bin(
    ComptonKernelSolver const& kernel,
    ComptonResult (ComptonKernelSolver::*eval)(double, double, double, double, double) const,
    double const E,
    double const Ep,
    double const xi_lo,
    double const xi_hi,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier,
    GaussLegendreRule const& active_xi_rule) const
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
            active_xi_rule, eps, span);
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
        return legendre_integrate(f, active_xi_rule, xi_lo, xi_hi);
    }

    if (peak_lo >= xi_hi) {
        return legendre_integrate(f, active_xi_rule, xi_lo, xi_hi);
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
            f, active_xi_rule, core_lo, core_hi);
    }

    if (core_hi < xi_hi) {
        double const span = xi_hi - core_hi;
        if (span > 1e-14 * bin_span) {
            result += legendre_integrate(f, xi_tail_rule_, core_hi, xi_hi);
        }
    }

    return result;
}

// ── Layer 2: E' + single ξ-bin integration for fixed E ──────────────────

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
    KernelMultiplier const& multiplier,
    GaussLegendreRule const& active_xi_rule) const
{
    auto const rb = compute_ridge_bounds(E, xi_lo, xi_hi, T);

    auto ep_integrand = [&](double const Ep) {
        return integrate_xi_bin(
            kernel, eval, E, Ep, xi_lo, xi_hi, T, Ne,
            multiplier, active_xi_rule);
    };

    return integrate_Ep_ridge(
        ep_integrand,
        Ep_lo, Ep_hi, rb,
        ep_k_cut_, ep_k_in_, ep_k_tail_,
        ep_edge_rule_, ep_interior_rule_, ep_elastic_core_rule_,
        ep_diagnostic_tails_, ep_diagnostic_tail_rule_);
}

// ── Single (g, gp) entry (simplified) ───────────────────────────────────

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
    GaussLegendreRule const& active_rule,
    GaussLegendreRule const& active_xi_rule,
    std::vector<double>& result) const
{
    int const G = num_groups();

    double const E_lo = group_boundaries_[g];
    double const E_hi = group_boundaries_[g + 1];
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
                T, Ne, multiplier, active_xi_rule);
            return w * inner;
        };

        double numerator = 0.0;

        {
            double const span = E_hi - E_lo;
            double const delta_lo = std::min(
                constants::E_BOUNDARY_LAYER_MULTIPLIER * thermal_half_width(E_lo, T),
                0.4 * span);
            double const delta_hi = std::min(
                constants::E_BOUNDARY_LAYER_MULTIPLIER * thermal_half_width(E_hi, T),
                0.4 * span);

            if (delta_lo > 1e-14 * span) {
                numerator += legendre_integrate(
                    E_integrand, active_rule, E_lo, E_lo + delta_lo);
            }

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

            if (delta_hi > 1e-14 * span) {
                numerator += legendre_integrate(
                    E_integrand, active_rule, E_hi - delta_hi, E_hi);
            }
        }

        std::size_t const idx =
            static_cast<std::size_t>(g) * G * num_angle_bins +
            static_cast<std::size_t>(gp) * num_angle_bins +
            static_cast<std::size_t>(a);
        result[idx] = 2.0 * std::numbers::pi * numerator * inv_denom;
        group_sum += std::abs(result[idx]);
    }
    return group_sum;
}

// ── Public ξ-bin integral API ───────────────────────────────────────────

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

    bool const is_cold = T < constants::COLD_TEMPERATURE_THRESHOLD;
    GaussLegendreRule const& active_xi_rule = is_cold ? xi_cold_rule_ : xi_rule_;
    double const dxi = 2.0 / static_cast<double>(num_xi_bins);

    std::vector<double> result(num_xi_bins);
    for (int a = 0; a < num_xi_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi = std::min(-1.0 + (a + 1) * dxi,
                                      1.0 - constants::XI_UPPER_EPS);
        result[a] = integrate_xi_bin(
            kernel, eval, E, Ep, xi_lo, xi_hi, T, Ne,
            multiplier, active_xi_rule);
    }
    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_xi_integral_sigma(
    ComptonKernelSolver const& kernel,
    double const E,
    double const Ep,
    int const num_xi_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_xi_integral_impl(
        kernel, &ComptonKernelSolver::sigma_E,
        E, Ep, num_xi_bins, T, Ne, multiplier);
}

std::vector<double> ComptonMultigroupKernel::compute_xi_integral_dsigma_dT(
    ComptonKernelSolver const& kernel,
    double const E,
    double const Ep,
    int const num_xi_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_xi_integral_impl(
        kernel, &ComptonKernelSolver::dsigma_E_dT,
        E, Ep, num_xi_bins, T, Ne, multiplier);
}

// ── Public E'+ξ-bin integral API ────────────────────────────────────────

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

    bool const is_cold = T < constants::COLD_TEMPERATURE_THRESHOLD;
    GaussLegendreRule const& active_xi_rule = is_cold ? xi_cold_rule_ : xi_rule_;
    double const dxi = 2.0 / static_cast<double>(num_xi_bins);

    std::vector<double> result(num_xi_bins);
    for (int a = 0; a < num_xi_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi = std::min(-1.0 + (a + 1) * dxi,
                                      1.0 - constants::XI_UPPER_EPS);
        result[a] = integrate_Ep_xi_bin(
            kernel, eval, E, Ep_lo, Ep_hi, xi_lo, xi_hi,
            T, Ne, multiplier, active_xi_rule);
    }
    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_Ep_xi_integral_sigma(
    ComptonKernelSolver const& kernel,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    int const num_xi_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_Ep_xi_integral_impl(
        kernel, &ComptonKernelSolver::sigma_E,
        E, Ep_lo, Ep_hi, num_xi_bins, T, Ne, multiplier);
}

std::vector<double> ComptonMultigroupKernel::compute_Ep_xi_integral_dsigma_dT(
    ComptonKernelSolver const& kernel,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    int const num_xi_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier) const
{
    return compute_Ep_xi_integral_impl(
        kernel, &ComptonKernelSolver::dsigma_E_dT,
        E, Ep_lo, Ep_hi, num_xi_bins, T, Ne, multiplier);
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

    // --- Weight-function denominators ---
    // D(g) = ∫_{E_lo}^{E_hi} w(E, T) dE normalises each row of the matrix
    // so that row sums represent physical scattering probabilities.
    std::vector<double> denominators(G);
    for (int g = 0; g < G; ++g) {
        denominators[g] = weight_func_->compute_denominator(
            group_boundaries_[g], group_boundaries_[g + 1], T);
    }

    // --- Cold-temperature rule selection ---
    // Below the threshold the kernel is extremely sharp (near-Thomson);
    // a higher-order rule is needed for the E and ξ axes.
    bool const is_cold = T < constants::COLD_TEMPERATURE_THRESHOLD;
    GaussLegendreRule const& active_rule = is_cold ? cold_rule_ : base_rule_;
    GaussLegendreRule const& active_xi_rule = is_cold ? xi_cold_rule_ : xi_rule_;

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
                kernel, eval, g, gp, num_angle_bins, dxi,
                T, Ne,
                inv_denom, multiplier, active_rule, active_xi_rule, result);
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
