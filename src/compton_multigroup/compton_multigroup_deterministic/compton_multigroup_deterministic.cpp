#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_common/compton_common.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "utilities/compute_logger.hpp"
#include "utilities/gauss_legendre.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace compton {

namespace constants {
static constexpr double XI_UPPER_EPS = 1e-10;
static constexpr double DOUBLE_PEAK_RATIO_THRESHOLD = 10.0;
} // namespace constants

// ── MGIntegrationConfig ─────────────────────────────────────────────────

MGIntegrationConfig::MGIntegrationConfig(
    std::optional<double> const cutoff_ratio,
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
    : xi_order(xi_order),
      xi_tail_order(xi_tail_order),
      cutoff_ratio(cutoff_ratio),
      xi_peak_k(xi_peak_k),
      ep_k_cut(ep_k_cut),
      ep_k_in(ep_k_in),
      ep_edge_order(ep_edge_order),
      ep_interior_order(ep_interior_order),
      e_panel_order(e_panel_order),
      log_e_panel_ratio(log_e_panel_ratio),
      e_boundary_k(e_boundary_k)
{
    if (cutoff_ratio.has_value() && !(cutoff_ratio.value() > 0.0)) {
        throw std::invalid_argument(
            "cutoff_ratio must be > 0 when provided; use nullopt to disable");
    }
    if (xi_order.has_value() && xi_order.value() < 1) {
        throw std::invalid_argument("xi_order must be >= 1");
    }
    if (!(xi_peak_k > 0.0)) {
        throw std::invalid_argument("xi_peak_k must be > 0");
    }
    if (!(ep_k_cut > 0.0)) {
        throw std::invalid_argument("ep_k_cut must be > 0");
    }
    if (!(ep_k_in >= 0.0)) {
        throw std::invalid_argument("ep_k_in must be >= 0");
    }
    if (!(ep_k_cut > ep_k_in)) {
        throw std::invalid_argument(
            "ep_k_cut must be > ep_k_in "
            "(semantic constraint: the edge region must be narrower than the "
            "retained interval)");
    }
    if (ep_edge_order.has_value() && ep_edge_order.value() < 1) {
        throw std::invalid_argument("ep_edge_order must be >= 1");
    }
    if (ep_interior_order.has_value() && ep_interior_order.value() < 1) {
        throw std::invalid_argument("ep_interior_order must be >= 1");
    }
    if (e_panel_order.has_value() && e_panel_order.value() < 1) {
        throw std::invalid_argument("e_panel_order must be >= 1");
    }
    if (!(log_e_panel_ratio > 1.0)) {
        throw std::invalid_argument("log_e_panel_ratio must be > 1.0");
    }
    if (!(e_boundary_k > 0.0)) {
        throw std::invalid_argument("e_boundary_k must be > 0");
    }
}

ComptonMultigroupKernel::ComptonMultigroupKernel(
    std::vector<double> const& energy_group_boundaries,
    std::shared_ptr<WeightFunction const> weight_function,
    MGIntegrationConfig const& config)
    : group_boundaries_(energy_group_boundaries),
      weight_func_(std::move(weight_function)),
      xi_rule_(compute_gauss_legendre(config.effective_xi_order())),
      xi_tail_rule_(compute_gauss_legendre(config.effective_xi_tail_order())),
      ep_edge_rule_(compute_gauss_legendre(config.effective_ep_edge_order())),
      ep_interior_rule_(
          compute_gauss_legendre(config.effective_ep_interior_order())),
      ep_elastic_core_rule_(compute_gauss_legendre(8)),
      e_panel_rule_(compute_gauss_legendre(config.effective_e_panel_order())),
      xi_peak_k_(config.xi_peak_k),
      ep_k_cut_(config.ep_k_cut),
      ep_k_in_(config.ep_k_in),
      group_cutoff_ratio_(config.cutoff_ratio),
      log_e_panel_ratio_(config.log_e_panel_ratio),
      e_boundary_k_(config.e_boundary_k)
{
    if (energy_group_boundaries.size() < 2) {
        throw std::invalid_argument("need at least 2 boundaries (1 group)");
    }

    for (auto const& boundary : energy_group_boundaries) {
        if (!(boundary > 0.0) || !std::isfinite(boundary)) {
            throw std::invalid_argument(
                "all boundaries must be finite and > 0");
        }
    }

    for (std::size_t i = 0; i + 1 < energy_group_boundaries.size(); ++i) {
        if (energy_group_boundaries[i] >= energy_group_boundaries[i + 1]) {
            throw std::invalid_argument(
                "boundaries must be strictly increasing");
        }
    }
}

namespace {

/// Moments of one kernel evaluation (one per multiplier) for the shared
/// integrations: component-wise arithmetic, so each component is computed
/// exactly as a double integrand would be.
template <std::size_t N>
struct KernelMoments {
    std::array<double, N> v{};

    KernelMoments& operator+=(KernelMoments const& other)
    {
        for (std::size_t k = 0; k < N; ++k) {
            v[k] += other.v[k];
        }
        return *this;
    }

    friend KernelMoments operator+(KernelMoments a, KernelMoments const& b)
    {
        a += b;
        return a;
    }

    friend KernelMoments operator*(double const s, KernelMoments a)
    {
        for (std::size_t k = 0; k < N; ++k) {
            a.v[k] = s * a.v[k];
        }
        return a;
    }

    friend KernelMoments operator*(KernelMoments a, double const s)
    {
        for (std::size_t k = 0; k < N; ++k) {
            a.v[k] = a.v[k] * s;
        }
        return a;
    }
};

template <typename F>
auto integrate_Ep_ridge(
    F const& f,
    double Ep_lo,
    double Ep_hi,
    RidgeBounds const& rb,
    double k_cut,
    double k_in,
    GaussLegendreRule const& edge_rule,
    GaussLegendreRule const& interior_rule,
    GaussLegendreRule const& elastic_core_rule)
{
    using Value = std::decay_t<decltype(f(Ep_lo))>;
    double const keep_lo = std::max(Ep_lo, rb.cold_lo - k_cut * rb.sigma_lo);
    double const keep_hi = std::min(
        Ep_hi,
        std::max(
            rb.cold_lo + k_cut * rb.sigma_lo,
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
        double const ec_lo =
            std::max(keep_lo, rb.cold_hi - k_cut * rb.sigma_hi);
        double const ec_hi =
            std::min(keep_hi, rb.cold_hi + k_cut * rb.sigma_hi);

        if (ec_hi > ec_lo && ec_hi > Ep_lo && ec_lo < Ep_hi) {
            Value dp_result{};

            if (keep_lo > Ep_lo) {
                assert(Ep_lo > 0.0);
                dp_result +=
                    rlog_legendre_integrate(f, edge_rule, Ep_lo, keep_lo);
            }

            if (ec_lo > keep_lo) {
                dp_result += legendre_integrate(f, edge_rule, keep_lo, ec_lo);
            }

            dp_result += legendre_integrate(f, elastic_core_rule, ec_lo, ec_hi);

            if (keep_hi > ec_hi) {
                dp_result +=
                    legendre_integrate(f, interior_rule, ec_hi, keep_hi);
            }

            if (Ep_hi > keep_hi) {
                assert(keep_hi > 0.0);
                if (Ep_hi / keep_hi > 2.0) {
                    dp_result +=
                        log_legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
                } else {
                    dp_result +=
                        legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
                }
            }

            return dp_result;
        }
    }

    // Ridge entirely outside [Ep_lo, Ep_hi]: integrate the full group range.
    if (keep_lo >= keep_hi) {
        assert(Ep_lo > 0.0);
        if (Ep_hi <= rb.cold_lo) {
            return rlog_legendre_integrate(f, edge_rule, Ep_lo, Ep_hi);
        }
        if (Ep_hi / Ep_lo > 2.0) {
            return log_legendre_integrate(f, edge_rule, Ep_lo, Ep_hi);
        }
        return legendre_integrate(f, edge_rule, Ep_lo, Ep_hi);
    }

    Value result{};

    if (keep_lo > Ep_lo) {
        assert(Ep_lo > 0.0);
        result += rlog_legendre_integrate(f, edge_rule, Ep_lo, keep_lo);
    }

    double const edge_lo = rb.cold_lo + k_in * rb.sigma_lo;
    double const edge_hi = rb.cold_hi - k_in * rb.sigma_hi;

    if (edge_lo >= edge_hi) {
        result += legendre_integrate(f, interior_rule, keep_lo, keep_hi);
    } else {
        double const left_hi = std::min(keep_hi, edge_lo);
        double const mid_lo = std::max(keep_lo, edge_lo);
        double const mid_hi = std::min(keep_hi, edge_hi);
        double const right_lo = std::max(keep_lo, edge_hi);

        if (left_hi > keep_lo) {
            result += legendre_integrate(f, edge_rule, keep_lo, left_hi);
        }
        if (mid_hi > mid_lo) {
            result += legendre_integrate(f, interior_rule, mid_lo, mid_hi);
        }
        if (keep_hi > right_lo) {
            result += legendre_integrate(f, edge_rule, right_lo, keep_hi);
        }
    }

    if (Ep_hi > keep_hi) {
        assert(keep_hi > 0.0);
        if (Ep_hi / keep_hi > 2.0) {
            result += log_legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
        } else {
            result += legendre_integrate(f, edge_rule, keep_hi, Ep_hi);
        }
    }

    return result;
}

} // anonymous namespace

template <typename Point>
auto ComptonMultigroupKernel::integrate_xi_bin_point(
    Point const& point,
    double const E,
    double const Ep,
    double const xi_lo,
    double const xi_hi,
    double const T) const
{
    using Value = std::decay_t<decltype(point(E, Ep, xi_lo))>;
    double const tau = T * units::k_boltz / units::me_c2;

    auto f = [&](double const xi) {
        return point(E, Ep, xi);
    };

    double const gamma = E / units::me_c2;
    double const gamma_p = Ep / units::me_c2;
    double const abs_dg = std::abs(gamma - gamma_p);

    double const peak_distance_from_one = abs_dg / (gamma * gamma_p);
    double const sigma_xi =
        std::sqrt(tau * abs_dg * (2.0 + abs_dg)) / (gamma * gamma_p);

    bool const endpoint_localized =
        endpoint_localized_xi(gamma, gamma_p, tau);

    if (endpoint_localized) {
        double const span = xi_hi - xi_lo;
        double const eps = span * 1e-14;
        return rlog_legendre_integrate(
            [&](double const s) { return f(xi_lo + s); },
            xi_rule_,
            eps,
            span);
    }

    double const xi_pk = 1.0 - peak_distance_from_one;
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

    Value result{};

    if (core_lo > xi_lo) {
        double const span = core_lo - xi_lo;
        if (span > 1e-14 * bin_span) {
            result += legendre_integrate(f, xi_tail_rule_, xi_lo, core_lo);
        }
    }

    if (core_hi > core_lo) {
        result += legendre_integrate(f, xi_rule_, core_lo, core_hi);
    }

    if (core_hi < xi_hi) {
        double const span = xi_hi - core_hi;
        if (span > 1e-14 * bin_span) {
            result += legendre_integrate(f, xi_tail_rule_, core_hi, xi_hi);
        }
    }

    return result;
}

template <typename Point>
auto ComptonMultigroupKernel::integrate_Ep_xi_bin_point(
    Point const& point,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    double const xi_lo,
    double const xi_hi,
    double const T) const
{
    auto const rb = compute_ridge_bounds(E, xi_lo, xi_hi, T);

    auto ep_integrand = [&](double const Ep) {
        return integrate_xi_bin_point(point, E, Ep, xi_lo, xi_hi, T);
    };

    return integrate_Ep_ridge(
        ep_integrand,
        Ep_lo,
        Ep_hi,
        rb,
        ep_k_cut_,
        ep_k_in_,
        ep_edge_rule_,
        ep_interior_rule_,
        ep_elastic_core_rule_);
}

template <typename Point>
auto ComptonMultigroupKernel::integrate_E_Ep_xi_bin_point(
    Point const& point,
    int const g,
    int const gp,
    double const xi_lo,
    double const xi_hi,
    double const T) const
{
    double const E_lo = group_boundaries_[g];
    double const E_hi = group_boundaries_[g + 1];
    double const Ep_lo = group_boundaries_[gp];
    double const Ep_hi = group_boundaries_[gp + 1];
    using Value = std::decay_t<decltype(point(E_lo, Ep_lo, xi_lo))>;

    auto E_integrand = [&](double const E) {
        double const w = weight_func_->weight(E, T);
        auto const inner = integrate_Ep_xi_bin_point(
            point,
            E,
            Ep_lo,
            Ep_hi,
            xi_lo,
            xi_hi,
            T);
        return w * inner;
    };

    auto integrate_sub = [&](double const a, double const b) -> Value {
        if (!(b > a)) {
            return Value{};
        }
        if (b / a <= log_e_panel_ratio_) {
            return legendre_integrate(E_integrand, e_panel_rule_, a, b);
        }
        if (weight_func_->weight(a, T) >= weight_func_->weight(b, T)) {
            return log_legendre_integrate(E_integrand, e_panel_rule_, a, b);
        }
        return rlog_legendre_integrate(E_integrand, e_panel_rule_, a, b);
    };

    auto integrate_middle = [&](double const a, double const b) -> Value {
        auto const Epk = weight_func_->peak_energy(T);
        if (Epk && *Epk > a && *Epk < b) {
            return integrate_sub(a, *Epk) + integrate_sub(*Epk, b);
        }
        return integrate_sub(a, b);
    };

    double const sigma_lo = ridge_thermal_width(E_lo, -1.0, T);
    double const sigma_hi = ridge_thermal_width(E_hi, -1.0, T);
    double const bl_lo = E_lo + e_boundary_k_ * sigma_lo;
    double const bl_hi = E_hi - e_boundary_k_ * sigma_hi;

    if (bl_lo < bl_hi) {
        return integrate_sub(E_lo, bl_lo) + integrate_middle(bl_lo, bl_hi) +
               integrate_sub(bl_hi, E_hi);
    }
    return integrate_middle(E_lo, E_hi);
}

double ComptonMultigroupKernel::integrate_xi_bin(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    double const E,
    double const Ep,
    double const xi_lo,
    double const xi_hi,
    double const T,
    KernelMultiplier const& multiplier) const
{
    auto const point = [&](double const e, double const ep, double const xi) {
        return multiplier(e, ep, xi) * (kernel.*eval)(e, ep, xi, T).value;
    };
    return integrate_xi_bin_point(point, E, Ep, xi_lo, xi_hi, T);
}

double ComptonMultigroupKernel::integrate_Ep_xi_bin(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    double const xi_lo,
    double const xi_hi,
    double const T,
    KernelMultiplier const& multiplier) const
{
    auto const point = [&](double const e, double const ep, double const xi) {
        return multiplier(e, ep, xi) * (kernel.*eval)(e, ep, xi, T).value;
    };
    return integrate_Ep_xi_bin_point(point, E, Ep_lo, Ep_hi, xi_lo, xi_hi, T);
}

double ComptonMultigroupKernel::integrate_E_Ep_xi_bin(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    int const g,
    int const gp,
    double const xi_lo,
    double const xi_hi,
    double const T,
    KernelMultiplier const& multiplier) const
{
    auto const point = [&](double const e, double const ep, double const xi) {
        return multiplier(e, ep, xi) * (kernel.*eval)(e, ep, xi, T).value;
    };
    return integrate_E_Ep_xi_bin_point(point, g, gp, xi_lo, xi_hi, T);
}

std::vector<double> ComptonMultigroupKernel::compute_xi_integral_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    double const E,
    double const Ep,
    int const num_xi_bins,
    double const T,
    KernelMultiplier const& multiplier) const
{
    if (num_xi_bins < 1) {
        throw std::invalid_argument("num_xi_bins must be >= 1");
    }
    if (!(E > 0.0) || !std::isfinite(E)) {
        throw std::invalid_argument("E must be finite and > 0");
    }
    if (!(Ep > 0.0) || !std::isfinite(Ep)) {
        throw std::invalid_argument("Ep must be finite and > 0");
    }

    double const dxi = 2.0 / static_cast<double>(num_xi_bins);

    std::vector<double> result(num_xi_bins);
    for (int a = 0; a < num_xi_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi =
            std::min(-1.0 + (a + 1) * dxi, 1.0 - constants::XI_UPPER_EPS);
        result[a] = integrate_xi_bin(
            kernel,
            eval,
            E,
            Ep,
            xi_lo,
            xi_hi,
            T,
            multiplier);
    }
    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_Ep_xi_integral_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    double const E,
    double const Ep_lo,
    double const Ep_hi,
    int const num_xi_bins,
    double const T,
    KernelMultiplier const& multiplier) const
{
    if (num_xi_bins < 1) {
        throw std::invalid_argument("num_xi_bins must be >= 1");
    }
    if (!(E > 0.0) || !std::isfinite(E)) {
        throw std::invalid_argument("E must be finite and > 0");
    }
    if (!(Ep_lo > 0.0) || !std::isfinite(Ep_lo)) {
        throw std::invalid_argument("Ep_lo must be finite and > 0");
    }
    if (!(Ep_hi > 0.0) || !std::isfinite(Ep_hi)) {
        throw std::invalid_argument("Ep_hi must be finite and > 0");
    }
    if (Ep_lo >= Ep_hi) {
        throw std::invalid_argument("Ep_lo must be < Ep_hi");
    }

    double const dxi = 2.0 / static_cast<double>(num_xi_bins);

    std::vector<double> result(num_xi_bins);
    for (int a = 0; a < num_xi_bins; ++a) {
        double const xi_lo = -1.0 + a * dxi;
        double const xi_hi =
            std::min(-1.0 + (a + 1) * dxi, 1.0 - constants::XI_UPPER_EPS);
        result[a] = integrate_Ep_xi_bin(
            kernel,
            eval,
            E,
            Ep_lo,
            Ep_hi,
            xi_lo,
            xi_hi,
            T,
            multiplier);
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
//          - When cutoff_ratio is set, expand rightward
//            (gp_peak+1, gp_peak+2, …) until the angle-summed magnitude
//            drops below cutoff_ratio × peak_sum, then leftward similarly.
//            Unevaluated groups are left at zero.
//          - When cutoff_ratio is nullopt, all target groups are evaluated.
//
//      This exploits the physical locality of Compton scattering:
//      most of the scattered energy stays near the incoming energy,
//      so distant target groups contribute negligibly.
//
//   3. Each selected (g, gp, angle) bin is evaluated by
//   integrate_E_Ep_xi_bin().
//
// OpenMP: each incoming group g is a task, and each angle bin of a
// (g, gp) pair is a nested task, so idle threads take angle bins from any g.
// A flat parallel loop over g caps the speedup at G and at the cost of the
// slowest group, and a parallel loop over angle bins alone leaves the
// angle-integrated matrix (num_angle_bins = 1) serial.  The gp traversal
// stays serial within g because each cutoff test needs the previous gp.
//
// E' uses fixed-order GL via integrate_Ep_ridge (no adaptive refinement).
// Convergence is controlled by ep_edge_order and ep_interior_order.

void ComptonMultigroupKernel::fill_row(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    int const g,
    int const num_angle_bins,
    double const T,
    KernelMultiplier const& multiplier,
    std::optional<double> const effective_cutoff,
    double* const row,
    std::vector<char> const* const target_mask) const
{
    int const G = num_groups();

    // Uniform angle-bin width: ξ ∈ [-1, 1] split into num_angle_bins slices.
    double const dxi = 2.0 / static_cast<double>(num_angle_bins);

    double const denom = weight_func_->compute_denominator(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const inv_denom = 1.0 / denom;

    auto do_group = [&](int const gp) {
        // row holds the G x num_angle_bins block of source group g.
        std::size_t const base =
            static_cast<std::size_t>(gp) * num_angle_bins;

#pragma omp taskloop grainsize(1) default(shared)
        for (int a = 0; a < num_angle_bins; ++a) {
            double const xi_lo = -1.0 + a * dxi;
            double const xi_hi = std::min(
                -1.0 + (a + 1) * dxi,
                1.0 - constants::XI_UPPER_EPS);

            double const numerator = integrate_E_Ep_xi_bin(
                kernel,
                eval,
                g,
                gp,
                xi_lo,
                xi_hi,
                T,
                multiplier);

            row[base + static_cast<std::size_t>(a)] =
                2.0 * std::numbers::pi * numerator * inv_denom;
        }

        // Summed in a fixed order after the taskloop so that the cutoff
        // decision does not depend on the thread count.
        double group_sum = 0.0;
        for (int a = 0; a < num_angle_bins; ++a) {
            group_sum +=
                std::abs(row[base + static_cast<std::size_t>(a)]);
        }
        return group_sum;
    };

    // A target mask (no cutoff) restricts the evaluation to the marked g'.
    if (target_mask != nullptr) {
        for (int gp = 0; gp < G; ++gp) {
            if ((*target_mask)[static_cast<std::size_t>(gp)] != 0) {
                do_group(gp);
            }
        }
        return;
    }

    // --- Outward-from-peak target-group traversal ---
    int const gp_peak = g;

    double const peak_sum = do_group(gp_peak);

    if (effective_cutoff.has_value()) {
        double const cutoff = effective_cutoff.value() * peak_sum;

        // Expand rightward (higher E' groups) until below cutoff.
        for (int gp = gp_peak + 1; gp < G; ++gp) {
            if (do_group(gp) < cutoff) {
                break;
            }
        }
        // Expand leftward (lower E' groups) until below cutoff.
        for (int gp = gp_peak - 1; gp >= 0; --gp) {
            if (do_group(gp) < cutoff) {
                break;
            }
        }
    } else {
        for (int gp = gp_peak + 1; gp < G; ++gp) {
            do_group(gp);
        }
        for (int gp = gp_peak - 1; gp >= 0; --gp) {
            do_group(gp);
        }
    }
}

std::vector<double> ComptonMultigroupKernel::compute_matrix_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    int const num_angle_bins,
    double const T,
    KernelMultiplier const& multiplier,
    std::optional<double> const effective_cutoff) const
{
    if (num_angle_bins < 1) {
        throw std::invalid_argument("num_angle_bins must be >= 1");
    }

    int const G = num_groups();

    // Output buffer: flat row-major [g][gp][angle_bin], initialised to zero.
    // Unevaluated entries (skipped by the cutoff) remain at zero.
    std::vector<double> result(
        static_cast<std::size_t>(G) * G * num_angle_bins,
        0.0);

    ComputeLogger logger(
        "deterministic",
        std::format(
            "G={}, angle_bins={}, T={:.4g} keV",
            G,
            num_angle_bins,
            T / units::kev_kelvin));

// --- Main loop over incoming groups g ---
// default(shared) on both taskloops: otherwise GCC makes the reference
// `multiplier` firstprivate, copying it as the abstract KernelMultiplier.
#pragma omp parallel
#pragma omp single
#pragma omp taskloop grainsize(1) default(shared)
    for (int g = 0; g < G; ++g) {
        fill_row(
            kernel,
            eval,
            g,
            num_angle_bins,
            T,
            multiplier,
            effective_cutoff,
            result.data() + static_cast<std::size_t>(g) * G * num_angle_bins,
            nullptr);
    }

    logger.done();

    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_row_impl(
    ComptonKernelSolver const& kernel,
    ComptonResult (
        ComptonKernelSolver::*eval)(double, double, double, double)
        const,
    int const g,
    int const num_angle_bins,
    double const T,
    KernelMultiplier const& multiplier,
    std::optional<double> const effective_cutoff,
    std::vector<char> const* const target_mask) const
{
    if (num_angle_bins < 1) {
        throw std::invalid_argument("num_angle_bins must be >= 1");
    }
    int const G = num_groups();
    if (target_mask != nullptr &&
        target_mask->size() != static_cast<std::size_t>(G)) {
        throw std::invalid_argument("target mask must have one entry per group");
    }
    if (g < 0 || g >= G) {
        throw std::invalid_argument("source group out of range");
    }

    std::vector<double> row(static_cast<std::size_t>(G) * num_angle_bins, 0.0);

// One source group: the angle-bin taskloop inside fill_row needs an
// enclosing parallel region.
#pragma omp parallel
#pragma omp single
    fill_row(
        kernel,
        eval,
        g,
        num_angle_bins,
        T,
        multiplier,
        effective_cutoff,
        row.data(),
        target_mask);

    return row;
}

std::vector<double> ComptonMultigroupKernel::compute_sigma_matrix(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    KernelMultiplier const& multiplier) const
{
    return compute_matrix_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        num_angle_bins,
        T,
        multiplier,
        group_cutoff_ratio_);
}

std::vector<double> ComptonMultigroupKernel::compute_kernel_derivative_contribution(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    KernelMultiplier const& multiplier) const
{
    return compute_matrix_impl(
        kernel,
        &ComptonKernelSolver::dsigma_E_dT,
        num_angle_bins,
        T,
        multiplier,
        group_cutoff_ratio_);
}

namespace {

class WeightDerivMultiplier : public KernelMultiplier {
    WeightFunction const& wf_;
    KernelMultiplier const& inner_;
    double T_;

  public:
    WeightDerivMultiplier(
        WeightFunction const& wf,
        KernelMultiplier const& inner,
        double T)
        : wf_(wf), inner_(inner), T_(T)
    {
    }

    double operator()(
        double const E,
        double const Ep,
        double const xi) const override
    {
        return inner_(E, Ep, xi) * wf_.d_log_weight_dT(E, T_);
    }
};

} // anonymous namespace

std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_matrix(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    KernelMultiplier const& multiplier) const
{
    std::optional<double> const no_cutoff = std::nullopt;

    // Term 1: kernel derivative 
    auto result = compute_matrix_impl(
        kernel,
        &ComptonKernelSolver::dsigma_E_dT,
        num_angle_bins,
        T,
        multiplier,
        no_cutoff);

    // Term 2: weight derivative via dlnw/dT multiplier 
    WeightDerivMultiplier wd_mult(*weight_func_, multiplier, T);
    auto const weight_deriv = compute_matrix_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        num_angle_bins,
        T,
        wd_mult,
        no_cutoff);

    // Term 3: sigma for denominator correction
    auto const sigma = compute_matrix_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        num_angle_bins,
        T,
        multiplier,
        no_cutoff);

    // Combine: full = (term1 + term2) - sigma * dWg/dT / Wg
    int const G = num_groups();
    for (int g = 0; g < G; ++g) {
        double const D = weight_func_->compute_denominator(
            group_boundaries_[g],
            group_boundaries_[g + 1],
            T);
        double const dD_dT = weight_func_->d_denominator_dT(
            group_boundaries_[g],
            group_boundaries_[g + 1],
            T);
        double const dD_over_D = dD_dT / D;

        for (int gp = 0; gp < G; ++gp) {
            for (int a = 0; a < num_angle_bins; ++a) {
                auto const idx =
                    static_cast<std::size_t>(g) * G * num_angle_bins +
                    static_cast<std::size_t>(gp) * num_angle_bins +
                    static_cast<std::size_t>(a);
                result[idx] += weight_deriv[idx] - sigma[idx] * dD_over_D;
            }
        }
    }
    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_sigma_row(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    int const g,
    KernelMultiplier const& multiplier) const
{
    return compute_row_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        g,
        num_angle_bins,
        T,
        multiplier,
        group_cutoff_ratio_);
}

std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_row(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    int const g,
    KernelMultiplier const& multiplier) const
{
    std::optional<double> const no_cutoff = std::nullopt;

    // The three terms of compute_dsigma_dT_matrix, for source group g only.
    auto result = compute_row_impl(
        kernel,
        &ComptonKernelSolver::dsigma_E_dT,
        g,
        num_angle_bins,
        T,
        multiplier,
        no_cutoff);
    WeightDerivMultiplier wd_mult(*weight_func_, multiplier, T);
    auto const weight_deriv = compute_row_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        g,
        num_angle_bins,
        T,
        wd_mult,
        no_cutoff);
    auto const sigma = compute_row_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        g,
        num_angle_bins,
        T,
        multiplier,
        no_cutoff);

    double const D = weight_func_->compute_denominator(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const dD_dT = weight_func_->d_denominator_dT(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const dD_over_D = dD_dT / D;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
        result[idx] += weight_deriv[idx] - sigma[idx] * dD_over_D;
    }
    return result;
}

std::vector<double> ComptonMultigroupKernel::compute_dsigma_dT_row(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    int const g,
    KernelMultiplier const& multiplier,
    std::vector<double> const& sigma_row) const
{
    int const G = num_groups();
    std::size_t const row_size = static_cast<std::size_t>(G) * num_angle_bins;
    if (sigma_row.size() != row_size) {
        throw std::invalid_argument("sigma_row must have G x num_angle_bins entries");
    }
    // Targets evaluated in sigma_row; the others stay zero in all terms.
    std::vector<char> mask(static_cast<std::size_t>(G), 0);
    for (int gp = 0; gp < G; ++gp) {
        for (int a = 0; a < num_angle_bins; ++a) {
            if (sigma_row[static_cast<std::size_t>(gp) * num_angle_bins +
                          static_cast<std::size_t>(a)] != 0.0) {
                mask[static_cast<std::size_t>(gp)] = 1;
            }
        }
    }
    std::optional<double> const no_cutoff = std::nullopt;
    auto result = compute_row_impl(
        kernel,
        &ComptonKernelSolver::dsigma_E_dT,
        g,
        num_angle_bins,
        T,
        multiplier,
        no_cutoff,
        &mask);
    WeightDerivMultiplier wd_mult(*weight_func_, multiplier, T);
    auto const weight_deriv = compute_row_impl(
        kernel,
        &ComptonKernelSolver::sigma_E,
        g,
        num_angle_bins,
        T,
        wd_mult,
        no_cutoff,
        &mask);

    double const D = weight_func_->compute_denominator(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const dD_dT = weight_func_->d_denominator_dT(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const dD_over_D = dD_dT / D;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
        result[idx] += weight_deriv[idx] - sigma_row[idx] * dD_over_D;
    }
    return result;
}

template <std::size_t M>
std::vector<SigmaRowWithDerivative>
ComptonMultigroupKernel::compute_rows_with_derivatives_impl(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    int const g,
    std::array<KernelMultiplier const*, M> const& multipliers) const
{
    int const G = num_groups();
    std::size_t const A = static_cast<std::size_t>(num_angle_bins);
    std::size_t const row_size = static_cast<std::size_t>(G) * A;
    double const dxi = 2.0 / static_cast<double>(num_angle_bins);
    double const denom = weight_func_->compute_denominator(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const inv_denom = 1.0 / denom;

    std::vector<WeightDerivMultiplier> weight_derivs;
    weight_derivs.reserve(M);
    for (std::size_t k = 0; k < M; ++k) {
        weight_derivs.emplace_back(*weight_func_, *multipliers[k], T);
    }
    auto const sigma_eval = &ComptonKernelSolver::sigma_E;
    auto const derivative_eval = &ComptonKernelSolver::dsigma_E_dT;

    // sigma_E sweep: sigma (k) and the weight-derivative term (M + k).
    auto const sigma_point = [&](double const E, double const Ep, double const xi) {
        double const value = (kernel.*sigma_eval)(E, Ep, xi, T).value;
        KernelMoments<2 * M> moments;
        for (std::size_t k = 0; k < M; ++k) {
            moments.v[k] = (*multipliers[k])(E, Ep, xi) * value;
            moments.v[M + k] = weight_derivs[k](E, Ep, xi) * value;
        }
        return moments;
    };
    // dsigma_E_dT sweep: the kernel-derivative term (k).
    auto const derivative_point = [&](double const E, double const Ep, double const xi) {
        double const value = (kernel.*derivative_eval)(E, Ep, xi, T).value;
        KernelMoments<M> moments;
        for (std::size_t k = 0; k < M; ++k) {
            moments.v[k] = (*multipliers[k])(E, Ep, xi) * value;
        }
        return moments;
    };

    std::vector<std::vector<double>> sigma(M, std::vector<double>(row_size, 0.0));
    std::vector<std::vector<double>> weight_term(M, std::vector<double>(row_size, 0.0));
    std::vector<std::vector<double>> derivative(M, std::vector<double>(row_size, 0.0));

    auto const clear = [&](std::vector<double>& row, int const gp) {
        std::size_t const base = static_cast<std::size_t>(gp) * A;
        std::fill(row.begin() + static_cast<std::ptrdiff_t>(base),
                  row.begin() + static_cast<std::ptrdiff_t>(base + A),
                  0.0);
    };

#pragma omp parallel
#pragma omp single
    {
        // As fill_row, for all moments at once; returns the angle-summed |sigma| per multiplier.
        auto const do_group = [&](int const gp) {
            std::size_t const base = static_cast<std::size_t>(gp) * A;
#pragma omp taskloop grainsize(1) default(shared)
            for (int a = 0; a < num_angle_bins; ++a) {
                double const xi_lo = -1.0 + a * dxi;
                double const xi_hi = std::min(
                    -1.0 + (a + 1) * dxi,
                    1.0 - constants::XI_UPPER_EPS);
                auto const numerator = integrate_E_Ep_xi_bin_point(
                    sigma_point, g, gp, xi_lo, xi_hi, T);
                for (std::size_t k = 0; k < M; ++k) {
                    sigma[k][base + static_cast<std::size_t>(a)] =
                        2.0 * std::numbers::pi * numerator.v[k] * inv_denom;
                    weight_term[k][base + static_cast<std::size_t>(a)] =
                        2.0 * std::numbers::pi * numerator.v[M + k] * inv_denom;
                }
            }
            std::array<double, M> sums{};
            for (std::size_t k = 0; k < M; ++k) {
                for (std::size_t a = 0; a < A; ++a) {
                    sums[k] += std::abs(sigma[k][base + a]);
                }
            }
            return sums;
        };

        // Outward traversal as fill_row; a multiplier whose own cutoff has
        // stopped keeps zeros (it would not have evaluated these targets).
        std::array<double, M> const peak = do_group(g);
        if (group_cutoff_ratio_.has_value()) {
            std::array<double, M> cutoff{};
            for (std::size_t k = 0; k < M; ++k) {
                cutoff[k] = group_cutoff_ratio_.value() * peak[k];
            }
            for (int const step : {1, -1}) {
                std::array<bool, M> active;
                active.fill(true);
                for (int gp = g + step; gp >= 0 && gp < G; gp += step) {
                    if (std::none_of(active.begin(), active.end(), [](bool const on) { return on; })) {
                        break;
                    }
                    std::array<double, M> const sums = do_group(gp);
                    for (std::size_t k = 0; k < M; ++k) {
                        if (!active[k]) {
                            clear(sigma[k], gp);
                            clear(weight_term[k], gp);
                        } else if (sums[k] < cutoff[k]) {
                            active[k] = false;
                        }
                    }
                }
            }
        } else {
            for (int gp = 0; gp < G; ++gp) {
                if (gp != g) {
                    do_group(gp);
                }
            }
        }

        // Derivative targets per multiplier: those with a nonzero sigma, as
        // compute_dsigma_dT_row(..., sigma_row).
        std::vector<std::vector<char>> mask(M, std::vector<char>(static_cast<std::size_t>(G), 0));
        std::vector<char> any_mask(static_cast<std::size_t>(G), 0);
        for (std::size_t k = 0; k < M; ++k) {
            for (int gp = 0; gp < G; ++gp) {
                std::size_t const base = static_cast<std::size_t>(gp) * A;
                for (std::size_t a = 0; a < A; ++a) {
                    if (sigma[k][base + a] != 0.0) {
                        mask[k][static_cast<std::size_t>(gp)] = 1;
                    }
                }
                if (mask[k][static_cast<std::size_t>(gp)] != 0) {
                    any_mask[static_cast<std::size_t>(gp)] = 1;
                } else {
                    clear(weight_term[k], gp);
                }
            }
        }

        for (int gp = 0; gp < G; ++gp) {
            if (any_mask[static_cast<std::size_t>(gp)] == 0) {
                continue;
            }
            std::size_t const base = static_cast<std::size_t>(gp) * A;
#pragma omp taskloop grainsize(1) default(shared)
            for (int a = 0; a < num_angle_bins; ++a) {
                double const xi_lo = -1.0 + a * dxi;
                double const xi_hi = std::min(
                    -1.0 + (a + 1) * dxi,
                    1.0 - constants::XI_UPPER_EPS);
                auto const numerator = integrate_E_Ep_xi_bin_point(
                    derivative_point, g, gp, xi_lo, xi_hi, T);
                for (std::size_t k = 0; k < M; ++k) {
                    if (mask[k][static_cast<std::size_t>(gp)] != 0) {
                        derivative[k][base + static_cast<std::size_t>(a)] =
                            2.0 * std::numbers::pi * numerator.v[k] * inv_denom;
                    }
                }
            }
        }
    }

    double const D = weight_func_->compute_denominator(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const dD_dT = weight_func_->d_denominator_dT(
        group_boundaries_[g],
        group_boundaries_[g + 1],
        T);
    double const dD_over_D = dD_dT / D;
    std::vector<SigmaRowWithDerivative> rows(M);
    for (std::size_t k = 0; k < M; ++k) {
        for (std::size_t idx = 0; idx < row_size; ++idx) {
            derivative[k][idx] += weight_term[k][idx] - sigma[k][idx] * dD_over_D;
        }
        rows[k].sigma = std::move(sigma[k]);
        rows[k].dsigma_dT = std::move(derivative[k]);
    }
    return rows;
}

std::vector<SigmaRowWithDerivative> ComptonMultigroupKernel::compute_rows_with_derivatives(
    ComptonKernelSolver const& kernel,
    int const num_angle_bins,
    double const T,
    int const g,
    std::vector<KernelMultiplier const*> const& multipliers) const
{
    if (num_angle_bins < 1) {
        throw std::invalid_argument("num_angle_bins must be >= 1");
    }
    if (g < 0 || g >= num_groups()) {
        throw std::invalid_argument("source group out of range");
    }
    for (KernelMultiplier const* const multiplier : multipliers) {
        if (multiplier == nullptr) {
            throw std::invalid_argument("null multiplier");
        }
    }
    if (multipliers.size() == 1) {
        return compute_rows_with_derivatives_impl<1>(
            kernel, num_angle_bins, T, g, {multipliers[0]});
    }
    if (multipliers.size() == 2) {
        return compute_rows_with_derivatives_impl<2>(
            kernel, num_angle_bins, T, g, {multipliers[0], multipliers[1]});
    }
    throw std::invalid_argument("compute_rows_with_derivatives takes one or two multipliers");
}

} // namespace compton
