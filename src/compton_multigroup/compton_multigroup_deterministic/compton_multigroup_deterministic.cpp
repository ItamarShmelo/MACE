#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_common/compton_common.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "utilities/compute_logger.hpp"
#include "utilities/gauss_legendre.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
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

template <typename F>
double integrate_Ep_ridge(
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
            double dp_result = 0.0;

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
    double const tau = T * units::k_boltz / units::me_c2;

    auto f = [&](double const xi) {
        return multiplier(E, Ep, xi) *
               (kernel.*eval)(E, Ep, xi, T).value;
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

    double result = 0.0;

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
    auto const rb = compute_ridge_bounds(E, xi_lo, xi_hi, T);

    auto ep_integrand = [&](double const Ep) {
        return integrate_xi_bin(
            kernel,
            eval,
            E,
            Ep,
            xi_lo,
            xi_hi,
            T,
            multiplier);
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
    double const E_lo = group_boundaries_[g];
    double const E_hi = group_boundaries_[g + 1];
    double const Ep_lo = group_boundaries_[gp];
    double const Ep_hi = group_boundaries_[gp + 1];

    auto E_integrand = [&](double const E) {
        double const w = weight_func_->weight(E, T);
        double const inner = integrate_Ep_xi_bin(
            kernel,
            eval,
            E,
            Ep_lo,
            Ep_hi,
            xi_lo,
            xi_hi,
            T,
            multiplier);
        return w * inner;
    };

    auto integrate_sub = [&](double const a, double const b) -> double {
        if (!(b > a)) {
            return 0.0;
        }
        if (b / a <= log_e_panel_ratio_) {
            return legendre_integrate(E_integrand, e_panel_rule_, a, b);
        }
        if (weight_func_->weight(a, T) >= weight_func_->weight(b, T)) {
            return log_legendre_integrate(E_integrand, e_panel_rule_, a, b);
        }
        return rlog_legendre_integrate(E_integrand, e_panel_rule_, a, b);
    };

    auto integrate_middle = [&](double const a, double const b) -> double {
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
// E' uses fixed-order GL via integrate_Ep_ridge (no adaptive refinement).
// Convergence is controlled by ep_edge_order and ep_interior_order.

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

    // Uniform angle-bin width: ξ ∈ [-1, 1] split into num_angle_bins slices.
    double const dxi = 2.0 / static_cast<double>(num_angle_bins);

    ComputeLogger logger(
        "deterministic",
        std::format(
            "G={}, angle_bins={}, T={:.4g} keV",
            G,
            num_angle_bins,
            T / units::kev_kelvin));

// --- Main loop over incoming groups g ---
#pragma omp parallel for schedule(dynamic)
    for (int g = 0; g < G; ++g) {
        double const denom = weight_func_->compute_denominator(
            group_boundaries_[g],
            group_boundaries_[g + 1],
            T);
        double const inv_denom = 1.0 / denom;

        auto do_group = [&](int const gp) {
            double group_sum = 0.0;

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

                std::size_t const idx =
                    static_cast<std::size_t>(g) * G * num_angle_bins +
                    static_cast<std::size_t>(gp) * num_angle_bins +
                    static_cast<std::size_t>(a);
                result[idx] = 2.0 * std::numbers::pi * numerator * inv_denom;
                group_sum += std::abs(result[idx]);
            }
            return group_sum;
        };

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

    logger.done();

    return result;
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

} // namespace compton
