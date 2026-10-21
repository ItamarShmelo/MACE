#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_common/compton_common.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <print>
#include <stdexcept>

namespace compton {

ComptonKernelSolver::ComptonKernelSolver(
    double const asymp_tau_alpha_threshold,
    double const power_series_self_tol,
    double const asymp_self_tol,
    double const dd_power_series_self_tol,
    double const dd_asymp_self_tol,
    double const dd_last_resort_self_tol,
    bool const verbose)
    : asymp_tau_alpha_threshold_(asymp_tau_alpha_threshold),
      power_series_self_tol_(power_series_self_tol),
      asymp_self_tol_(asymp_self_tol),
      dd_power_series_self_tol_(dd_power_series_self_tol),
      dd_asymp_self_tol_(dd_asymp_self_tol),
      dd_last_resort_self_tol_(dd_last_resort_self_tol),
      verbose_(verbose),
      asymp_series_(false),
      asymp_series_dd_(true),
      power_series_(false),
      power_series_dd_(true, 1e-8, 4, 500)
{}

namespace {

/// Allowed disagreement of a cross-regime result with the in-regime DD
/// result, in units of the in-regime error estimate.
constexpr double CROSS_REGIME_AGREEMENT = 10.0;

template <ComptonKernelSolver::KernelOp Op, typename Solver>
ComptonResult eval_kernel(
    Solver const& s,
    double E,
    double Ep,
    double xi,
    double T)
{
    if constexpr (Op == ComptonKernelSolver::KernelOp::sigma) {
        return s.sigma_E(E, Ep, xi, T);
    } else {
        return s.dsigma_E_dT(E, Ep, xi, T);
    }
}

} // namespace

template <ComptonKernelSolver::KernelOp Op>
ComptonResult ComptonKernelSolver::dispatch(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    double const tau = T * units::k_boltz / units::me_c2;
    auto const p = compute_params<double>(
        E / units::me_c2,
        E_prime / units::me_c2,
        xi,
        tau);
    double const tau_alpha_max =
        std::max(tau * p.alpha_plus, tau * p.alpha_minus);

    auto const evaluate = [&](bool const asymp, bool const dd)
        -> std::optional<ComptonResult> {
        try {
            if (asymp) {
                return eval_kernel<Op>(
                    dd ? asymp_series_dd_ : asymp_series_, E, E_prime, xi, T);
            }
            return eval_kernel<Op>(
                dd ? power_series_dd_ : power_series_, E, E_prime, xi, T);
        } catch (...) { // NOLINT(bugprone-empty-catch) -- fallthrough to next
                        // backend
        }
        return std::nullopt;
    };
    auto const sign_ok = [](bool const asymp, ComptonResult const& r) {
        return asymp || Op == KernelOp::dsigma_dT || r.value >= 0.0;
    };
    auto const accepted = [&](bool const asymp,
                              std::optional<ComptonResult> const& r,
                              double const tol) {
        return r && sign_ok(asymp, *r) && r->estimated_rel_error < tol;
    };
    // Outside its regime a series can stall and report a tiny estimate while
    // being wrong by many orders of magnitude, so a cross-regime result must
    // also agree with the in-regime DD result to within that result's own
    // error estimate.
    auto const consistent = [](ComptonResult const& candidate,
                               std::optional<ComptonResult> const& reference) {
        return reference &&
               std::abs(candidate.value - reference->value) <=
                   CROSS_REGIME_AGREEMENT * reference->estimated_rel_error *
                       std::abs(reference->value);
    };

    bool const asymp_regime = tau_alpha_max < asymp_tau_alpha_threshold_;
    // A1/P1: double precision series of the regime.
    double const double_tol =
        asymp_regime ? asymp_self_tol_ : power_series_self_tol_;
    if (auto const r = evaluate(asymp_regime, false);
        accepted(asymp_regime, r, double_tol)) {
        return *r;
    }
    // A2/P2: double-double series of the regime.
    double const dd_tol =
        asymp_regime ? dd_asymp_self_tol_ : dd_power_series_self_tol_;
    auto const in_regime = evaluate(asymp_regime, true);
    if (accepted(asymp_regime, in_regime, dd_tol)) {
        return *in_regime;
    }
    // A3/P3: double-double series of the other regime.  The asymptotic series
    // is divergent, so near the threshold A2 is truncation-limited.
    double const cross_tol =
        asymp_regime ? dd_power_series_self_tol_ : dd_asymp_self_tol_;
    if (auto const r = evaluate(!asymp_regime, true);
        accepted(!asymp_regime, r, cross_tol) && consistent(*r, in_regime)) {
        return *r;
    }
    if (accepted(asymp_regime, in_regime, dd_last_resort_self_tol_)) {
        return *in_regime;
    }

    throw std::runtime_error("all kernel backends failed");
}

ComptonResult ComptonKernelSolver::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    ComptonResult result{};
    try {
        result = dispatch<KernelOp::sigma>(E, E_prime, xi, T);
    } catch (...) {
        if (verbose_) {
            double const gamma = E / units::me_c2;
            double const gamma_p = E_prime / units::me_c2;
            double const T_kev = T / units::kev_kelvin;
            double const tau = T * units::k_boltz / units::me_c2;
            std::println(
                stderr,
                "WARNING sigma_E: all backends failed, returning 0"
                "  (gamma={:.6e}, gamma'={:.6e}, T={:.6e} keV, tau={:.6e}, "
                "xi={:.6e})",
                gamma,
                gamma_p,
                T_kev,
                tau,
                xi);
        }
        return ComptonResult{0.0, 1.0, 1.0, 0};
    }
    if (result.value < 0.0) {
        if (verbose_) {
            double const gamma = E / units::me_c2;
            double const gamma_p = E_prime / units::me_c2;
            double const T_kev = T / units::kev_kelvin;
            double const tau = T * units::k_boltz / units::me_c2;
            std::println(
                stderr,
                "WARNING sigma_E: {:.6e} clamped to 0"
                "  (gamma={:.6e}, gamma'={:.6e}, T={:.6e} keV, tau={:.6e}, "
                "xi={:.6e}, "
                "err={:.3e})",
                result.value,
                gamma,
                gamma_p,
                T_kev,
                tau,
                xi,
                result.estimated_rel_error);
        }
        result.value = 0.0;
    }
    return result;
}

ComptonResult ComptonKernelSolver::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    try {
        return dispatch<KernelOp::dsigma_dT>(E, E_prime, xi, T);
    } catch (...) {
        if (verbose_) {
            double const gamma = E / units::me_c2;
            double const gamma_p = E_prime / units::me_c2;
            double const T_kev = T / units::kev_kelvin;
            double const tau = T * units::k_boltz / units::me_c2;
            std::println(
                stderr,
                "WARNING dsigma_E_dT: all backends failed, returning 0"
                "  (gamma={:.6e}, gamma'={:.6e}, T={:.6e} keV, tau={:.6e}, "
                "xi={:.6e})",
                gamma,
                gamma_p,
                T_kev,
                tau,
                xi);
        }
        return ComptonResult{0.0, 1.0, 1.0, 0};
    }
}

} // namespace compton
