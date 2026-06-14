#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"

#include <cstdio>

namespace compton {

ComptonKernelSolver::ComptonKernelSolver(
    double const asymp_tau_alpha_threshold,
    double const gamma_double_precision_safe,
    double const quadrature_self_tol,
    double const asymp_gamma_dd_threshold,
    double const asymp_self_tol,
    double const asymp_gamma_dd_cross_val_threshold)
    : asymp_tau_alpha_threshold_(asymp_tau_alpha_threshold)
    , gamma_double_precision_safe_(gamma_double_precision_safe)
    , quadrature_self_tol_(quadrature_self_tol)
    , asymp_gamma_dd_threshold_(asymp_gamma_dd_threshold)
    , asymp_self_tol_(asymp_self_tol)
    , asymp_gamma_dd_cross_val_threshold_(asymp_gamma_dd_cross_val_threshold)
    , asymp_series_(false)
    , asymp_series_dd_(true)
    , power_series_(false)
    , power_series_dd_(true)
    , quadrature_(64)
{}

namespace {

/// Compile-time selector: calls sigma_E or dsigma_E_dT on any solver
/// backend, deducing the concrete solver type from the argument.
template <ComptonKernelSolver::KernelOp Op, typename Solver>
ComptonResult eval_kernel(Solver const& s,
                          double E, double Ep, double xi,
                          double T, double Ne)
{
    if constexpr (Op == ComptonKernelSolver::KernelOp::sigma)
        return s.sigma_E(E, Ep, xi, T, Ne);
    else
        return s.dsigma_E_dT(E, Ep, xi, T, Ne);
}

} // anon namespace

template <ComptonKernelSolver::KernelOp Op>
ComptonResult ComptonKernelSolver::dispatch(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    double const tau     = T * units::k_boltz / units::me_c2;
    double const gamma   = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    auto const p = compute_params<double>(gamma, gamma_p, xi, tau);

    double const tau_alpha_max = std::max(tau * p.alpha_plus,
                                          tau * p.alpha_minus);
    double const gamma_min = std::min(gamma, gamma_p);

    if (tau_alpha_max < asymp_tau_alpha_threshold_) {
      try {
        auto const a_result =
            (gamma_min < asymp_gamma_dd_threshold_)
                ? eval_kernel<Op>(asymp_series_dd_, E, E_prime, xi, T, Ne)
                : eval_kernel<Op>(asymp_series_,    E, E_prime, xi, T, Ne);
        if (a_result.estimated_rel_error < asymp_self_tol_) {
            // Near the dispatch boundary at ultra-low gamma the
            // asymptotic error estimate can silently underestimate the
            // true error.  Cross-validate against DD power series, but
            // only when tau_alpha is close to the threshold -- deep in
            // the asymptotic regime the kernel can have narrow spikes
            // that are correct point-wise but unresolvable by GL
            // quadrature, so using asymptotic there avoids artifacts.
            // Only prefer PS-DD when it also reports lower self-error
            // than the asymptotic series, guarding against the PS-DD
            // error estimator underreporting at extreme kinematics.
            if (gamma_min < asymp_gamma_dd_cross_val_threshold_
                && tau_alpha_max > 0.4 * asymp_tau_alpha_threshold_) {
                try {
                    auto const ps = eval_kernel<Op>(power_series_dd_,
                                                    E, E_prime, xi, T, Ne);
                    if (ps.estimated_rel_error < quadrature_self_tol_
                        && ps.estimated_rel_error < a_result.estimated_rel_error)
                        return ps;
                } catch (...) {}
            }
            return a_result;
        }
      } catch (...) {}
    }

    if (gamma_min >= gamma_double_precision_safe_)
        return eval_kernel<Op>(power_series_, E, E_prime, xi, T, Ne);

    auto const q_result = eval_kernel<Op>(quadrature_, E, E_prime, xi, T, Ne);
    if (q_result.estimated_rel_error < quadrature_self_tol_)
        return q_result;

    ComptonResult ps_result;
    bool ps_ok = false;
    try {
        ps_result = eval_kernel<Op>(power_series_dd_, E, E_prime, xi, T, Ne);
        ps_ok = ps_result.estimated_rel_error < 1.0;
    } catch (...) {}

    // Always try asymp-DD as a competing estimate.  At ultra-low gamma
    // and large tau*alpha the PS-DD error estimator can underreport,
    // passing the tolerance gate with a wrong-sign result while
    // asymp-DD carries the correct answer with even lower self-error.
    ComptonResult a_last;
    bool a_ok = false;
    try {
        a_last = eval_kernel<Op>(asymp_series_dd_, E, E_prime, xi, T, Ne);
        a_ok = a_last.estimated_rel_error < 1.0;
    } catch (...) {}

    if (ps_ok && a_ok) {
        auto const& better = (a_last.estimated_rel_error <= ps_result.estimated_rel_error)
                                 ? a_last : ps_result;
        if (better.estimated_rel_error < quadrature_self_tol_)
            return better;
        return (a_last.estimated_rel_error <= ps_result.estimated_rel_error)
                   ? a_last : ps_result;
    }
    if (ps_ok) return ps_result;
    if (a_ok)  return a_last;
    throw std::runtime_error("all kernel backends failed");
}

ComptonResult ComptonKernelSolver::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    ComptonResult result;
    try {
        result = dispatch<KernelOp::sigma>(E, E_prime, xi, T, Ne);
    } catch (...) {
        double const gamma   = E       / units::me_c2;
        double const gamma_p = E_prime / units::me_c2;
        double const T_kev   = T / units::kev_kelvin;
        double const tau     = T * units::k_boltz / units::me_c2;
        std::fprintf(stderr,
            "WARNING sigma_E: all backends failed, returning 0"
            "  (gamma=%.6e, gamma'=%.6e, T=%.6e keV, tau=%.6e, xi=%.6e)\n",
            gamma, gamma_p, T_kev, tau, xi);
        return ComptonResult{0.0, 1.0};
    }
    if (result.value < 0.0) {
        double const gamma   = E       / units::me_c2;
        double const gamma_p = E_prime / units::me_c2;
        double const T_kev   = T / units::kev_kelvin;
        double const tau     = T * units::k_boltz / units::me_c2;
        std::fprintf(stderr,
            "WARNING sigma_E: %.6e clamped to 0"
            "  (gamma=%.6e, gamma'=%.6e, T=%.6e keV, tau=%.6e, xi=%.6e, err=%.3e)\n",
            result.value, gamma, gamma_p, T_kev, tau, xi, result.estimated_rel_error);
        result.value = 0.0;
    }
    return result;
}

ComptonResult ComptonKernelSolver::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    try {
        return dispatch<KernelOp::dsigma_dT>(E, E_prime, xi, T, Ne);
    } catch (...) {
        double const gamma   = E       / units::me_c2;
        double const gamma_p = E_prime / units::me_c2;
        double const T_kev   = T / units::kev_kelvin;
        double const tau     = T * units::k_boltz / units::me_c2;
        std::fprintf(stderr,
            "WARNING dsigma_E_dT: all backends failed, returning 0"
            "  (gamma=%.6e, gamma'=%.6e, T=%.6e keV, tau=%.6e, xi=%.6e)\n",
            gamma, gamma_p, T_kev, tau, xi);
        return ComptonResult{0.0, 1.0};
    }
}

} // namespace compton
