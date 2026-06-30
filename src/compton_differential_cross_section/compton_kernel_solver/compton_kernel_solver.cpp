#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"

#include <cstdio>

namespace compton {

ComptonKernelSolver::ComptonKernelSolver(
    double const asymp_tau_alpha_threshold,
    double const power_series_self_tol,
    double const asymp_self_tol,
    double const dd_power_series_self_tol,
    double const dd_asymp_self_tol)
    : asymp_tau_alpha_threshold_(asymp_tau_alpha_threshold),
      power_series_self_tol_(power_series_self_tol),
      asymp_self_tol_(asymp_self_tol),
      dd_power_series_self_tol_(dd_power_series_self_tol),
      dd_asymp_self_tol_(dd_asymp_self_tol),
      asymp_series_(false),
      asymp_series_dd_(true),
      power_series_(false),
      power_series_dd_(true, 1e-8, 4, 500)
{}

namespace {

template <ComptonKernelSolver::KernelOp Op, typename Solver>
ComptonResult eval_kernel(
    Solver const& s,
    double E,
    double Ep,
    double xi,
    double T,
    double Ne)
{
    if constexpr (Op == ComptonKernelSolver::KernelOp::sigma)
        return s.sigma_E(E, Ep, xi, T, Ne);
    else
        return s.dsigma_E_dT(E, Ep, xi, T, Ne);
}

} // namespace

template <ComptonKernelSolver::KernelOp Op>
ComptonResult ComptonKernelSolver::dispatch(
    double const E,
    double const E_prime,
    double const xi,
    double const T,
    double const Ne) const
{
    double const tau = T * units::k_boltz / units::me_c2;
    auto const p = compute_params<double>(
        E / units::me_c2,
        E_prime / units::me_c2,
        xi,
        tau);
    double const tau_alpha_max =
        std::max(tau * p.alpha_plus, tau * p.alpha_minus);

    if (tau_alpha_max < asymp_tau_alpha_threshold_) {
        // --- Asymptotic regime ---

        // A1: double asymptotic (roundoff estimator flags cancellation)
        try {
            auto const r =
                eval_kernel<Op>(asymp_series_, E, E_prime, xi, T, Ne);
            if (r.estimated_rel_error < asymp_self_tol_)
                return r;
        } catch (...) {
        }

        // A2: DD asymptotic
        try {
            auto const r =
                eval_kernel<Op>(asymp_series_dd_, E, E_prime, xi, T, Ne);
            if (r.estimated_rel_error < dd_asymp_self_tol_)
                return r;
        } catch (...) {
        }
    } else {
        // --- Power series regime ---

        // P1: double power series
        try {
            auto const r =
                eval_kernel<Op>(power_series_, E, E_prime, xi, T, Ne);
            if (r.estimated_rel_error < power_series_self_tol_ &&
                (Op == KernelOp::dsigma_dT || r.value >= 0.0))
                return r;
        } catch (...) {
        }

        // P2: DD power series
        try {
            auto const r =
                eval_kernel<Op>(power_series_dd_, E, E_prime, xi, T, Ne);
            if (r.estimated_rel_error < dd_power_series_self_tol_ &&
                (Op == KernelOp::dsigma_dT || r.value >= 0.0))
                return r;
        } catch (...) {
        }
    }

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
        double const gamma = E / units::me_c2;
        double const gamma_p = E_prime / units::me_c2;
        double const T_kev = T / units::kev_kelvin;
        double const tau = T * units::k_boltz / units::me_c2;
        std::fprintf(
            stderr,
            "WARNING sigma_E: all backends failed, returning 0"
            "  (gamma=%.6e, gamma'=%.6e, T=%.6e keV, tau=%.6e, xi=%.6e)\n",
            gamma,
            gamma_p,
            T_kev,
            tau,
            xi);
        return ComptonResult{0.0, 1.0};
    }
    if (result.value < 0.0) {
        double const gamma = E / units::me_c2;
        double const gamma_p = E_prime / units::me_c2;
        double const T_kev = T / units::kev_kelvin;
        double const tau = T * units::k_boltz / units::me_c2;
        std::fprintf(
            stderr,
            "WARNING sigma_E: %.6e clamped to 0"
            "  (gamma=%.6e, gamma'=%.6e, T=%.6e keV, tau=%.6e, xi=%.6e, "
            "err=%.3e)\n",
            result.value,
            gamma,
            gamma_p,
            T_kev,
            tau,
            xi,
            result.estimated_rel_error);
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
        double const gamma = E / units::me_c2;
        double const gamma_p = E_prime / units::me_c2;
        double const T_kev = T / units::kev_kelvin;
        double const tau = T * units::k_boltz / units::me_c2;
        std::fprintf(
            stderr,
            "WARNING dsigma_E_dT: all backends failed, returning 0"
            "  (gamma=%.6e, gamma'=%.6e, T=%.6e keV, tau=%.6e, xi=%.6e)\n",
            gamma,
            gamma_p,
            T_kev,
            tau,
            xi);
        return ComptonResult{0.0, 1.0};
    }
}

} // namespace compton
