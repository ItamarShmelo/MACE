/**
 * @file compton_kernel_solver.cpp
 * @brief Robust adaptive solver for the Compton scattering kernel.
 *
 * Cascade: asymptotic -> power -> quadrature-256
 */

#include "compton_kernel_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace compton {

ComptonKernelSolver::ComptonKernelSolver(double target_rel_tol, double target_abs_tol)
    : target_rel_tol_(target_rel_tol),
      target_abs_tol_(target_abs_tol),
      series_(SeriesMethod::Auto, 1e-12, 4, 200),
      quad256_(256, QuadratureForm::PostIntegrationByParts)
{}

SolverResult ComptonKernelSolver::sigma_E(
    double E, double E_prime, double xi, double tau, double Ne) const
{
    if (!(E > 0.0) || !std::isfinite(E))
        throw std::invalid_argument("E must be finite and > 0");
    if (!(E_prime > 0.0) || !std::isfinite(E_prime))
        throw std::invalid_argument("E_prime must be finite and > 0");
    if (!(tau > 0.0) || !std::isfinite(tau))
        throw std::invalid_argument("tau must be finite and > 0");
    if (!(xi > -1.0 && xi < 1.0) || !std::isfinite(xi))
        throw std::invalid_argument("xi must be finite and strictly inside (-1, 1)");
    if (!std::isfinite(Ne))
        throw std::invalid_argument("Ne must be finite");

    const double gamma = E / units::me_c2;
    const double gamma_p = E_prime / units::me_c2;

    KershawParams p = compute_params(gamma, gamma_p, xi, tau);

    const double tau_alpha_max = std::max(tau * p.alpha_plus, tau * p.alpha_minus);

    auto make_result = [&](double value, double abs_err, double rel_err,
                           int terms, SolverMethod method, bool fallback,
                           bool target_met_val) -> SolverResult {
        bool clamped = false;
        if (value < 0.0 && std::abs(value) < abs_err) {
            clamped = true;
            value = 0.0;
        }
        return SolverResult{value, abs_err, rel_err, terms, method,
                            fallback, target_met_val, clamped,
                            tau_alpha_max};
    };

    // --- Phase 1: Try asymptotic if in its natural regime ---
    if (tau_alpha_max < ASYMP_TAU_ALPHA_THRESHOLD) {
        ComptonKernelSeries asym_series(SeriesMethod::Asymptotic, 1e-12, 4, 200);
        SeriesResult ar = asym_series.sigma_E(E, E_prime, xi, tau, Ne);

        if (std::abs(ar.value) < target_abs_tol_) {
            return make_result(0.0, 0.0, 0.0, ar.terms_used,
                               SolverMethod::Asymptotic, false, true);
        }
        if (ar.estimated_rel_error < target_rel_tol_) {
            return make_result(ar.value, ar.estimated_abs_error,
                               ar.estimated_rel_error, ar.terms_used,
                               SolverMethod::Asymptotic, false, true);
        }
    }

    // --- Phase 2: Try power series ---
    {
        ComptonKernelSeries pow_series(SeriesMethod::PowerSeries, 1e-12, 4, 1000);
        SeriesResult pr = pow_series.sigma_E(E, E_prime, xi, tau, Ne);

        if (std::abs(pr.value) < target_abs_tol_) {
            return make_result(0.0, 0.0, 0.0, pr.terms_used,
                               SolverMethod::PowerSeries, false, true);
        }

        if (pr.estimated_rel_error < target_rel_tol_) {
            return make_result(pr.value, pr.estimated_abs_error,
                               pr.estimated_rel_error, pr.terms_used,
                               SolverMethod::PowerSeries, false, true);
        }

        // --- Phase 3: Quadrature safety net ---
        SigmaResult qr = quad256_.sigma_E(E, E_prime, xi, tau, Ne);
        double q_rel_err = qr.estimated_rel_error;
        bool q_target_met = (q_rel_err < target_rel_tol_);

        return make_result(qr.value, qr.estimated_abs_error, q_rel_err,
                           256, SolverMethod::Quadrature, false,
                           q_target_met);
    }
}

} // namespace compton
