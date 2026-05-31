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

SigmaResult ComptonKernelSolver::sigma_E(
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

    double const gamma = E / units::me_c2;
    double const gamma_p = E_prime / units::me_c2;

    KershawParams<double> p = compute_params<double>(gamma, gamma_p, xi, tau);

    double const tau_alpha_max = std::max(tau * p.alpha_plus, tau * p.alpha_minus);

    auto clamp_negative = [](SigmaResult r) -> SigmaResult {
        if (r.value < 0.0 && std::abs(r.value) < r.estimated_abs_error) {
            r.value = 0.0;
        }
        return r;
    };

    // --- Phase 1: Try asymptotic if in its natural regime ---
    if (tau_alpha_max < ASYMP_TAU_ALPHA_THRESHOLD) {
        try {
            ComptonKernelSeries asym_series(SeriesMethod::Asymptotic, 1e-12, 4, 200);
            SigmaResult ar = asym_series.sigma_E(E, E_prime, xi, tau, Ne);

            if (std::abs(ar.value) < target_abs_tol_) {
                return SigmaResult{0.0, 0.0, 0.0};
            }
            if (ar.estimated_rel_error < target_rel_tol_) {
                return clamp_negative(ar);
            }
        } catch (std::runtime_error const&) {
        }
    }

    // --- Phase 2: Try power series ---
    try {
        ComptonKernelSeries pow_series(SeriesMethod::PowerSeriesHighPrecision, 1e-12, 4, 1000);
        SigmaResult pr = pow_series.sigma_E(E, E_prime, xi, tau, Ne);

        if (std::abs(pr.value) < target_abs_tol_) {
            return SigmaResult{0.0, 0.0, 0.0};
        }

        if (pr.estimated_rel_error < target_rel_tol_) {
            return clamp_negative(pr);
        }
    } catch (std::runtime_error const&) {
    }

    // --- Phase 3: Quadrature safety net ---
    SigmaResult qr = quad256_.sigma_E(E, E_prime, xi, tau, Ne);
    return clamp_negative(qr);
}

} // namespace compton
