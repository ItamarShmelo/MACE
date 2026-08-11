#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"

#include "compton_common/compton_common.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace compton {

ComptonKernelApproximateSolver::ComptonKernelApproximateSolver(
    bool const verbose)
    : verbose_(verbose),
      asymptotic_(false),
      asymptotic_dd_(true),
      power_(false),
      power_dd_(true, 1e-8, 4, 500)
{}

namespace {

template <ComptonKernelApproximateSolver::KernelOp Op, typename Kernel>
ComptonResult evaluate_kernel(
    Kernel const& kernel,
    double const E,
    double const E_prime,
    double const xi,
    double const T)
{
    if constexpr (Op == ComptonKernelApproximateSolver::KernelOp::sigma) {
        return kernel.sigma_E(E, E_prime, xi, T);
    } else {
        return kernel.dsigma_E_dT(E, E_prime, xi, T);
    }
}

template <ComptonKernelApproximateSolver::KernelOp Op, typename Kernel>
bool acceptable(ComptonResult const& result, double const tolerance)
{
    return result.estimated_rel_error < tolerance &&
           (Op == ComptonKernelApproximateSolver::KernelOp::dsigma_dT ||
            result.value >= 0.0);
}

struct DispatchParameters {
    double gamma;
    double gamma_prime;
    double tau;
    double tau_alpha_max;
};

DispatchParameters dispatch_parameters(
    double const E,
    double const E_prime,
    double const xi,
    double const T)
{
    double const gamma = E / units::me_c2;
    double const gamma_prime = E_prime / units::me_c2;
    double const tau = T * units::k_boltz / units::me_c2;
    double const a = 1.0 - xi;
    double const A = gamma * gamma_prime * a;
    double const delta = gamma_prime - gamma;
    double const root =
        std::sqrt((1.0 + 0.5 * A) * (1.0 + delta * delta / (2.0 * A)));
    double lambda_plus = 0.0;
    if (delta >= 0.0) {
        lambda_plus = 0.5 * delta + root;
    } else {
        double const numerator =
            1.0 + 0.5 * A + delta * delta / (2.0 * A);
        lambda_plus = numerator / (root - 0.5 * delta);
    }
    constexpr double LAMBDA_TOLERANCE =
        256.0 * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(lambda_plus) ||
        lambda_plus < 1.0 - LAMBDA_TOLERANCE) {
        throw std::runtime_error("approximate solver kinematics failed");
    }
    lambda_plus = std::max(lambda_plus, 1.0);

    double const omega2 = (1.0 + xi) / a;
    double const rho_plus = lambda_plus + gamma;
    double const rho_minus = lambda_plus - gamma_prime;
    double const alpha_plus =
        1.0 / std::sqrt(rho_plus * rho_plus + omega2);
    double const alpha_minus =
        1.0 / std::sqrt(rho_minus * rho_minus + omega2);
    return {
        gamma,
        gamma_prime,
        tau,
        tau * std::max(alpha_plus, alpha_minus)};
}

} // namespace

template <ComptonKernelApproximateSolver::KernelOp Op>
ComptonResult ComptonKernelApproximateSolver::dispatch(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    assert_parameters(E, E_prime, xi, T);

    DispatchParameters const parameters =
        dispatch_parameters(E, E_prime, xi, T);
    constexpr double double_series_tolerance =
        Op == KernelOp::sigma
            ? approximate_solver_constants::DOUBLE_SERIES_SELF_TOL
            : approximate_solver_constants::DOUBLE_DERIVATIVE_SERIES_SELF_TOL;
    bool const in_asymptotic_domain =
        parameters.tau_alpha_max <
        approximate_solver_constants::ASYMP_TAU_ALPHA_THRESHOLD;
    bool const asymptotic_is_fastest =
        in_asymptotic_domain &&
        parameters.tau <
            approximate_solver_constants::FAST_ASYMP_TAU_THRESHOLD &&
        std::min(parameters.gamma, parameters.gamma_prime) >=
            approximate_solver_constants::FAST_ASYMP_MIN_GAMMA;

    ComptonResult approximate_result{};
    bool approximate_is_accurate = false;
    if (!asymptotic_is_fastest) {
        approximate_result =
            evaluate_kernel<Op>(approximate_, E, E_prime, xi, T);
        constexpr double approximate_tolerance =
            Op == KernelOp::sigma
                ? approximate_solver_constants::
                      APPROXIMATE_PADE_DISAGREEMENT_THRESHOLD
                : approximate_solver_constants::
                      APPROXIMATE_DERIVATIVE_PADE_DISAGREEMENT_THRESHOLD;
        approximate_is_accurate =
            approximate_result.estimated_abs_error != 1.0 &&
            approximate_result.estimated_rel_error < approximate_tolerance;
    }

    enum class Backend { asymptotic, approximate, power };
    Backend const backend =
        asymptotic_is_fastest ||
                (in_asymptotic_domain && !approximate_is_accurate)
            ? Backend::asymptotic
            : approximate_is_accurate ? Backend::approximate : Backend::power;

    // Case 1: asymptotic series, with DD fallback in the cancellation region.
    if (backend == Backend::asymptotic) {
        try {
            auto const result =
                evaluate_kernel<Op>(asymptotic_, E, E_prime, xi, T);
            if (acceptable<Op, ComptonKernelAsymptoticSeries>(
                    result,
                    double_series_tolerance)) {
                return result;
            }
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }

        auto const result =
            evaluate_kernel<Op>(asymptotic_dd_, E, E_prime, xi, T);
        if (acceptable<Op, ComptonKernelAsymptoticSeries>(
                result,
                approximate_solver_constants::DD_SERIES_SELF_TOL)) {
            return result;
        }
        throw std::runtime_error("asymptotic kernel backends failed");
    } else if (backend == Backend::approximate) {
        // Case 2: explicit coefficients and the Padé pair pass their gates.
        return approximate_result;
    }

    // Case 3: convergent power series, with DD fallback.
    try {
        auto const result = evaluate_kernel<Op>(power_, E, E_prime, xi, T);
        if (acceptable<Op, ComptonPowerSeries>(
                result,
                double_series_tolerance)) {
            return result;
        }
    } catch (...) { // NOLINT(bugprone-empty-catch)
    }

    auto const result = evaluate_kernel<Op>(power_dd_, E, E_prime, xi, T);
    if (acceptable<Op, ComptonPowerSeries>(
            result,
            approximate_solver_constants::DD_SERIES_SELF_TOL)) {
        return result;
    }
    throw std::runtime_error("power-series kernel backends failed");
}

ComptonResult ComptonKernelApproximateSolver::sigma_E(
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
            std::fprintf(
                stderr,
                "WARNING approximate solver sigma_E failed "
                "(gamma=%.6e, gamma'=%.6e, T=%.6e keV, xi=%.6e)\n",
                E / units::me_c2,
                E_prime / units::me_c2,
                T / units::kev_kelvin,
                xi);
        }
        return {0.0, 1.0, 1.0, 0};
    }

    if (result.value < 0.0) {
        result.value = 0.0;
    }
    return result;
}

ComptonResult ComptonKernelApproximateSolver::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    try {
        return dispatch<KernelOp::dsigma_dT>(E, E_prime, xi, T);
    } catch (...) {
        if (verbose_) {
            std::fprintf(
                stderr,
                "WARNING approximate solver dsigma_E_dT failed "
                "(gamma=%.6e, gamma'=%.6e, T=%.6e keV, xi=%.6e)\n",
                E / units::me_c2,
                E_prime / units::me_c2,
                T / units::kev_kelvin,
                xi);
        }
        return {0.0, 1.0, 1.0, 0};
    }
}

} // namespace compton
