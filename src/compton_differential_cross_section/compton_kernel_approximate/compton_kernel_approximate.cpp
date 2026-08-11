#include "compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.hpp"

#include "utilities/units.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace compton {
namespace {

constexpr std::size_t APPROXIMATION_ORDER = 5;

inline double multiply_add(double const a, double const b, double const c)
{
#if defined(__FP_FAST_FMA)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
}

bool solve_2x2(
    double a00,
    double a01,
    double a10,
    double a11,
    double b0,
    double b1,
    double& x0,
    double& x1)
{
    double const scale =
        std::max({std::abs(a00), std::abs(a01), std::abs(a10), std::abs(a11)});
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return false;
    }

    if (std::abs(a00) < std::abs(a10)) {
        std::swap(a00, a10);
        std::swap(a01, a11);
        std::swap(b0, b1);
    }

    constexpr double RELATIVE_PIVOT =
        64.0 * std::numeric_limits<double>::epsilon();
    if (std::abs(a00) <= RELATIVE_PIVOT * scale) {
        return false;
    }

    double const factor = a10 / a00;
    a11 = multiply_add(-factor, a01, a11);
    b1 = multiply_add(-factor, b0, b1);
    if (std::abs(a11) <= RELATIVE_PIVOT * scale) {
        return false;
    }

    x1 = b1 / a11;
    x0 = (b0 - a01 * x1) / a00;
    return std::isfinite(x0) && std::isfinite(x1);
}

bool solve_3x3(
    std::array<std::array<double, 3>, 3> matrix,
    std::array<double, 3> rhs,
    std::array<double, 3>& solution)
{
    double scale = 0.0;
    for (auto const& row : matrix) {
        for (double const value : row) {
            scale = std::max(scale, std::abs(value));
        }
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return false;
    }

    constexpr double RELATIVE_PIVOT =
        64.0 * std::numeric_limits<double>::epsilon();
    for (std::size_t column = 0; column < 3; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 3; ++row) {
            if (std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) <= RELATIVE_PIVOT * scale) {
            return false;
        }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(rhs[pivot], rhs[column]);
        }
        for (std::size_t row = column + 1; row < 3; ++row) {
            double const factor =
                matrix[row][column] / matrix[column][column];
            for (std::size_t k = column + 1; k < 3; ++k) {
                matrix[row][k] = multiply_add(
                    -factor,
                    matrix[column][k],
                    matrix[row][k]);
            }
            rhs[row] = multiply_add(-factor, rhs[column], rhs[row]);
        }
    }

    for (int row = 2; row >= 0; --row) {
        auto const row_index = static_cast<std::size_t>(row);
        double value = rhs[row_index];
        for (std::size_t column = row_index + 1; column < 3; ++column) {
            value = multiply_add(
                -matrix[row_index][column],
                solution[column],
                value);
        }
        solution[row_index] = value / matrix[row_index][row_index];
    }

    return std::ranges::all_of(solution, [](double const value) {
        return std::isfinite(value);
    });
}

double stable_lambda_plus(double const delta, double const A)
{
    double const root =
        std::sqrt((1.0 + 0.5 * A) * (1.0 + delta * delta / (2.0 * A)));
    if (delta >= 0.0) {
        return 0.5 * delta + root;
    }

    double const numerator = 1.0 + 0.5 * A + delta * delta / (2.0 * A);
    return numerator / (root - 0.5 * delta);
}

/** Explicit derivatives F(n,p,rho) from the finite Section 9 formula. */
template <int P>
std::array<double, APPROXIMATION_ORDER + 1> inverse_power_derivatives(
    double const rho,
    double const omega2)
{
    static_assert(P == 1 || P == 3);
    constexpr double p = static_cast<double>(P);
    double const R = rho * rho + omega2;
    double const inverse_R = 1.0 / R;
    double const inverse_s = std::sqrt(inverse_R);
    double const base = P == 1 ? inverse_s : inverse_s * inverse_R;
    double const g1 = base * inverse_R;
    double const g2 = g1 * inverse_R;
    double const g3 = g2 * inverse_R;
    double const g4 = g3 * inverse_R;
    double const g5 = g4 * inverse_R;
    double const rho2 = rho * rho;
    double const rho3 = rho2 * rho;
    double const rho4 = rho2 * rho2;
    double const rho5 = rho4 * rho;
    double const p2 = p * (p + 2.0);
    double const p4 = p2 * (p + 4.0);
    double const p6 = p4 * (p + 6.0);
    double const p8 = p6 * (p + 8.0);

    return {
        base,
        -p * rho * g1,
        -p * g1 + p2 * rho2 * g2,
        3.0 * p2 * rho * g2 - p4 * rho3 * g3,
        3.0 * p2 * g2 - 6.0 * p4 * rho2 * g3 + p6 * rho4 * g4,
        -15.0 * p4 * rho * g3 + 10.0 * p6 * rho3 * g4 -
            p8 * rho5 * g5};
}

struct CoefficientEvaluation {
    std::array<double, APPROXIMATION_ORDER + 1> value;
    double estimated_rel_roundoff;
};

CoefficientEvaluation explicit_coefficients(
    double const gamma,
    double const gamma_prime,
    double const xi,
    double const A,
    double const q,
    double const lambda_plus)
{
    double const omega2 = (1.0 + xi) / (1.0 - xi);
    double const rho_minus = lambda_plus - gamma_prime;
    double const rho_plus = lambda_plus + gamma;
    double const rho_square_difference =
        rho_plus * rho_plus - rho_minus * rho_minus;
    double const q2 = q * q;
    double const delta_minus = 0.5 * (rho_square_difference - q2);
    double const delta_plus = 0.5 * (rho_square_difference + q2);
    double const A2 = A * A;
    double const B = (A2 - 2.0 * A - 2.0) / A2;
    auto const F1_minus = inverse_power_derivatives<1>(rho_minus, omega2);
    auto const F1_plus = inverse_power_derivatives<1>(rho_plus, omega2);
    auto const F3_minus = inverse_power_derivatives<3>(rho_minus, omega2);
    auto const F3_plus = inverse_power_derivatives<3>(rho_plus, omega2);

    CoefficientEvaluation result{};
    double const epsilon = std::numeric_limits<double>::epsilon();
    for (std::size_t n = 0; n <= APPROXIMATION_ORDER; ++n) {
        double const leading = n == 0 ? 2.0 / q : 0.0;
        double const difference1 = F1_minus[n] - F1_plus[n];
        double const boundary1 = B * difference1;
        double const product_minus = delta_minus * F3_minus[n];
        double const product_plus = delta_plus * F3_plus[n];
        double const boundary3 = (product_minus + product_plus) / A2;
        double derivative_boundary = 0.0;
        double derivative_operand_sum = 0.0;
        if (n > 0) {
            double const derivative_sum =
                F3_minus[n - 1] + F3_plus[n - 1];
            derivative_boundary =
                static_cast<double>(n) * (gamma + gamma_prime) / A2 *
                derivative_sum;
            derivative_operand_sum =
                static_cast<double>(n) * (gamma + gamma_prime) / A2 *
                (std::abs(F3_minus[n - 1]) +
                 std::abs(F3_plus[n - 1]));
        }

        double const coefficient =
            ((leading + boundary1) + boundary3) + derivative_boundary;
        result.value[n] = coefficient;

        double const absolute_roundoff_scale =
            std::abs(leading) +
            std::abs(B) * (std::abs(F1_minus[n]) + std::abs(F1_plus[n])) +
            (std::abs(product_minus) + std::abs(product_plus)) / A2 +
            derivative_operand_sum;
        double const relative_roundoff =
            epsilon * absolute_roundoff_scale /
            (std::abs(coefficient) + constants::REL_ERROR_TINY_SCALE);
        result.estimated_rel_roundoff =
            std::max(result.estimated_rel_roundoff, relative_roundoff);
    }

    if (!std::ranges::all_of(result.value, [](double const value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error(
            "approximate kernel explicit coefficients are nonfinite");
    }
    return result;
}

struct PadeEvaluation {
    double amplitude;
    double derivative;
    double relative_disagreement;
};

PadeEvaluation evaluate_pade(
    std::array<double, APPROXIMATION_ORDER + 1> const& C,
    double const tau)
{
    double b1 = 0.0;
    double b2 = 0.0;
    if (!solve_2x2(C[3], C[2], C[4], C[3], -C[4], -C[5], b1, b2)) {
        throw std::runtime_error("approximate kernel [3/2] Pade solve failed");
    }

    std::array<double, 3> e{};
    if (!solve_3x3(
            {{{C[2], C[1], C[0]},
              {C[3], C[2], C[1]},
              {C[4], C[3], C[2]}}},
            {{-C[3], -C[4], -C[5]}},
            e)) {
        throw std::runtime_error("approximate kernel [2/3] Pade solve failed");
    }

    double const tau2 = tau * tau;
    double const tau3 = tau2 * tau;
    double const p32_1 = C[1] + b1 * C[0];
    double const p32_2 = C[2] + b1 * C[1] + b2 * C[0];
    double const p32_3 = C[3] + b1 * C[2] + b2 * C[1];
    double const P32 = C[0] + p32_1 * tau + p32_2 * tau2 + p32_3 * tau3;
    double const dP32 = p32_1 + 2.0 * p32_2 * tau + 3.0 * p32_3 * tau2;
    double const D32 = 1.0 + b1 * tau + b2 * tau2;
    double const dD32 = b1 + 2.0 * b2 * tau;

    double const p23_1 = C[1] + e[0] * C[0];
    double const p23_2 = C[2] + e[0] * C[1] + e[1] * C[0];
    double const P23 = C[0] + p23_1 * tau + p23_2 * tau2;
    double const dP23 = p23_1 + 2.0 * p23_2 * tau;
    double const D23 = 1.0 + e[0] * tau + e[1] * tau2 + e[2] * tau3;
    double const dD23 = e[0] + 2.0 * e[1] * tau + 3.0 * e[2] * tau2;

    double const scale = std::max(std::abs(D32), std::abs(D23));
    if (!(scale > 0.0) || !std::isfinite(scale) || !std::isfinite(P32) ||
        !std::isfinite(P23)) {
        throw std::runtime_error("approximate kernel Pade evaluation failed");
    }

    double const d32 = D32 / scale;
    double const d23 = D23 / scale;
    double const dd32 = dD32 / scale;
    double const dd23 = dD23 / scale;
    double const d32_2 = d32 * d32;
    double const d23_2 = d23 * d23;
    double const blend_numerator =
        d32_2 * d32 * P32 + d23_2 * d23 * P23;
    double const blend_denominator =
        scale * (d32_2 * d32_2 + d23_2 * d23_2);
    double const derivative_numerator =
        3.0 * d32_2 * dd32 * P32 + d32_2 * d32 * dP32 +
        3.0 * d23_2 * dd23 * P23 + d23_2 * d23 * dP23;
    double const derivative_denominator =
        scale *
        (4.0 * d32_2 * d32 * dd32 + 4.0 * d23_2 * d23 * dd23);

    double const amplitude = blend_numerator / blend_denominator;
    double const derivative =
        (derivative_numerator * blend_denominator -
         blend_numerator * derivative_denominator) /
        (blend_denominator * blend_denominator);
    if (!(amplitude > 0.0) || !std::isfinite(amplitude) ||
        !std::isfinite(derivative)) {
        throw std::runtime_error("approximate kernel produced invalid amplitude");
    }

    double disagreement = std::numeric_limits<double>::infinity();
    if (D32 != 0.0 && D23 != 0.0) {
        double const approximation32 = P32 / D32;
        double const approximation23 = P23 / D23;
        disagreement =
            std::abs(approximation32 - approximation23) / std::abs(amplitude);
    }
    return {amplitude, derivative, disagreement};
}

} // namespace

ComptonKernelApproximate::Evaluation ComptonKernelApproximate::evaluate(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    assert_parameters(E, E_prime, xi, T);

    double const gamma = E / units::me_c2;
    double const gamma_prime = E_prime / units::me_c2;
    double const tau = T * units::k_boltz / units::me_c2;
    double const one_minus_xi = 1.0 - xi;
    double const A = gamma * gamma_prime * one_minus_xi;
    double const delta = gamma_prime - gamma;
    double const q = std::hypot(delta, std::sqrt(2.0 * A));
    if (!(A > 0.0) || !(q > 0.0) || !std::isfinite(A) ||
        !std::isfinite(q)) {
        throw std::runtime_error("approximate kernel kinematics failed");
    }

    double lambda_plus = stable_lambda_plus(delta, A);
    constexpr double LAMBDA_TOLERANCE =
        256.0 * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(lambda_plus) ||
        lambda_plus < 1.0 - LAMBDA_TOLERANCE) {
        throw std::runtime_error("approximate kernel lambda_plus is invalid");
    }
    lambda_plus = std::max(lambda_plus, 1.0);

    CoefficientEvaluation const coefficients = explicit_coefficients(
        gamma,
        gamma_prime,
        xi,
        A,
        q,
        lambda_plus);
    PadeEvaluation const pade = evaluate_pade(coefficients.value, tau);

    double const tau2 = tau * tau;
    double const normalization_numerator =
        1.0 + (141.0 / 208.0) * tau - (441.0 / 3328.0) * tau2;
    double const normalization_denominator =
        1.0 + (531.0 / 208.0) * tau + (6519.0 / 3328.0) * tau2;
    double const normalization =
        normalization_numerator / normalization_denominator;
    if (!(normalization > 0.0) || !std::isfinite(normalization)) {
        throw std::runtime_error(
            "approximate kernel normalization is nonpositive");
    }
    double const dnormalization_numerator =
        (141.0 / 208.0) - 2.0 * (441.0 / 3328.0) * tau;
    double const dnormalization_denominator =
        (531.0 / 208.0) + 2.0 * (6519.0 / 3328.0) * tau;
    double const dnormalization =
        (dnormalization_numerator * normalization_denominator -
         normalization_numerator * dnormalization_denominator) /
        (normalization_denominator * normalization_denominator);

    double const estimated_rel_error = std::max(
        pade.relative_disagreement,
        coefficients.estimated_rel_roundoff);
    double const exponential = std::exp(-(lambda_plus - 1.0) / tau);
    if (exponential == 0.0) {
        return {0.0, 0.0, estimated_rel_error};
    }

    double const value =
        units::r_e2 * gamma_prime / (4.0 * E) *
        std::sqrt(2.0 / std::numbers::pi) / std::sqrt(tau) * normalization *
        pade.amplitude * exponential;
    if (!std::isfinite(value)) {
        throw std::runtime_error("approximate kernel result is nonfinite");
    }

    double const logarithmic_tau_derivative =
        pade.derivative / pade.amplitude +
        dnormalization / normalization - 0.5 / tau +
        (lambda_plus - 1.0) / tau2;
    double const dvalue_dT =
        value * logarithmic_tau_derivative * units::k_boltz / units::me_c2;
    if (!std::isfinite(dvalue_dT)) {
        throw std::runtime_error(
            "approximate kernel temperature derivative is nonfinite");
    }

    return {value, dvalue_dT, estimated_rel_error};
}

ComptonResult ComptonKernelApproximate::sigma_E(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    try {
        Evaluation const result = evaluate(E, E_prime, xi, T);
        double const abs_error =
            std::abs(result.value) * result.estimated_rel_error;
        return {result.value, abs_error, result.estimated_rel_error, 6};
    } catch (...) {
        return {0.0, 1.0, 1.0, 0};
    }
}

ComptonResult ComptonKernelApproximate::dsigma_E_dT(
    double const E,
    double const E_prime,
    double const xi,
    double const T) const
{
    try {
        Evaluation const result = evaluate(E, E_prime, xi, T);
        double const abs_error =
            std::abs(result.dvalue_dT) * result.estimated_rel_error;
        return {result.dvalue_dT, abs_error, result.estimated_rel_error, 6};
    } catch (...) {
        return {0.0, 1.0, 1.0, 0};
    }
}

} // namespace compton
