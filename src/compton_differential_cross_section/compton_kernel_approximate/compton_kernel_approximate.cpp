#include "compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.hpp"

#include "utilities/units.hpp"

#include <array>
#include <cmath>
#include <cfloat>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace compton {

namespace {

// Precomputed generalized binomial coefficients binom(-1/2, j) for j=0..5.
// binom(z, j) = z*(z-1)*...*(z-j+1) / j!
constexpr std::array<double, 6> BINOM_NEG_HALF = {
    1.0,         // binom(-1/2, 0)
    -0.5,        // binom(-1/2, 1)
    3.0 / 8.0,   // binom(-1/2, 2)
    -5.0 / 16.0, // binom(-1/2, 3)
    35.0 / 128.0,  // binom(-1/2, 4)
    -63.0 / 256.0  // binom(-1/2, 5)
};

// Precomputed generalized binomial coefficients binom(-3/2, j) for j=0..5.
constexpr std::array<double, 6> BINOM_NEG_THREE_HALF = {
    1.0,            // binom(-3/2, 0)
    -3.0 / 2.0,    // binom(-3/2, 1)
    15.0 / 8.0,    // binom(-3/2, 2)
    -35.0 / 16.0,  // binom(-3/2, 3)
    315.0 / 128.0,  // binom(-3/2, 4)
    -693.0 / 256.0  // binom(-3/2, 5)
};

// Standard binomial coefficients binom(j, k) for j=0..5, k=0..5.
// Only j >= k entries are meaningful.
constexpr std::array<std::array<int, 6>, 6> BINOM_STANDARD = {{
    {1, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0},
    {1, 2, 1, 0, 0, 0},
    {1, 3, 3, 1, 0, 0},
    {1, 4, 6, 4, 1, 0},
    {1, 5, 10, 10, 5, 1}
}};

constexpr std::array<int, 6> FACTORIAL = {1, 1, 2, 6, 24, 120};

/**
 * Evaluate the finite derivative polynomial F_n^(p)(rho).
 * p must be 1 or 3.
 * omega2 = (1+xi)/(1-xi).
 */
double
F_np(int n, int p, double rho, double omega2)
{
    double const s2 = rho * rho + omega2;
    double const half_p = static_cast<double>(p) / 2.0;

    std::array<double, 6> const& gen_binom =
        (p == 1) ? BINOM_NEG_HALF : BINOM_NEG_THREE_HALF;

    int const j_min = (n + 1) / 2; // ceil(n/2) for non-negative n
    double sum = 0.0;

    for (int j = j_min; j <= n; ++j) {
        double const binom_gen = gen_binom[j];
        double const binom_std =
            static_cast<double>(BINOM_STANDARD[j][n - j]);
        double const two_rho_power = std::pow(2.0 * rho, (2 * j) - n);
        double const s2_power = std::pow(s2, -half_p - static_cast<double>(j));

        sum += binom_gen * binom_std * two_rho_power * s2_power;
    }

    return static_cast<double>(FACTORIAL[n]) * sum;
}

/**
 * Solve a 2x2 linear system with partial pivoting.
 * | a00  a01 | | x0 |   | b0 |
 * | a10  a11 | | x1 | = | b1 |
 *
 * Throws if the system is singular (pivot below threshold).
 */
void
solve_2x2(
    double a00,
    double a01,
    double a10,
    double a11,
    double b0,
    double b1,
    double& x0,
    double& x1,
    double scale)
{
    // Partial pivoting
    if (std::abs(a00) < std::abs(a10)) {
        std::swap(a00, a10);
        std::swap(a01, a11);
        std::swap(b0, b1);
    }

    if (std::abs(a00) < 1e-15 * scale) {
        throw std::runtime_error("Pade [3/2] system singular");
    }

    double const factor = a10 / a00;
    a11 -= factor * a01;
    b1 -= factor * b0;

    if (std::abs(a11) < 1e-15 * scale) {
        throw std::runtime_error("Pade [3/2] system singular");
    }

    x1 = b1 / a11;
    x0 = (b0 - a01 * x1) / a00;
}

/**
 * Solve a 3x3 linear system with partial pivoting.
 * Throws if singular.
 */
void
solve_3x3(
    std::array<std::array<double, 3>, 3>& A,
    std::array<double, 3>& b,
    double scale)
{
    for (int col = 0; col < 3; ++col) {
        // Find pivot
        int pivot_row = col;
        double max_val = std::abs(A[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(A[row][col]) > max_val) {
                max_val = std::abs(A[row][col]);
                pivot_row = row;
            }
        }

        if (max_val < 1e-15 * scale) {
            throw std::runtime_error("Pade [2/3] system singular");
        }

        if (pivot_row != col) {
            std::swap(A[col], A[pivot_row]);
            std::swap(b[col], b[pivot_row]);
        }

        // Eliminate below
        for (int row = col + 1; row < 3; ++row) {
            double const factor = A[row][col] / A[col][col];
            for (int k = col + 1; k < 3; ++k) {
                A[row][k] -= factor * A[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    for (int row = 2; row >= 0; --row) {
        for (int col = row + 1; col < 3; ++col) {
            b[row] -= A[row][col] * b[col];
        }
        b[row] /= A[row][row];
    }
}

} // anonymous namespace

ComptonResult
ComptonKernelApproximate::sigma_E(double E, double E_prime, double xi, double T) const
{
    try {
        double const gamma = E / units::me_c2;
        double const gamma_prime = E_prime / units::me_c2;
        double const tau = units::k_boltz * T / units::me_c2;

        double const value = evaluate_sigma_E(gamma, gamma_prime, xi, tau);
        return ComptonResult{value, 0.0, 0.0, 0};
    } catch (...) {
        return ComptonResult{0.0, 1.0, 0.0, 0};
    }
}

double
ComptonKernelApproximate::evaluate_sigma_E(
    double gamma,
    double gamma_prime,
    double xi,
    double tau) const
{
    // Step 1: Input validation
    if (std::isnan(gamma) || std::isnan(gamma_prime) || std::isnan(xi) ||
        std::isnan(tau) || std::isinf(gamma) || std::isinf(gamma_prime) ||
        std::isinf(xi) || std::isinf(tau)) {
        throw std::invalid_argument("NaN or inf input");
    }
    if (gamma <= 0.0 || gamma_prime <= 0.0) {
        throw std::invalid_argument("non-positive photon energy");
    }
    if (tau < 0.0) {
        throw std::invalid_argument("negative temperature");
    }
    if (tau == 0.0) {
        throw std::runtime_error("zero-temperature distributional limit");
    }
    if (xi < -1.0 || xi > 1.0) {
        throw std::invalid_argument("xi out of range [-1, 1]");
    }
    if (xi == 1.0) {
        throw std::runtime_error("forward-scattering distributional limit");
    }

    // Step 2: Kinematic quantities
    double const a = 1.0 - xi;
    double const A = gamma * gamma_prime * a;
    double const Delta = gamma_prime - gamma;
    double const q = std::hypot(Delta, std::sqrt(2.0 * A));
    double const omega2 = (1.0 + xi) / a;

    // Step 3: Minimum electron Lorentz factor
    double lambda_min =
        0.5 * Delta +
        std::sqrt((1.0 + 0.5 * A) * (1.0 + Delta * Delta / (2.0 * A)));

    if (lambda_min >= 1.0 - 128.0 * DBL_EPSILON) {
        lambda_min = std::max(lambda_min, 1.0);
    } else {
        throw std::runtime_error("lambda_min below 1 by more than 128 eps");
    }

    // Step 4: Endpoint variables
    double const rho_minus = lambda_min - gamma_prime;
    double const rho_plus = lambda_min + gamma;
    double const rho_plus_sq = rho_plus * rho_plus;
    double const rho_minus_sq = rho_minus * rho_minus;
    double const q_sq = q * q;
    double const delta_minus = (rho_plus_sq - rho_minus_sq - q_sq) / 2.0;
    double const delta_plus = (rho_plus_sq - rho_minus_sq + q_sq) / 2.0;
    double const A_sq = A * A;
    double const B = (A_sq - 2.0 * A - 2.0) / A_sq;
    double const sum_gamma = gamma + gamma_prime;
    double const inv_A_sq = 1.0 / A_sq;

    // Step 5: Explicit coefficients C[0..5]
    std::array<double, 6> C{};
    for (int n = 0; n <= 5; ++n) {
        double const kronecker_n0 = (n == 0) ? 1.0 : 0.0;
        double const F_n_1_minus = F_np(n, 1, rho_minus, omega2);
        double const F_n_1_plus = F_np(n, 1, rho_plus, omega2);
        double const F_n_3_minus = F_np(n, 3, rho_minus, omega2);
        double const F_n_3_plus = F_np(n, 3, rho_plus, omega2);

        double F_nm1_3_minus = 0.0;
        double F_nm1_3_plus = 0.0;
        if (n > 0) {
            F_nm1_3_minus = F_np(n - 1, 3, rho_minus, omega2);
            F_nm1_3_plus = F_np(n - 1, 3, rho_plus, omega2);
        }

        C[n] = (2.0 * kronecker_n0) / q +
               B * (F_n_1_minus - F_n_1_plus) +
               (delta_minus * F_n_3_minus + delta_plus * F_n_3_plus) *
                   inv_A_sq +
               static_cast<double>(n) * sum_gamma * inv_A_sq *
                   (F_nm1_3_minus + F_nm1_3_plus);
    }

    // Step 6: [3/2] Padé continuation
    double const coeff_scale =
        std::max({std::abs(C[0]), std::abs(C[1]), std::abs(C[2]),
                  std::abs(C[3]), std::abs(C[4]), std::abs(C[5])});

    double b1 = 0.0;
    double b2 = 0.0;
    solve_2x2(C[3], C[2], C[4], C[3], -C[4], -C[5], b1, b2, coeff_scale);

    double const tau2 = tau * tau;
    double const tau3 = tau2 * tau;
    double const D32 = 1.0 + b1 * tau + b2 * tau2;
    double const P32 = C[0] + (C[1] + b1 * C[0]) * tau +
                       (C[2] + b1 * C[1] + b2 * C[0]) * tau2 +
                       (C[3] + b1 * C[2] + b2 * C[1]) * tau3;

    // Step 7: [2/3] Padé continuation
    std::array<std::array<double, 3>, 3> mat = {{
        {C[2], C[1], C[0]},
        {C[3], C[2], C[1]},
        {C[4], C[3], C[2]}
    }};
    std::array<double, 3> rhs = {-C[3], -C[4], -C[5]};
    solve_3x3(mat, rhs, coeff_scale);
    double const e1 = rhs[0];
    double const e2 = rhs[1];
    double const e3 = rhs[2];

    double const D23 = 1.0 + e1 * tau + e2 * tau2 + e3 * tau3;
    double const P23 =
        C[0] + (C[1] + e1 * C[0]) * tau + (C[2] + e1 * C[1] + e2 * C[0]) * tau2;

    // Step 8: Pole-suppressed amplitude
    double const D32_2 = D32 * D32;
    double const D32_3 = D32_2 * D32;
    double const D32_4 = D32_2 * D32_2;
    double const D23_2 = D23 * D23;
    double const D23_3 = D23_2 * D23;
    double const D23_4 = D23_2 * D23_2;
    double const denom = D32_4 + D23_4;

    if (denom < 1e-300) {
        throw std::runtime_error("pole-suppressed denominator underflow");
    }

    double const A_G5 = (D32_3 * P32 + D23_3 * P23) / denom;

    if (A_G5 <= 0.0) {
        throw std::runtime_error("nonpositive amplitude");
    }

    // Step 9: Maxwell-Jüttner normalization approximation
    double const N_tau =
        (1.0 + (141.0 / 208.0) * tau - (441.0 / 3328.0) * tau2) /
        (1.0 + (531.0 / 208.0) * tau + (6519.0 / 3328.0) * tau2);

    // Step 10: Direct kernel assembly
    double const E = gamma * units::me_c2;
    double const result = units::r_e2 * gamma_prime / (4.0 * E) *
                          std::sqrt(2.0 / std::numbers::pi) *
                          std::pow(tau, -0.5) * N_tau * A_G5 *
                          std::exp(-(lambda_min - 1.0) / tau);

    return result;
}

} // namespace compton
