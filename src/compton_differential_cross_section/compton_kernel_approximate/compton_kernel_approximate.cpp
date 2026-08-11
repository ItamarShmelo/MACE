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

// Precomputed combined coefficients for F_n^(p)(rho) evaluation.
// Each entry stores: FACTORIAL[n] * gen_binom[j] * BINOM_STANDARD[j][n-j]
// for the valid (n, j) pairs, separately for p=1 and p=3.
//
// Within F_np, the sum is:
//   sum_j  coeff * (2*rho)^(2j-n) * s2^(-(p/2 + j))
//
// With power tables:
//   p=1: s2_nhalf[j]      (s2^{-(0.5+j)})
//   p=3: s2_nhalf[j + 1]  (s2^{-(1.5+j)})
//   (2*rho)^k: tr_pow[2*j - n]

struct FnpTerm {
    int tr_idx;   // index into tr_pow: 2*j - n
    int s2_idx;   // index into s2_nhalf: j for p=1, j+1 for p=3
    double w1;    // combined weight for p=1
    double w3;    // combined weight for p=3
};

// Term groups by n. j_min = ceil(n/2), j_max = n.
// n=0: 1 term; n=1: 1 term; n=2: 2 terms; n=3: 2 terms;
// n=4: 3 terms; n=5: 3 terms.  Total: 12 entries.
constexpr std::array<FnpTerm, 12> FNP_TABLE = {{
    // n=0: j=0
    {0, 0,  1.0,  1.0},
    // n=1: j=1
    {1, 1, -0.5, -1.5},
    // n=2: j=1
    {0, 1, -1.0, -3.0},
    // n=2: j=2
    {2, 2,  0.75,  3.75},
    // n=3: j=2
    {1, 2,  4.5,  22.5},
    // n=3: j=3
    {3, 3, -1.875, -13.125},
    // n=4: j=2
    {0, 2,  9.0,  45.0},
    // n=4: j=3
    {2, 3, -22.5, -157.5},
    // n=4: j=4
    {4, 4,  6.5625,  59.0625},
    // n=5: j=3
    {1, 3, -112.5, -787.5},
    // n=5: j=4
    {3, 4,  131.25,  1181.25},
    // n=5: j=5
    {5, 5, -29.53125, -324.84375},
}};

// Start index in FNP_TABLE for each n=0..5, plus sentinel at end.
constexpr std::array<int, 7> FNP_OFFSETS = {0, 1, 2, 4, 6, 9, 12};

// Precomputed power tables for a single endpoint (rho, omega2).
struct EndpointPowers {
    std::array<double, 6> tr_pow;   // (2*rho)^k for k=0..5
    std::array<double, 7> s2_nhalf; // s2^{-(k+0.5)} for k=0..6
};

inline EndpointPowers
compute_endpoint_powers(double rho, double omega2)
{
    EndpointPowers ep{};

    double const s2 = (rho * rho) + omega2;
    double const inv_sqrt_s2 = 1.0 / std::sqrt(s2);
    double const inv_s2 = inv_sqrt_s2 * inv_sqrt_s2;

    ep.s2_nhalf[0] = inv_sqrt_s2;
    for (int k = 1; k < 7; ++k) {
        ep.s2_nhalf[k] = ep.s2_nhalf[k - 1] * inv_s2;
    }

    double const two_rho = 2.0 * rho;
    ep.tr_pow[0] = 1.0;
    for (int k = 1; k < 6; ++k) {
        ep.tr_pow[k] = ep.tr_pow[k - 1] * two_rho;
    }

    return ep;
}

// Evaluate F_n^(1) and F_n^(3) simultaneously using precomputed tables.
inline void
eval_fnp_pair(
    int n,
    EndpointPowers const& ep,
    double& f_p1,
    double& f_p3)
{
    f_p1 = 0.0;
    f_p3 = 0.0;
    int const start = FNP_OFFSETS[n];
    int const end = FNP_OFFSETS[n + 1];

    for (int i = start; i < end; ++i) {
        FnpTerm const& t = FNP_TABLE[i];
        double const base = ep.tr_pow[t.tr_idx];
        f_p1 += t.w1 * base * ep.s2_nhalf[t.s2_idx];
        f_p3 += t.w3 * base * ep.s2_nhalf[t.s2_idx + 1];
    }
}

/**
 * Solve a 2x2 linear system with partial pivoting.
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

        for (int row = col + 1; row < 3; ++row) {
            double const factor = A[row][col] / A[col][col];
            for (int k = col + 1; k < 3; ++k) {
                A[row][k] -= factor * A[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

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
    double const q = std::sqrt(Delta * Delta + 2.0 * A);
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

    // Step 5: Precompute power tables for both endpoints
    EndpointPowers const ep_minus = compute_endpoint_powers(rho_minus, omega2);
    EndpointPowers const ep_plus = compute_endpoint_powers(rho_plus, omega2);

    // Step 6: Compute coefficients C[0..5] using flattened tables,
    // caching F_(n-1)^(3) across iterations.
    std::array<double, 6> C{};
    double prev_f3_minus = 0.0;
    double prev_f3_plus = 0.0;

    for (int n = 0; n <= 5; ++n) {
        double f1_minus = 0.0;
        double f3_minus = 0.0;
        double f1_plus = 0.0;
        double f3_plus = 0.0;
        eval_fnp_pair(n, ep_minus, f1_minus, f3_minus);
        eval_fnp_pair(n, ep_plus, f1_plus, f3_plus);

        double const kronecker_n0 = (n == 0) ? 1.0 : 0.0;

        C[n] = (2.0 * kronecker_n0) / q +
               B * (f1_minus - f1_plus) +
               (delta_minus * f3_minus + delta_plus * f3_plus) * inv_A_sq +
               static_cast<double>(n) * sum_gamma * inv_A_sq *
                   (prev_f3_minus + prev_f3_plus);

        prev_f3_minus = f3_minus;
        prev_f3_plus = f3_plus;
    }

    // Step 7: [3/2] Padé continuation
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

    // Step 8: [2/3] Padé continuation
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

    // Step 9: Pole-suppressed amplitude
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

    // Step 10: Maxwell-Jüttner normalization approximation
    double const N_tau =
        (1.0 + (141.0 / 208.0) * tau - (441.0 / 3328.0) * tau2) /
        (1.0 + (531.0 / 208.0) * tau + (6519.0 / 3328.0) * tau2);

    // Step 11: Direct kernel assembly
    double const inv_sqrt_tau = 1.0 / std::sqrt(tau);
    double const E = gamma * units::me_c2;
    double const result = units::r_e2 * gamma_prime / (4.0 * E) *
                          std::sqrt(2.0 / std::numbers::pi) *
                          inv_sqrt_tau * N_tau * A_G5 *
                          std::exp(-(lambda_min - 1.0) / tau);

    return result;
}

} // namespace compton
