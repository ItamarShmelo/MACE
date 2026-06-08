#ifndef GAUSS_LEGENDRE_HPP
#define GAUSS_LEGENDRE_HPP
/**
 * @file gauss_legendre.hpp
 * @brief N-point Gauss-Legendre quadrature via the Golub-Welsch algorithm.
 *
 * Computes nodes {x_i} and weights {w_i} on [−1, 1] such that
 *
 *     ∫₋₁¹  f(x) dx  ≈  Σᵢ  w_i f(x_i)
 *
 * is exact for polynomials f of degree ≤ 2N−1.
 *
 * A convenience function maps the rule to an arbitrary finite interval [a, b].
 *
 * ─────────────────────────────────────────────────────────────────────────
 * MATHEMATICAL BACKGROUND
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The Legendre polynomials P_n(x) satisfy the three-term recurrence
 *
 *     (n+1) P_{n+1}(x)  =  (2n+1) x P_n(x)  −  n P_{n−1}(x).
 *
 * Rewriting in the monic form p_{n+1} = (x − a_n) p_n − b_n p_{n−1}
 * gives recurrence coefficients:
 *
 *     a_n = 0                       (diagonal of the Jacobi matrix)
 *     b_n = n² / (4n² − 1)         (square of sub-diagonal elements)
 *
 * so sub-diagonal  =  sqrt(b_n) = n / sqrt(4n² − 1).
 *
 * The Golub-Welsch algorithm states:
 *
 *     Nodes  x_i   =  eigenvalues of the N×N Jacobi matrix
 *     Weights w_i  =  μ₀ · (v_i^{(0)})²
 *
 * where μ₀ = ∫₋₁¹ dx = 2 and v_i^{(0)} is the first component of the
 * i-th normalized eigenvector.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * IMPLEMENTATION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Reuses the tql2 implicit-QL eigensolver from gauss_laguerre.hpp.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REFERENCE
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
 *   Mathematics of Computation 23 (106): 221–230, 1969.
 */

#include "compton_kernel_quadrature/gauss_laguerre.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>

namespace compton {

struct GaussLegendreRule {
    std::vector<double> nodes;   ///< Quadrature nodes on [−1, 1]
    std::vector<double> weights; ///< Corresponding quadrature weights

    GaussLegendreRule() = default;
    explicit GaussLegendreRule(int const N) : nodes(N), weights(N) {}
};

/**
 * @brief Compute N-point Gauss-Legendre quadrature rule.
 *
 * Builds the N×N Jacobi tridiagonal matrix for Legendre polynomials on
 * [−1, 1], then diagonalizes it via the QL algorithm to extract the N
 * quadrature nodes and weights.
 *
 * @param N  Number of quadrature points (must be ≥ 1).
 * @return   GaussLegendreRule with nodes and weights vectors of length N.
 * @throws   std::invalid_argument if N < 1.
 */
inline GaussLegendreRule compute_gauss_legendre(int const N) {
    if (N < 1) throw std::invalid_argument("N must be >= 1");

    std::vector<double> diag(N, 0.0);
    std::vector<double> offdiag(N, 0.0);

    for (int i = 1; i < N; ++i) {
        double const n = static_cast<double>(i);
        offdiag[i] = n / std::sqrt(4.0 * n * n - 1.0);
    }

    detail::Tql2Result const eig = detail::tql2(N, std::move(diag), std::move(offdiag));

    GaussLegendreRule rule(N);

    constexpr double mu0 = 2.0;
    for (int i = 0; i < N; ++i) {
        rule.nodes[i] = eig.eigenvalues[i];
        double const v0 = eig.eigenvectors[0 * N + i];
        rule.weights[i] = mu0 * v0 * v0;
    }

    return rule;
}

/**
 * @brief Integrate f over [a, b] using a precomputed Gauss-Legendre rule.
 *
 * Maps the reference nodes from [−1, 1] to [a, b] via the affine
 * transformation x = (b−a)/2 · t + (a+b)/2, with Jacobian (b−a)/2.
 *
 * @param integrand  Callable f(x) → double.
 * @param rule       Precomputed rule on [−1, 1].
 * @param a          Lower integration limit.
 * @param b          Upper integration limit.
 * @return           Approximate value of ∫_a^b f(x) dx.
 */
template<typename F>
inline double legendre_integrate(F&& integrand,
                                 GaussLegendreRule const& rule,
                                 double const a,
                                 double const b) {
    double const half_width = 0.5 * (b - a);
    double const midpoint   = 0.5 * (a + b);

    double sum = 0.0;
    int const n = static_cast<int>(rule.nodes.size());
    for (int i = 0; i < n; ++i) {
        double const x = half_width * rule.nodes[i] + midpoint;
        sum += rule.weights[i] * integrand(x);
    }
    return half_width * sum;
}

/**
 * @brief Adaptive Gauss-Legendre integration via recursive bisection.
 *
 * Estimates the integral over [a, b] by comparing the single-panel GL
 * result against the sum of two half-panel results.  If the difference
 * exceeds the relative tolerance, each half is refined independently.
 *
 * @param integrand   Callable f(x) -> double.
 * @param rule        Precomputed GL rule (base panel order).
 * @param a           Lower integration limit.
 * @param b           Upper integration limit.
 * @param tol         Relative tolerance for convergence.
 * @param max_depth   Maximum recursion depth (prevents runaway subdivision).
 * @return            Approximate value of integral_a^b f(x) dx.
 */
template<typename F>
double adaptive_legendre_integrate(F&& integrand,
                                   GaussLegendreRule const& rule,
                                   double const a,
                                   double const b,
                                   double const tol,
                                   int const max_depth = 15)
{
    constexpr double abs_floor = 1e-300;

    double const I_whole = legendre_integrate(integrand, rule, a, b);

    double const mid = 0.5 * (a + b);
    double const I_left  = legendre_integrate(integrand, rule, a, mid);
    double const I_right = legendre_integrate(integrand, rule, mid, b);
    double const I_halves = I_left + I_right;

    double const error = std::abs(I_halves - I_whole);
    double const scale = std::max(std::abs(I_halves), abs_floor);

    if (error <= tol * scale || max_depth <= 0) {
        return I_halves;
    }

    double const refined_left = adaptive_legendre_integrate(
        integrand, rule, a, mid, tol, max_depth - 1);
    double const refined_right = adaptive_legendre_integrate(
        integrand, rule, mid, b, tol, max_depth - 1);
    return refined_left + refined_right;
}

} // namespace compton

#endif
