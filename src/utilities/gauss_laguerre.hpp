#ifndef GAUSS_LAGUERRE_HPP
#define GAUSS_LAGUERRE_HPP
/**
 * @file gauss_laguerre.hpp
 * @brief N-point Gauss-Laguerre quadrature via the Golub-Welsch algorithm.
 *
 * Computes nodes {x_i} and weights {w_i} such that
 *
 *     ∫₀^∞  f(x) e^{-x} dx  ≈  Σᵢ  w_i f(x_i)
 *
 * is exact for polynomials f of degree ≤ 2N−1.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * MATHEMATICAL BACKGROUND
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The generalized Laguerre polynomials L_n^{(α)}(x) satisfy a three-term
 * recurrence.  For α = 0 (our case):
 *
 *     (n+1) L_{n+1}(x)  =  (2n + 1 − x) L_n(x)  −  n L_{n−1}(x)
 *
 * Rewriting in monic form p_{n+1} = (x − a_n) p_n − b_n p_{n−1} gives
 * recurrence coefficients:
 *
 *     a_n = 2n + 1       (diagonal of the Jacobi matrix)
 *     b_n = n²           (square of sub-diagonal elements)
 *
 * The Golub-Welsch algorithm (Golub & Welsch 1969) states:
 *
 *     Nodes  x_i   =  eigenvalues of the N×N symmetric tridiagonal
 *                      Jacobi matrix  J  with  J_{ii} = a_i,
 *                      J_{i,i+1} = J_{i+1,i} = sqrt(b_{i+1}) = i+1.
 *
 *     Weights w_i  =  μ₀ · (v_i^{(0)})²
 *
 * where v_i^{(0)} is the first component of the i-th normalized eigenvector,
 * and μ₀ = ∫₀^∞ e^{-x} dx = 1 (zeroth moment of the weight function).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * IMPLEMENTATION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * 1. Build the Jacobi matrix in tridiagonal form (diagonal + sub-diagonal).
 *
 * 2. Solve the eigenvalue problem using the implicit QL algorithm with
 *    implicit shifts (tql2).  This is a standard O(N²) method for symmetric
 *    tridiagonal matrices that simultaneously produces all eigenvalues and
 *    eigenvectors.  Our implementation follows the EISPACK tql2 routine:
 *      - Find an unreduced submatrix [l..m]
 *      - Apply Wilkinson shift
 *      - Execute Givens rotations (the "implicit QL step") sweeping from
 *        m−1 down to l, accumulating the rotation in the eigenvector matrix Z
 *      - Iterate until sub-diagonal elements are negligible
 *
 * 3. Sort eigenvalues/eigenvectors ascending and extract nodes and weights.
 *
 * Why not use Boost?  The system Boost (1.83) does not ship
 * boost/math/quadrature/gauss_laguerre.hpp, so we provide our own.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REFERENCE
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
 *   Mathematics of Computation 23 (106): 221–230, 1969.
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace compton {

struct GaussLaguerreRule {
    std::vector<double> nodes;   ///< Quadrature nodes (zeros of L_N)
    std::vector<double> weights; ///< Corresponding quadrature weights

    GaussLaguerreRule() = default;
    explicit GaussLaguerreRule(int const N)
        : nodes(N),
          weights(N)
    {}
};

namespace detail {

struct Tql2Result {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
};

/**
 * @brief Implicit QL algorithm for symmetric tridiagonal eigenvalue problem.
 *
 * Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal
 * matrix.  On entry:
 *   - diag[0..n-1]     : diagonal elements
 *   - offdiag[0..n-1]  : sub-diagonal in positions [1..n-1]; offdiag[0]
 *                        ignored on entry (kept 0.0 by convention)
 *
 * Returns:
 *   - eigenvalues[]   : eigenvalues in ascending order
 *   - eigenvectors[]  : eigenvector matrix in column-major form, where
 *                       eigenvectors[k*n+i] is the k-th component of the
 *                       i-th eigenvector
 *
 * The algorithm converges cubically and typically needs 2–3 iterations per
 * eigenvalue.  A hard limit of 300 iterations per eigenvalue guards against
 * pathological cases.
 */
inline Tql2Result
tql2(int const n, std::vector<double> diag, std::vector<double> offdiag)
{
    std::vector<double> Z(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        Z[i * n + i] = 1.0;
    }

    if (n == 1) {
        return Tql2Result{std::move(diag), std::move(Z)};
    }

    // Shift offdiag so that subdiagonal is in offdiag[0..n-2]
    // (offdiag passed in has subdiag in [1..n-1])
    for (int i = 0; i < n - 1; ++i) {
        offdiag[i] = offdiag[i + 1];
    }
    offdiag[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;
        constexpr int max_iter = 300;

        while (true) {
            int m = l;
            for (; m < n - 1; ++m) {
                double const dd = std::abs(diag[m]) + std::abs(diag[m + 1]);
                if (std::abs(offdiag[m]) <= 1e-15 * dd) {
                    break;
                }
            }
            if (m == l) {
                break;
            }

            if (iter++ >= max_iter) {
                throw std::runtime_error("tql2: too many iterations");
            }

            double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
            double r = std::hypot(g, 1.0);
            g = diag[m] - diag[l] + offdiag[l] / (g + std::copysign(r, g));

            double s = 1.0;
            double c = 1.0;
            double p = 0.0;

            for (int i = m - 1; i >= l; --i) {
                double f = s * offdiag[i];
                double const b = c * offdiag[i];

                r = std::hypot(f, g);
                offdiag[i + 1] = r;

                if (r == 0.0) {
                    diag[i + 1] -= p;
                    offdiag[m] = 0.0;
                    break;
                }

                s = f / r;
                c = g / r;
                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;

                for (int k = 0; k < n; ++k) {
                    f = Z[k * n + (i + 1)];
                    Z[k * n + (i + 1)] = s * Z[k * n + i] + c * f;
                    Z[k * n + i] = c * Z[k * n + i] - s * f;
                }
            }

            if (r == 0.0 && iter > 0) {
                continue;
            }

            diag[l] -= p;
            offdiag[l] = g;
            offdiag[m] = 0.0;
        }
    }

    // Sort eigenvalues and eigenvectors by ascending eigenvalue
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return diag[a] < diag[b];
    });

    std::vector<double> sorted_diag(n);
    std::vector<double> sorted_Z(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        sorted_diag[i] = diag[idx[i]];
        for (int k = 0; k < n; ++k) {
            sorted_Z[k * n + i] = Z[k * n + idx[i]];
        }
    }
    diag = std::move(sorted_diag);
    Z = std::move(sorted_Z);
    return Tql2Result{std::move(diag), std::move(Z)};
}

} // namespace detail

/**
 * @brief Compute N-point Gauss-Laguerre quadrature rule (α = 0).
 *
 * Builds the N×N Jacobi tridiagonal matrix for the Laguerre weight e^{-x}
 * on [0,∞), then diagonalizes it via the QL algorithm to extract the N
 * quadrature nodes and weights.
 *
 * @param N  Number of quadrature points (must be ≥ 1).
 * @return   GaussLaguerreRule with nodes and weights vectors of length N.
 * @throws   std::invalid_argument if N < 1.
 */
inline GaussLaguerreRule compute_gauss_laguerre(int N)
{
    if (N < 1) {
        throw std::invalid_argument("N must be >= 1");
    }

    // For Laguerre polynomials L_n(x) with alpha=0:
    // Three-term recurrence: (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
    // Monic form: p_{n+1} = (x - a_n) p_n - b_n p_{n-1}
    // where a_n = 2n+1, b_n = n^2
    // Jacobi matrix: diagonal = a_n, off-diagonal = sqrt(b_n) = n

    std::vector<double> diag(N);
    std::vector<double> offdiag(N, 0.0);

    // subdiag in positions 1..N-1; offdiag[0] is ignored on entry by tql2.
    for (int i = 0; i < N; ++i) {
        diag[i] = 2.0 * i + 1.0;
        offdiag[i] = (i == 0) ? 0.0 : static_cast<double>(i);
    }

    detail::Tql2Result const eig =
        detail::tql2(N, std::move(diag), std::move(offdiag));

    GaussLaguerreRule rule(N);

    // mu_0 = integral of e^{-x} from 0 to inf = 1
    for (int i = 0; i < N; ++i) {
        rule.nodes[i] = eig.eigenvalues[i];
        double const v0 = eig.eigenvectors[0 * N + i];
        rule.weights[i] = v0 * v0;
    }

    return rule;
}

/**
 * @brief Integrate a function using a precomputed Gauss-Laguerre rule.
 */
template <typename F>
inline double laguerre_integrate(F&& integrand, GaussLaguerreRule const& rule)
{
    double sum = 0.0;
    int const n = static_cast<int>(rule.nodes.size());
    for (int i = 0; i < n; ++i) {
        sum += rule.weights[i] * integrand(rule.nodes[i]);
    }
    return sum;
}

/**
 * @brief Return cached Gauss-Laguerre rules for supported orders.
 */
inline GaussLaguerreRule const& get_rule(int const N)
{
    static GaussLaguerreRule const rule_16 = compute_gauss_laguerre(16);
    static GaussLaguerreRule const rule_32 = compute_gauss_laguerre(32);
    static GaussLaguerreRule const rule_64 = compute_gauss_laguerre(64);
    static GaussLaguerreRule const rule_128 = compute_gauss_laguerre(128);
    static GaussLaguerreRule const rule_256 = compute_gauss_laguerre(256);

    switch (N) {
    case 16:
        return rule_16;
    case 32:
        return rule_32;
    case 64:
        return rule_64;
    case 128:
        return rule_128;
    case 256:
        return rule_256;
    default:
        throw std::invalid_argument("N must be one of: 16, 32, 64, 128, 256");
    }
}

} // namespace compton

#endif
