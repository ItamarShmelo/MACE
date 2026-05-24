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

#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace compton {

struct GaussLaguerreRule {
    std::vector<double> nodes;   ///< Quadrature nodes (zeros of L_N)
    std::vector<double> weights; ///< Corresponding quadrature weights
};

namespace detail {

/**
 * @brief Implicit QL algorithm for symmetric tridiagonal eigenvalue problem.
 *
 * Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal
 * matrix.  On entry:
 *   - diag[0..n-1]     : diagonal elements
 *   - offdiag[0..n-1]  : sub-diagonal in positions [1..n-1]; offdiag[0] unused
 *
 * On exit:
 *   - diag[]   : eigenvalues in ascending order
 *   - Z[k*n+i] : k-th component of the i-th eigenvector
 *
 * The algorithm converges cubically and typically needs 2–3 iterations per
 * eigenvalue.  A hard limit of 300 iterations per eigenvalue guards against
 * pathological cases.
 */
inline void tql2(int n, std::vector<double>& diag, std::vector<double>& offdiag,
                 std::vector<double>& Z) {
    Z.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) Z[i * n + i] = 1.0;

    if (n == 1) return;

    // Shift offdiag so that subdiagonal is in offdiag[0..n-2]
    // (offdiag passed in has subdiag in [1..n-1])
    for (int i = 0; i < n - 1; ++i)
        offdiag[i] = offdiag[i + 1];
    offdiag[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;
        constexpr int max_iter = 300;

        while (true) {
            int m = l;
            for (; m < n - 1; ++m) {
                double dd = std::abs(diag[m]) + std::abs(diag[m + 1]);
                if (std::abs(offdiag[m]) <= 1e-15 * dd) break;
            }
            if (m == l) break;

            if (iter++ >= max_iter)
                throw std::runtime_error("tql2: too many iterations");

            double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
            double r = std::hypot(g, 1.0);
            g = diag[m] - diag[l] + offdiag[l] / (g + std::copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;

            for (int i = m - 1; i >= l; --i) {
                double f = s * offdiag[i];
                double b = c * offdiag[i];

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

            if (r == 0.0 && iter > 0) continue;

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
    std::vector<double> sorted_Z(n * n);
    for (int i = 0; i < n; ++i) {
        sorted_diag[i] = diag[idx[i]];
        for (int k = 0; k < n; ++k) {
            sorted_Z[k * n + i] = Z[k * n + idx[i]];
        }
    }
    diag = std::move(sorted_diag);
    Z = std::move(sorted_Z);
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
inline GaussLaguerreRule compute_gauss_laguerre(int N) {
    if (N < 1) throw std::invalid_argument("N must be >= 1");

    // For Laguerre polynomials L_n(x) with alpha=0:
    // Three-term recurrence: (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
    // Monic form: p_{n+1} = (x - a_n) p_n - b_n p_{n-1}
    // where a_n = 2n+1, b_n = n^2
    // Jacobi matrix: diagonal = a_n, off-diagonal = sqrt(b_n) = n

    std::vector<double> diag(N);
    std::vector<double> offdiag(N, 0.0);

    for (int i = 0; i < N; ++i)
        diag[i] = 2.0 * i + 1.0;

    // subdiag in positions 1..N-1 (position 0 unused, tql2 will shift)
    for (int i = 1; i < N; ++i)
        offdiag[i] = static_cast<double>(i);

    std::vector<double> Z;
    detail::tql2(N, diag, offdiag, Z);

    GaussLaguerreRule rule;
    rule.nodes.resize(N);
    rule.weights.resize(N);

    // mu_0 = integral of e^{-x} from 0 to inf = 1
    for (int i = 0; i < N; ++i) {
        rule.nodes[i] = diag[i];
        double v0 = Z[0 * N + i];
        rule.weights[i] = v0 * v0;
    }

    return rule;
}

} // namespace compton

#endif
