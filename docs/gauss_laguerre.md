# Gauss-Laguerre Quadrature Implementation

## Purpose

This document explains the custom Gauss-Laguerre quadrature implementation in `src/gauss_laguerre.hpp`.  The module computes N-point quadrature rules for integrals of the form:

```
∫₀^∞  f(x) · e^{−x} dx  ≈  Σᵢ₌₁ᴺ  w_i · f(x_i)
```

where the rule is exact for polynomials f of degree ≤ 2N−1.

---

## Why Custom Implementation?

The system Boost (v1.83) does not include `boost/math/quadrature/gauss_laguerre.hpp` (this header was introduced in Boost 1.84+).  Rather than requiring a newer Boost version or vendoring the entire Boost.Math repository, we implement the quadrature rule generation from first principles using the **Golub-Welsch algorithm**.

---

## Mathematical Foundation

### Gauss Quadrature and Orthogonal Polynomials

Any N-point Gaussian quadrature rule for a weight function w(x) on [a,b] can be derived from the family of orthogonal polynomials {p_n(x)} satisfying:

```
∫_a^b  p_m(x) · p_n(x) · w(x) dx  =  h_n · δ_{mn}
```

The N quadrature **nodes** are the zeros of p_N(x), and the **weights** are determined by the Christoffel-Darboux theorem.

### Laguerre Polynomials

For weight w(x) = e^{−x} on [0, ∞), the orthogonal polynomials are the Laguerre polynomials L_n(x) (with α = 0).  They satisfy the three-term recurrence:

```
(n+1) L_{n+1}(x)  =  (2n + 1 − x) L_n(x)  −  n L_{n−1}(x)
```

### The Jacobi (Tridiagonal) Matrix

Rewriting in monic form p_{n+1}(x) = (x − a_n) p_n(x) − b_n p_{n−1}(x):

```
a_n = 2n + 1       (n = 0, 1, ..., N−1)
b_n = n²           (n = 1, 2, ..., N−1)
```

The **Jacobi matrix** J is the N×N symmetric tridiagonal matrix:

```
        ┌ a₀   √b₁               ┐
        │ √b₁  a₁   √b₂          │
J   =   │      √b₂  a₂   √b₃     │
        │           ⋱    ⋱        │
        └              √b_{N-1} a_{N-1} ┘
```

For Laguerre: diagonal elements are `{1, 3, 5, 7, ...}` and sub-diagonal elements are `{1, 2, 3, 4, ...}`.

---

## The Golub-Welsch Algorithm

**Theorem (Golub & Welsch, 1969):**  The N-point Gaussian quadrature nodes are the eigenvalues of the Jacobi matrix J, and the weights are:

```
w_i = μ₀ · (v_i^{(1)})²
```

where:
- v_i^{(1)} is the first component of the i-th normalized eigenvector of J
- μ₀ = ∫₀^∞ e^{−x} dx = 1 (zeroth moment of the weight function)

This reduces quadrature rule generation to a symmetric tridiagonal eigenvalue problem.

---

## Eigenvalue Algorithm: Implicit QL (tql2)

We solve the tridiagonal eigenvalue problem using the **implicit QL algorithm**, adapted from the EISPACK `tql2` routine.  This is an iterative method with cubic convergence.

### Algorithm Outline

For each eigenvalue (indexed by l = 0, ..., N−1):

1. **Find unreduced submatrix:** Scan from l to find the smallest m such that the sub-diagonal element |e_m| is negligible relative to |d_m| + |d_{m+1}|.

2. **Convergence check:** If m == l, eigenvalue l is isolated; proceed to l+1.

3. **Wilkinson shift:** Compute a shift g that accelerates convergence:
   ```
   g = (d_{l+1} − d_l) / (2 · e_l)
   r = hypot(g, 1)
   g = d_m − d_l + e_l / (g + copysign(r, g))
   ```

4. **Implicit QL step (Givens rotations):** Sweep from i = m−1 down to l, applying a sequence of Givens rotations that chase a "bulge" along the tridiagonal.  Each rotation:
   - Updates the diagonal and sub-diagonal elements
   - Accumulates in the eigenvector matrix Z

5. **Iterate** until convergence (sub-diagonal element becomes negligible).

### Complexity

- Each eigenvalue typically requires 2–3 iterations
- Each iteration is O(N) for the sweep
- Total: O(N²) for all eigenvalues and eigenvectors
- Hard limit: 300 iterations per eigenvalue (guards against pathological cases)

### Numerical Properties

- Uses `std::hypot` for stable computation of rotations
- Convergence criterion: `|e_m| ≤ 10⁻¹⁵ · (|d_m| + |d_{m+1}|)`
- After all eigenvalues are found, a sort step orders them ascending

---

## Implementation Details

### Data Layout

```cpp
struct GaussLaguerreRule {
    std::vector<double> nodes;   // x_i: quadrature nodes
    std::vector<double> weights; // w_i: quadrature weights
};
```

### compute_gauss_laguerre(N)

1. Allocate `diag[N]` and `offdiag[N]`:
   ```cpp
   diag[i] = 2*i + 1           // a_i = 2i+1
   offdiag[i] = i              // √b_i = i (for i ≥ 1)
   ```
   Note: `offdiag[0]` is unused; the sub-diagonal occupies positions [1..N−1].

2. Call `tql2(N, diag, offdiag, Z)` which:
   - Shifts offdiag internally so sub-diagonal is in [0..N−2]
   - Computes eigenvalues (stored in `diag`) and eigenvectors (stored in `Z`)
   - Sorts by ascending eigenvalue

3. Extract results:
   ```cpp
   nodes[i]   = diag[i]           // eigenvalue = quadrature node
   weights[i] = Z[0*N + i]²      // (first eigenvector component)²
   ```

### Caching

In the main kernel code (`compton_kernel_quadrature.cpp`), rules are computed once and cached as static objects:

```cpp
static const GaussLaguerreRule& get_rule(int N) {
    static const GaussLaguerreRule rule_64  = compute_gauss_laguerre(64);
    static const GaussLaguerreRule rule_128 = compute_gauss_laguerre(128);
    // ...
}
```

A 256-point rule takes ~1ms to compute.

---

## Validation

The quadrature rule can be validated by checking:

1. **Moment conditions:** ∫₀^∞ x^k e^{−x} dx = k! for k = 0, 1, ..., 2N−1
   ```
   Σᵢ w_i · x_i^k  =  k!    (should be exact to machine precision)
   ```

2. **Weight sum:** Σᵢ w_i = 1 (= μ₀ = Γ(1))

3. **Positive nodes:** All x_i > 0

4. **Positive weights:** All w_i > 0

5. **Convergence:** For smooth integrands, doubling N should not change the result by more than O(10⁻⁶) for N ≥ 64.

---

## Reference

G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
Mathematics of Computation **23** (106): 221–230, 1969.
