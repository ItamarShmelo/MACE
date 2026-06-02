"""
Tests for the Gauss-Legendre and Gauss-Laguerre quadrature implementations.

Validates node/weight properties, polynomial exactness, and convergence
against known analytic integrals.
"""

import sys
import math

import numpy as np
import pytest
from scipy.special import exp1

sys.path.insert(0, "cpp_modules")

import _compton_kernel_quadrature as cq
import _compton_multigroup as cm


def _legendre_integrate(f, nodes, weights, a, b):
    """Affine-mapped Gauss-Legendre quadrature on [a, b]."""
    half_width = 0.5 * (b - a)
    midpoint = 0.5 * (a + b)
    x = half_width * nodes + midpoint
    return half_width * np.dot(weights, f(x))


class TestGaussLaguerre:
    """Tests for gauss_laguerre.hpp via cq.gauss_laguerre_rule."""

    def test_rule_properties(self):
        nodes, weights = cq.gauss_laguerre_rule(32)
        assert weights.sum() == pytest.approx(1.0, abs=1e-14)
        assert np.all(nodes > 0)
        assert np.all(weights > 0)

        n1, w1 = cq.gauss_laguerre_rule(1)
        assert n1[0] == pytest.approx(1.0, abs=1e-14)
        assert w1[0] == pytest.approx(1.0, abs=1e-14)

        with pytest.raises(ValueError):
            cq.gauss_laguerre_rule(0)

    def test_polynomial_exactness(self):
        """∫₀^∞ xᵏ e⁻ˣ dx = k!, exact for k = 0..2N-1."""
        N = 16
        nodes, weights = cq.gauss_laguerre_rule(N)
        worst_rel = 0.0
        for k in range(2 * N):
            approx = float(np.dot(weights, nodes**k))
            exact = float(math.factorial(k))
            rel = abs(approx - exact) / exact if exact else abs(approx)
            worst_rel = max(worst_rel, rel)
        assert worst_rel < 1e-10, f"worst relative error = {worst_rel:.2e}"

    def test_known_integrals(self):
        """Non-polynomial integrals with convergence check (N=8 vs N=64)."""
        cases = [
            (np.cos, 0.5),
            (lambda x: 1.0 / (1.0 + x), math.e * exp1(1.0)),
        ]
        for func, exact in cases:
            n8, w8 = cq.gauss_laguerre_rule(8)
            err_8 = abs(float(np.dot(w8, func(n8))) - exact)
            n64, w64 = cq.gauss_laguerre_rule(64)
            err_64 = abs(float(np.dot(w64, func(n64))) - exact)

            assert err_64 < 1e-12, f"N=64 error = {err_64:.2e}"
            assert err_64 < err_8, "N=64 should beat N=8"


class TestGaussLegendre:
    """Tests for gauss_legendre.hpp via cm.gauss_legendre_rule."""

    def test_rule_properties(self):
        nodes, weights = cm.gauss_legendre_rule(32)
        assert weights.sum() == pytest.approx(2.0, abs=1e-14)
        assert np.all(nodes > -1) and np.all(nodes < 1)
        assert np.all(weights > 0)

        s = np.argsort(nodes)
        sorted_n = nodes[s]
        sorted_w = weights[s]
        np.testing.assert_allclose(sorted_n, -sorted_n[::-1], atol=1e-14)
        np.testing.assert_allclose(sorted_w, sorted_w[::-1], atol=1e-14)

        n1, w1 = cm.gauss_legendre_rule(1)
        assert n1[0] == pytest.approx(0.0, abs=1e-14)
        assert w1[0] == pytest.approx(2.0, abs=1e-14)

        with pytest.raises(ValueError):
            cm.gauss_legendre_rule(0)

    def test_polynomial_exactness(self):
        """∫₋₁¹ xᵏ dx: 0 for odd k, 2/(k+1) for even k. Exact for k = 0..2N-1."""
        N = 16
        nodes, weights = cm.gauss_legendre_rule(N)
        worst_rel = 0.0
        for k in range(2 * N):
            approx = float(np.dot(weights, nodes**k))
            exact = 0.0 if k % 2 else 2.0 / (k + 1)
            err = abs(approx - exact)
            rel = err / abs(exact) if exact else err
            worst_rel = max(worst_rel, rel)
        assert worst_rel < 1e-10, f"worst relative error = {worst_rel:.2e}"

    def test_mapped_interval(self):
        """Affine-mapped integration on finite intervals."""
        nodes, weights = cm.gauss_legendre_rule(32)

        val = _legendre_integrate(np.exp, nodes, weights, 0.0, 1.0)
        assert val == pytest.approx(math.e - 1, rel=1e-14)

        val = _legendre_integrate(np.sin, nodes, weights, 0.0, math.pi)
        assert val == pytest.approx(2.0, rel=1e-14)

    def test_convergence(self):
        """Higher order gives smaller error for exp(x) on [-1, 1]."""
        exact = math.e - 1.0 / math.e
        n4, w4 = cm.gauss_legendre_rule(4)
        n32, w32 = cm.gauss_legendre_rule(32)
        err_4 = abs(float(np.dot(w4, np.exp(n4))) - exact)
        err_32 = abs(float(np.dot(w32, np.exp(n32))) - exact)
        assert err_32 < err_4, "N=32 should beat N=4"
