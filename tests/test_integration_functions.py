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


class TestAdaptiveLegendre:
    """Tests for adaptive_legendre_integrate via cm.adaptive_legendre_integrate."""

    def test_smooth_function(self):
        """Adaptive integration of exp(x) on [0, 1] meets tolerance."""
        exact = math.e - 1.0
        for tol in [1e-3, 1e-6, 1e-10]:
            result = cm.adaptive_legendre_integrate(
                math.exp, base_order=4, a=0.0, b=1.0, tol=tol)
            rel_err = abs(result - exact) / exact
            assert rel_err < tol, (
                f"tol={tol:.0e}: rel_err={rel_err:.2e}")

    def test_peaked_function(self):
        """Adaptive integration of a peaked function requiring subdivision."""
        exact = math.atan(100.0)  # integral of 1/(1+100*x^2) from 0 to inf... no
        # ∫₀¹ 1/(1 + (10*(x-0.5))^2) dx  -- peaked at x=0.5
        from scipy.integrate import quad as scipy_quad
        f = lambda x: 1.0 / (1.0 + (10.0 * (x - 0.5))**2)
        exact, _ = scipy_quad(f, 0.0, 1.0)

        for tol in [1e-3, 1e-6]:
            result = cm.adaptive_legendre_integrate(
                f, base_order=4, a=0.0, b=1.0, tol=tol)
            rel_err = abs(result - exact) / exact
            assert rel_err < 10 * tol, (
                f"peaked tol={tol:.0e}: rel_err={rel_err:.2e}")

    def test_oscillatory_function(self):
        """Adaptive integration of sin(20x) on [0, 1] -- needs many panels."""
        exact = (1.0 - math.cos(20.0)) / 20.0  # ~0.0979
        f = lambda x: math.sin(20.0 * x)

        result = cm.adaptive_legendre_integrate(
            f, base_order=8, a=0.0, b=1.0, tol=1e-6)
        rel_err = abs(result - exact) / abs(exact)
        assert rel_err < 1e-5, f"oscillatory: rel_err={rel_err:.2e}"

    @pytest.mark.parametrize("base_order", [4, 8, 16])
    def test_base_order_all_converge(self, base_order):
        """All base orders converge to the correct answer for a smooth integral."""
        exact = 2.0  # ∫₀^π sin(x) dx
        result = cm.adaptive_legendre_integrate(
            math.sin, base_order=base_order, a=0.0, b=math.pi, tol=1e-8)
        rel_err = abs(result - exact) / exact
        assert rel_err < 1e-7, (
            f"base_order={base_order}: rel_err={rel_err:.2e}")

    def test_near_zero_integral(self):
        """Adaptive integration handles near-zero integrals without division issues."""
        f = lambda x: math.sin(2.0 * math.pi * x)  # ∫₀¹ sin(2πx) dx = 0
        result = cm.adaptive_legendre_integrate(
            f, base_order=4, a=0.0, b=1.0, tol=1e-8)
        assert abs(result) < 1e-12, f"expected ~0, got {result:.2e}"


class TestAdaptiveLogLegendre:
    """Tests for adaptive_log_legendre_integrate (clusters nodes near lower end)."""

    def test_smooth_function(self):
        """Log-space integration of exp(x) on [1, 10] meets tolerance."""
        from scipy.integrate import quad as scipy_quad
        exact, _ = scipy_quad(math.exp, 1.0, 10.0)
        for tol in [1e-3, 1e-6, 1e-10]:
            result = cm.adaptive_log_legendre_integrate(
                math.exp, base_order=8, a=1.0, b=10.0, tol=tol)
            rel_err = abs(result - exact) / exact
            assert rel_err < 10 * tol, (
                f"tol={tol:.0e}: rel_err={rel_err:.2e}")

    def test_exp_decay(self):
        """Log-space integration handles exponential decay well (clusters near a)."""
        from scipy.integrate import quad as scipy_quad
        f = lambda x: math.exp(-x)
        exact, _ = scipy_quad(f, 1.0, 100.0)
        result = cm.adaptive_log_legendre_integrate(
            f, base_order=8, a=1.0, b=100.0, tol=1e-6)
        rel_err = abs(result - exact) / exact
        assert rel_err < 1e-4, f"exp-decay: rel_err={rel_err:.2e}"

    def test_power_law(self):
        """∫₁^e x^(-2) dx = 1 - 1/e."""
        exact = 1.0 - 1.0 / math.e
        result = cm.adaptive_log_legendre_integrate(
            lambda x: x**(-2), base_order=4, a=1.0, b=math.e, tol=1e-8)
        rel_err = abs(result - exact) / exact
        assert rel_err < 1e-6, f"power-law: rel_err={rel_err:.2e}"

    def test_narrow_interval(self):
        """Integration over a very narrow interval [1.0, 1.001]."""
        from scipy.integrate import quad as scipy_quad
        f = lambda x: math.sin(x)
        exact, _ = scipy_quad(f, 1.0, 1.001)
        result = cm.adaptive_log_legendre_integrate(
            f, base_order=4, a=1.0, b=1.001, tol=1e-8)
        rel_err = abs(result - exact) / abs(exact) if exact != 0 else abs(result)
        assert rel_err < 1e-6, f"narrow: rel_err={rel_err:.2e}"

    def test_polynomial_exactness(self):
        """Polynomial x^3 on [1, 5]: should match to high precision."""
        exact = (5**4 - 1**4) / 4.0
        result = cm.adaptive_log_legendre_integrate(
            lambda x: x**3, base_order=16, a=1.0, b=5.0, tol=1e-10)
        rel_err = abs(result - exact) / exact
        assert rel_err < 1e-8, f"poly: rel_err={rel_err:.2e}"


class TestAdaptiveRlogLegendre:
    """Tests for adaptive_rlog_legendre_integrate (clusters nodes near upper end)."""

    def test_smooth_function(self):
        """Rlog-space integration of exp(x) on [1, 10] meets tolerance."""
        from scipy.integrate import quad as scipy_quad
        exact, _ = scipy_quad(math.exp, 1.0, 10.0)
        for tol in [1e-3, 1e-6, 1e-10]:
            result = cm.adaptive_rlog_legendre_integrate(
                math.exp, base_order=8, a=1.0, b=10.0, tol=tol)
            rel_err = abs(result - exact) / exact
            assert rel_err < 10 * tol, (
                f"tol={tol:.0e}: rel_err={rel_err:.2e}")

    def test_exp_growth_near_upper(self):
        """Rlog-space should handle integrands peaked near upper end well.

        f(x) = exp(-(b-x)) peaks at x=b; rlog clusters nodes there.
        """
        from scipy.integrate import quad as scipy_quad
        b = 10.0
        f = lambda x: math.exp(-(b - x))
        exact, _ = scipy_quad(f, 1.0, b)
        result = cm.adaptive_rlog_legendre_integrate(
            f, base_order=8, a=1.0, b=b, tol=1e-6)
        rel_err = abs(result - exact) / exact
        assert rel_err < 1e-4, f"exp-growth: rel_err={rel_err:.2e}"

    def test_power_law(self):
        """∫₁^e x^(-2) dx = 1 - 1/e (same integral, different node clustering)."""
        exact = 1.0 - 1.0 / math.e
        result = cm.adaptive_rlog_legendre_integrate(
            lambda x: x**(-2), base_order=4, a=1.0, b=math.e, tol=1e-8)
        rel_err = abs(result - exact) / exact
        assert rel_err < 1e-6, f"power-law: rel_err={rel_err:.2e}"

    def test_matches_log_on_smooth(self):
        """Both log and rlog should give the same answer for a smooth integrand."""
        from scipy.integrate import quad as scipy_quad
        f = lambda x: math.sin(x)
        exact, _ = scipy_quad(f, 1.0, 5.0)

        r_log = cm.adaptive_log_legendre_integrate(
            f, base_order=8, a=1.0, b=5.0, tol=1e-8)
        r_rlog = cm.adaptive_rlog_legendre_integrate(
            f, base_order=8, a=1.0, b=5.0, tol=1e-8)

        assert abs(r_log - exact) / abs(exact) < 1e-6
        assert abs(r_rlog - exact) / abs(exact) < 1e-6
        assert abs(r_log - r_rlog) / abs(exact) < 1e-6

    def test_narrow_interval(self):
        """Integration over a very narrow interval [5.0, 5.001]."""
        from scipy.integrate import quad as scipy_quad
        f = lambda x: math.cos(x)
        exact, _ = scipy_quad(f, 5.0, 5.001)
        result = cm.adaptive_rlog_legendre_integrate(
            f, base_order=4, a=5.0, b=5.001, tol=1e-8)
        rel_err = abs(result - exact) / abs(exact) if exact != 0 else abs(result)
        assert rel_err < 1e-6, f"narrow: rel_err={rel_err:.2e}"
