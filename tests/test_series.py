"""
Phase 4: C++ series module validation tests.

Tests the C++ series implementation against:
  - C++ quadrature (NL=256)
  - Python series (pointwise agreement)
  - scipy.special.expn (ehat_expn validation)
  - Physical constraints (detailed balance, positivity)
"""

import math
import sys

import numpy as np
import pytest
from scipy.special import expn

sys.path.insert(0, "cpp_modules")
sys.path.insert(0, "src/python")

import _compton_kernel_quadrature as cq
import _compton_kernel_series as cs
from pycompton.compton_kernel_quadrature import me_c2
from pycompton.compton_kernel_series import sigma_E_series as py_sigma_E_series

kev = 1.602176634e-9

TEST_POINTS = [
    (1.0, 1.01, 0.0),
    (1.0, 0.99, 0.0),
    (1.0, 2.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 1.01, 0.5),
    (1.0, 1.01, -0.5),
    (10.0, 10.5, 0.0),
    (10.0, 9.5, 0.0),
    (100.0, 101.0, 0.0),
]

TEMPS_KEV = [0.1, 1.0, 5.0, 20.0, 100.0]


# ═══════════════════════════════════════════════════════════════════════════════
# C++ ehat_expn vs scipy
# ═══════════════════════════════════════════════════════════════════════════════


class TestEhatExpn:
    @pytest.mark.parametrize("m", [1, 2, 5, 10])
    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 10.0, 30.0, 49.0])
    def test_vs_scipy(self, m, x):
        val = cs.ehat_expn(m, x)
        ref = math.exp(x) * float(expn(m, x))
        reldiff = abs(val - ref) / (abs(ref) + 1e-300)
        assert reldiff < 1e-12, f"m={m}, x={x}: reldiff={reldiff}"

    @pytest.mark.parametrize("m", [1, 2, 5, 10])
    @pytest.mark.parametrize("x", [51.0, 100.0, 500.0])
    def test_large_x_positive_finite(self, m, x):
        val = cs.ehat_expn(m, x)
        assert val > 0 and math.isfinite(val)

    def test_invalid_inputs(self):
        with pytest.raises(Exception):
            cs.ehat_expn(1, -1.0)
        with pytest.raises(Exception):
            cs.ehat_expn(0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# C++ power series vs C++ quadrature
# ═══════════════════════════════════════════════════════════════════════════════


class TestPowerSeriesVsQuadrature:
    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi",
        [(1.0, 1.01, 0.0), (10.0, 10.5, 0.0), (100.0, 101.0, 0.0)],
    )
    def test_high_temp(self, E_kev, Ep_kev, xi):
        T_kev = 100.0
        tau = T_kev * kev / me_c2
        E, Ep = E_kev * kev, Ep_kev * kev

        quad = cq.ComptonKernelQuadrature(256)
        qres = quad.sigma_E(E, Ep, xi, tau, 1.0)

        series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeries)
        sres = series.sigma_E(E, Ep, xi, tau, 1.0)

        reldiff = abs(sres.value - qres.value) / (abs(qres.value) + 1e-300)
        assert sres.converged
        assert reldiff < 1e-4, f"reldiff={reldiff}"


# ═══════════════════════════════════════════════════════════════════════════════
# C++ asymptotic vs C++ quadrature
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsymptoticVsQuadrature:
    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", [0.1, 1.0, 5.0])
    def test_low_temp(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E, Ep = E_kev * kev, Ep_kev * kev

        quad = cq.ComptonKernelQuadrature(256)
        qres = quad.sigma_E(E, Ep, xi, tau, 1.0)

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Asymptotic)
        sres = series.sigma_E(E, Ep, xi, tau, 1.0)

        reldiff = abs(sres.value - qres.value) / (abs(qres.value) + 1e-300)
        assert sres.converged, (
            f"Asymptotic not converged: T={T_kev}, E={E_kev}, Ep={Ep_kev}, xi={xi}"
        )
        assert reldiff < 1e-3, f"reldiff={reldiff}"


# ═══════════════════════════════════════════════════════════════════════════════
# C++ auto mode vs C++ quadrature
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoSwitching:
    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", TEMPS_KEV)
    def test_auto_vs_quad(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E, Ep = E_kev * kev, Ep_kev * kev

        quad = cq.ComptonKernelQuadrature(256)
        qres = quad.sigma_E(E, Ep, xi, tau, 1.0)

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        sres = series.sigma_E(E, Ep, xi, tau, 1.0)

        reldiff = abs(sres.value - qres.value) / (abs(qres.value) + 1e-300)
        assert sres.converged
        assert reldiff < 1e-3, (
            f"reldiff={reldiff}: T={T_kev}, method={sres.method_used}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence flags
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvergenceFlags:
    def test_power_series_converges(self):
        E, Ep = 10.0 * kev, 10.5 * kev
        tau = 100.0 * kev / me_c2
        series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeries)
        r = series.sigma_E(E, Ep, 0.0, tau, 1.0)
        assert r.converged
        assert r.terms_used > 0

    def test_asymptotic_converges(self):
        E, Ep = 1.0 * kev, 1.01 * kev
        tau = 0.1 * kev / me_c2
        series = cs.ComptonKernelSeries(cs.SeriesMethod.Asymptotic)
        r = series.sigma_E(E, Ep, 0.0, tau, 1.0)
        assert r.converged

    def test_error_estimates_nonneg(self):
        E, Ep = 1.0 * kev, 1.5 * kev
        tau = 10.0 * kev / me_c2
        series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        r = series.sigma_E(E, Ep, 0.0, tau, 1.0)
        assert r.estimated_abs_error >= 0
        assert r.estimated_rel_error >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Detailed balance
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetailedBalance:
    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi",
        [(1.0, 1.5, 0.0), (10.0, 12.0, 0.3), (1.0, 3.0, -0.5)],
    )
    @pytest.mark.parametrize("T_kev", [1.0, 20.0, 100.0])
    def test_detailed_balance(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E, Ep = E_kev * kev, Ep_kev * kev

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        r_fwd = series.sigma_E(E, Ep, xi, tau, 1.0)
        r_rev = series.sigma_E(Ep, E, xi, tau, 1.0)

        lhs = E * E * r_fwd.value * math.exp(-E / (tau * me_c2))
        rhs = Ep * Ep * r_rev.value * math.exp(-Ep / (tau * me_c2))

        if abs(lhs) < 1e-300 and abs(rhs) < 1e-300:
            return

        reldiff = abs(lhs - rhs) / (max(abs(lhs), abs(rhs)) + 1e-300)
        assert reldiff < 1e-3, f"reldiff={reldiff}"


# ═══════════════════════════════════════════════════════════════════════════════
# Positivity
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositivity:
    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", TEMPS_KEV)
    def test_positive(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E, Ep = E_kev * kev, Ep_kev * kev
        series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        r = series.sigma_E(E, Ep, xi, tau, 1.0)
        assert r.value >= 0, f"Negative: {r.value}"


# ═══════════════════════════════════════════════════════════════════════════════
# Python vs C++ agreement
# ═══════════════════════════════════════════════════════════════════════════════


class TestPythonVsCpp:
    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", TEMPS_KEV)
    def test_agreement(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E, Ep = E_kev * kev, Ep_kev * kev

        cpp_series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        cpp_res = cpp_series.sigma_E(E, Ep, xi, tau, 1.0)

        py_res = py_sigma_E_series(E, Ep, xi, tau, 1.0, method="auto")

        if abs(cpp_res.value) < 1e-300 and abs(py_res.value) < 1e-300:
            return

        reldiff = abs(cpp_res.value - py_res.value) / (
            max(abs(cpp_res.value), abs(py_res.value)) + 1e-300
        )
        assert reldiff < 1e-6, (
            f"Python vs C++ reldiff={reldiff}: T={T_kev}, E={E_kev}, "
            f"Ep={Ep_kev}, xi={xi}"
        )
