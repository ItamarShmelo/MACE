"""
Consolidated validation: series methods vs Gauss-Laguerre quadrature (Q256).

Tests each series evaluation method against the reference Q256 post-IBP
quadrature in the regime where that method is convergent:
  - ComptonPowerSeries (double)           at high T (hot plasma)
  - ComptonPowerSeries (DD)               at high T (hot plasma)
  - ComptonKernelAsymptoticSeries         at low T  (cold plasma)
  - ComptonKernelSolver (adaptive)        across all T

Also validates the temperature derivative dsigma_E_dT for each method
against the same quadrature reference.
"""

import sys

import pytest

sys.path.insert(0, "cpp_modules")

import _compton_kernel_quadrature as cq
from _compton_power_series import ComptonPowerSeries
from _compton_kernel_asymptotic_series import ComptonKernelAsymptoticSeries
from _compton_kernel_solver import ComptonKernelSolver
from _units import kev, kev_kelvin

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

TEMPS_HOT_KEV = [100.0]
TEMPS_COLD_KEV = [0.1, 1.0, 5.0]
TEMPS_ALL_KEV = [0.1, 1.0, 5.0, 20.0, 100.0]

QUAD_REF = cq.ComptonKernelQuadrature(256, cq.QuadratureForm.PostIBP)

QUAD_REL_ERR_SKIP = 1e-5


def _rel_diff(series_val, quad_val):
    return abs(series_val - quad_val) / (abs(quad_val) + 1e-300)


class TestPowerSeriesVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_HOT_KEV)
    def test_double(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonPowerSeries()
        sres = series.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-4, f"reldiff={rd:.2e}"


class TestPowerSeriesHPVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_HOT_KEV)
    def test_double_double(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonPowerSeries(high_precision=True)
        sres = series.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-4, f"reldiff={rd:.2e}"


class TestAsymptoticVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_COLD_KEV)
    def test_low_temp(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonKernelAsymptoticSeries()
        sres = series.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


class TestAsymptoticHPVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_COLD_KEV)
    def test_low_temp(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonKernelAsymptoticSeries(high_precision=True)
        sres = series.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


class TestAsymptoticHPDDvsDouble:
    """Verify DD gives equal or better results than double at cold temps."""

    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_COLD_KEV)
    def test_dd_close_to_double(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        series_dbl = ComptonKernelAsymptoticSeries(high_precision=False)
        series_dd = ComptonKernelAsymptoticSeries(high_precision=True)

        res_dbl = series_dbl.sigma_E(E, Ep, xi, T, 1.0)
        res_dd = series_dd.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(res_dd.value, res_dbl.value)
        assert rd < 1e-3, f"double vs DD reldiff={rd:.2e}"


class TestSolverVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_ALL_KEV)
    def test_solver(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        solver = ComptonKernelSolver()
        sres = solver.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


# ---------------------------------------------------------------------------
# Temperature derivative: dsigma_E_dT
# ---------------------------------------------------------------------------


class TestPowerSeriesDerivativeVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_HOT_KEV)
    def test_double(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonPowerSeries()
        sres = series.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-4, f"reldiff={rd:.2e}"


class TestPowerSeriesHPDerivativeVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_HOT_KEV)
    def test_double_double(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonPowerSeries(high_precision=True)
        sres = series.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-4, f"reldiff={rd:.2e}"


class TestAsymptoticDerivativeVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_COLD_KEV)
    def test_low_temp(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonKernelAsymptoticSeries()
        sres = series.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


class TestAsymptoticHPDerivativeVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_COLD_KEV)
    def test_low_temp(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = ComptonKernelAsymptoticSeries(high_precision=True)
        sres = series.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


class TestSolverDerivativeVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_ALL_KEV)
    def test_solver(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        solver = ComptonKernelSolver()
        sres = solver.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"
