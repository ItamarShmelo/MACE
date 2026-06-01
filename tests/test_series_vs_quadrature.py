"""
Consolidated validation: series methods vs Gauss-Laguerre quadrature (Q256).

Tests each series evaluation method against the reference Q256 post-IBP
quadrature in the regime where that method is convergent:
  - PowerSeries (double)              at high T (hot plasma)
  - PowerSeriesHighPrecision (DD)     at high T (hot plasma)
  - Asymptotic                        at low T  (cold plasma)
  - Auto                              across all T

Also validates the temperature derivative dsigma_E_dT for each method
against the same quadrature reference.
"""

import sys

import pytest

sys.path.insert(0, "cpp_modules")

import _compton_kernel_quadrature as cq
import _compton_kernel_series as cs
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

        series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeries)
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

        series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeriesHighPrecision)
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

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Asymptotic)
        sres = series.sigma_E(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


class TestAutoVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_ALL_KEV)
    def test_auto(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        sres = series.sigma_E(E, Ep, xi, T, 1.0)

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

        series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeries)
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

        series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeriesHighPrecision)
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

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Asymptotic)
        sres = series.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"


class TestAutoDerivativeVsQuadrature:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi", TEST_POINTS)
    @pytest.mark.parametrize("T_kev", TEMPS_ALL_KEV)
    def test_auto(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin

        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > QUAD_REL_ERR_SKIP:
            pytest.skip("quadrature reference unreliable")

        series = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
        sres = series.dsigma_E_dT(E, Ep, xi, T, 1.0)

        rd = _rel_diff(sres.value, qres.value)
        assert rd < 1e-3, f"reldiff={rd:.2e}"
