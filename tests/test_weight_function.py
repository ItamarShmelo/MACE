"""
Unit tests for weight functions (Planck, Uniform, Wien).

Validates:
  1. weight(E, T) against analytic formulae.
  2. compute_denominator(E_left, E_right, T) against scipy adaptive quadrature.
"""

import sys

import numpy as np
import pytest
from scipy.integrate import quad as scipy_quad

sys.path.insert(0, "cpp_modules")

import _compton_multigroup as cm
from _units import k_boltz, kev_kelvin

CAP_X = 25.0
W0 = CAP_X**3 / np.expm1(CAP_X)


# ---------------------------------------------------------------------------
# 1. weight(E, T)
# ---------------------------------------------------------------------------

class TestWeight:
    """Verify the capped Planck weight function."""

    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 10.0, 24.9])
    def test_below_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.PlanckWeightFunction(cap_x=CAP_X)

        result = wf.weight(E, T)
        expected = x**3 / np.expm1(x)

        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("x", [25.0, 30.0, 50.0])
    def test_at_and_above_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.PlanckWeightFunction(cap_x=CAP_X)

        result = wf.weight(E, T)

        assert result == pytest.approx(W0, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. compute_denominator(E_left, E_right, T)
# ---------------------------------------------------------------------------

class TestComputeDenominator:
    """Verify the denominator integral against scipy adaptive quadrature."""

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_below_cap(self, T_kev):
        """Group entirely below x = 25 threshold."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = 0.1, 5.0
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.PlanckWeightFunction(cap_x=CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref, _ = scipy_quad(lambda x: x**3 / np.expm1(x), x_lo, x_hi)
        expected = kT * ref

        assert computed == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_above_cap(self, T_kev):
        """Group entirely above x = 25 threshold."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = 26.0, 30.0
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.PlanckWeightFunction(cap_x=CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        expected = kT * W0 * (x_hi - x_lo)

        assert computed == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_straddling(self, T_kev):
        """Group spanning the cap threshold."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = 20.0, 30.0
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.PlanckWeightFunction(cap_x=CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref_below, _ = scipy_quad(lambda x: x**3 / np.expm1(x), x_lo, CAP_X)
        expected = kT * (ref_below + W0 * (x_hi - CAP_X))

        assert computed == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. UniformWeightFunction
# ---------------------------------------------------------------------------

class TestUniformWeight:
    """Verify the uniform (flat) weight function."""

    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 10.0, 30.0])
    def test_weight_is_one(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.UniformWeightFunction()

        assert wf.weight(E, T) == 1.0


class TestUniformDenominator:
    """Verify the uniform denominator equals E_right - E_left."""

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize("x_range", [(0.1, 5.0), (1.0, 20.0), (10.0, 50.0)])
    def test_denominator(self, T_kev, x_range):
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = x_range
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.UniformWeightFunction()
        computed = wf.compute_denominator(E_lo, E_hi, T)

        assert computed == pytest.approx(E_hi - E_lo, rel=1e-14)


# ---------------------------------------------------------------------------
# 4. WienWeightFunction
# ---------------------------------------------------------------------------

WIEN_CAP_X = 25.0
WIEN_W0 = WIEN_CAP_X**3 * np.exp(-WIEN_CAP_X)


class TestWienWeight:
    """Verify the capped Wien weight w(E,T) = x^3 * exp(-x)."""

    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 10.0, 24.9])
    def test_below_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.WienWeightFunction(cap_x=WIEN_CAP_X)

        result = wf.weight(E, T)
        expected = x**3 * np.exp(-x)

        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("x", [25.0, 30.0, 50.0])
    def test_at_and_above_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.WienWeightFunction(cap_x=WIEN_CAP_X)

        result = wf.weight(E, T)

        assert result == pytest.approx(WIEN_W0, rel=1e-12)


class TestWienDenominator:
    """Verify the Wien denominator against scipy adaptive quadrature."""

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize("x_range", [(0.1, 5.0), (1.0, 20.0)])
    def test_below_cap(self, T_kev, x_range):
        """Group entirely below x = 25 threshold."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = x_range
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.WienWeightFunction(cap_x=WIEN_CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref, _ = scipy_quad(lambda x: x**3 * np.exp(-x), x_lo, x_hi)
        expected = kT * ref

        assert computed == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_above_cap(self, T_kev):
        """Group entirely above x = 25 threshold."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = 26.0, 30.0
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.WienWeightFunction(cap_x=WIEN_CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        expected = kT * WIEN_W0 * (x_hi - x_lo)

        assert computed == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_straddling(self, T_kev):
        """Group spanning the cap threshold."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = 20.0, 30.0
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.WienWeightFunction(cap_x=WIEN_CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref_below, _ = scipy_quad(lambda x: x**3 * np.exp(-x), x_lo, WIEN_CAP_X)
        expected = kT * (ref_below + WIEN_W0 * (x_hi - WIEN_CAP_X))

        assert computed == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize(
        "x_range", [(1e-6, 1e-5), (0.001, 0.01), (0.01, 0.1)]
    )
    def test_small_x(self, T_kev, x_range):
        """Small-x regime where the closed-form antiderivative loses digits."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = x_range
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.WienWeightFunction(cap_x=WIEN_CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref, _ = scipy_quad(lambda x: x**3 * np.exp(-x), x_lo, x_hi)
        expected = kT * ref

        assert computed == pytest.approx(expected, rel=1e-9, abs=0)


# ---------------------------------------------------------------------------
# 5. peak_energy()
# ---------------------------------------------------------------------------

class TestPeakEnergy:
    """Verify peak_energy(T) for all weight functions."""

    def test_planck_peak_energy(self):
        wf = cm.PlanckWeightFunction(cap_x=25.0)
        T = 10.0 * kev_kelvin
        E_peak = wf.peak_energy(T)
        assert E_peak == pytest.approx(2.821439 * k_boltz * T, rel=1e-6)

    def test_uniform_peak_energy(self):
        wf = cm.UniformWeightFunction()
        assert wf.peak_energy(10.0 * kev_kelvin) is None

    def test_wien_peak_energy(self):
        wf = cm.WienWeightFunction(cap_x=25.0)
        T = 10.0 * kev_kelvin
        E_peak = wf.peak_energy(T)
        assert E_peak == pytest.approx(3.0 * k_boltz * T, rel=1e-6)

    @pytest.mark.parametrize("T_kev", [0.1, 1.0, 10.0, 100.0])
    def test_planck_peak_scales_linearly(self, T_kev):
        """Peak energy should scale linearly with T."""
        wf = cm.PlanckWeightFunction(cap_x=25.0)
        T = T_kev * kev_kelvin
        E_peak = wf.peak_energy(T)
        T2 = 2.0 * T
        E_peak2 = wf.peak_energy(T2)
        assert E_peak2 == pytest.approx(2.0 * E_peak, rel=1e-14)
