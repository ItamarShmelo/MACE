"""
Unit tests for PlanckWeightFunction.

Validates:
  1. weight(E, T) against the analytic formula x^3/(e^x - 1) and the cap constant.
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
        wf = cm.PlanckWeightFunction(CAP_X)

        result = wf.weight(E, T)
        expected = x**3 / np.expm1(x)

        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("x", [25.0, 30.0, 50.0])
    def test_at_and_above_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.PlanckWeightFunction(CAP_X)

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

        wf = cm.PlanckWeightFunction(CAP_X)
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

        wf = cm.PlanckWeightFunction(CAP_X)
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

        wf = cm.PlanckWeightFunction(CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref_below, _ = scipy_quad(lambda x: x**3 / np.expm1(x), x_lo, CAP_X)
        expected = kT * (ref_below + W0 * (x_hi - CAP_X))

        assert computed == pytest.approx(expected, rel=1e-10)
