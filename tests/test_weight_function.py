"""
Unit tests for weight functions (Planck, Uniform, Wien).

Validates:
  1. weight(E, T) against analytic formulae.
  2. compute_denominator(E_left, E_right, T) against scipy adaptive quadrature.
"""

import compton_matrix._compton_multigroup as cm
import numpy as np
import pytest
from compton_matrix._units import k_boltz, kev_kelvin
from scipy.integrate import quad as scipy_quad

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
        wf = cm.CappedPlanckWeightFunction(cap_x=CAP_X)

        result = wf.weight(E, T)
        expected = x**3 / np.expm1(x)

        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("x", [25.0, 30.0, 50.0])
    def test_at_and_above_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.CappedPlanckWeightFunction(cap_x=CAP_X)

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

        wf = cm.CappedPlanckWeightFunction(cap_x=CAP_X)
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

        wf = cm.CappedPlanckWeightFunction(cap_x=CAP_X)
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

        wf = cm.CappedPlanckWeightFunction(cap_x=CAP_X)
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
# 4. CappedWienWeightFunction
# ---------------------------------------------------------------------------

WIEN_CAP_X = 25.0
WIEN_W0 = WIEN_CAP_X**3 * np.exp(-WIEN_CAP_X)


class TestWienWeight:
    """Verify the capped Wien weight w(E,T) = x^3 * exp(-x)."""

    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 10.0, 24.9])
    def test_below_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.CappedWienWeightFunction(cap_x=WIEN_CAP_X)

        result = wf.weight(E, T)
        expected = x**3 * np.exp(-x)

        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("x", [25.0, 30.0, 50.0])
    def test_at_and_above_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = cm.CappedWienWeightFunction(cap_x=WIEN_CAP_X)

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

        wf = cm.CappedWienWeightFunction(cap_x=WIEN_CAP_X)
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

        wf = cm.CappedWienWeightFunction(cap_x=WIEN_CAP_X)
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

        wf = cm.CappedWienWeightFunction(cap_x=WIEN_CAP_X)
        computed = wf.compute_denominator(E_lo, E_hi, T)

        ref_below, _ = scipy_quad(lambda x: x**3 * np.exp(-x), x_lo, WIEN_CAP_X)
        expected = kT * (ref_below + WIEN_W0 * (x_hi - WIEN_CAP_X))

        assert computed == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize("x_range", [(1e-6, 1e-5), (0.001, 0.01), (0.01, 0.1)])
    def test_small_x(self, T_kev, x_range):
        """Small-x regime where the closed-form antiderivative loses digits."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = x_range
        E_lo, E_hi = x_lo * kT, x_hi * kT

        wf = cm.CappedWienWeightFunction(cap_x=WIEN_CAP_X)
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
        wf = cm.CappedPlanckWeightFunction(cap_x=25.0)
        T = 10.0 * kev_kelvin
        E_peak = wf.peak_energy(T)
        assert E_peak == pytest.approx(2.821439 * k_boltz * T, rel=1e-6)

    def test_uniform_peak_energy(self):
        wf = cm.UniformWeightFunction()
        assert wf.peak_energy(10.0 * kev_kelvin) is None

    def test_wien_peak_energy(self):
        wf = cm.CappedWienWeightFunction(cap_x=25.0)
        T = 10.0 * kev_kelvin
        E_peak = wf.peak_energy(T)
        assert E_peak == pytest.approx(3.0 * k_boltz * T, rel=1e-6)

    @pytest.mark.parametrize("T_kev", [0.1, 1.0, 10.0, 100.0])
    def test_planck_peak_scales_linearly(self, T_kev):
        """Peak energy should scale linearly with T."""
        wf = cm.CappedPlanckWeightFunction(cap_x=25.0)
        T = T_kev * kev_kelvin
        E_peak = wf.peak_energy(T)
        T2 = 2.0 * T
        E_peak2 = wf.peak_energy(T2)
        assert E_peak2 == pytest.approx(2.0 * E_peak, rel=1e-14)


# ---------------------------------------------------------------------------
# 6. d_weight_dT
# ---------------------------------------------------------------------------


class TestDWeightDT:
    """Verify d_weight_dT against central finite differences."""

    @pytest.mark.parametrize("wf_cls,cap_x", [
        (cm.CappedPlanckWeightFunction, CAP_X),
        (cm.CappedWienWeightFunction, WIEN_CAP_X),
    ])
    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 5.0, 10.0, 20.0])
    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_below_cap_fd(self, wf_cls, cap_x, x, T_kev):
        """Compare d_weight_dT against central FD well below the cap."""
        T = T_kev * kev_kelvin
        E = x * k_boltz * T
        wf = wf_cls(cap_x=cap_x)

        h = T * 1e-6
        fd = (wf.weight(E, T + h) - wf.weight(E, T - h)) / (2 * h)
        analytic = wf.d_weight_dT(E, T)

        assert analytic == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("wf_cls,cap_x", [
        (cm.CappedPlanckWeightFunction, CAP_X),
        (cm.CappedWienWeightFunction, WIEN_CAP_X),
    ])
    @pytest.mark.parametrize("x", [26.0, 30.0, 50.0])
    def test_above_cap_zero(self, wf_cls, cap_x, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = wf_cls(cap_x=cap_x)
        assert wf.d_weight_dT(E, T) == pytest.approx(0.0, abs=1e-30)

    def test_uniform_zero(self):
        wf = cm.UniformWeightFunction()
        T = 10.0 * kev_kelvin
        E = 5.0 * k_boltz * T
        assert wf.d_weight_dT(E, T) == pytest.approx(0.0, abs=1e-30)

    @pytest.mark.parametrize("wf_cls,cap_x", [
        (cm.CappedPlanckWeightFunction, CAP_X),
        (cm.CappedWienWeightFunction, WIEN_CAP_X),
    ])
    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 5.0, 10.0, 20.0])
    def test_d_log_weight_dT_cross_check(self, wf_cls, cap_x, x):
        """Verify d_log_weight_dT(E,T) * w(E,T) == d_weight_dT(E,T)."""
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = wf_cls(cap_x=cap_x)

        w = wf.weight(E, T)
        dlnw = wf.d_log_weight_dT(E, T)
        dw = wf.d_weight_dT(E, T)

        assert dlnw * w == pytest.approx(dw, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. d_denominator_dT
# ---------------------------------------------------------------------------


class TestDDenominatorDT:
    """Verify d_denominator_dT against FD and integral of d_weight_dT."""

    @pytest.mark.parametrize("wf_cls,cap_x", [
        (cm.CappedPlanckWeightFunction, CAP_X),
        (cm.CappedWienWeightFunction, WIEN_CAP_X),
    ])
    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize("x_range,label", [
        ((0.1, 5.0), "below"),
        ((26.0, 30.0), "above"),
        ((20.0, 30.0), "straddling"),
    ])
    def test_fd(self, wf_cls, cap_x, T_kev, x_range, label):
        """Compare d_denominator_dT against central FD."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = x_range
        E_lo, E_hi = x_lo * kT, x_hi * kT
        wf = wf_cls(cap_x=cap_x)

        h = T * 1e-6
        D_plus = wf.compute_denominator(E_lo, E_hi, T + h)
        D_minus = wf.compute_denominator(E_lo, E_hi, T - h)
        fd = (D_plus - D_minus) / (2 * h)

        analytic = wf.d_denominator_dT(E_lo, E_hi, T)

        if label == "above":
            assert analytic == pytest.approx(0.0, abs=1e-30)
            assert fd == pytest.approx(0.0, abs=max(abs(D_plus) * 1e-6, 1e-30))
        else:
            assert analytic == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("wf_cls,cap_x", [
        (cm.CappedPlanckWeightFunction, CAP_X),
        (cm.CappedWienWeightFunction, WIEN_CAP_X),
    ])
    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize("x_range", [
        (0.1, 5.0),
        (20.0, 30.0),
    ])
    def test_integral_cross_check(self, wf_cls, cap_x, T_kev, x_range):
        """Cross-check d_denominator_dT against integral of d_weight_dT."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        x_lo, x_hi = x_range
        E_lo, E_hi = x_lo * kT, x_hi * kT
        wf = wf_cls(cap_x=cap_x)

        E_cap = cap_x * kT
        points = [E_cap] if E_lo < E_cap < E_hi else []
        integral, _ = scipy_quad(
            lambda E: wf.d_weight_dT(E, T), E_lo, E_hi, points=points
        )

        analytic = wf.d_denominator_dT(E_lo, E_hi, T)

        assert analytic == pytest.approx(integral, rel=1e-8)

    def test_uniform_zero(self):
        wf = cm.UniformWeightFunction()
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        E_lo, E_hi = 0.1 * kT, 5.0 * kT
        assert wf.d_denominator_dT(E_lo, E_hi, T) == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# 8. Shifted WienWeightFunction
# ---------------------------------------------------------------------------

from compton_matrix._units import kev

SHIFTED_BOUNDS = [0.1 * kev, 1.0 * kev, 10.0 * kev, 100.0 * kev]


def _make_shifted_wien():
    return cm.WienWeightFunction(group_boundaries=SHIFTED_BOUNDS)


def _make_shifted_planck():
    return cm.PlanckWeightFunction(cap_x=CAP_X, group_boundaries=SHIFTED_BOUNDS)


def _find_x_lo_py(E, T, bounds):
    """Python equivalent of find_x_lo: binary search for group lower boundary."""
    import bisect
    kT = k_boltz * T
    idx = bisect.bisect_right(bounds, E) - 1
    idx = max(0, min(idx, len(bounds) - 2))
    return bounds[idx] / kT


class TestShiftedWienWeight:
    """Verify shifted Wien weight: w(E,T) = x^3 exp(-(x - x_lo))."""

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 5.0, 10.0])
    def test_moderate_x(self, x):
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        E = x * kT
        wf = _make_shifted_wien()

        x_lo = _find_x_lo_py(E, T, SHIFTED_BOUNDS)
        expected = x**3 * np.exp(-(x - x_lo))
        assert wf.weight(E, T) == pytest.approx(expected, rel=1e-12)

    def test_large_x_no_underflow(self):
        """At large x the unshifted Wien would underflow; shifted doesn't."""
        large_bounds = [0.1 * kev, 1.0 * kev, 10.0 * kev, 100.0 * kev,
                        1000.0 * kev, 10000.0 * kev]
        wf = cm.WienWeightFunction(group_boundaries=large_bounds)
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        for x in [100.0, 500.0]:
            E = x * kT
            w = wf.weight(E, T)
            assert np.isfinite(w)
            assert w > 0

    def test_weight_in_second_group(self):
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        x = 30.0
        E = x * kT
        wf = _make_shifted_wien()
        x_lo = _find_x_lo_py(E, T, SHIFTED_BOUNDS)
        expected = x**3 * np.exp(-(x - x_lo))
        assert wf.weight(E, T) == pytest.approx(expected, rel=1e-12)


class TestShiftedWienDenominator:
    """Verify shifted Wien denominator against scipy quadrature."""

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_vs_scipy(self, T_kev):
        T = T_kev * kev_kelvin
        wf = _make_shifted_wien()

        for i in range(len(SHIFTED_BOUNDS) - 1):
            E_lo = SHIFTED_BOUNDS[i]
            E_hi = SHIFTED_BOUNDS[i + 1]
            computed = wf.compute_denominator(E_lo, E_hi, T)
            ref, _ = scipy_quad(lambda E: wf.weight(E, T), E_lo, E_hi)
            assert computed == pytest.approx(ref, rel=1e-10), (
                f"group {i}, T={T_kev} keV"
            )

    @pytest.mark.parametrize("T_kev", [1.0, 10.0])
    def test_narrow_groups_taylor(self, T_kev):
        """Small-delta groups exercise the Taylor branch of I_n."""
        T = T_kev * kev_kelvin
        kT = k_boltz * T
        narrow_bounds = [0.1 * kev, 0.11 * kev, 0.12 * kev]
        wf = cm.WienWeightFunction(group_boundaries=narrow_bounds)

        for i in range(len(narrow_bounds) - 1):
            E_lo = narrow_bounds[i]
            E_hi = narrow_bounds[i + 1]
            computed = wf.compute_denominator(E_lo, E_hi, T)
            ref, _ = scipy_quad(lambda E: wf.weight(E, T), E_lo, E_hi)
            assert computed == pytest.approx(ref, rel=1e-10), (
                f"narrow group {i}, T={T_kev} keV"
            )


# ---------------------------------------------------------------------------
# 9. Shifted PlanckWeightFunction
# ---------------------------------------------------------------------------


class TestShiftedPlanckWeight:
    """Verify shifted Planck weight in all three regimes."""

    @pytest.mark.parametrize("x", [0.5, 1.0, 5.0, 10.0, 20.0, 24.9])
    def test_below_cap(self, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = _make_shifted_planck()
        expected = x**3 / np.expm1(x)
        assert wf.weight(E, T) == pytest.approx(expected, rel=1e-12)

    def test_above_cap_shifted(self):
        """In a group fully above cap_x, uses shifted Wien."""
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        x = 80.0
        E = x * kT
        wf = _make_shifted_planck()
        x_lo = SHIFTED_BOUNDS[2] / kT
        expected = x**3 * np.exp(-(x - x_lo))
        assert wf.weight(E, T) == pytest.approx(expected, rel=1e-12)

    def test_straddling_unshifted_wien(self):
        """In a straddling group (x_lo < cap_x), uses unshifted Wien."""
        T = 1.0 * kev_kelvin
        kT = k_boltz * T
        x_lo_dim = SHIFTED_BOUNDS[0] / kT
        if x_lo_dim >= CAP_X:
            pytest.skip("x_lo >= cap_x at this T")
        x = CAP_X + 1.0
        E = x * kT
        wf = _make_shifted_planck()
        expected = x**3 * np.exp(-x)
        assert wf.weight(E, T) == pytest.approx(expected, rel=1e-12)


class TestShiftedPlanckDenominator:
    """Verify shifted Planck denominator against scipy quadrature."""

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_vs_scipy(self, T_kev):
        T = T_kev * kev_kelvin
        wf = _make_shifted_planck()

        for i in range(len(SHIFTED_BOUNDS) - 1):
            E_lo = SHIFTED_BOUNDS[i]
            E_hi = SHIFTED_BOUNDS[i + 1]
            computed = wf.compute_denominator(E_lo, E_hi, T)
            ref, _ = scipy_quad(lambda E: wf.weight(E, T), E_lo, E_hi)
            assert computed == pytest.approx(ref, rel=1e-10), (
                f"group {i}, T={T_kev} keV"
            )


# ---------------------------------------------------------------------------
# 10. Peak energy for shifted classes
# ---------------------------------------------------------------------------


class TestShiftedPeakEnergy:
    def test_wien_peak(self):
        wf = _make_shifted_wien()
        T = 10.0 * kev_kelvin
        assert wf.peak_energy(T) == pytest.approx(3.0 * k_boltz * T, rel=1e-6)

    def test_planck_peak(self):
        wf = _make_shifted_planck()
        T = 10.0 * kev_kelvin
        assert wf.peak_energy(T) == pytest.approx(2.821439 * k_boltz * T, rel=1e-6)


# ---------------------------------------------------------------------------
# 11. Group boundary behavior
# ---------------------------------------------------------------------------


class TestGroupBoundaryBehavior:
    """Verify weight jumps and half-open group convention."""

    def test_weight_jump_at_internal_boundary(self):
        """Weight jumps upward by exp(x_g - x_{g-1}) at group boundaries."""
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        wf = _make_shifted_wien()
        E_boundary = SHIFTED_BOUNDS[1]
        eps = E_boundary * 1e-12
        w_below = wf.weight(E_boundary - eps, T)
        w_at = wf.weight(E_boundary, T)
        x_prev = SHIFTED_BOUNDS[0] / kT
        x_curr = SHIFTED_BOUNDS[1] / kT
        expected_ratio = np.exp(x_curr - x_prev)
        actual_ratio = w_at / w_below
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_half_open_convention(self):
        """E exactly at boundary uses the group that starts there."""
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        wf = _make_shifted_wien()
        E = SHIFTED_BOUNDS[1]
        x = E / kT
        x_lo = SHIFTED_BOUNDS[1] / kT
        expected = x**3 * np.exp(-(x - x_lo))
        assert expected == pytest.approx(x**3, rel=1e-12)
        assert wf.weight(E, T) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 12. Invalid constructor inputs
# ---------------------------------------------------------------------------


class TestInvalidConstructorInputs:
    def test_fewer_than_2_boundaries(self):
        with pytest.raises(ValueError):
            cm.WienWeightFunction(group_boundaries=[1.0])

    def test_non_increasing_boundaries(self):
        with pytest.raises(ValueError):
            cm.WienWeightFunction(group_boundaries=[1.0, 1.0, 2.0])

    def test_nan_boundary(self):
        with pytest.raises(ValueError):
            cm.WienWeightFunction(group_boundaries=[1.0, float("nan"), 3.0])

    def test_inf_boundary(self):
        with pytest.raises(ValueError):
            cm.WienWeightFunction(group_boundaries=[1.0, float("inf"), 3.0])

    def test_negative_boundary(self):
        with pytest.raises(ValueError):
            cm.WienWeightFunction(group_boundaries=[-1.0, 1.0, 2.0])

    def test_planck_non_positive_cap_x(self):
        with pytest.raises(ValueError):
            cm.PlanckWeightFunction(cap_x=0.0, group_boundaries=[1.0, 2.0])

    def test_planck_nan_cap_x(self):
        with pytest.raises(ValueError):
            cm.PlanckWeightFunction(
                cap_x=float("nan"), group_boundaries=[1.0, 2.0]
            )

    def test_planck_inf_cap_x(self):
        with pytest.raises(ValueError):
            cm.PlanckWeightFunction(
                cap_x=float("inf"), group_boundaries=[1.0, 2.0]
            )


# ---------------------------------------------------------------------------
# 13. Exhaustive derivative FD tests for shifted classes
# ---------------------------------------------------------------------------


class TestShiftedDWeightDT:
    """Verify d_weight_dT for shifted classes against central FD."""

    @pytest.mark.parametrize("wf_factory", [
        _make_shifted_wien,
        _make_shifted_planck,
    ], ids=["Wien", "Planck"])
    @pytest.mark.parametrize("x", [
        0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 24.9,
        25.1, 30.0, 50.0, 100.0, 200.0, 500.0,
    ])
    @pytest.mark.parametrize("T_kev", [0.1, 1.0, 10.0, 100.0])
    def test_fd(self, wf_factory, x, T_kev):
        T = T_kev * kev_kelvin
        E = x * k_boltz * T
        wf = wf_factory()

        h = T * 1e-6
        w_plus = wf.weight(E, T + h)
        w_minus = wf.weight(E, T - h)
        if abs(w_plus) < 1e-300 and abs(w_minus) < 1e-300:
            pytest.skip("weight underflows at this (x, T)")
        fd = (w_plus - w_minus) / (2 * h)
        analytic = wf.d_weight_dT(E, T)

        assert analytic == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("wf_factory", [
        _make_shifted_wien,
        _make_shifted_planck,
    ], ids=["Wien", "Planck"])
    @pytest.mark.parametrize("x", [
        0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 24.9,
        25.1, 30.0, 50.0, 100.0, 200.0, 500.0,
    ])
    def test_d_log_cross_check(self, wf_factory, x):
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T
        wf = wf_factory()

        w = wf.weight(E, T)
        dlnw = wf.d_log_weight_dT(E, T)
        dw = wf.d_weight_dT(E, T)

        assert dlnw * w == pytest.approx(dw, rel=1e-12)


class TestShiftedDDenominatorDT:
    """Verify d_denominator_dT for shifted classes against FD and scipy."""

    @pytest.mark.parametrize("wf_factory", [
        _make_shifted_wien,
        _make_shifted_planck,
    ], ids=["Wien", "Planck"])
    @pytest.mark.parametrize("T_kev", [0.1, 1.0, 10.0, 100.0])
    def test_fd_all_groups(self, wf_factory, T_kev):
        """FD check over every group in SHIFTED_BOUNDS."""
        T = T_kev * kev_kelvin
        wf = wf_factory()

        for i in range(len(SHIFTED_BOUNDS) - 1):
            E_lo = SHIFTED_BOUNDS[i]
            E_hi = SHIFTED_BOUNDS[i + 1]
            h = T * 1e-6
            D_plus = wf.compute_denominator(E_lo, E_hi, T + h)
            D_minus = wf.compute_denominator(E_lo, E_hi, T - h)
            fd = (D_plus - D_minus) / (2 * h)

            analytic = wf.d_denominator_dT(E_lo, E_hi, T)

            assert analytic == pytest.approx(fd, rel=1e-5), (
                f"group {i}, T={T_kev} keV"
            )

    @pytest.mark.parametrize("wf_factory", [
        _make_shifted_wien,
        _make_shifted_planck,
    ], ids=["Wien", "Planck"])
    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_integral_cross_check(self, wf_factory, T_kev):
        """Cross-check d_denominator_dT against integral of d_weight_dT."""
        T = T_kev * kev_kelvin
        wf = wf_factory()

        for i in range(len(SHIFTED_BOUNDS) - 1):
            E_lo = SHIFTED_BOUNDS[i]
            E_hi = SHIFTED_BOUNDS[i + 1]
            integral, _ = scipy_quad(
                lambda E: wf.d_weight_dT(E, T), E_lo, E_hi
            )
            analytic = wf.d_denominator_dT(E_lo, E_hi, T)

            assert analytic == pytest.approx(integral, rel=1e-8), (
                f"group {i}, T={T_kev} keV"
            )


class TestShiftedDerivativeConsistency:
    """Additional consistency checks for shifted weight implementations."""

    def test_regime_transition_continuity(self):
        """At x = cap_x, Planck and Wien weights nearly match."""
        T = 10.0 * kev_kelvin
        kT = k_boltz * T
        wf = _make_shifted_planck()
        eps = 1e-8
        E_below = (CAP_X - eps) * kT
        E_above = (CAP_X + eps) * kT
        w_below = wf.weight(E_below, T)
        w_above = wf.weight(E_above, T)
        rel_jump = abs(w_above - w_below) / max(abs(w_below), 1e-300)
        assert rel_jump < 1e-6

    @pytest.mark.parametrize("x", [50.0, 100.0])
    def test_large_x_stability(self, x):
        """At large x, shifted weight and derivatives are finite and nonzero."""
        large_bounds = [0.1 * kev, 1.0 * kev, 10.0 * kev, 100.0 * kev,
                        1000.0 * kev, 10000.0 * kev]
        wf = cm.WienWeightFunction(group_boundaries=large_bounds)
        T = 10.0 * kev_kelvin
        E = x * k_boltz * T

        w = wf.weight(E, T)
        dw = wf.d_weight_dT(E, T)
        dlnw = wf.d_log_weight_dT(E, T)

        assert np.isfinite(w) and w > 0
        assert np.isfinite(dw)
        assert np.isfinite(dlnw)
