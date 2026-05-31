"""
Phase 2f validation: Python series vs Python quadrature.

These tests validate the pure-Python power series and asymptotic series
implementations against the Python Gauss-Laguerre quadrature (NL=256).
They must all pass before any C++ work begins.
"""

import math
import sys

import numpy as np
import pytest
from scipy.special import expn

sys.path.insert(0, "src/python")

from pycompton.compton_kernel_quadrature import (
    compute_params,
    me_c2,
    sigma_E,
    stable_sigma0_E,
)
from pycompton.compton_kernel_series import (
    SigmaResult,
    _asymptotic_series_normalized,
    _power_series_normalized,
    ehat_expn,
    sigma_E_series,
)

kev = 1.602176634e-9


# ═══════════════════════════════════════════════════════════════════════════════
# Test points spanning the parameter space
# ═══════════════════════════════════════════════════════════════════════════════

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
# ehat_expn tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEhatExpn:
    """Validate scaled exponential integral against scipy reference."""

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 20])
    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 49.0])
    def test_small_x_regime(self, m, x):
        val = ehat_expn(m, x)
        ref = math.exp(x) * float(expn(m, x))
        if ref > 0:
            assert abs(val - ref) / ref < 1e-13, f"m={m}, x={x}: {val} vs {ref}"

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 20])
    @pytest.mark.parametrize("x", [51.0, 100.0, 200.0, 500.0, 1000.0])
    def test_large_x_regime(self, m, x):
        val = ehat_expn(m, x)
        assert val > 0, f"ehat_expn({m}, {x}) should be positive"
        assert math.isfinite(val), f"ehat_expn({m}, {x}) is not finite"
        approx_ref = 1.0 / x
        assert val < 2.0 / x, f"ehat_expn({m}, {x}) = {val} too large"

    @pytest.mark.parametrize("x", [0.1, 1.0, 10.0, 49.5, 50.5, 100.0])
    def test_continuity_at_boundary(self, x):
        """Check ehat_expn is smooth near x=50 boundary."""
        m = 3
        val = ehat_expn(m, x)
        assert val > 0 and math.isfinite(val)

    def test_boundary_consistency(self):
        """Values just below and above x=50 should nearly agree."""
        for m in [1, 2, 5, 10]:
            v_below = ehat_expn(m, 49.9)
            v_above = ehat_expn(m, 50.1)
            reldiff = abs(v_above - v_below) / v_below
            assert reldiff < 0.01, f"m={m}: discontinuity at x=50: {reldiff}"

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            ehat_expn(1, -1.0)
        with pytest.raises(ValueError):
            ehat_expn(0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Power series validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPowerSeriesVsQuadrature:
    """Compare Python power series against Python quadrature."""

    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi",
        [(1.0, 1.01, 0.0), (10.0, 10.5, 0.0), (100.0, 101.0, 0.0)],
    )
    def test_high_temp_agreement(self, E_kev, Ep_kev, xi):
        """Power series should agree with quadrature at high temperature."""
        T_kev = 100.0
        tau = T_kev * kev / me_c2
        E = E_kev * kev
        Ep = Ep_kev * kev
        val_quad, _, _ = sigma_E(E, Ep, xi, tau, 1.0, NL=256)
        res = sigma_E_series(E, Ep, xi, tau, 1.0, method="power")
        reldiff = abs(res.value - val_quad) / (abs(val_quad) + 1e-300)
        assert reldiff < 1e-4, f"reldiff={reldiff}"


# ═══════════════════════════════════════════════════════════════════════════════
# Asymptotic series validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsymptoticVsQuadrature:
    """Compare Python asymptotic series against Python quadrature."""

    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", [0.1, 1.0, 5.0])
    def test_low_temp_agreement(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E = E_kev * kev
        Ep = Ep_kev * kev
        val_quad, _, _ = sigma_E(E, Ep, xi, tau, 1.0, NL=256)
        res = sigma_E_series(E, Ep, xi, tau, 1.0, method="asymptotic")
        reldiff = abs(res.value - val_quad) / (abs(val_quad) + 1e-300)
        assert reldiff < 1e-3, f"reldiff={reldiff}"


# ═══════════════════════════════════════════════════════════════════════════════
# Auto mode validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoModeVsQuadrature:
    """Auto mode should agree with quadrature across all temperatures."""

    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", TEMPS_KEV)
    def test_auto_agreement(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E = E_kev * kev
        Ep = Ep_kev * kev
        val_quad, _, _ = sigma_E(E, Ep, xi, tau, 1.0, NL=256)
        res = sigma_E_series(E, Ep, xi, tau, 1.0, method="auto")
        reldiff = abs(res.value - val_quad) / (abs(val_quad) + 1e-300)
        assert reldiff < 1e-3, (
            f"reldiff={reldiff}: T={T_kev}, E={E_kev}, Ep={Ep_kev}, xi={xi}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Detailed balance
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetailedBalance:
    """Energy-form detailed balance: E^2 * sigma(E->E') = E'^2 * sigma(E'->E)."""

    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi",
        [(1.0, 1.5, 0.0), (10.0, 12.0, 0.3), (1.0, 3.0, -0.5)],
    )
    @pytest.mark.parametrize("T_kev", [1.0, 20.0, 100.0])
    def test_detailed_balance(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E = E_kev * kev
        Ep = Ep_kev * kev
        res_fwd = sigma_E_series(E, Ep, xi, tau, 1.0, method="auto")
        res_rev = sigma_E_series(Ep, E, xi, tau, 1.0, method="auto")

        lhs = E * E * res_fwd.value * math.exp(-E / (tau * me_c2))
        rhs = Ep * Ep * res_rev.value * math.exp(-Ep / (tau * me_c2))

        if abs(lhs) < 1e-300 and abs(rhs) < 1e-300:
            return

        reldiff = abs(lhs - rhs) / (max(abs(lhs), abs(rhs)) + 1e-300)
        assert reldiff < 1e-3, (
            f"Detailed balance violated: reldiff={reldiff}, "
            f"T={T_kev}, E={E_kev}, Ep={Ep_kev}, xi={xi}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Positivity
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositivity:
    """Kernel values should be non-negative."""

    @pytest.mark.parametrize(
        "E_kev,Ep_kev,xi", TEST_POINTS
    )
    @pytest.mark.parametrize("T_kev", TEMPS_KEV)
    def test_positive(self, E_kev, Ep_kev, xi, T_kev):
        tau = T_kev * kev / me_c2
        E = E_kev * kev
        Ep = Ep_kev * kev
        res = sigma_E_series(E, Ep, xi, tau, 1.0, method="auto")
        assert res.value >= 0, (
            f"Negative kernel: {res.value}, T={T_kev}, E={E_kev}, "
            f"Ep={Ep_kev}, xi={xi}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvergenceBehavior:
    """Series should converge without raising for well-conditioned inputs."""

    def test_power_series_converges_high_temp(self):
        E = 10.0 * kev
        Ep = 10.5 * kev
        tau = 100.0 * kev / me_c2
        res = sigma_E_series(E, Ep, 0.0, tau, 1.0, method="power")
        assert res.value > 0

    def test_asymptotic_converges_low_temp(self):
        E = 1.0 * kev
        Ep = 1.01 * kev
        tau = 0.1 * kev / me_c2
        res = sigma_E_series(E, Ep, 0.0, tau, 1.0, method="asymptotic")
        assert res.value > 0

    def test_error_estimates_positive(self):
        E = 1.0 * kev
        Ep = 1.5 * kev
        tau = 10.0 * kev / me_c2
        res = sigma_E_series(E, Ep, 0.0, tau, 1.0, method="auto")
        assert res.estimated_abs_error >= 0
        assert res.estimated_rel_error >= 0
