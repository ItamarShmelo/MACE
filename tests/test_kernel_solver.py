"""
Tests for ComptonKernelSolver adaptive dispatch.

Validates that the solver produces the correct result across all dispatch
regimes (asymptotic, double power series, Q64-accepted, DD fallback) for
both sigma_E and dsigma_E_dT, and that custom threshold parameters work.
"""

import sys

import pytest

sys.path.insert(0, "cpp_modules")

import _compton_kernel_quadrature as cq
import _compton_kernel_series as cs
from _compton_kernel_solver import ComptonKernelSolver
from _units import kev, kev_kelvin, me_c2

QUAD_REF = cq.ComptonKernelQuadrature(256, cq.QuadratureForm.PostIBP)
SERIES_AUTO = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
SOLVER = ComptonKernelSolver()


def _rel_diff(a, b):
    return abs(a - b) / (abs(b) + 1e-300)


# ── Asymptotic regime: low T, moderate gamma ─────────────────────────────

ASYMP_POINTS = [
    (10.0, 10.5, 0.0, 0.5),
    (10.0, 9.5, 0.0, 1.0),
    (50.0, 51.0, 0.3, 0.1),
    (100.0, 101.0, -0.5, 5.0),
]


class TestAsymptoticRegime:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", ASYMP_POINTS)
    def test_sigma_E_vs_q256(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > 1e-5:
            pytest.skip("quadrature reference unreliable")
        sres = SOLVER.sigma_E(E, Ep, xi, T, 1.0)
        assert _rel_diff(sres.value, qres.value) < 1e-3

    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", ASYMP_POINTS)
    def test_dsigma_E_dT_vs_q256(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > 1e-5:
            pytest.skip("quadrature reference unreliable")
        sres = SOLVER.dsigma_E_dT(E, Ep, xi, T, 1.0)
        assert _rel_diff(sres.value, qres.value) < 1e-3


# ── Double power series regime: high gamma, hot plasma ───────────────────

DOUBLE_PS_POINTS = [
    (100.0, 101.0, 0.0, 100.0),
    (50.0, 55.0, 0.5, 50.0),
    (20.0, 18.0, -0.3, 80.0),
    (30.0, 35.0, 0.7, 200.0),
]


class TestDoublePowerSeriesRegime:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", DOUBLE_PS_POINTS)
    def test_sigma_E_vs_q256(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        qres = QUAD_REF.sigma_E(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > 1e-5:
            pytest.skip("quadrature reference unreliable")
        sres = SOLVER.sigma_E(E, Ep, xi, T, 1.0)
        assert _rel_diff(sres.value, qres.value) < 1e-4

    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", DOUBLE_PS_POINTS)
    def test_dsigma_E_dT_vs_q256(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        qres = QUAD_REF.dsigma_E_dT(E, Ep, xi, T, 1.0)
        if qres.estimated_rel_error > 1e-5:
            pytest.skip("quadrature reference unreliable")
        sres = SOLVER.dsigma_E_dT(E, Ep, xi, T, 1.0)
        assert _rel_diff(sres.value, qres.value) < 1e-4


# ── Q64-accepted regime: DD regime where quadrature converges ────────────
# gamma < 0.02 (E < ~10 keV), moderate T, not extreme forward scattering

Q64_ACCEPTED_POINTS = [
    (5.0, 5.5, 0.0, 30.0),
    (3.0, 2.5, 0.3, 40.0),
    (1.0, 0.9, 0.5, 25.0),
    (7.0, 8.0, -0.5, 20.0),
]


class TestQ64AcceptedRegime:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", Q64_ACCEPTED_POINTS)
    def test_sigma_E_matches_auto(self, E_kev, Ep_kev, xi, T_kev):
        """Solver should be at least as accurate as Series(Auto) here."""
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        auto_res = SERIES_AUTO.sigma_E(E, Ep, xi, T, 1.0)
        solver_res = SOLVER.sigma_E(E, Ep, xi, T, 1.0)
        assert _rel_diff(solver_res.value, auto_res.value) < 1e-5

    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", Q64_ACCEPTED_POINTS)
    def test_dsigma_E_dT_matches_auto(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        auto_res = SERIES_AUTO.dsigma_E_dT(E, Ep, xi, T, 1.0)
        solver_res = SOLVER.dsigma_E_dT(E, Ep, xi, T, 1.0)
        assert _rel_diff(solver_res.value, auto_res.value) < 1e-5


# ── Solver vs Series(Auto) across the full parameter space ───────────────

FULL_SPACE_POINTS = [
    # Asymptotic regime
    (10.0, 10.5, 0.0, 0.5),
    (50.0, 51.0, 0.3, 1.0),
    # Double PS regime
    (100.0, 101.0, 0.0, 100.0),
    (20.0, 18.0, -0.3, 50.0),
    # DD regime (Q64 likely accepted)
    (5.0, 5.5, 0.0, 30.0),
    (3.0, 2.5, 0.3, 40.0),
    # DD regime (potentially harder)
    (1.0, 0.7, 0.7, 50.0),
    (0.5, 0.5, 0.0, 20.0),
]


class TestSolverVsAuto:
    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", FULL_SPACE_POINTS)
    def test_sigma_E(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        auto_res = SERIES_AUTO.sigma_E(E, Ep, xi, T, 1.0)
        solver_res = SOLVER.sigma_E(E, Ep, xi, T, 1.0)
        rd = _rel_diff(solver_res.value, auto_res.value)
        assert rd < 1e-4, f"reldiff={rd:.2e}"

    @pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", FULL_SPACE_POINTS)
    def test_dsigma_E_dT(self, E_kev, Ep_kev, xi, T_kev):
        E, Ep, T = E_kev * kev, Ep_kev * kev, T_kev * kev_kelvin
        auto_res = SERIES_AUTO.dsigma_E_dT(E, Ep, xi, T, 1.0)
        solver_res = SOLVER.dsigma_E_dT(E, Ep, xi, T, 1.0)
        rd = _rel_diff(solver_res.value, auto_res.value)
        assert rd < 1e-4, f"reldiff={rd:.2e}"


# ── Custom threshold: tighter quadrature tolerance ───────────────────────


class TestCustomThresholds:
    def test_tighter_quadrature_tol(self):
        """A tighter self-tol falls back to DD more often but still accurate."""
        tight = ComptonKernelSolver(quadrature_self_tol=1e-7)
        E, Ep, T = 5.0 * kev, 5.5 * kev, 30.0 * kev_kelvin
        tight_res = tight.sigma_E(E, Ep, xi=0.0, T=T, Ne=1.0)
        default_res = SOLVER.sigma_E(E, Ep, xi=0.0, T=T, Ne=1.0)
        assert _rel_diff(tight_res.value, default_res.value) < 1e-5

    def test_zero_quadrature_tol_forces_dd(self):
        """With tol=0 the quadrature is never accepted; result = DD."""
        dd_only = ComptonKernelSolver(quadrature_self_tol=0.0)
        dd_series = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeriesHighPrecision)
        E, Ep, T = 5.0 * kev, 5.5 * kev, 30.0 * kev_kelvin
        solver_res = dd_only.sigma_E(E, Ep, xi=0.0, T=T, Ne=1.0)
        dd_res = dd_series.sigma_E(E, Ep, xi=0.0, T=T, Ne=1.0)
        assert _rel_diff(solver_res.value, dd_res.value) < 1e-12

    def test_large_quadrature_tol_accepts_everything(self):
        """With tol=1.0 the quadrature is always accepted in DD regime."""
        q_always = ComptonKernelSolver(quadrature_self_tol=1.0)
        q64 = cq.ComptonKernelQuadrature(64)
        E, Ep, T = 5.0 * kev, 5.5 * kev, 30.0 * kev_kelvin
        solver_res = q_always.sigma_E(E, Ep, xi=0.0, T=T, Ne=1.0)
        q_res = q64.sigma_E(E, Ep, xi=0.0, T=T, Ne=1.0)
        assert _rel_diff(solver_res.value, q_res.value) < 1e-12
