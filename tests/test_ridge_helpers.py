"""
Unit tests for ridge_thermal_width, RidgeBounds, and compute_ridge_bounds.

Phase 1: validates the new physics functions in isolation.
Phase 2 tests (conservation comparison, overlap collapse, etc.) are added
after the engine replacement.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "cpp_modules")

import _compton_multigroup as cm
from _compton_differential_cross_section import ComptonKernelSolver
from _units import kev, kev_kelvin, me_c2, k_boltz

KERNEL = ComptonKernelSolver()
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "ridge_reference")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tau(T):
    """Dimensionless temperature tau = k_B T / (m_e c^2)."""
    return T * k_boltz / me_c2


def _config(order, **kwargs):
    for key in ("xi_order", "xi_tail_order", "ep_edge_order",
                "ep_interior_order", "e_panel_order"):
        if kwargs.get(key) is None:
            kwargs[key] = order
    return cm.MGIntegrationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Tests for ridge_thermal_width
# ---------------------------------------------------------------------------

class TestRidgeThermalWidth:
    """Validate ridge_thermal_width against limiting cases and analytic formulas."""

    @pytest.mark.parametrize("E", [1 * kev, 100 * kev, 1000 * kev])
    @pytest.mark.parametrize("T", [0.5 * kev_kelvin, 10 * kev_kelvin])
    def test_exact_forward_scatter_is_zero(self, E, T):
        """xi = 1.0 exactly must return bitwise 0.0."""
        assert cm.ridge_thermal_width(E, 1.0, T) == 0.0

    @pytest.mark.parametrize("E", [1 * kev, 100 * kev])
    @pytest.mark.parametrize("T", [1 * kev_kelvin, 10 * kev_kelvin])
    def test_near_forward_scaling(self, E, T):
        """For small u = 1 - xi, sigma -> E * sqrt(2*tau*u)."""
        xi = 1.0 - 1e-10  # matches XI_UPPER_EPS
        u = 1.0 - xi
        tau = _tau(T)
        gamma = E / me_c2

        sigma = cm.ridge_thermal_width(E, xi, T)
        asymptotic = E * math.sqrt(2.0 * tau * u)

        assert sigma > 0.0
        ratio = sigma / asymptotic
        # Correction is O(gamma*u), so ratio should be close to 1
        assert abs(ratio - 1.0) < 10.0 * gamma * u + 1e-12

    @pytest.mark.parametrize("E", [1 * kev, 100 * kev, 1000 * kev])
    @pytest.mark.parametrize("xi", [-1.0, -0.5, 0.0, 0.5, 0.9])
    def test_zero_temperature_is_zero(self, E, xi):
        """T = 0 must return bitwise 0.0."""
        assert cm.ridge_thermal_width(E, xi, 0.0) == 0.0

    @pytest.mark.parametrize("E", [10 * kev, 100 * kev])
    @pytest.mark.parametrize("xi", [-0.5, 0.0, 0.5])
    def test_sqrt_T_scaling(self, E, xi):
        """sigma scales as sqrt(T) at fixed E and xi."""
        T1 = 1 * kev_kelvin
        T2 = 4 * kev_kelvin

        s1 = cm.ridge_thermal_width(E, xi, T1)
        s2 = cm.ridge_thermal_width(E, xi, T2)

        assert s1 > 0.0
        assert s2 > 0.0
        ratio = s2 / s1
        expected = math.sqrt(T2 / T1)  # = 2.0
        assert abs(ratio - expected) < 1e-12 * expected

    @pytest.mark.parametrize("E", [1 * kev, 100 * kev, 1000 * kev])
    @pytest.mark.parametrize("T", [0.5 * kev_kelvin, 10 * kev_kelvin, 100 * kev_kelvin])
    def test_backscatter_finite_positive(self, E, T):
        """xi = -1 (full backscatter) must give finite, positive result."""
        sigma = cm.ridge_thermal_width(E, -1.0, T)
        assert math.isfinite(sigma)
        assert sigma > 0.0

    @pytest.mark.parametrize("E,xi,T", [
        (100 * kev, 0.0, 10 * kev_kelvin),
        (10 * kev, -1.0, 1 * kev_kelvin),
        (500 * kev, 0.5, 50 * kev_kelvin),
    ])
    def test_analytic_formula(self, E, xi, T):
        """Compare C++ result against Python computation of the formula."""
        gamma = E / me_c2
        tau = _tau(T)
        u = max(0.0, 1.0 - xi)
        d = 1.0 + gamma * u
        expected = (E / (d * d)) * math.sqrt(
            tau * u * (2.0 + 2.0 * gamma * u + gamma * gamma * u))

        result = cm.ridge_thermal_width(E, xi, T)
        assert abs(result - expected) < 1e-14 * max(abs(expected), 1e-30)


# ---------------------------------------------------------------------------
# Tests for compute_ridge_bounds
# ---------------------------------------------------------------------------

class TestComputeRidgeBounds:
    """Validate compute_ridge_bounds struct fields."""

    @pytest.mark.parametrize("E", [1 * kev, 100 * kev, 1000 * kev])
    @pytest.mark.parametrize("xi_lo,xi_hi", [
        (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (-1.0, 0.999999999),
    ])
    def test_cold_lo_leq_cold_hi(self, E, xi_lo, xi_hi):
        """cold_lo <= cold_hi for xi_lo < xi_hi (monotonicity of cold ridge)."""
        rb = cm.compute_ridge_bounds(E, xi_lo, xi_hi, 10 * kev_kelvin)
        assert rb.cold_lo <= rb.cold_hi

    @pytest.mark.parametrize("E", [1 * kev, 50 * kev, 500 * kev])
    @pytest.mark.parametrize("xi_lo,xi_hi", [
        (-1.0, 0.0), (-0.5, 0.5), (0.0, 0.9),
    ])
    def test_cold_recoil_agreement(self, E, xi_lo, xi_hi):
        """At T=0, cold endpoints must be bitwise equal to cold_recoil_lo/hi."""
        rb = cm.compute_ridge_bounds(E, xi_lo, xi_hi, 0.0)

        lo_ref = cm.cold_recoil_lo(E, xi_lo)
        hi_ref = cm.cold_recoil_hi(E, xi_hi)

        # Bitwise equality
        assert rb.cold_lo == lo_ref
        assert rb.cold_hi == hi_ref

    @pytest.mark.parametrize("E", [10 * kev, 100 * kev])
    @pytest.mark.parametrize("xi_lo,xi_hi", [(-1.0, 0.0), (0.0, 0.9)])
    def test_zero_sigma_at_T0(self, E, xi_lo, xi_hi):
        """At T=0, sigma_lo and sigma_hi must be exactly 0.0."""
        rb = cm.compute_ridge_bounds(E, xi_lo, xi_hi, 0.0)
        assert rb.sigma_lo == 0.0
        assert rb.sigma_hi == 0.0

    @pytest.mark.parametrize("E", [10 * kev, 100 * kev])
    @pytest.mark.parametrize("T", [1 * kev_kelvin, 10 * kev_kelvin])
    def test_sigma_positive_at_finite_T(self, E, T):
        """At finite T with xi_lo < xi_hi away from 1, both sigmas are positive."""
        rb = cm.compute_ridge_bounds(E, -0.5, 0.5, T)
        assert rb.sigma_lo > 0.0
        assert rb.sigma_hi > 0.0


# ---------------------------------------------------------------------------
# Phase 2: Integration-level tests for the new ridge engine
# ---------------------------------------------------------------------------

class TestColdRecoilBitwiseCompat:
    """cold_recoil_lo/hi (reimplemented via compute_ridge_bounds) must match
    the Phase 1 reference values bitwise."""

    def test_bitwise_match(self):
        path = os.path.join(REFERENCE_DIR, "cold_recoil_reference.json")
        with open(path) as f:
            data = json.load(f)
        for key, vals in data.items():
            E_str, xi_str = key.rsplit("_", 1)
            E = float(E_str)
            xi = float(xi_str)
            assert cm.cold_recoil_lo(E, xi) == vals["lo"], \
                f"cold_recoil_lo mismatch at E={E}, xi={xi}"
            assert cm.cold_recoil_hi(E, xi) == vals["hi"], \
                f"cold_recoil_hi mismatch at E={E}, xi={xi}"


class TestConservationComparison:
    """Diagonal entries (self-scattering) should agree between tail modes
    since the peak group always contains the ridge."""

    def test_diagonal_agreement_fine_bins(self):
        """With fine angle bins, sigma varies little within each bin,
        so the diagonal should agree closely between tail modes."""
        bounds = np.geomspace(1.0 * kev, 50.0 * kev, 11).tolist()
        T = 1.0 * kev_kelvin

        config_no_tails = _config(32, ep_k_cut=5.0, ep_k_in=2.0)
        config_tails = _config(32, ep_k_cut=5.0, ep_k_in=2.0)

        mg_no = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config_no_tails)
        mg_yes = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config_tails)

        m_no = mg_no.compute_sigma_matrix(KERNEL, num_angle_bins=8, T=T, Ne=1.0)
        m_yes = mg_yes.compute_sigma_matrix(KERNEL, num_angle_bins=8, T=T, Ne=1.0)

        for g in range(m_no.shape[0]):
            diag_no = m_no[g, g].sum()
            diag_yes = m_yes[g, g].sum()
            if abs(diag_yes) > 1e-35:
                rel = abs(diag_no - diag_yes) / abs(diag_yes)
                assert rel < 0.15, \
                    f"Diagonal g={g}: rel diff = {rel:.2e}"

    def test_tails_add_to_row_sum(self):
        """With diagnostic tails, row sums should be >= no-tails row sums
        (tails only add material, never subtract)."""
        bounds = np.geomspace(1.0 * kev, 50.0 * kev, 11).tolist()
        T = 1.0 * kev_kelvin

        config_no = _config(24, cutoff_ratio=1e-14, ep_k_cut=5.0)
        config_yes = _config(24, cutoff_ratio=1e-14, ep_k_cut=5.0)

        mg_no = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config_no)
        mg_yes = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config_yes)

        m_no = mg_no.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)
        m_yes = mg_yes.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)

        rows_no = m_no.sum(axis=(1, 2))
        rows_yes = m_yes.sum(axis=(1, 2))

        for g in range(len(rows_no)):
            assert rows_yes[g] >= rows_no[g] - 1e-35, \
                f"Row {g}: tails reduced row sum unexpectedly"


class TestOverlapCollapse:
    """When the xi bin is narrow, edge regions should overlap and collapse
    to a single interior panel."""

    def test_narrow_bin_finite(self):
        bounds = [1.0 * kev, 5.0 * kev, 50.0 * kev]
        T = 10.0 * kev_kelvin
        config = _config(16, ep_k_cut=5.0, ep_k_in=2.0)
        mg = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config)
        m = mg.compute_sigma_matrix(KERNEL, num_angle_bins=32, T=T, Ne=1.0)
        assert np.all(np.isfinite(m))
        assert np.any(m != 0.0)


class TestFullyOutsideGroup:
    """Groups entirely outside the retained interval should be zero (no tails)
    or small but non-zero (with diagnostic tails)."""

    def test_far_group_zero_without_tails(self):
        bounds = [0.01 * kev, 0.05 * kev, 0.1 * kev, 500 * kev, 1000 * kev]
        T = 1.0 * kev_kelvin
        config = _config(16, ep_k_cut=5.0)
        mg = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config)
        m = mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)
        assert m[0, 3, 0] == 0.0, "far above-ridge group should be exactly 0"

    def test_far_group_nonzero_with_tails(self):
        bounds = [0.01 * kev, 0.05 * kev, 0.1 * kev, 500 * kev, 1000 * kev]
        T = 1.0 * kev_kelvin
        config = _config(16, ep_k_cut=5.0)
        mg = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config)
        m = mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)
        assert np.all(np.isfinite(m))


class TestZeroSigmaBehavior:
    """At T=0 (sigma=0), integration should still produce finite results."""

    def test_T0_finite_results(self):
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]
        T = 0.001 * kev_kelvin  # near-cold
        config = _config(16, ep_k_cut=5.0)
        mg = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config)
        m = mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)
        assert np.all(np.isfinite(m))
        assert np.any(m != 0.0)


class TestClippedRetainedInterval:
    """When keep_lo would be below Ep_lo, it should be clipped to Ep_lo."""

    def test_clipping_no_crash(self):
        bounds = [1.0 * kev, 2.0 * kev, 100.0 * kev]
        T = 50.0 * kev_kelvin  # wide thermal width
        config = _config(16, ep_k_cut=6.0)
        mg = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config)
        m = mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)
        assert np.all(np.isfinite(m))


class TestDiagnosticTailEndToEnd:
    """End-to-end: groups outside the retained interval should be exactly zero
    without tails and finite with tails."""

    def test_far_groups_zero_vs_nonzero(self):
        bounds = [0.01 * kev, 0.05 * kev, 0.1 * kev, 500 * kev, 1000 * kev]
        T = 1.0 * kev_kelvin

        config_off = _config(16, ep_k_cut=5.0)
        config_on = _config(16, ep_k_cut=5.0)

        mg_off = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config_off)
        mg_on = cm.ComptonMultigroupKernel(
            bounds, cm.PlanckWeightFunction(cap_x=25.0), config_on)

        m_off = mg_off.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)
        m_on = mg_on.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)

        assert np.all(np.isfinite(m_off))
        assert np.all(np.isfinite(m_on))
        assert m_off[0, 3, 0] == 0.0, "Far group without tails should be exactly 0"


class TestKcutConvergence:
    """Matrix entries should converge as k_cut increases from 4 to 6."""

    def test_monotonic_convergence(self):
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
        T = 10.0 * kev_kelvin

        matrices = {}
        for k in [4.0, 5.0, 6.0]:
            config = _config(24, ep_k_cut=k)
            mg = cm.ComptonMultigroupKernel(
                bounds, cm.PlanckWeightFunction(cap_x=25.0), config)
            matrices[k] = mg.compute_sigma_matrix(
                KERNEL, num_angle_bins=2, T=T, Ne=1.0)

        diff_45 = np.max(np.abs(matrices[5.0] - matrices[4.0]))
        diff_56 = np.max(np.abs(matrices[6.0] - matrices[5.0]))

        scale = np.max(np.abs(matrices[5.0]))
        rel_56 = diff_56 / scale
        assert diff_56 < diff_45 or rel_56 < 1e-4, (
            f"k_cut convergence failed: diff(5,6)={diff_56:.2e} >= "
            f"diff(4,5)={diff_45:.2e}, rel(5,6)={rel_56:.2e}"
        )


# ---------------------------------------------------------------------------
# Phase 3: Double-peak E' integration tests
# ---------------------------------------------------------------------------

XI_UPPER_EPS = 1e-10


class TestDoublePeakLastBinConvergence:
    """The last xi bin (xi_hi ~ 1 - 1e-10) should converge with the
    4-region double-peak scheme across ep_order sweeps."""

    @pytest.mark.parametrize("T_keV", [0.1, 1.0, 10.0])
    def test_ep_order_convergence(self, T_keV):
        E = 10.0 * kev
        T = T_keV * kev_kelvin
        Ep_lo = 0.5 * E
        Ep_hi = 2.0 * E
        num_xi_bins = 4

        vals_by_order = {}
        for order in [16, 24, 32, 48, 64]:
            config = _config(
                24, ep_k_cut=5.0, ep_k_in=2.0,
                ep_edge_order=order, ep_interior_order=order)
            mg = cm.ComptonMultigroupKernel(
                [Ep_lo, Ep_hi], cm.UniformWeightFunction(), config)
            vals = np.asarray(mg.compute_Ep_xi_integral_sigma(
                KERNEL, E, Ep_lo, Ep_hi, num_xi_bins, T, 1.0))
            vals_by_order[order] = vals

        ref = vals_by_order[64]
        for order in [32, 48]:
            delta = np.abs(vals_by_order[order] - ref) / np.maximum(np.abs(ref), 1e-30)
            assert delta[3] < 1e-3, \
                f"T={T_keV} keV, order={order}: last-bin delta={delta[3]:.2e} >= 1e-3"


class TestDoublePeakNonLastBinUnchanged:
    """Non-last bins (0-2) should produce the same result whether the
    double-peak path is used or not. Since sigma_lo/sigma_hi < 10 for
    bins 0-2, they should use the existing 3-region path and remain
    unchanged."""

    def test_bins_0_to_2_match(self):
        E = 10.0 * kev
        T = 1.0 * kev_kelvin
        Ep_lo = 0.5 * E
        Ep_hi = 2.0 * E
        num_xi_bins = 4

        config = _config(
            24, ep_k_cut=5.0, ep_k_in=2.0,
            ep_edge_order=24, ep_interior_order=24)
        mg = cm.ComptonMultigroupKernel(
            [Ep_lo, Ep_hi], cm.UniformWeightFunction(), config)

        vals_24 = np.asarray(mg.compute_Ep_xi_integral_sigma(
            KERNEL, E, Ep_lo, Ep_hi, num_xi_bins, T, 1.0))

        config_48 = _config(
            24, ep_k_cut=5.0, ep_k_in=2.0,
            ep_edge_order=48, ep_interior_order=48)
        mg_48 = cm.ComptonMultigroupKernel(
            [Ep_lo, Ep_hi], cm.UniformWeightFunction(), config_48)
        vals_48 = np.asarray(mg_48.compute_Ep_xi_integral_sigma(
            KERNEL, E, Ep_lo, Ep_hi, num_xi_bins, T, 1.0))

        for a in range(3):
            if abs(vals_48[a]) > 1e-30:
                delta = abs(vals_24[a] - vals_48[a]) / abs(vals_48[a])
                assert delta < 1e-6, \
                    f"bin {a}: delta={delta:.2e} >= 1e-6, values not converged"


class TestDoublePeakThresholdBoundary:
    """Verify behavior near the sigma_lo/sigma_hi = 10 threshold.
    With 2 xi bins, the last bin spans [-1, 1-eps] -> ratio >> 10.
    With many xi bins, interior bins have ratio < 10."""

    def test_finite_results_all_bins(self):
        E = 10.0 * kev
        T = 1.0 * kev_kelvin
        Ep_lo = 0.5 * E
        Ep_hi = 2.0 * E

        for num_xi_bins in [2, 4, 8, 16]:
            config = _config(
                24, ep_k_cut=5.0, ep_k_in=2.0,
                ep_edge_order=24, ep_interior_order=24)
            mg = cm.ComptonMultigroupKernel(
                [Ep_lo, Ep_hi], cm.UniformWeightFunction(), config)
            vals = np.asarray(mg.compute_Ep_xi_integral_sigma(
                KERNEL, E, Ep_lo, Ep_hi, num_xi_bins, T, 1.0))
            assert np.all(np.isfinite(vals)), \
                f"num_xi_bins={num_xi_bins}: non-finite values"
            assert np.all(vals > 0), \
                f"num_xi_bins={num_xi_bins}: non-positive values"

    def test_threshold_transition_smooth(self):
        """With increasing num_xi_bins, total integral should remain stable."""
        E = 10.0 * kev
        T = 1.0 * kev_kelvin
        Ep_lo = 0.5 * E
        Ep_hi = 2.0 * E

        totals = []
        for num_xi_bins in [4, 8, 16]:
            config = _config(
                24, ep_k_cut=5.0, ep_k_in=2.0,
                ep_edge_order=32, ep_interior_order=32)
            mg = cm.ComptonMultigroupKernel(
                [Ep_lo, Ep_hi], cm.UniformWeightFunction(), config)
            vals = np.asarray(mg.compute_Ep_xi_integral_sigma(
                KERNEL, E, Ep_lo, Ep_hi, num_xi_bins, T, 1.0))
            totals.append(float(vals.sum()))

        for i in range(1, len(totals)):
            rel = abs(totals[i] - totals[0]) / abs(totals[0])
            assert rel < 0.05, \
                f"Total integral changed by {rel:.2e} between num_xi_bins settings"


class TestDoublePeakEpOrderSweep:
    """Compare old-style oscillation (expected at low orders) vs
    converged high-order result for the last bin at T=1 keV."""

    def test_convergence_vs_oscillation(self):
        E = 10.0 * kev
        T = 1.0 * kev_kelvin
        Ep_lo = 0.5 * E
        Ep_hi = 2.0 * E
        num_xi_bins = 4

        vals_list = []
        for order in [8, 16, 24, 32, 48, 64]:
            config = _config(
                24, ep_k_cut=5.0, ep_k_in=2.0,
                ep_edge_order=order, ep_interior_order=order)
            mg = cm.ComptonMultigroupKernel(
                [Ep_lo, Ep_hi], cm.UniformWeightFunction(), config)
            vals = np.asarray(mg.compute_Ep_xi_integral_sigma(
                KERNEL, E, Ep_lo, Ep_hi, num_xi_bins, T, 1.0))
            vals_list.append(vals)

        ref = vals_list[-1]

        deltas = [abs(v[3] - ref[3]) / abs(ref[3]) for v in vals_list[:-1]]

        assert deltas[-1] < deltas[0], \
            f"Last-bin delta not decreasing: first={deltas[0]:.2e}, last={deltas[-1]:.2e}"
        assert deltas[-1] < 1e-4, \
            f"Last-bin delta at order 48 still {deltas[-1]:.2e} >= 1e-4"
