"""
Tests for the xi (mu) integration strategy.

Validates the integration branches:
  1. Endpoint-localized (peak near xi=1 and narrow): rlog over full bin
  2. Interior peak (three-region split): left tail + core + right tail
  3. Peak entirely left: plain GL over full bin
  4. Peak entirely right: plain GL over full bin
  5. Extreme energy ratios: no exceptions, positive finite results
  6. Log/rlog clustering sanity check
  7. Endpoint-localized predicate: direct unit tests
  8. Regression sweep: last-bin accuracy across temperatures
"""

import math

import compton_matrix._compton_multigroup as cm
import numpy as np
import pytest
from compton_matrix._compton_differential_cross_section import ComptonKernelSolver
from compton_matrix._units import kev, kev_kelvin

KERNEL = ComptonKernelSolver()


def _config(order, **kwargs):
    for key in ("xi_order", "xi_tail_order", "ep_edge_order", "ep_interior_order", "e_panel_order"):
        if kwargs.get(key) is None:
            kwargs[key] = order
    return cm.MGIntegrationConfig(**kwargs)


def _make_mg(bounds_erg, *, order=24, xi_order=None, tol=1e-4, xi_peak_k=10.0):
    """Helper to build a ComptonMultigroupKernel with given config."""
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds_erg,
        weight_function=cm.UniformWeightFunction(),
        config=_config(order, xi_order=xi_order, xi_peak_k=xi_peak_k),
    )


def _reference_mg(bounds_erg, *, order=128):
    """Build a high-order reference ComptonMultigroupKernel."""
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds_erg,
        weight_function=cm.UniformWeightFunction(),
        config=_config(order, xi_peak_k=10.0),
    )


# ---------------------------------------------------------------------------
# 1. Endpoint-localized branch (peak near xi=1 and narrow)
# ---------------------------------------------------------------------------


class TestEndpointLocalized:
    """gamma = gamma_p = 0.1 (d = 0): endpoint-localized rlog integration."""

    def test_nonzero_finite(self):
        """Endpoint-localized integration produces nonzero, finite results."""
        E = 0.1 * 511.0 * kev
        bounds = [0.95 * E, 1.05 * E]
        T = 0.1 * kev_kelvin

        mg = _make_mg(bounds, order=48, xi_order=48)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=8, T=T)

        last_bin = S[0, 0, -1]
        assert np.isfinite(last_bin), "endpoint-localized result is not finite"
        assert last_bin != 0.0, "endpoint-localized result is zero"

    def test_vs_reference(self):
        """Endpoint-localized result agrees with high-order GL reference."""
        E = 0.1 * 511.0 * kev
        bounds = [0.95 * E, 1.05 * E]
        T = 0.1 * kev_kelvin
        n_bins = 8

        mg = _make_mg(bounds, order=48, xi_order=48)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        mg_ref = _reference_mg(bounds)
        S_ref = mg_ref.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        peak = np.max(np.abs(S_ref))
        mask = np.abs(S_ref) > 1e-4 * peak
        if not np.any(mask):
            pytest.skip("no significant entries in reference")

        rel = np.max(np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask]))
        assert rel < 0.01, f"endpoint-localized vs ref: max rel diff = {rel:.2e}"

    def test_broad_sigma_no_rlog(self):
        """Large sigma_xi: rlog NOT selected, peak-splitting path converges."""
        E_in = 1.0 * 511.0 * kev
        E_out = 2.0 * 511.0 * kev
        bounds = sorted({0.9 * E_in, 1.1 * E_in, 0.9 * E_out, 1.1 * E_out})
        T = 511.0 * kev_kelvin
        n_bins = 8

        mg = _make_mg(bounds, order=48, xi_order=48)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        mg_ref = _reference_mg(bounds)
        S_ref = mg_ref.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        peak = np.max(np.abs(S_ref))
        mask = np.abs(S_ref) > 1e-4 * peak
        if not np.any(mask):
            pytest.skip("no significant entries in reference")

        rel = np.max(np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask]))
        assert rel < 0.05, f"broad-sigma vs ref: max rel diff = {rel:.2e}"

    def test_constants_exposed(self):
        """The calibrated constants are accessible from Python."""
        assert cm.XI_ENDPOINT_EPS == 0.1
        assert cm.XI_CUSP_TAU == 0.001


# ---------------------------------------------------------------------------
# 2. Interior peak (three-region split)
# ---------------------------------------------------------------------------


class TestInteriorPeak:
    """gamma = 10, gamma_p = 15, T = 0.1 keV: narrow peak inside last bin.

    d = 5, xi_pk = 1 - 5/150 ≈ 0.967, sigma_xi ≈ 5.5e-4,
    half_w ≈ 0.013.  Peak window [0.954, 0.980] sits inside the last
    8-bin angle bin [0.75, 1 - 1e-10].
    """

    def test_vs_reference(self):
        E_in = 10.0 * 511.0 * kev
        E_out = 15.0 * 511.0 * kev
        bounds = sorted({0.95 * E_in, 1.05 * E_in, 0.95 * E_out, 1.05 * E_out})
        T = 0.1 * kev_kelvin
        n_bins = 8

        mg = _make_mg(bounds, order=48, xi_order=48)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        mg_ref = _reference_mg(bounds)
        S_ref = mg_ref.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        peak = np.max(np.abs(S_ref))
        mask = np.abs(S_ref) > 1e-4 * peak
        if not np.any(mask):
            pytest.skip("no significant entries in reference")

        rel = np.max(np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask]))
        assert rel < 0.05, f"interior-peak vs ref: max rel diff = {rel:.2e}"

    def test_all_bins_finite(self):
        """All angle bins produce finite results."""
        E_in = 10.0 * 511.0 * kev
        E_out = 15.0 * 511.0 * kev
        bounds = sorted({0.95 * E_in, 1.05 * E_in, 0.95 * E_out, 1.05 * E_out})
        T = 0.1 * kev_kelvin

        mg = _make_mg(bounds, order=24)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=8, T=T)
        assert np.all(np.isfinite(S)), "non-finite entries in interior-peak"


# ---------------------------------------------------------------------------
# 3. Peak entirely left
# ---------------------------------------------------------------------------


class TestPeakLeft:
    """gamma = 0.1, gamma_p = 0.5, T = 0.1 keV: xi_pk = -7.

    At T = 0.1 keV: half_w ≈ 6.5, peak_hi = -0.5 < xi_lo for mid bins.
    """

    def test_vs_reference(self):
        E_in = 0.1 * 511.0 * kev
        E_out = 0.5 * 511.0 * kev
        bounds = sorted({0.9 * E_in, 1.1 * E_in, 0.9 * E_out, 1.1 * E_out})
        T = 0.1 * kev_kelvin
        n_bins = 8

        mg = _make_mg(bounds, order=48, xi_order=48)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        mg_ref = _reference_mg(bounds)
        S_ref = mg_ref.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        peak = np.max(np.abs(S_ref))
        mask = np.abs(S_ref) > 1e-4 * peak
        if not np.any(mask):
            pytest.skip("no significant entries in reference")

        rel = np.max(np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask]))
        assert rel < 0.05, f"peak-left vs ref: max rel diff = {rel:.2e}"


# ---------------------------------------------------------------------------
# 4. Peak entirely right
# ---------------------------------------------------------------------------


class TestPeakRight:
    """gamma = 10, gamma_p = 15, T = 0.1 keV, first bin [-1, -0.75].

    xi_pk ≈ 0.967.  peak_lo ≈ 0.954 >> xi_hi = -0.75.
    """

    def test_vs_reference(self):
        E_in = 10.0 * 511.0 * kev
        E_out = 15.0 * 511.0 * kev
        bounds = sorted({0.95 * E_in, 1.05 * E_in, 0.95 * E_out, 1.05 * E_out})
        T = 0.1 * kev_kelvin
        n_bins = 8

        mg = _make_mg(bounds, order=48, xi_order=48)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        mg_ref = _reference_mg(bounds)
        S_ref = mg_ref.compute_sigma_matrix(KERNEL, num_angle_bins=n_bins, T=T)

        g_in = 0
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            if lo <= E_in <= hi:
                g_in = i
                break
        g_out = 0
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            if lo <= E_out <= hi:
                g_out = i
                break

        first_bin_val = S[g_in, g_out, 0]
        first_bin_ref = S_ref[g_in, g_out, 0]

        assert np.isfinite(first_bin_val), "peak-right result not finite"
        if abs(first_bin_ref) > 1e-35:
            rel = abs(first_bin_val - first_bin_ref) / abs(first_bin_ref)
            assert rel < 0.05, f"peak-right first bin vs ref: rel diff = {rel:.2e}"


# ---------------------------------------------------------------------------
# 5. Extreme energy ratios
# ---------------------------------------------------------------------------


class TestExtremeEnergyRatios:
    """gamma_p/gamma in {0.01, 100}: xi_pk far left, results positive/finite."""

    @pytest.mark.parametrize("ratio", [0.01, 100.0])
    def test_positive_finite(self, ratio):
        E_in = 10.0 * kev
        E_out = ratio * E_in
        lo = min(E_in, E_out) * 0.5
        hi = max(E_in, E_out) * 2.0
        mid = math.sqrt(lo * hi)
        bounds = sorted({lo, mid, hi})
        T = 1.0 * kev_kelvin

        mg = _make_mg(bounds, order=24)
        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=4, T=T)

        assert np.all(np.isfinite(S)), "non-finite entries at extreme ratio"
        row_sums = S.sum(axis=(1, 2))
        assert np.all(row_sums >= 0), "negative row sums at extreme ratio"


# ---------------------------------------------------------------------------
# 6. Log/rlog clustering sanity check
# ---------------------------------------------------------------------------


class TestLogRlogClustering:
    """Verify log clusters near lower end and rlog near upper end."""

    def test_left_peaked_function(self):
        """log should outperform rlog for f(x) = exp(-1000*x)."""
        span = 1.0
        eps = 1e-10

        def f(x):
            return math.exp(-1000.0 * x)

        from scipy.integrate import quad

        ref, _ = quad(f, eps, span)

        log_val = cm.adaptive_log_legendre_integrate(f, 8, eps, span, tol=1e-15, max_depth=0)
        rlog_val = cm.adaptive_rlog_legendre_integrate(f, 8, eps, span, tol=1e-15, max_depth=0)

        log_err = abs(log_val - ref) / abs(ref)
        rlog_err = abs(rlog_val - ref) / abs(ref)

        assert log_err < rlog_err, (
            f"log should be more accurate for left-peaked: log_err={log_err:.2e}, rlog_err={rlog_err:.2e}"
        )

    def test_right_peaked_function(self):
        """rlog should outperform log for g(x) = exp(-1000*(span-x))."""
        span = 1.0
        eps = 1e-10

        def g(x):
            return math.exp(-1000.0 * (span - x))

        from scipy.integrate import quad

        ref, _ = quad(g, eps, span)

        log_val = cm.adaptive_log_legendre_integrate(g, 8, eps, span, tol=1e-15, max_depth=0)
        rlog_val = cm.adaptive_rlog_legendre_integrate(g, 8, eps, span, tol=1e-15, max_depth=0)

        log_err = abs(log_val - ref) / abs(ref)
        rlog_err = abs(rlog_val - ref) / abs(ref)

        assert rlog_err < log_err, (
            f"rlog should be more accurate for right-peaked: log_err={log_err:.2e}, rlog_err={rlog_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 7. Endpoint-localized predicate: direct unit tests
# ---------------------------------------------------------------------------


class TestEndpointLocalizedPredicate:
    """Direct unit tests for the endpoint-localized branch selection predicate."""

    def test_exactly_elastic(self):
        """gamma = gamma_p: peak_distance = 0, sigma_xi = 0 -> True."""
        assert cm.endpoint_localized_xi(1.0, 1.0, 0.001) is True

    def test_narrow_near_endpoint(self):
        """Small delta_gamma, small sigma_xi -> True."""
        assert cm.endpoint_localized_xi(1.0, 1.001, 0.001) is True

    def test_broad_kernel_rejected(self):
        """Large sigma_xi (~0.87) exceeds XI_ENDPOINT_EPS -> False."""
        assert cm.endpoint_localized_xi(1.0, 2.0, 1.0) is False

    def test_peak_far_from_endpoint(self):
        """Peak distance > sigma_xi -> False, even if sigma_xi is small."""
        assert cm.endpoint_localized_xi(0.1, 0.5, 1e-6) is False

    def test_zero_tau_elastic(self):
        """T=0, gamma=gamma_p: sigma_xi=0, peak_distance=0 -> True."""
        assert cm.endpoint_localized_xi(1.0, 1.0, 0.0) is True

    def test_zero_tau_nonelastic(self):
        """T=0, gamma != gamma_p: sigma_xi=0 but peak_distance > 0 -> False."""
        assert cm.endpoint_localized_xi(1.0, 2.0, 0.0) is False

    def test_near_elastic_cusp_hot(self):
        """Hot T, same-group: |dg|/gamma <= eps and tau > cusp_tau -> True."""
        # gamma=0.1, gamma_p=0.105, tau=2.0
        # |dg|/gamma = 0.05 <= 0.1, tau=2.0 > 0.001
        assert cm.endpoint_localized_xi(0.1, 0.105, 2.0) is True

    def test_near_elastic_cusp_cold_rejected(self):
        """Cold T, same-group: tau < cusp_tau so cusp condition does not fire."""
        # gamma=0.1, gamma_p=0.105, tau=1e-8
        # |dg|/gamma = 0.05 <= 0.1 but tau=1e-8 < 0.001
        assert cm.endpoint_localized_xi(0.1, 0.105, 1e-8) is False

    def test_far_group_hot(self):
        """Hot T, far groups: |dg|/gamma >> eps -> False."""
        # gamma=0.1, gamma_p=0.5, tau=2.0
        # |dg|/gamma = 0.4/0.1 = 4.0 >> 0.1
        assert cm.endpoint_localized_xi(0.1, 0.5, 2.0) is False


# ---------------------------------------------------------------------------
# 8. Regression sweep: last-bin accuracy across temperatures
# ---------------------------------------------------------------------------


class TestEndpointLocalizedRegression:
    """Lightweight regression: 9 log-spaced T points, last-bin convergence.

    The full 50-point sweep runs via scripts/regression_xi_sweep.py
    under Slurm.  This subset covers all 6 old regime boundaries and is
    fast enough for regular pytest runs.
    """

    _E = 0.1 * 511.0 * kev
    _bounds = [0.95 * _E, 1.05 * _E]
    _n_bins = 8

    @pytest.fixture(scope="class")
    @classmethod
    def mg(cls):
        return _make_mg(cls._bounds, order=48, xi_order=48)

    @pytest.fixture(scope="class")
    @classmethod
    def mg_ref(cls):
        return _reference_mg(cls._bounds, order=64)

    @pytest.mark.parametrize(
        "T_kev",
        [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    def test_last_bin_convergence(self, T_kev, mg, mg_ref):
        T = T_kev * kev_kelvin

        S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=self._n_bins, T=T)
        S_ref = mg_ref.compute_sigma_matrix(
            KERNEL, num_angle_bins=self._n_bins, T=T
        )

        last_bin = S[0, 0, -1]
        last_bin_ref = S_ref[0, 0, -1]

        assert np.isfinite(last_bin), f"last bin not finite at T={T_kev} keV"
        if abs(last_bin_ref) > 1e-35:
            rel = abs(last_bin - last_bin_ref) / abs(last_bin_ref)
            assert rel < 1e-3, (
                f"T={T_kev} keV: last-bin rel err = {rel:.2e} (threshold 1e-3)"
            )
