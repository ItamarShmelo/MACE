"""
Bit-exactness verification for the compute_group_entry refactoring.

Compares post-refactor matrix output against pre-refactor snapshots using
bitwise equality (np.testing.assert_array_equal).  The snapshots are
generated once by running this file with --snapshot and committed alongside
the test.

Also includes smoke tests for the four new public integral methods.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "cpp_modules")

import _compton_multigroup as cm
from _compton_differential_cross_section import ComptonKernelSolver
from _units import kev, kev_kelvin

KERNEL = ComptonKernelSolver()
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "refactor_snapshots")


def _snapshot_path(name: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{name}.npy")


def _save(name: str, arr: np.ndarray):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    np.save(_snapshot_path(name), arr)


def _load(name: str) -> np.ndarray:
    return np.load(_snapshot_path(name))


# ---------------------------------------------------------------------------
# Snapshot scenarios
# ---------------------------------------------------------------------------

def _hot_sigma():
    bounds = [0.1 * kev, 0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    T = 10.0 * kev_kelvin
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3))
    return mg.compute_sigma_matrix(KERNEL, num_angle_bins=4, T=T, Ne=1.0)


def _hot_dsigma():
    bounds = [0.1 * kev, 0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    T = 10.0 * kev_kelvin
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3))
    return mg.compute_dsigma_dT_matrix(KERNEL, num_angle_bins=4, T=T, Ne=1.0)


def _cold_sigma():
    bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    T = 0.5 * kev_kelvin
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3))
    return mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)


def _flat_ep_sigma():
    bounds = [0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    T = 10.0 * kev_kelvin
    flat = cm.FlatEpConfig(density=64.0, min_points=8, max_points=256,
                           mode=cm.FlatEpDensityMode.points_per_decade,
                           flat_E=False)
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3,
                                      flat_ep=flat))
    return mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)


def _single_bin_sigma():
    bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]
    T = 10.0 * kev_kelvin
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3))
    return mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)


def _wien_sigma():
    bounds = [0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    T = 10.0 * kev_kelvin
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.WienWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3))
    return mg.compute_sigma_matrix(KERNEL, num_angle_bins=2, T=T, Ne=1.0)


SCENARIOS = {
    "hot_sigma":       _hot_sigma,
    "hot_dsigma":      _hot_dsigma,
    "cold_sigma":      _cold_sigma,
    "flat_ep_sigma":   _flat_ep_sigma,
    "single_bin_sigma": _single_bin_sigma,
    "wien_sigma":      _wien_sigma,
}


# ---------------------------------------------------------------------------
# Snapshot generation (run once before refactoring)
# ---------------------------------------------------------------------------

def generate_snapshots():
    """Generate .npy snapshots for all scenarios."""
    for name, func in SCENARIOS.items():
        arr = func()
        _save(name, arr)
        print(f"  saved {name}: shape={arr.shape}")
    print(f"Snapshots written to {SNAPSHOT_DIR}/")


# ---------------------------------------------------------------------------
# Bit-exact comparison tests
# ---------------------------------------------------------------------------

class TestBitExact:
    """Post-refactor output must be bitwise identical to pre-refactor snapshots."""

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_bitwise_equal(self, name):
        snap_file = _snapshot_path(name)
        if not os.path.exists(snap_file):
            pytest.skip(f"snapshot {name}.npy not found; run generate_snapshots()")
        expected = _load(name)
        actual = SCENARIOS[name]()
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"bit-exact mismatch for scenario '{name}'")


# ---------------------------------------------------------------------------
# New public API smoke tests
# ---------------------------------------------------------------------------

class TestNewPublicAPI:
    """Smoke tests for the four new compute_*_integral_* methods."""

    @pytest.fixture()
    def mg(self):
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]
        return cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=cm.UniformWeightFunction(),
            config=cm.MGIntegrationConfig(base_order=8, integration_tolerance=1e-3))

    def test_xi_integral_sigma_shape(self, mg):
        E = 5.0 * kev
        Ep = 4.0 * kev
        T = 10.0 * kev_kelvin
        n_bins = 4
        result = mg.compute_xi_integral_sigma(
            KERNEL, E=E, Ep=Ep, num_xi_bins=n_bins, T=T, Ne=1.0)
        assert result.shape == (n_bins,)
        assert np.all(np.isfinite(result))
        assert np.any(result != 0.0)

    def test_xi_integral_dsigma_dT_shape(self, mg):
        E = 5.0 * kev
        Ep = 4.0 * kev
        T = 10.0 * kev_kelvin
        n_bins = 4
        result = mg.compute_xi_integral_dsigma_dT(
            KERNEL, E=E, Ep=Ep, num_xi_bins=n_bins, T=T, Ne=1.0)
        assert result.shape == (n_bins,)
        assert np.all(np.isfinite(result))

    def test_Ep_xi_integral_sigma_shape(self, mg):
        E = 5.0 * kev
        T = 10.0 * kev_kelvin
        n_bins = 4
        result = mg.compute_Ep_xi_integral_sigma(
            KERNEL, E=E, Ep_lo=1.0 * kev, Ep_hi=10.0 * kev,
            num_xi_bins=n_bins, T=T, Ne=1.0)
        assert result.shape == (n_bins,)
        assert np.all(np.isfinite(result))
        assert np.any(result != 0.0)

    def test_Ep_xi_integral_dsigma_dT_shape(self, mg):
        E = 5.0 * kev
        T = 10.0 * kev_kelvin
        n_bins = 4
        result = mg.compute_Ep_xi_integral_dsigma_dT(
            KERNEL, E=E, Ep_lo=1.0 * kev, Ep_hi=10.0 * kev,
            num_xi_bins=n_bins, T=T, Ne=1.0)
        assert result.shape == (n_bins,)
        assert np.all(np.isfinite(result))

    def test_xi_integral_bin_sum_consistency(self, mg):
        """Sum of N-bin xi integrals should approximate 1-bin result."""
        E = 5.0 * kev
        Ep = 4.0 * kev
        T = 10.0 * kev_kelvin

        result_1 = mg.compute_xi_integral_sigma(
            KERNEL, E=E, Ep=Ep, num_xi_bins=1, T=T, Ne=1.0)
        result_4 = mg.compute_xi_integral_sigma(
            KERNEL, E=E, Ep=Ep, num_xi_bins=4, T=T, Ne=1.0)

        total_1 = result_1.sum()
        total_4 = result_4.sum()
        if abs(total_1) > 1e-35:
            rel = abs(total_4 - total_1) / abs(total_1)
            assert rel < 0.05, f"bin-sum consistency: rel diff = {rel:.2e}"


# ---------------------------------------------------------------------------
# CLI entry point for snapshot generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--snapshot" in sys.argv:
        generate_snapshots()
    else:
        print("Usage: python test_refactor_bit_exact.py --snapshot")
        print("  Generates .npy reference files in tests/refactor_snapshots/")
