"""
Monte Carlo multigroup kernel tests.

Validates ComptonMonteCarloKernel against:
  1. CMMC Monte Carlo (optional -- skip if unavailable)
  2. Weight function invariance (standalone)
  3. Kernel multiplier correctness (standalone)
  4. Seed reproducibility (standalone)
"""

import math
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "cpp_modules")

import _compton_monte_carlo as mc
import _compton_multigroup as cm
from _units import kev, kev_kelvin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BOUNDARIES_KEV = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
BOUNDARIES_ERG = [b * kev for b in BOUNDARIES_KEV]
G = len(BOUNDARIES_ERG) - 1

NUM_SAMPLES = 2_000_000
SEED = 42


def _make_mc(bounds, *, weight_function=None, num_samples=NUM_SAMPLES, seed=SEED):
    if weight_function is None:
        weight_function = cm.WienWeightFunction(cap_x=25.0)
    return mc.ComptonMonteCarloKernel(
        energy_group_boundaries=bounds,
        weight_function=weight_function,
        config=mc.MCIntegrationConfig(
            num_samples=num_samples,
            seed=seed,
            discard_out_of_grid=True))


# ---------------------------------------------------------------------------
# CMMC import guard (same pattern as test_multigroup.py)
# ---------------------------------------------------------------------------

def _try_import_cmmc():
    try:
        sys.path.insert(0, "external/CMMC/cpp_modules")
        import _compton_matrix_mc
        return _compton_matrix_mc
    except (ImportError, OSError):
        return None


def _cmmc_smoke_test():
    code = (
        "import sys; sys.path.insert(0,'external/CMMC/cpp_modules'); "
        "sys.path.insert(0,'cpp_modules'); "
        "import _compton_matrix_mc as mc; "
        "mc.ComptonMatrixMC("
        "energy_groups_centers=[1e-9,5e-9],"
        "energy_groups_boundaries=[5e-10,2e-9,1e-8],"
        "num_of_samples=10,"
        "force_detailed_balance=False,"
        "seed=1).calculate_S_matrix(temperature=1e8)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, timeout=30)
    return result.returncode == 0


_cmmc = _try_import_cmmc()
_cmmc_functional = _cmmc_smoke_test() if _cmmc is not None else False


# ---------------------------------------------------------------------------
# 1. CMMC comparison (optional)
# ---------------------------------------------------------------------------

class TestCMMCComparison:
    """Compare row sums against CMMC Monte Carlo."""

    @pytest.fixture(autouse=True)
    def _require_mc(self):
        if _cmmc is None:
            pytest.skip("_compton_matrix_mc not available")
        if not _cmmc_functional:
            pytest.skip("_compton_matrix_mc segfaults (pre-existing issue)")

    @pytest.mark.parametrize("T_kev", [1.0, 10.0, 100.0])
    def test_bitwise_exact_match(self, T_kev):
        """Same seed must produce bitwise identical matrices."""
        T = T_kev * kev_kelvin
        mc_bounds_kev = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        mc_bounds_erg = [b * kev for b in mc_bounds_kev]
        G_local = len(mc_bounds_erg) - 1
        centers = [math.sqrt(mc_bounds_erg[i] * mc_bounds_erg[i + 1])
                   for i in range(G_local)]

        cmmc_obj = _cmmc.ComptonMatrixMC(
            energy_groups_centers=centers,
            energy_groups_boundaries=mc_bounds_erg,
            num_of_samples=NUM_SAMPLES,
            force_detailed_balance=False,
            seed=SEED,
            discard_out_of_grid=True)
        S_cmmc = np.array(cmmc_obj.calculate_S_matrix(temperature=T))

        mc_obj = _make_mc(mc_bounds_erg)
        S_mc = mc_obj.compute_sigma_matrix(T=T, Ne=1.0)

        np.testing.assert_array_equal(
            S_mc, S_cmmc,
            err_msg=f"bitwise mismatch at T={T_kev} keV")

    def test_row_sum_agreement(self):
        """Row sums should agree within ~5% relative."""
        T_kev = 10.0
        T = T_kev * kev_kelvin

        mc_bounds_kev = [1.0, 5.0, 10.0, 50.0]
        mc_bounds_erg = [b * kev for b in mc_bounds_kev]
        G_local = len(mc_bounds_erg) - 1
        centers = [math.sqrt(mc_bounds_erg[i] * mc_bounds_erg[i + 1])
                   for i in range(G_local)]

        cmmc_obj = _cmmc.ComptonMatrixMC(
            energy_groups_centers=centers,
            energy_groups_boundaries=mc_bounds_erg,
            num_of_samples=NUM_SAMPLES,
            force_detailed_balance=False,
            seed=SEED,
            discard_out_of_grid=True)
        S_cmmc = np.array(cmmc_obj.calculate_S_matrix(temperature=T))

        mc_obj = _make_mc(mc_bounds_erg)
        S_mc = mc_obj.compute_sigma_matrix(T=T, Ne=1.0)

        rs_cmmc = S_cmmc.sum(axis=1)
        rs_mc = S_mc.sum(axis=1)

        mask = np.abs(rs_cmmc) > 1e-35
        if not np.any(mask):
            pytest.skip("all row sums near zero")

        rel_err = np.abs(rs_cmmc[mask] - rs_mc[mask]) / np.abs(rs_cmmc[mask])
        assert np.max(rel_err) < 0.05, (
            f"row-sum max rel error = {np.max(rel_err):.3f}")

    def test_angle_cdf_agreement(self):
        """Angular CDFs should have median max-difference < 0.15."""
        T_kev = 10.0
        T = T_kev * kev_kelvin

        mc_bounds_kev = [1.0, 5.0, 10.0]
        mc_bounds_erg = [b * kev for b in mc_bounds_kev]
        G_local = len(mc_bounds_erg) - 1
        centers = [math.sqrt(mc_bounds_erg[i] * mc_bounds_erg[i + 1])
                   for i in range(G_local)]

        NUM_ANGLE_BINS = _cmmc.ComptonMatrixMC.NUM_ANGLE_BINS

        cmmc_obj = _cmmc.ComptonMatrixMC(
            energy_groups_centers=centers,
            energy_groups_boundaries=mc_bounds_erg,
            num_of_samples=NUM_SAMPLES,
            force_detailed_balance=False,
            seed=SEED,
            discard_out_of_grid=True)
        cmmc_obj.set_tables(temperature_grid=[T * 0.9, T, T * 1.1])

        mc_obj = _make_mc(mc_bounds_erg, num_samples=NUM_SAMPLES)
        S_mc = mc_obj.compute_sigma_matrix(
            num_angle_bins=NUM_ANGLE_BINS, T=T, Ne=1.0)

        max_cdf_diffs = []
        for g0 in range(G_local):
            for g in range(G_local):
                row = S_mc[g0, g, :]
                total = row.sum()
                if total < 1e-35:
                    continue

                cdf_mc_new = np.zeros(NUM_ANGLE_BINS + 1)
                cdf_mc_new[1:] = np.cumsum(row) / total
                cdf_mc_new[-1] = 1.0

                cdf_cmmc = np.array(
                    cmmc_obj.get_angle_cdf(temperature=T, g0=g0, g=g))

                max_cdf_diffs.append(np.max(np.abs(cdf_mc_new - cdf_cmmc)))

        if not max_cdf_diffs:
            pytest.skip("no significant transitions")

        median_diff = np.median(max_cdf_diffs)
        assert median_diff < 0.15, (
            f"median max CDF diff = {median_diff:.3f}")


# ---------------------------------------------------------------------------
# 2. Weight function invariance
# ---------------------------------------------------------------------------

class TestWeightFunctionInvariance:
    """Wien vs Planck produce different but valid matrices."""

    def test_different_weights_different_results(self):
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        mc_wien = _make_mc(
            bounds, weight_function=cm.WienWeightFunction(cap_x=25.0))
        mc_planck = _make_mc(
            bounds, weight_function=cm.PlanckWeightFunction(cap_x=25.0))

        S_wien = mc_wien.compute_sigma_matrix(T=T, Ne=1.0)
        S_planck = mc_planck.compute_sigma_matrix(T=T, Ne=1.0)

        assert S_wien.shape == S_planck.shape
        assert not np.allclose(S_wien, S_planck, rtol=0.01, atol=0), \
            "Wien and Planck should produce meaningfully different matrices"

    def test_all_weights_positive(self):
        """All weight functions produce non-negative matrices."""
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        for wf in [cm.WienWeightFunction(cap_x=25.0),
                    cm.PlanckWeightFunction(cap_x=25.0),
                    cm.UniformWeightFunction()]:
            mc_obj = _make_mc(bounds, weight_function=wf)
            S = mc_obj.compute_sigma_matrix(T=T, Ne=1.0)
            assert np.all(S >= 0), f"negative entries with {type(wf).__name__}"

    def test_row_sums_positive(self):
        """Row sums are positive for all weight functions."""
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        for wf in [cm.WienWeightFunction(cap_x=25.0),
                    cm.PlanckWeightFunction(cap_x=25.0),
                    cm.UniformWeightFunction()]:
            mc_obj = _make_mc(bounds, weight_function=wf)
            S = mc_obj.compute_sigma_matrix(T=T, Ne=1.0)
            rs = S.sum(axis=1)
            assert np.all(rs > 0), \
                f"non-positive row sums with {type(wf).__name__}"


# ---------------------------------------------------------------------------
# 3. Kernel multiplier correctness
# ---------------------------------------------------------------------------

class TestKernelMultiplier:
    """Explicit ConstantMultiplier matches default-argument call."""

    def test_constant_multiplier_matches_default(self):
        T = 10.0 * kev_kelvin

        mc_default = _make_mc(BOUNDARIES_ERG, seed=99)
        mc_explicit = _make_mc(BOUNDARIES_ERG, seed=99)

        S_default = mc_default.compute_sigma_matrix(T=T, Ne=1.0)
        S_explicit = mc_explicit.compute_sigma_matrix(
            T=T, Ne=1.0, multiplier=cm.ConstantMultiplier())

        np.testing.assert_array_equal(S_default, S_explicit)

    def test_constant_multiplier_multiangle(self):
        T = 10.0 * kev_kelvin

        mc_default = _make_mc(BOUNDARIES_ERG, seed=99)
        mc_explicit = _make_mc(BOUNDARIES_ERG, seed=99)

        S_default = mc_default.compute_sigma_matrix(
            num_angle_bins=4, T=T, Ne=1.0)
        S_explicit = mc_explicit.compute_sigma_matrix(
            num_angle_bins=4, T=T, Ne=1.0,
            multiplier=cm.ConstantMultiplier())

        np.testing.assert_array_equal(S_default, S_explicit)


# ---------------------------------------------------------------------------
# 4. Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    """Same seed produces bitwise identical results."""

    def test_same_seed_same_result(self):
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        mc1 = _make_mc(bounds, seed=123)
        mc2 = _make_mc(bounds, seed=123)

        S1 = mc1.compute_sigma_matrix(T=T, Ne=1.0)
        S2 = mc2.compute_sigma_matrix(T=T, Ne=1.0)

        np.testing.assert_array_equal(S1, S2)

    def test_different_seed_different_result(self):
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        mc1 = _make_mc(bounds, seed=123)
        mc2 = _make_mc(bounds, seed=456)

        S1 = mc1.compute_sigma_matrix(T=T, Ne=1.0)
        S2 = mc2.compute_sigma_matrix(T=T, Ne=1.0)

        assert not np.array_equal(S1, S2), \
            "different seeds should produce different results"


# ---------------------------------------------------------------------------
# 5. Basic sanity checks
# ---------------------------------------------------------------------------

class TestBasicSanity:
    """Smoke tests for API and output shape."""

    def test_num_groups(self):
        mc_obj = _make_mc(BOUNDARIES_ERG)
        assert mc_obj.num_groups == G

    def test_group_centers_geometric_mean(self):
        mc_obj = _make_mc(BOUNDARIES_ERG)
        for i in range(G):
            expected = math.sqrt(BOUNDARIES_ERG[i] * BOUNDARIES_ERG[i + 1])
            assert mc_obj.group_centers[i] == pytest.approx(expected)

    def test_output_shape_2d(self):
        mc_obj = _make_mc(BOUNDARIES_ERG, num_samples=1000)
        S = mc_obj.compute_sigma_matrix(T=10.0 * kev_kelvin, Ne=1.0)
        assert S.shape == (G, G)

    def test_output_shape_3d(self):
        mc_obj = _make_mc(BOUNDARIES_ERG, num_samples=1000)
        S = mc_obj.compute_sigma_matrix(
            num_angle_bins=4, T=10.0 * kev_kelvin, Ne=1.0)
        assert S.shape == (G, G, 4)

    def test_angle_sum_matches_integrated(self):
        """Sum over angle bins ~ angle-integrated result."""
        T = 10.0 * kev_kelvin
        mc_obj = _make_mc(BOUNDARIES_ERG)

        S_int = mc_obj.compute_sigma_matrix(T=T, Ne=1.0)

        mc_obj2 = _make_mc(BOUNDARIES_ERG, seed=SEED)
        S_ang = mc_obj2.compute_sigma_matrix(
            num_angle_bins=1, T=T, Ne=1.0)
        S_summed = S_ang.sum(axis=2)

        np.testing.assert_array_equal(S_int, S_summed)

    def test_invalid_boundaries(self):
        with pytest.raises(Exception):
            mc.ComptonMonteCarloKernel(
                energy_group_boundaries=[5.0 * kev, 1.0 * kev],
                weight_function=cm.WienWeightFunction(cap_x=25.0))

    def test_invalid_temperature(self):
        mc_obj = _make_mc(BOUNDARIES_ERG, num_samples=100)
        with pytest.raises(Exception):
            mc_obj.compute_sigma_matrix(T=-1.0, Ne=1.0)


# ---------------------------------------------------------------------------
# 6. Derivative golden-value regression
# ---------------------------------------------------------------------------

class TestDerivativeGolden:
    """Seed-locked golden values for compute_dsigma_dT_matrix."""

    def test_derivative_seed_reproducibility(self):
        """Same seed produces bitwise identical derivative matrices."""
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        mc1 = _make_mc(bounds, seed=42, num_samples=100_000)
        mc2 = _make_mc(bounds, seed=42, num_samples=100_000)

        dS1 = mc1.compute_dsigma_dT_matrix(T=T, Ne=1.0)
        dS2 = mc2.compute_dsigma_dT_matrix(T=T, Ne=1.0)

        np.testing.assert_array_equal(dS1, dS2)

    def test_derivative_golden_values(self):
        """Lock current output to detect unintentional behavior changes."""
        T = 10.0 * kev_kelvin
        bounds = [1.0 * kev, 5.0 * kev, 10.0 * kev]

        mc_obj = _make_mc(bounds, seed=42, num_samples=100_000)
        dS = mc_obj.compute_dsigma_dT_matrix(T=T, Ne=1.0)

        expected = np.array([
            [-6.35909511869973661076e-34,  6.31800766422992232794e-34],
            [ 7.46477093608892448577e-35, -7.43325988878795136172e-34],
        ])

        np.testing.assert_array_equal(
            dS, expected,
            err_msg="derivative golden values changed -- "
                    "update if intentional")

    def test_derivative_output_shape_2d(self):
        mc_obj = _make_mc(BOUNDARIES_ERG, num_samples=1000)
        dS = mc_obj.compute_dsigma_dT_matrix(T=10.0 * kev_kelvin, Ne=1.0)
        assert dS.shape == (G, G)

    def test_derivative_output_shape_3d(self):
        mc_obj = _make_mc(BOUNDARIES_ERG, num_samples=1000)
        dS = mc_obj.compute_dsigma_dT_matrix(
            num_angle_bins=4, T=10.0 * kev_kelvin, Ne=1.0)
        assert dS.shape == (G, G, 4)
