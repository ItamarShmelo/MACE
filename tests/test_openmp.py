"""
OpenMP parallelization correctness tests.

Validates that:
  1. Deterministic results are bitwise identical across thread counts.
  2. Monte Carlo results agree statistically across thread counts.

Since OMP_NUM_THREADS is read at process startup, each case runs in a
separate subprocess that writes results to a temp .npy file.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

WORKER_DETERMINISTIC = textwrap.dedent("""\
    import sys, os, numpy as np
    import compton_matrix._compton_multigroup as cm
    import compton_matrix._compton_differential_cross_section as cq
    from compton_matrix._units import kev, kev_kelvin

    boundaries = [0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    wf = cm.CappedWienWeightFunction(cap_x=25.0)
    kernel = cq.ComptonKernelSolver()
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries,
        weight_function=wf,
        config=cm.MGIntegrationConfig(
            xi_order=8,
            xi_tail_order=8,
            ep_edge_order=8,
            ep_interior_order=8,
            e_panel_order=8))

    T = 10.0 * kev_kelvin
    S = mg.compute_sigma_matrix(kernel=kernel, num_angle_bins=2, T=T)
    np.save(sys.argv[1], S)
""")

WORKER_MONTE_CARLO = textwrap.dedent("""\
    import sys, os, numpy as np
    import compton_matrix._compton_multigroup as cm
    from compton_matrix._units import kev, kev_kelvin

    boundaries = [0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]
    wf = cm.CappedWienWeightFunction(cap_x=25.0)
    mc = cm.ComptonMonteCarloKernel(
        energy_group_boundaries=boundaries,
        weight_function=wf,
        config=cm.MCIntegrationConfig(num_samples=2_000_000, seed=42))

    T = 10.0 * kev_kelvin
    S = mc.compute_sigma_matrix(num_angle_bins=2, T=T)
    np.save(sys.argv[1], S)
""")


def _run_worker(script: str, num_threads: int, out_path: str):
    """Run a Python worker script in a subprocess with given OMP_NUM_THREADS."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    result = subprocess.run([sys.executable, "-c", script, out_path], env=env, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"Worker failed (OMP_NUM_THREADS={num_threads}):\n"
            f"stdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}"
        )


class TestDeterministicParallel:
    """Deterministic results must be bitwise identical across thread counts."""

    def test_1_vs_4_threads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_1 = os.path.join(tmpdir, "det_1.npy")
            path_4 = os.path.join(tmpdir, "det_4.npy")

            _run_worker(WORKER_DETERMINISTIC, 1, path_1)
            _run_worker(WORKER_DETERMINISTIC, 4, path_4)

            S1 = np.load(path_1)
            S4 = np.load(path_4)

            np.testing.assert_array_equal(S1, S4, err_msg="Deterministic results differ between 1 and 4 threads")


class TestMonteCarloParallel:
    """MC results must agree statistically across thread counts."""

    def test_1_vs_4_threads_relative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_1 = os.path.join(tmpdir, "mc_1.npy")
            path_4 = os.path.join(tmpdir, "mc_4.npy")

            _run_worker(WORKER_MONTE_CARLO, 1, path_1)
            _run_worker(WORKER_MONTE_CARLO, 4, path_4)

            S1 = np.load(path_1)
            S4 = np.load(path_4)

            assert S1.shape == S4.shape

            denom = np.maximum(np.abs(S1), np.abs(S4))
            # Only compare bins with significant signal (> 1% of peak)
            peak = np.max(denom)
            mask = denom > 0.01 * peak

            if not np.any(mask):
                pytest.skip("all entries near zero")

            rel_diff = np.abs(S1[mask] - S4[mask]) / denom[mask]
            max_rel = np.max(rel_diff)

            assert max_rel < 0.05, f"max relative difference = {max_rel:.4e}, expected < 0.05"

    def test_1_vs_4_threads_row_sums(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_1 = os.path.join(tmpdir, "mc_1.npy")
            path_4 = os.path.join(tmpdir, "mc_4.npy")

            _run_worker(WORKER_MONTE_CARLO, 1, path_1)
            _run_worker(WORKER_MONTE_CARLO, 4, path_4)

            S1 = np.load(path_1)
            S4 = np.load(path_4)

            rs1 = S1.reshape(S1.shape[0], -1).sum(axis=1)
            rs4 = S4.reshape(S4.shape[0], -1).sum(axis=1)

            denom = np.maximum(np.abs(rs1), np.abs(rs4))
            mask = denom > 1e-35

            if not np.any(mask):
                pytest.skip("all row sums near zero")

            rel_diff = np.abs(rs1[mask] - rs4[mask]) / denom[mask]
            max_rel = np.max(rel_diff)

            assert max_rel < 0.005, f"row-sum max relative difference = {max_rel:.4e}, expected < 0.005"
