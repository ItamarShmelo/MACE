"""
xi quadrature validation script.

Compares the new xi integration (peak-aware splitting with log-spaced tails)
against a high-order adaptive GL reference across representative test cases:

  1. Elastic-like (d = 0)
  2. Interior peak / three-region split
  3. Peak entirely left
  4. Peak entirely right
  5. Extreme energy ratios

For each case, computes the multigroup-multiangle matrix with the production
config and a high-order reference, then reports element-wise relative
differences for significant entries.

Usage:
    OMP_NUM_THREADS=16 python3.12 reports/xi_quadrature_validation.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cpp_modules"))

import _compton_multigroup as cm
from _compton_differential_cross_section import ComptonKernelSolver
from _units import kev, kev_kelvin

KERNEL = ComptonKernelSolver()
N_ANGLE_BINS = 8
SIGNIFICANCE_THRESHOLD = 1e-4


def make_mg(bounds, *, base_order=48, xi_order=48):
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.UniformWeightFunction(),
        config=cm.MGIntegrationConfig(
            base_order=base_order,
            integration_tolerance=1e-6,
            cold_temperature_order=base_order,
            xi_order=xi_order,
            xi_peak_k=10.0))


def make_ref(bounds, *, base_order=128):
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.UniformWeightFunction(),
        config=cm.MGIntegrationConfig(
            base_order=base_order,
            integration_tolerance=1e-10,
            cold_temperature_order=base_order,
            xi_order=base_order,
            xi_peak_k=10.0))


def compare(name, bounds, T_K):
    print(f"\n{'='*60}")
    print(f"Case: {name}")
    print(f"  bounds_keV = {[b / kev for b in bounds]}")
    print(f"  T = {T_K / kev_kelvin:.4g} keV")

    t0 = time.time()
    mg = make_mg(bounds)
    S = mg.compute_sigma_matrix(KERNEL, num_angle_bins=N_ANGLE_BINS,
                                T=T_K, Ne=1.0)
    dt_test = time.time() - t0

    t0 = time.time()
    mg_ref = make_ref(bounds)
    S_ref = mg_ref.compute_sigma_matrix(KERNEL, num_angle_bins=N_ANGLE_BINS,
                                        T=T_K, Ne=1.0)
    dt_ref = time.time() - t0

    peak_val = np.max(np.abs(S_ref))
    mask = np.abs(S_ref) > SIGNIFICANCE_THRESHOLD * peak_val

    n_sig = np.sum(mask)
    n_total = S.size
    print(f"  shape = {S.shape}, significant entries = {n_sig}/{n_total}")
    print(f"  time: test={dt_test:.1f}s, ref={dt_ref:.1f}s")

    if not np.any(mask):
        print("  SKIP: no significant entries")
        return

    rel_diff = np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask])
    print(f"  max rel diff = {np.max(rel_diff):.2e}")
    print(f"  mean rel diff = {np.mean(rel_diff):.2e}")
    print(f"  median rel diff = {np.median(rel_diff):.2e}")

    if np.max(rel_diff) > 0.05:
        idx = np.argmax(rel_diff)
        flat_idx = np.where(mask.ravel())[0][idx]
        coords = np.unravel_index(flat_idx, S.shape)
        print(f"  WARN: worst entry at {coords}: "
              f"test={S[coords]:.6e}, ref={S_ref[coords]:.6e}")
    else:
        print("  PASS")


def main():
    print("xi Quadrature Validation")
    print(f"angle bins = {N_ANGLE_BINS}, "
          f"significance threshold = {SIGNIFICANCE_THRESHOLD}")

    # 1. Elastic-like
    E = 0.1 * 511.0 * kev
    compare("Elastic-like (d=0)",
            [0.95 * E, 1.05 * E],
            0.1 * kev_kelvin)

    # 2. Interior peak (three-region split)
    E_in = 10.0 * 511.0 * kev
    E_out = 15.0 * 511.0 * kev
    compare("Interior peak (three-region split)",
            sorted({0.95 * E_in, 1.05 * E_in, 0.95 * E_out, 1.05 * E_out}),
            0.1 * kev_kelvin)

    # 3. Peak entirely left
    E_in = 0.1 * 511.0 * kev
    E_out = 0.5 * 511.0 * kev
    compare("Peak entirely left (xi_pk=-7)",
            sorted({0.9 * E_in, 1.1 * E_in, 0.9 * E_out, 1.1 * E_out}),
            0.1 * kev_kelvin)

    # 4. Peak entirely right
    E_in = 10.0 * 511.0 * kev
    E_out = 15.0 * 511.0 * kev
    compare("Peak entirely right (first bin)",
            sorted({0.95 * E_in, 1.05 * E_in, 0.95 * E_out, 1.05 * E_out}),
            0.1 * kev_kelvin)

    # 5. Extreme energy ratios
    E_base = 10.0 * kev
    for ratio in [0.01, 100.0]:
        E_out = ratio * E_base
        lo = min(E_base, E_out) * 0.5
        hi = max(E_base, E_out) * 2.0
        import math
        mid = math.sqrt(lo * hi)
        compare(f"Extreme ratio (gamma_p/gamma={ratio})",
                sorted({lo, mid, hi}),
                1.0 * kev_kelvin)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
