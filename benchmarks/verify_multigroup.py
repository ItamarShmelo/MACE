"""
Multigroup matrix benchmark: timing and accuracy verification.

Replicates benchmark_multigroup.cpp configuration.
Accepts --threads argument to label output.
Outputs JSON to stdout.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import compton_matrix as cm


def build_multigroup():
    """Build the 4-group, uniform-weight multigroup kernel matching C++ benchmark."""
    boundaries = [0.1, 1.0, 10.0, 100.0, 1000.0]
    boundaries_erg = [b * cm.kev for b in boundaries]

    config = cm.MGIntegrationConfig(
        cutoff_ratio=None,
        xi_order=24,
        xi_peak_k=5.0,
        xi_tail_order=16,
        ep_k_cut=5.0,
        ep_k_in=2.0,
        ep_edge_order=24,
        ep_interior_order=24,
        e_panel_order=8,
        log_e_panel_ratio=2.0,
        e_boundary_k=5.0,
    )

    mg = cm.ComptonMultigroupKernel(
        boundaries_erg,
        cm.UniformWeightFunction(),
        config,
    )
    return mg, len(boundaries) - 1


def measure(mg, kernel, angle_bins, T, trials=3):
    """Time matrix computation, return (best_ms, matrix)."""
    best_ms = float("inf")
    best_matrix = None
    for _ in range(trials):
        start = time.perf_counter()
        matrix = mg.compute_sigma_matrix(kernel, angle_bins, T)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_matrix = matrix
    return best_ms, best_matrix


def compute_accuracy(candidate_matrix, reference_matrix, groups, angle_bins):
    """Compute L1 relative, max significant-cell relative, and max row-sum relative errors."""
    cand = np.asarray(candidate_matrix, dtype=np.float64).ravel()
    ref = np.asarray(reference_matrix, dtype=np.float64).ravel()

    finite_mask = np.isfinite(ref) & np.isfinite(cand)
    cand = np.where(finite_mask, cand, 0.0)
    ref = np.where(finite_mask, ref, 0.0)

    reference_max = np.max(np.abs(ref)) if ref.size > 0 else 0.0
    differences = np.abs(cand - ref)

    l1_difference = np.sum(differences)
    l1_reference = np.sum(np.abs(ref))
    l1_relative = l1_difference / (l1_reference + 1e-300)

    significant_mask = np.abs(ref) > 1e-8 * reference_max
    if np.any(significant_mask):
        max_significant_relative = float(
            np.max(differences[significant_mask] / np.abs(ref[significant_mask]))
        )
    else:
        max_significant_relative = 0.0

    max_row_sum_relative = 0.0
    for g in range(groups):
        cand_sum = 0.0
        ref_sum = 0.0
        for gp in range(groups):
            for angle in range(angle_bins):
                idx = g * groups * angle_bins + gp * angle_bins + angle
                cand_sum += cand[idx]
                ref_sum += ref[idx]
        row_err = abs(cand_sum - ref_sum) / (abs(ref_sum) + 1e-300)
        max_row_sum_relative = max(max_row_sum_relative, row_err)

    return {
        "l1_relative": l1_relative,
        "max_significant_relative": max_significant_relative,
        "max_row_sum_relative": max_row_sum_relative,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    omp_threads = int(os.environ.get("OMP_NUM_THREADS", args.threads))
    print(f"OMP_NUM_THREADS={omp_threads}, label={args.threads}", file=sys.stderr)

    mg, groups = build_multigroup()
    angle_bins = 4
    temperatures_kev = [1.0, 10.0, 100.0]

    reference_solver = cm.ComptonKernelSolver()
    approximate_solver = cm.ComptonKernelApproximateSolver()

    results = {"threads": args.threads, "measurements": []}

    for T_kev in temperatures_kev:
        T = T_kev * cm.kev_kelvin
        print(f"  T={T_kev} keV: timing reference...", file=sys.stderr)
        ref_ms, ref_matrix = measure(mg, reference_solver, angle_bins, T)
        print(f"  T={T_kev} keV: timing approximate...", file=sys.stderr)
        approx_ms, approx_matrix = measure(mg, approximate_solver, angle_bins, T)

        acc = compute_accuracy(approx_matrix, ref_matrix, groups, angle_bins)

        entry = {
            "T_kev": T_kev,
            "original_ms": ref_ms,
            "approximate_ms": approx_ms,
            "speedup": ref_ms / approx_ms,
            "accuracy": acc,
        }
        results["measurements"].append(entry)
        print(
            f"  T={T_kev} keV: original={ref_ms:.2f}ms, "
            f"approx={approx_ms:.2f}ms, speedup={ref_ms/approx_ms:.2f}x, "
            f"max_sig_err={acc['max_significant_relative']:.2e}",
            file=sys.stderr,
        )

    print(json.dumps(results, indent=2))
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
