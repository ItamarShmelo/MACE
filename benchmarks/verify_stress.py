"""
Stress benchmark: 4 matrix families x 14 temperatures, 5% significant-cell gate.

Replicates benchmark_multigroup_stress.cpp configuration.
Exits with code 1 if any significant-cell error >= 5%.
Outputs JSON to stdout.
"""

import json
import math
import sys
import time

import numpy as np

import compton_matrix as cm


SCENARIOS = [
    {
        "name": "standard_uniform",
        "boundaries_kev": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "angle_bins": 4,
        "wien": False,
    },
    {
        "name": "fine_uniform",
        "boundaries_kev": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "angle_bins": 8,
        "wien": False,
    },
    {
        "name": "broad_uniform",
        "boundaries_kev": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "angle_bins": 8,
        "wien": False,
    },
    {
        "name": "fine_wien",
        "boundaries_kev": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "angle_bins": 8,
        "wien": True,
    },
]

TEMPERATURES_KEV = [
    1.0, 10.0, 50.0, 100.0, 150.0, 200.0,
    220.0, 224.0, 225.0, 228.0, 229.0, 229.9, 230.0, 250.0,
]


def build_config():
    """Build MGIntegrationConfig matching the C++ stress benchmark."""
    return cm.MGIntegrationConfig(
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


def compute_accuracy(candidate_matrix, reference_matrix, groups, angle_bins):
    """Compute accuracy metrics matching C++ benchmark."""
    cand = np.asarray(candidate_matrix, dtype=np.float64).ravel()
    ref = np.asarray(reference_matrix, dtype=np.float64).ravel()

    finite_mask = np.isfinite(ref) & np.isfinite(cand)
    cand = np.where(finite_mask, cand, 0.0)
    ref = np.where(finite_mask, ref, 0.0)

    reference_max = np.max(np.abs(ref)) if ref.size > 0 else 0.0
    differences = np.abs(cand - ref)

    l1_difference = float(np.sum(differences))
    l1_reference = float(np.sum(np.abs(ref)))
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
    config = build_config()
    reference_solver = cm.ComptonKernelSolver()
    candidate_solver = cm.ComptonKernelApproximateSolver()

    passed = True
    results = {"scenarios": []}

    for scenario in SCENARIOS:
        boundaries_erg = [b * cm.kev for b in scenario["boundaries_kev"]]
        groups = len(boundaries_erg) - 1

        if scenario["wien"]:
            weight = cm.CappedWienWeightFunction(cap_x=700.0)
        else:
            weight = cm.UniformWeightFunction()

        mg = cm.ComptonMultigroupKernel(boundaries_erg, weight, config)

        scenario_results = {"name": scenario["name"], "temperatures": []}

        for T_kev in TEMPERATURES_KEV:
            T = T_kev * cm.kev_kelvin

            print(
                f"  {scenario['name']} T={T_kev} keV ...",
                file=sys.stderr,
                flush=True,
            )

            start = time.perf_counter()
            ref_matrix = mg.compute_sigma_matrix(
                reference_solver, scenario["angle_bins"], T
            )
            ref_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            cand_matrix = mg.compute_sigma_matrix(
                candidate_solver, scenario["angle_bins"], T
            )
            cand_ms = (time.perf_counter() - start) * 1000.0

            acc = compute_accuracy(
                cand_matrix, ref_matrix, groups, scenario["angle_bins"]
            )

            is_finite = math.isfinite(acc["max_significant_relative"])
            under_threshold = acc["max_significant_relative"] < 0.05
            if not (is_finite and under_threshold):
                passed = False

            entry = {
                "T_kev": T_kev,
                "original_ms": ref_ms,
                "approximate_ms": cand_ms,
                "speedup": ref_ms / cand_ms if cand_ms > 0 else float("inf"),
                "l1_relative": acc["l1_relative"],
                "max_significant_relative": acc["max_significant_relative"],
                "max_row_sum_relative": acc["max_row_sum_relative"],
                "pass": is_finite and under_threshold,
            }
            scenario_results["temperatures"].append(entry)

        results["scenarios"].append(scenario_results)

    results["overall_pass"] = passed
    print(json.dumps(results, indent=2))

    if not passed:
        print("FAILED: significant-cell error exceeded 5%", file=sys.stderr)
        sys.exit(1)
    else:
        print("PASSED: all significant-cell errors below 5%", file=sys.stderr)


if __name__ == "__main__":
    main()
