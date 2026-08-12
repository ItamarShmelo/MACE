"""
Point-kernel benchmark: accuracy and timing verification.

Replicates the structured 3024-point grid from benchmark_approximate_solver.cpp.
Outputs JSON to stdout.
"""

import json
import math
import sys
import time

import numpy as np

import compton_matrix as cm


def make_grid():
    """Construct the same 3024-point grid as the C++ benchmark."""
    temperatures_kev = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 70.0, 100.0]
    ratios = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0]
    angles = [-0.999, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999]
    offsets = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

    points = []
    for kT_kev in temperatures_kev:
        tau = kT_kev * cm.kev / cm.me_c2
        for ratio in ratios:
            E = ratio * kT_kev * cm.kev
            gamma = E / cm.me_c2
            for xi in angles:
                cold_line = gamma / (1.0 + gamma * (1.0 - xi))
                ep_list = []
                for offset in offsets:
                    gamma_prime = cold_line * math.exp(
                        offset * math.sqrt(2.0 * tau * (1.0 - xi))
                    )
                    ep_list.append(gamma_prime * cm.me_c2)
                points.append({
                    "E": E,
                    "xi": xi,
                    "T": kT_kev * cm.kev_kelvin,
                    "E_primes": ep_list,
                })
    return points


def compute_accuracy(groups):
    """Evaluate accuracy of ApproximateSolver vs Solver on the full grid."""
    solver = cm.ComptonKernelSolver()
    approx_solver = cm.ComptonKernelApproximateSolver()
    approximate = cm.ComptonKernelApproximate()

    relative_errors = []
    approx_raw_relative_errors = []
    failures = 0
    approximate_failures = 0
    approximate_rejections = 0
    approximate_accepted = 0

    for group in groups:
        E = group["E"]
        xi = group["xi"]
        T = group["T"]
        for Ep in group["E_primes"]:
            ref = solver.sigma_E(E, Ep, xi, T)
            cand = approx_solver.sigma_E(E, Ep, xi, T)

            if cand.estimated_abs_error == 1.0 and cand.value == 0.0:
                failures += 1
            elif ref.value != 0.0:
                relative_errors.append(
                    abs(cand.value - ref.value) / abs(ref.value)
                )

            raw = approximate.sigma_E(E, Ep, xi, T)
            if raw.estimated_abs_error == 1.0:
                approximate_failures += 1
            elif raw.estimated_rel_error >= 3e-4:
                approximate_rejections += 1
            else:
                approximate_accepted += 1
                if ref.value != 0.0:
                    approx_raw_relative_errors.append(
                        abs(raw.value - ref.value) / abs(ref.value)
                    )

    relative_errors.sort()
    n = len(relative_errors)

    def percentile(arr, frac):
        idx = int(frac * (len(arr) - 1))
        return arr[idx]

    results = {
        "points": sum(len(g["E_primes"]) for g in groups),
        "solver_failures": failures,
        "approximate_accepted": approximate_accepted,
        "approximate_failures": approximate_failures,
        "approximate_rejections": approximate_rejections,
        "relative_median": percentile(relative_errors, 0.5),
        "relative_p95": percentile(relative_errors, 0.95),
        "relative_p99": percentile(relative_errors, 0.99),
        "relative_max": max(relative_errors),
    }
    if approx_raw_relative_errors:
        results["approximate_relative_max"] = max(approx_raw_relative_errors)

    return results


def time_solver_vec(groups, kernel, repeats=120, samples=9):
    """Time kernel using sigma_E_vec, returning ns/eval statistics."""
    total_evals = sum(len(g["E_primes"]) for g in groups) * repeats
    timings = []

    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            for group in groups:
                E = group["E"]
                xi = group["xi"]
                T = group["T"]
                ep_arr = np.array(group["E_primes"], dtype=np.float64)
                kernel.sigma_E_vec(E, ep_arr, xi, T)
        elapsed_ns = time.perf_counter_ns() - start
        timings.append(elapsed_ns / total_evals)

    timings.sort()
    return {
        "min_ns": timings[0],
        "median_ns": timings[len(timings) // 2],
        "max_ns": timings[-1],
    }


def main():
    print("Building grid...", file=sys.stderr)
    groups = make_grid()
    total_points = sum(len(g["E_primes"]) for g in groups)
    print(f"Grid: {len(groups)} groups, {total_points} points", file=sys.stderr)

    print("Computing accuracy...", file=sys.stderr)
    accuracy = compute_accuracy(groups)

    print("Timing original solver (vec)...", file=sys.stderr)
    original_timing = time_solver_vec(groups, cm.ComptonKernelSolver())

    print("Timing approximate solver (vec)...", file=sys.stderr)
    approx_solver_timing = time_solver_vec(groups, cm.ComptonKernelApproximateSolver())

    # For raw approximate, filter to accepted points only
    approximate = cm.ComptonKernelApproximate()
    solver = cm.ComptonKernelSolver()
    accepted_groups = []
    for group in groups:
        E = group["E"]
        xi = group["xi"]
        T = group["T"]
        accepted_eps = []
        for Ep in group["E_primes"]:
            raw = approximate.sigma_E(E, Ep, xi, T)
            if raw.estimated_abs_error != 1.0 and raw.estimated_rel_error < 3e-4:
                accepted_eps.append(Ep)
        if accepted_eps:
            accepted_groups.append({"E": E, "xi": xi, "T": T, "E_primes": accepted_eps})

    print("Timing raw approximate (vec, accepted subset)...", file=sys.stderr)
    approx_raw_timing = time_solver_vec(accepted_groups, cm.ComptonKernelApproximate())

    results = {
        "accuracy": accuracy,
        "timing": {
            "original_solver": original_timing,
            "approximate_solver": approx_solver_timing,
            "raw_approximate_accepted_subset": approx_raw_timing,
        },
        "speedup_solver": original_timing["median_ns"] / approx_solver_timing["median_ns"],
        "speedup_raw": original_timing["median_ns"] / approx_raw_timing["median_ns"],
    }

    print(json.dumps(results, indent=2))
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
