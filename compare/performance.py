"""Performance comparison: ComptonKernelApproximate vs ComptonKernelSolver.

Measures evaluation throughput for both solvers across representative
parameter regimes with reproducible timing protocol.
"""

import time
from pathlib import Path

import numpy as np

from compton_matrix._compton_differential_cross_section import (
    ComptonKernelApproximate,
    ComptonKernelSolver,
)
from compton_matrix._units import kev, kev_kelvin

ME_C2 = 9.109383713928e-28 * (2.99792458e10) ** 2

# Representative timing grid
TEMPERATURES_KEV = [0.1, 10.0, 100.0]  # cold, warm, hot
INCIDENT_RATIOS = [0.01, 1.0, 10.0]  # low, unity, high
XI_VALUES = [-1.0, 0.0, 0.9]  # backscatter, side, forward

N_EPRIME = 20
N_WARMUP = 5
N_BATCHES = 10
N_EVALS_PER_BATCH = 100

APPROX = ComptonKernelApproximate()
SOLVER = ComptonKernelSolver()


def cold_compton_line(E, xi):
    gamma = E / ME_C2
    return E / (1.0 + gamma * (1.0 - xi))


def build_timing_grid(E, xi):
    """Build a fixed grid of outgoing energies for timing."""
    E_prime_C = cold_compton_line(E, xi)
    return np.geomspace(E_prime_C * 0.5, E_prime_C * 2.0, N_EPRIME)


def time_solver(solver, E, E_prime_arr, xi, T):
    """Time a solver with warm-up, batches, and per-evaluation timing."""
    # Warm-up
    for _ in range(N_WARMUP):
        for Ep in E_prime_arr:
            solver.sigma_E(E, Ep, xi, T)

    batch_times = []
    for _ in range(N_BATCHES):
        t0 = time.perf_counter_ns()
        for _ in range(N_EVALS_PER_BATCH):
            for Ep in E_prime_arr:
                solver.sigma_E(E, Ep, xi, T)
        t1 = time.perf_counter_ns()
        batch_times.append((t1 - t0) / (N_EVALS_PER_BATCH * len(E_prime_arr)))

    return np.array(batch_times)


def time_solver_vec(solver, E, E_prime_arr, xi, T):
    """Time the vectorized interface."""
    # Warm-up
    for _ in range(N_WARMUP):
        solver.sigma_E_vec(E, E_prime_arr, xi, T)

    batch_times = []
    for _ in range(N_BATCHES):
        t0 = time.perf_counter_ns()
        for _ in range(N_EVALS_PER_BATCH):
            solver.sigma_E_vec(E, E_prime_arr, xi, T)
        t1 = time.perf_counter_ns()
        batch_times.append((t1 - t0) / (N_EVALS_PER_BATCH * len(E_prime_arr)))

    return np.array(batch_times)


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    total = len(TEMPERATURES_KEV) * len(INCIDENT_RATIOS) * len(XI_VALUES)
    print(f"Running performance comparison over {total} configurations...")
    print(f"  {N_WARMUP} warm-up calls, {N_BATCHES} batches x "
          f"{N_EVALS_PER_BATCH} evaluations x {N_EPRIME} E' points")
    print()
    print(f"{'T_keV':>8} {'E/kT':>8} {'xi':>6} | "
          f"{'Approx_ns':>10} {'Solver_ns':>10} {'Speedup':>8} | "
          f"{'Apx_vec_ns':>10} {'Sol_vec_ns':>10} {'Speedup_v':>9}")
    print("-" * 95)

    for T_kev in TEMPERATURES_KEV:
        T = T_kev * kev_kelvin
        for ratio in INCIDENT_RATIOS:
            E = ratio * T_kev * kev
            for xi in XI_VALUES:
                E_prime_arr = build_timing_grid(E, xi)

                # Scalar timing
                approx_times = time_solver(APPROX, E, E_prime_arr, xi, T)
                solver_times = time_solver(SOLVER, E, E_prime_arr, xi, T)

                # Vectorized timing
                approx_vec_times = time_solver_vec(APPROX, E, E_prime_arr, xi, T)
                solver_vec_times = time_solver_vec(SOLVER, E, E_prime_arr, xi, T)

                approx_median = np.median(approx_times)
                solver_median = np.median(solver_times)
                speedup = solver_median / approx_median if approx_median > 0 else np.inf

                approx_vec_median = np.median(approx_vec_times)
                solver_vec_median = np.median(solver_vec_times)
                speedup_vec = (
                    solver_vec_median / approx_vec_median
                    if approx_vec_median > 0
                    else np.inf
                )

                results.append({
                    "T_kev": T_kev,
                    "E_over_kT": ratio,
                    "xi": xi,
                    "approx_median_ns": approx_median,
                    "approx_iqr_ns": np.percentile(approx_times, 75)
                    - np.percentile(approx_times, 25),
                    "solver_median_ns": solver_median,
                    "solver_iqr_ns": np.percentile(solver_times, 75)
                    - np.percentile(solver_times, 25),
                    "speedup": speedup,
                    "approx_vec_median_ns": approx_vec_median,
                    "solver_vec_median_ns": solver_vec_median,
                    "speedup_vec": speedup_vec,
                })

                print(
                    f"{T_kev:>8.1f} {ratio:>8.3g} {xi:>6.2f} | "
                    f"{approx_median:>10.0f} {solver_median:>10.0f} "
                    f"{speedup:>8.2f}x | "
                    f"{approx_vec_median:>10.0f} {solver_vec_median:>10.0f} "
                    f"{speedup_vec:>9.2f}x"
                )

    # Summary
    speedups = [r["speedup"] for r in results if np.isfinite(r["speedup"])]
    vec_speedups = [
        r["speedup_vec"] for r in results if np.isfinite(r["speedup_vec"])
    ]

    print("\n" + "=" * 95)
    print("SUMMARY (scalar)")
    print("=" * 95)
    print(f"Median speedup: {np.median(speedups):.2f}x")
    print(f"Mean speedup:   {np.mean(speedups):.2f}x")
    print(f"Min speedup:    {np.min(speedups):.2f}x")
    print(f"Max speedup:    {np.max(speedups):.2f}x")

    print(f"\nSUMMARY (vectorized)")
    print(f"Median speedup: {np.median(vec_speedups):.2f}x")
    print(f"Mean speedup:   {np.mean(vec_speedups):.2f}x")
    print(f"Min speedup:    {np.min(vec_speedups):.2f}x")
    print(f"Max speedup:    {np.max(vec_speedups):.2f}x")

    # Save results
    np.savez(
        output_dir / "performance_results.npz",
        T_kev=np.array([r["T_kev"] for r in results]),
        E_over_kT=np.array([r["E_over_kT"] for r in results]),
        xi=np.array([r["xi"] for r in results]),
        approx_median_ns=np.array([r["approx_median_ns"] for r in results]),
        solver_median_ns=np.array([r["solver_median_ns"] for r in results]),
        speedup=np.array([r["speedup"] for r in results]),
        approx_vec_median_ns=np.array([r["approx_vec_median_ns"] for r in results]),
        solver_vec_median_ns=np.array(
            [r["solver_vec_median_ns"] for r in results]
        ),
        speedup_vec=np.array([r["speedup_vec"] for r in results]),
    )
    print(f"\nResults saved to {output_dir / 'performance_results.npz'}")

    # Generate bar chart if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [
            f"T={r['T_kev']},r={r['E_over_kT']},xi={r['xi']:.1f}"
            for r in results
        ]
        speedup_vals = [r["speedup"] for r in results]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(labels))
        bars = ax.bar(x, speedup_vals, color="steelblue")
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Speedup (Solver time / Approximate time)")
        ax.set_title("ComptonKernelApproximate speedup over ComptonKernelSolver")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(output_dir / "performance_speedup.png", dpi=150)
        plt.close(fig)
        print(f"Bar chart saved to {output_dir / 'performance_speedup.png'}")

    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
