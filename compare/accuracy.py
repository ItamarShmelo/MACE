"""Accuracy comparison: ComptonKernelApproximate vs ComptonKernelSolver.

Evaluates both solvers across the declared benchmark domain with an adaptive
outgoing-energy grid, then reports per-regime accuracy metrics.
"""

import sys
from pathlib import Path

import numpy as np

from compton_matrix._compton_differential_cross_section import (
    ComptonKernelApproximate,
    ComptonKernelSolver,
)
from compton_matrix._units import kev, kev_kelvin

ME_C2 = 9.109383713928e-28 * (2.99792458e10) ** 2

# Benchmark domain
TEMPERATURES_KEV = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 70.0, 100.0]
INCIDENT_RATIOS = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0]
XI_VALUES = [-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999]

APPROX = ComptonKernelApproximate()
SOLVER = ComptonKernelSolver()


def cold_compton_line(E, xi):
    gamma = E / ME_C2
    return E / (1.0 + gamma * (1.0 - xi))


def build_adaptive_grid(E, xi, T, n_coarse=30, n_fine=50):
    """Two-pass adaptive outgoing-energy grid centered on the cold line."""
    E_prime_C = cold_compton_line(E, xi)

    lo = max(E_prime_C * 0.01, 1e-30)
    hi = E_prime_C * 100.0
    E_prime_coarse = np.geomspace(lo, hi, n_coarse)

    ref_values = np.zeros(n_coarse)
    for i, Ep in enumerate(E_prime_coarse):
        r = SOLVER.sigma_E(E, Ep, xi, T)
        ref_values[i] = r.value if r.estimated_abs_error < 1.0 else 0.0

    peak_val = np.max(ref_values)
    if peak_val <= 0:
        return E_prime_coarse

    threshold = peak_val * 1e-6
    mask = ref_values > threshold
    if not np.any(mask):
        return E_prime_coarse

    indices = np.where(mask)[0]
    lo_fine = E_prime_coarse[max(indices[0] - 1, 0)]
    hi_fine = E_prime_coarse[min(indices[-1] + 1, n_coarse - 1)]

    return np.geomspace(lo_fine, hi_fine, n_fine)


def evaluate_triple(E, xi, T):
    """Evaluate approximate solver and reference on an adaptive grid, return metrics."""
    E_prime_arr = build_adaptive_grid(E, xi, T)
    n = len(E_prime_arr)

    approx_vals = np.zeros(n)
    ref_vals = np.zeros(n)
    ref_errs = np.zeros(n)
    approx_failures = 0

    for i, Ep in enumerate(E_prime_arr):
        approx_r = APPROX.sigma_E(E, Ep, xi, T)
        ref_r = SOLVER.sigma_E(E, Ep, xi, T)

        if approx_r.estimated_abs_error == 1.0:
            approx_failures += 1
            approx_vals[i] = np.nan
        else:
            approx_vals[i] = approx_r.value

        if ref_r.estimated_abs_error >= 1.0:
            ref_vals[i] = np.nan
            ref_errs[i] = np.nan
        else:
            ref_vals[i] = ref_r.value
            ref_errs[i] = ref_r.estimated_abs_error

    peak_val = np.nanmax(ref_vals) if np.any(np.isfinite(ref_vals)) else 0.0
    if peak_val <= 0:
        return {
            "peak_val": 0.0,
            "envelope_fraction": np.nan,
            "peak_abs_error": np.nan,
            "profile_weighted_error": np.nan,
            "max_rel_error_masked": np.nan,
            "approx_failures": approx_failures,
            "n_points": n,
        }

    mask_threshold = 1e-4 * peak_val
    valid = (
        np.isfinite(approx_vals)
        & np.isfinite(ref_vals)
        & (ref_vals > mask_threshold)
        & (ref_errs < ref_vals)
    )

    if not np.any(valid):
        return {
            "peak_val": peak_val,
            "envelope_fraction": np.nan,
            "peak_abs_error": np.nan,
            "profile_weighted_error": np.nan,
            "max_rel_error_masked": np.nan,
            "approx_failures": approx_failures,
            "n_points": n,
        }

    abs_diff = np.abs(approx_vals[valid] - ref_vals[valid])
    within_envelope = abs_diff <= ref_errs[valid]
    envelope_fraction = np.mean(within_envelope)

    peak_idx = np.nanargmax(ref_vals)
    peak_abs_error = (
        abs(approx_vals[peak_idx] - ref_vals[peak_idx])
        if np.isfinite(approx_vals[peak_idx])
        else np.nan
    )

    ref_v = ref_vals[valid]
    profile_weighted = np.sum(abs_diff * ref_v) / np.sum(ref_v**2)

    rel_errors = abs_diff / np.abs(ref_v)
    max_rel_error = np.max(rel_errors)

    return {
        "peak_val": peak_val,
        "envelope_fraction": envelope_fraction,
        "peak_abs_error": peak_abs_error,
        "profile_weighted_error": profile_weighted,
        "max_rel_error_masked": max_rel_error,
        "approx_failures": approx_failures,
        "n_points": n,
    }


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    cancellation_regimes = []

    total = len(TEMPERATURES_KEV) * len(INCIDENT_RATIOS) * len(XI_VALUES)
    count = 0

    print(f"Running accuracy comparison over {total} parameter triples...")
    print(f"{'T_keV':>8} {'E/kT':>8} {'xi':>8} | {'env_frac':>9} {'max_rel':>9} "
          f"{'prof_wt':>9} {'failures':>8}")
    print("-" * 80)

    for T_kev in TEMPERATURES_KEV:
        T = T_kev * kev_kelvin
        for ratio in INCIDENT_RATIOS:
            E = ratio * T_kev * kev
            for xi in XI_VALUES:
                count += 1
                metrics = evaluate_triple(E, xi, T)
                metrics["T_kev"] = T_kev
                metrics["E_over_kT"] = ratio
                metrics["xi"] = xi
                results.append(metrics)

                env_frac = metrics["envelope_fraction"]
                max_rel = metrics["max_rel_error_masked"]
                prof_wt = metrics["profile_weighted_error"]
                failures = metrics["approx_failures"]

                env_str = f"{env_frac:.4f}" if np.isfinite(env_frac) else "N/A"
                rel_str = f"{max_rel:.2e}" if np.isfinite(max_rel) else "N/A"
                pwt_str = f"{prof_wt:.2e}" if np.isfinite(prof_wt) else "N/A"

                print(
                    f"{T_kev:>8.2f} {ratio:>8.3g} {xi:>8.3f} | "
                    f"{env_str:>9} {rel_str:>9} {pwt_str:>9} {failures:>8}"
                )

                if np.isfinite(max_rel) and max_rel > 0.10:
                    cancellation_regimes.append(metrics)

    # Summary statistics
    valid_results = [
        r for r in results if np.isfinite(r["max_rel_error_masked"])
    ]

    if valid_results:
        all_max_rel = [r["max_rel_error_masked"] for r in valid_results]
        all_env_frac = [r["envelope_fraction"] for r in valid_results]
        all_prof_wt = [r["profile_weighted_error"] for r in valid_results]
        total_failures = sum(r["approx_failures"] for r in results)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total parameter triples evaluated: {total}")
        print(f"Triples with valid comparison:     {len(valid_results)}")
        print(f"Total approximate failures:        {total_failures}")
        print()
        print(f"Max relative error (masked):  median={np.median(all_max_rel):.3e}, "
              f"mean={np.mean(all_max_rel):.3e}, max={np.max(all_max_rel):.3e}")
        print(f"Envelope fraction:            median={np.median(all_env_frac):.4f}, "
              f"mean={np.mean(all_env_frac):.4f}, min={np.min(all_env_frac):.4f}")
        print(f"Profile-weighted error:       median={np.median(all_prof_wt):.3e}, "
              f"mean={np.mean(all_prof_wt):.3e}, max={np.max(all_prof_wt):.3e}")

    if cancellation_regimes:
        print(f"\nCANCELLATION-PRONE REGIMES (max_rel > 10%): "
              f"{len(cancellation_regimes)} triples")
        print(f"{'T_keV':>8} {'E/kT':>8} {'xi':>8} {'max_rel':>10}")
        for r in cancellation_regimes:
            print(f"{r['T_kev']:>8.2f} {r['E_over_kT']:>8.3g} "
                  f"{r['xi']:>8.3f} {r['max_rel_error_masked']:>10.3e}")

    # Save raw results
    np.savez(
        output_dir / "accuracy_results.npz",
        T_kev=np.array([r["T_kev"] for r in results]),
        E_over_kT=np.array([r["E_over_kT"] for r in results]),
        xi=np.array([r["xi"] for r in results]),
        envelope_fraction=np.array(
            [r["envelope_fraction"] for r in results]
        ),
        max_rel_error=np.array(
            [r["max_rel_error_masked"] for r in results]
        ),
        profile_weighted_error=np.array(
            [r["profile_weighted_error"] for r in results]
        ),
        g5_failures=np.array([r["approx_failures"] for r in results]),
    )
    print(f"\nResults saved to {output_dir / 'accuracy_results.npz'}")

    # Generate plots if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Heatmap: max relative error vs (T, E/kT) at xi=0
        xi_target = 0.0
        subset = [
            r
            for r in results
            if r["xi"] == xi_target and np.isfinite(r["max_rel_error_masked"])
        ]
        if subset:
            T_vals = sorted(set(r["T_kev"] for r in subset))
            ratio_vals = sorted(set(r["E_over_kT"] for r in subset))
            grid = np.full((len(ratio_vals), len(T_vals)), np.nan)
            for r in subset:
                i = ratio_vals.index(r["E_over_kT"])
                j = T_vals.index(r["T_kev"])
                grid[i, j] = np.log10(r["max_rel_error_masked"] + 1e-16)

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                extent=[0, len(T_vals), 0, len(ratio_vals)],
                cmap="RdYlGn_r",
            )
            ax.set_xticks(np.arange(len(T_vals)) + 0.5)
            ax.set_xticklabels([f"{t}" for t in T_vals])
            ax.set_yticks(np.arange(len(ratio_vals)) + 0.5)
            ax.set_yticklabels([f"{r}" for r in ratio_vals])
            ax.set_xlabel("kT_e [keV]")
            ax.set_ylabel("h*nu / (kT_e)")
            ax.set_title(f"log10(max relative error) at xi={xi_target}")
            plt.colorbar(im, ax=ax, label="log10(rel error)")
            fig.tight_layout()
            fig.savefig(output_dir / "accuracy_heatmap_xi0.png", dpi=150)
            plt.close(fig)
            print(f"Heatmap saved to {output_dir / 'accuracy_heatmap_xi0.png'}")

    except ImportError:
        print("matplotlib not available, skipping plots")

    sys.exit(0 if not cancellation_regimes else 1)


if __name__ == "__main__":
    main()
