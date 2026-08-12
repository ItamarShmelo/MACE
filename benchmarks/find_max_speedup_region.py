"""
Find the group-to-group region with maximum speedup between
ComptonKernelApproximateSolver and ComptonKernelSolver.

Tests each (g -> g') pair by constructing focused 2-boundary kernels.
"""

import json
import sys
import time

import numpy as np

import compton_matrix as cm


def make_config():
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


def time_pair(E_lo, E_hi, Ep_lo, Ep_hi, angle_bins, T, config, trials=3):
    """Time a single incident-group -> outgoing-group computation."""
    boundaries_in = [E_lo, E_hi]
    boundaries_out = [Ep_lo, Ep_hi]

    if E_lo == Ep_lo and E_hi == Ep_hi:
        boundaries = [E_lo, E_hi]
    elif Ep_hi <= E_lo:
        boundaries = [Ep_lo, Ep_hi, E_lo, E_hi]
    elif E_hi <= Ep_lo:
        boundaries = [E_lo, E_hi, Ep_lo, Ep_hi]
    elif Ep_lo <= E_lo and Ep_hi >= E_hi:
        boundaries = sorted(set([Ep_lo, E_lo, E_hi, Ep_hi]))
    elif E_lo <= Ep_lo and E_hi >= Ep_hi:
        boundaries = sorted(set([E_lo, Ep_lo, Ep_hi, E_hi]))
    else:
        boundaries = sorted(set([E_lo, E_hi, Ep_lo, Ep_hi]))

    mg = cm.ComptonMultigroupKernel(boundaries, cm.UniformWeightFunction(), config)
    groups = len(boundaries) - 1

    ref_solver = cm.ComptonKernelSolver()
    approx_solver = cm.ComptonKernelApproximateSolver()

    best_ref = float("inf")
    best_approx = float("inf")
    ref_matrix = None
    approx_matrix = None

    for _ in range(trials):
        start = time.perf_counter()
        ref_matrix = mg.compute_sigma_matrix(ref_solver, angle_bins, T)
        elapsed = time.perf_counter() - start
        best_ref = min(best_ref, elapsed)

    for _ in range(trials):
        start = time.perf_counter()
        approx_matrix = mg.compute_sigma_matrix(approx_solver, angle_bins, T)
        elapsed = time.perf_counter() - start
        best_approx = min(best_approx, elapsed)

    return best_ref * 1000, best_approx * 1000, ref_matrix, approx_matrix


def main():
    group_boundaries_kev = [0.1, 1.0, 10.0, 100.0, 1000.0]
    group_boundaries_erg = [b * cm.kev for b in group_boundaries_kev]
    temperatures_kev = [1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 229.9]
    angle_bins = 4
    config = make_config()

    results = []

    for T_kev in temperatures_kev:
        T = T_kev * cm.kev_kelvin
        print(f"\n=== T = {T_kev} keV ===", file=sys.stderr)

        # Time the full matrix to get total reference
        full_mg = cm.ComptonMultigroupKernel(
            group_boundaries_erg, cm.UniformWeightFunction(), config
        )
        ref_solver = cm.ComptonKernelSolver()
        approx_solver = cm.ComptonKernelApproximateSolver()

        best_full_ref = float("inf")
        best_full_approx = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            full_mg.compute_sigma_matrix(ref_solver, angle_bins, T)
            best_full_ref = min(best_full_ref, time.perf_counter() - start)
        for _ in range(3):
            start = time.perf_counter()
            full_mg.compute_sigma_matrix(approx_solver, angle_bins, T)
            best_full_approx = min(best_full_approx, time.perf_counter() - start)

        full_speedup = (best_full_ref / best_full_approx) if best_full_approx > 0 else 0
        print(
            f"  Full matrix: ref={best_full_ref*1000:.1f}ms, "
            f"approx={best_full_approx*1000:.1f}ms, speedup={full_speedup:.2f}x",
            file=sys.stderr,
        )

        # Now time each incident group row using a per-row kernel
        n_groups = len(group_boundaries_kev) - 1
        for g in range(n_groups):
            E_lo = group_boundaries_erg[g]
            E_hi = group_boundaries_erg[g + 1]

            # Create a kernel with the incident group as one boundary pair
            # and ALL outgoing groups
            row_boundaries = sorted(
                set(group_boundaries_erg)
            )
            row_mg = cm.ComptonMultigroupKernel(
                row_boundaries, cm.UniformWeightFunction(), config
            )

            # Time just this row by timing the full kernel (OpenMP off for row isolation)
            # Instead, use a simpler approach: time single-group incident kernels
            incident_boundaries = [E_lo, E_hi] + [
                b for b in group_boundaries_erg if b < E_lo or b > E_hi
            ]
            incident_boundaries = sorted(set(incident_boundaries))

            # Better: just time individual g->g' using Ep_xi integral
            for gp in range(n_groups):
                Ep_lo = group_boundaries_erg[gp]
                Ep_hi = group_boundaries_erg[gp + 1]

                # Use the partial integral API for a representative E in the group
                E_mid = np.sqrt(E_lo * E_hi)  # geometric mean

                best_ref_t = float("inf")
                best_approx_t = float("inf")

                for _ in range(5):
                    start = time.perf_counter()
                    full_mg.compute_Ep_xi_integral_sigma(
                        ref_solver, E_mid, Ep_lo, Ep_hi, angle_bins, T
                    )
                    best_ref_t = min(best_ref_t, time.perf_counter() - start)

                for _ in range(5):
                    start = time.perf_counter()
                    full_mg.compute_Ep_xi_integral_sigma(
                        approx_solver, E_mid, Ep_lo, Ep_hi, angle_bins, T
                    )
                    best_approx_t = min(best_approx_t, time.perf_counter() - start)

                speedup = best_ref_t / best_approx_t if best_approx_t > 0 else 0

                entry = {
                    "T_kev": T_kev,
                    "g": g,
                    "gp": gp,
                    "E_range_kev": [group_boundaries_kev[g], group_boundaries_kev[g + 1]],
                    "Ep_range_kev": [group_boundaries_kev[gp], group_boundaries_kev[gp + 1]],
                    "ref_ms": best_ref_t * 1000,
                    "approx_ms": best_approx_t * 1000,
                    "speedup": speedup,
                }
                results.append(entry)

                label = (
                    f"  [{group_boundaries_kev[g]}-{group_boundaries_kev[g+1]}] -> "
                    f"[{group_boundaries_kev[gp]}-{group_boundaries_kev[gp+1]}] keV"
                )
                print(
                    f"{label}: ref={best_ref_t*1000:.2f}ms, "
                    f"approx={best_approx_t*1000:.2f}ms, speedup={speedup:.2f}x",
                    file=sys.stderr,
                )

    # Sort by speedup and print top results
    results.sort(key=lambda x: x["speedup"], reverse=True)

    print("\n\n=== TOP 10 GROUP-TO-GROUP SPEEDUPS ===\n", file=sys.stderr)
    for i, r in enumerate(results[:10]):
        print(
            f"  {i+1}. T={r['T_kev']} keV, "
            f"[{r['E_range_kev'][0]}-{r['E_range_kev'][1]}] -> "
            f"[{r['Ep_range_kev'][0]}-{r['Ep_range_kev'][1]}] keV: "
            f"{r['speedup']:.2f}x  (ref={r['ref_ms']:.1f}ms, approx={r['approx_ms']:.1f}ms)",
            file=sys.stderr,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
