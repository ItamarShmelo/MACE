"""
Before/after comparison scan for P3 removal experiment.

Evaluates the solver on a dense parameter-space grid and saves raw results
to .npz files for precise diffing.

Usage:
    python3 reports/p3_removal_scan.py --output before   # save baseline
    python3 reports/p3_removal_scan.py --output after    # save post-change
    python3 reports/p3_removal_scan.py --compare         # diff the two
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_differential_cross_section import ComptonKernelSolver
from _units import kev, kev_kelvin

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
os.makedirs(GEN_DIR, exist_ok=True)

E_REF_KEV = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
RATIO_GRID = np.logspace(-0.7, 1.0, 80)
TAU_GRID_KEV = np.logspace(-2, 2.7, 80)
XI_VALUES = np.array([-0.95, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9, 0.95])


def run_scan():
    solver = ComptonKernelSolver()

    n_E = len(E_REF_KEV)
    n_ratio = len(RATIO_GRID)
    n_tau = len(TAU_GRID_KEV)
    n_xi = len(XI_VALUES)
    total = n_E * n_ratio * n_tau * n_xi

    values = np.full((n_E, n_xi, n_tau, n_ratio), np.nan)
    errors = np.full((n_E, n_xi, n_tau, n_ratio), np.nan)
    failed = np.zeros((n_E, n_xi, n_tau, n_ratio), dtype=bool)

    print(f"Running scan: {n_E} energies x {n_xi} angles x {n_tau} temps x {n_ratio} ratios = {total} points")
    t0 = time.time()
    done = 0

    for ie, E_kev in enumerate(E_REF_KEV):
        E = E_kev * kev
        for ixi, xi in enumerate(XI_VALUES):
            for it, T_kev in enumerate(TAU_GRID_KEV):
                T = T_kev * kev_kelvin
                for ir, ratio in enumerate(RATIO_GRID):
                    Ep = E * ratio
                    try:
                        r = solver.sigma_E(E, Ep, xi, T, 1.0)
                        values[ie, ixi, it, ir] = r.value
                        errors[ie, ixi, it, ir] = r.estimated_rel_error
                    except Exception:
                        failed[ie, ixi, it, ir] = True
                    done += 1

            elapsed = time.time() - t0
            frac = done / total
            if frac > 0:
                eta = elapsed / frac * (1 - frac)
                print(f"  [{100*frac:5.1f}%] E={E_kev} keV, xi={xi:.2f} done "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"Scan complete: {total} points in {elapsed:.1f}s")
    n_failed = np.sum(failed)
    if n_failed > 0:
        print(f"  {n_failed} points failed ({100*n_failed/total:.2f}%)")

    return values, errors, failed


def save_scan(tag):
    values, errors, failed = run_scan()
    path = os.path.join(GEN_DIR, f"p3_scan_{tag}.npz")
    np.savez(path,
             values=values, errors=errors, failed=failed,
             E_ref_kev=E_REF_KEV, ratio_grid=RATIO_GRID,
             tau_grid_kev=TAU_GRID_KEV, xi_values=XI_VALUES)
    print(f"Saved: {path}")


def compare():
    path_before = os.path.join(GEN_DIR, "p3_scan_before.npz")
    path_after = os.path.join(GEN_DIR, "p3_scan_after.npz")

    if not os.path.exists(path_before):
        print(f"ERROR: {path_before} not found. Run with --output before first.")
        sys.exit(1)
    if not os.path.exists(path_after):
        print(f"ERROR: {path_after} not found. Run with --output after first.")
        sys.exit(1)

    before = np.load(path_before)
    after = np.load(path_after)

    v_b, v_a = before['values'], after['values']
    e_b, e_a = before['errors'], after['errors']
    f_b, f_a = before['failed'], after['failed']

    total = v_b.size

    newly_failed = f_a & ~f_b
    newly_ok = f_b & ~f_a

    valid_both = ~f_b & ~f_a
    abs_diff = np.abs(v_a - v_b)
    denom = np.maximum(np.abs(v_b), 1e-300)
    rel_diff = np.where(valid_both, abs_diff / denom, 0.0)

    changed = valid_both & (rel_diff > 1e-15)

    print("=" * 70)
    print("P3 REMOVAL COMPARISON REPORT")
    print("=" * 70)
    print(f"Total grid points: {total}")
    print()

    print(f"Newly failed (after but not before): {np.sum(newly_failed)}")
    print(f"Newly OK (before failed, after OK):  {np.sum(newly_ok)}")
    print()

    n_changed = np.sum(changed)
    print(f"Points with value change (rel > 1e-15): {n_changed}")

    if n_changed > 0:
        rd = rel_diff[changed]
        print(f"  Max relative difference:  {np.max(rd):.3e}")
        print(f"  Mean relative difference: {np.mean(rd):.3e}")
        print(f"  Median relative diff:     {np.median(rd):.3e}")
        print()

        idx = np.argwhere(changed)
        top_indices = np.argsort(rel_diff[changed])[-min(20, n_changed):][::-1]
        print(f"Top {min(20, n_changed)} largest differences:")
        print(f"  {'E(keV)':>8} {'xi':>6} {'T(keV)':>10} {'E\'/E':>8} "
              f"{'val_before':>12} {'val_after':>12} {'rel_diff':>10}")
        print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")

        E_ref = before['E_ref_kev']
        ratios = before['ratio_grid']
        taus = before['tau_grid_kev']
        xis = before['xi_values']

        for ti in top_indices:
            ie, ixi, it, ir = idx[ti]
            print(f"  {E_ref[ie]:8.3f} {xis[ixi]:6.2f} {taus[it]:10.4f} "
                  f"{ratios[ir]:8.4f} {v_b[ie,ixi,it,ir]:12.5e} "
                  f"{v_a[ie,ixi,it,ir]:12.5e} {rel_diff[ie,ixi,it,ir]:10.3e}")
    else:
        print("  -> Results are IDENTICAL.")

    print()
    err_change = valid_both & (np.abs(e_a - e_b) > 1e-15)
    n_err_changed = np.sum(err_change)
    print(f"Points with error-estimate change: {n_err_changed}")

    if n_err_changed > 0:
        worse = valid_both & (e_a > e_b) & (e_a - e_b > 1e-15)
        better = valid_both & (e_b > e_a) & (e_b - e_a > 1e-15)
        print(f"  Error got worse:  {np.sum(worse)}")
        print(f"  Error got better: {np.sum(better)}")

    print()
    if n_changed == 0 and np.sum(newly_failed) == 0:
        print("CONCLUSION: P3 removal has NO EFFECT on solver outputs.")
    else:
        print("CONCLUSION: P3 removal AFFECTS solver outputs. See details above.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3 removal before/after scan")
    parser.add_argument("--output", choices=["before", "after"],
                        help="Run scan and save as 'before' or 'after'")
    parser.add_argument("--compare", action="store_true",
                        help="Compare before/after results")
    args = parser.parse_args()

    if args.output:
        save_scan(args.output)
    elif args.compare:
        compare()
    else:
        parser.print_help()
