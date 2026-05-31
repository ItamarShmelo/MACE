"""
Solver accuracy colorplot report.

Produces 2D colorplots of the solver's reported error and true error (vs Q256)
over the (E'/E, tau) plane for several xi values, plus a summary accuracy report.

Usage:
    python3 reports/solver_error_colorplot.py

Output:
    reports/generated/solver_error_colorplot.md  (+ .png plots in figs/)
"""

import sys
import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterSciNotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_kernel_series import ComptonKernelSeries, SeriesMethod
from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm

ME_C2 = 9.109383713928e-28 * (2.99792458e10)**2
KEV = 1.602176634e-9

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

solver = ComptonKernelSeries(SeriesMethod.Auto)
quad256 = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)

E_REF_KEV = 10.0
E_REF = E_REF_KEV * KEV

RATIO_GRID = np.logspace(-0.7, 1.0, 60)
TAU_GRID_KEV = np.logspace(-1, 2.7, 60)
XI_VALUES = [-0.5, 0.0, 0.5, 0.9]

lines = []


def emit(s=""):
    lines.append(s)


def evaluate_grid(E, xi_val, ratio_grid, tau_kev_grid):
    """Evaluate solver and Q256 on a 2D (ratio, tau) grid for fixed E and xi."""
    n_tau = len(tau_kev_grid)
    n_ratio = len(ratio_grid)

    reported_rel = np.full((n_tau, n_ratio), np.nan)
    true_rel = np.full((n_tau, n_ratio), np.nan)
    values = np.full((n_tau, n_ratio), np.nan)
    failed = np.zeros((n_tau, n_ratio), dtype=bool)

    for i, T_keV in enumerate(tau_kev_grid):
        tau = T_keV * KEV / ME_C2
        for j, ratio in enumerate(ratio_grid):
            Ep = E * ratio
            try:
                sr = solver.sigma_E(E, Ep, xi_val, tau, 1.0)
                values[i, j] = sr.value
                reported_rel[i, j] = sr.estimated_rel_error

                qr = quad256.sigma_E(E, Ep, xi_val, tau, 1.0)
                if abs(qr.value) > 1e-300:
                    true_rel[i, j] = abs(sr.value - qr.value) / abs(qr.value)
            except Exception:
                failed[i, j] = True

    return reported_rel, true_rel, values, failed


def plot_error_panels(ratio_grid, tau_kev_grid, data_dict, title, filename,
                      vmin=1e-16, vmax=1e0, cbar_label="Relative error"):
    """Plot a grid of colormaps for multiple xi values."""
    n_xi = len(data_dict)
    fig, axes = plt.subplots(1, n_xi, figsize=(5 * n_xi + 1.5, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.cm.inferno_r.copy()
    cmap.set_bad('lightgray')

    for k, (xi_val, data) in enumerate(data_dict.items()):
        ax = axes[k]
        safe_data = np.where(np.isfinite(data) & (data > 0), data, np.nan)
        pcm = ax.pcolormesh(ratio_grid, tau_kev_grid, safe_data,
                            norm=norm, cmap=cmap, shading='nearest')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("E'/E")
        if k == 0:
            ax.set_ylabel("T  [keV]")
        ax.set_title(f"$\\xi = {xi_val}$")

    cbar = fig.colorbar(pcm, ax=axes.tolist(), shrink=0.85, pad=0.03)
    cbar.set_label(cbar_label)
    fig.suptitle(f"{title}   (E = {E_REF_KEV:.0f} keV)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_reliability_panels(ratio_grid, tau_kev_grid, reliability_dict, filename):
    """Plot ratio = true_error / reported_error; values >1 mean underestimated."""
    n_xi = len(reliability_dict)
    fig, axes = plt.subplots(1, n_xi, figsize=(5 * n_xi + 1.5, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    norm = LogNorm(vmin=1e-6, vmax=1e6, clip=True)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('lightgray')

    for k, (xi_val, data) in enumerate(reliability_dict.items()):
        ax = axes[k]
        safe_data = np.where(np.isfinite(data) & (data > 0), data, np.nan)
        pcm = ax.pcolormesh(ratio_grid, tau_kev_grid, safe_data,
                            norm=norm, cmap=cmap, shading='nearest')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("E'/E")
        if k == 0:
            ax.set_ylabel("T  [keV]")
        ax.set_title(f"$\\xi = {xi_val}$")

    cbar = fig.colorbar(pcm, ax=axes.tolist(), shrink=0.85, pad=0.03)
    cbar.set_label("True error / Reported error")
    fig.suptitle(
        f"Error Estimate Reliability   (E = {E_REF_KEV:.0f} keV)\n"
        "Red > 1 = underestimated,  Blue < 1 = conservative",
        fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def collect_statistics(all_reported, all_true, all_failed):
    """Compute summary statistics from the grid evaluations."""
    reported_flat = []
    true_flat = []
    reliability_flat = []
    n_failed = 0
    n_total = 0
    n_underestimated = 0

    for xi_val in XI_VALUES:
        rep = all_reported[xi_val]
        tru = all_true[xi_val]
        fail = all_failed[xi_val]

        n_total += rep.size
        n_failed += fail.sum()

        valid = np.isfinite(rep) & (rep > 0) & ~fail
        reported_flat.extend(rep[valid].tolist())

        valid_true = np.isfinite(tru) & (tru > 0) & ~fail
        true_flat.extend(tru[valid_true].tolist())

        both = valid & valid_true
        if both.any():
            ratio = tru[both] / rep[both]
            reliability_flat.extend(ratio.tolist())
            n_underestimated += (ratio > 10).sum()

    return {
        'n_total': n_total,
        'n_failed': n_failed,
        'n_evaluated': n_total - n_failed,
        'reported': np.array(reported_flat) if reported_flat else np.array([]),
        'true': np.array(true_flat) if true_flat else np.array([]),
        'reliability': np.array(reliability_flat) if reliability_flat else np.array([]),
        'n_underestimated_10x': n_underestimated,
    }


def main():
    emit("# Solver Error Accuracy Report")
    emit()
    emit(f"Reference energy E = {E_REF_KEV:.0f} keV.  "
         f"Grid: {len(RATIO_GRID)} E'/E values "
         f"x {len(TAU_GRID_KEV)} temperatures "
         f"x {len(XI_VALUES)} xi values "
         f"= {len(RATIO_GRID) * len(TAU_GRID_KEV) * len(XI_VALUES)} points.")
    emit()

    # ── Evaluate on grid ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    all_reported = {}
    all_true = {}
    all_values = {}
    all_failed = {}
    all_reliability = {}

    for xi_val in XI_VALUES:
        print(f"  evaluating xi={xi_val} ...")
        rep, tru, vals, fail = evaluate_grid(
            E_REF, xi_val, RATIO_GRID, TAU_GRID_KEV)
        all_reported[xi_val] = rep
        all_true[xi_val] = tru
        all_values[xi_val] = vals
        all_failed[xi_val] = fail

        eps = 1e-300
        all_reliability[xi_val] = np.where(
            (rep > eps) & (tru > eps), tru / rep, np.nan)

    wall_time = time.perf_counter() - t0
    print(f"  grid evaluation took {wall_time:.1f}s")

    # ── Section 1: Reported relative error ───────────────────────────────
    emit("## 1. Solver Reported Relative Error")
    emit()
    emit("The solver's self-reported `estimated_rel_error` across the (E'/E, T) plane.")
    emit()

    plot_error_panels(
        RATIO_GRID, TAU_GRID_KEV, all_reported,
        "Solver Reported Relative Error",
        "solver_reported_error.png",
        cbar_label="Reported relative error")

    emit("![Reported relative error](figs/solver_reported_error.png)")
    emit()

    # ── Section 2: True relative error vs Q256 ───────────────────────────
    emit("## 2. True Relative Error  (Solver vs Q256)")
    emit()
    emit("Actual discrepancy `|solver - Q256| / |Q256|` using 256-point "
         "Gauss-Laguerre quadrature as reference.")
    emit()

    plot_error_panels(
        RATIO_GRID, TAU_GRID_KEV, all_true,
        "True Relative Error  (Solver vs Q256)",
        "solver_true_error.png",
        cbar_label="True relative error")

    emit("![True relative error](figs/solver_true_error.png)")
    emit()

    # ── Section 3: Reliability ratio ─────────────────────────────────────
    emit("## 3. Error Estimate Reliability  (True / Reported)")
    emit()
    emit("Ratio of true error to reported error.  Values > 1 (red) mean the "
         "solver *underestimates* its error; values < 1 (blue) mean the "
         "estimate is conservative.  Gray cells have no valid comparison.")
    emit()

    plot_reliability_panels(
        RATIO_GRID, TAU_GRID_KEV, all_reliability,
        "solver_error_reliability.png")

    emit("![Error reliability](figs/solver_error_reliability.png)")
    emit()

    # ── Section 4: Summary statistics ────────────────────────────────────
    stats = collect_statistics(all_reported, all_true, all_failed)

    emit("## 4. Summary Statistics")
    emit()
    emit(f"| Metric | Value |")
    emit(f"|--------|-------|")
    emit(f"| Grid points evaluated | {stats['n_evaluated']} / {stats['n_total']} |")
    emit(f"| Failed (exception) | {stats['n_failed']} |")
    emit(f"| Wall-clock time | {wall_time:.1f} s |")
    emit()

    if len(stats['reported']) > 0:
        rep = stats['reported']
        emit("### Reported error distribution")
        emit()
        emit(f"| Statistic | Value |")
        emit(f"|-----------|-------|")
        emit(f"| Median | {np.median(rep):.2e} |")
        emit(f"| Mean | {np.mean(rep):.2e} |")
        emit(f"| Min | {rep.min():.2e} |")
        emit(f"| Max | {rep.max():.2e} |")
        emit(f"| 90th percentile | {np.percentile(rep, 90):.2e} |")
        emit(f"| 99th percentile | {np.percentile(rep, 99):.2e} |")
        emit()

    if len(stats['true']) > 0:
        tru = stats['true']
        emit("### True error distribution  (vs Q256)")
        emit()
        emit(f"| Statistic | Value |")
        emit(f"|-----------|-------|")
        emit(f"| Median | {np.median(tru):.2e} |")
        emit(f"| Mean | {np.mean(tru):.2e} |")
        emit(f"| Min | {tru.min():.2e} |")
        emit(f"| Max | {tru.max():.2e} |")
        emit(f"| 90th percentile | {np.percentile(tru, 90):.2e} |")
        emit(f"| 99th percentile | {np.percentile(tru, 99):.2e} |")
        pct_below_1e8 = 100 * (tru < 1e-8).sum() / len(tru)
        pct_below_1e6 = 100 * (tru < 1e-6).sum() / len(tru)
        emit(f"| Points with error < 1e-8 | {pct_below_1e8:.1f}% |")
        emit(f"| Points with error < 1e-6 | {pct_below_1e6:.1f}% |")
        emit()

    if len(stats['reliability']) > 0:
        rel = stats['reliability']
        emit("### Error estimate reliability  (true / reported)")
        emit()
        emit(f"| Statistic | Value |")
        emit(f"|-----------|-------|")
        emit(f"| Median | {np.median(rel):.2e} |")
        emit(f"| Mean | {np.mean(rel):.2e} |")
        emit(f"| Max | {rel.max():.2e} |")
        pct_conservative = 100 * (rel < 1).sum() / len(rel)
        pct_within_10x = 100 * (rel < 10).sum() / len(rel)
        emit(f"| Conservative (ratio < 1) | {pct_conservative:.1f}% |")
        emit(f"| Within 10x (ratio < 10) | {pct_within_10x:.1f}% |")
        emit(f"| Severely underestimated (ratio > 10) | {stats['n_underestimated_10x']} points |")
        emit()

    # ── Section 5: Histogram of true error ───────────────────────────────
    emit("## 5. True Error Histogram")
    emit()

    if len(stats['true']) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        log_errs = np.log10(np.clip(stats['true'], 1e-16, None))
        ax.hist(log_errs, bins=40, edgecolor='black', alpha=0.75, color='steelblue')
        ax.axvline(np.log10(1e-8), color='red', ls='--', lw=1.5, label='1e-8')
        ax.axvline(np.log10(1e-6), color='orange', ls='--', lw=1.5, label='1e-6')
        ax.set_xlabel("log$_{10}$ (true relative error)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of True Relative Error  (Solver vs Q256)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, 'solver_true_error_hist.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        emit("![True error histogram](figs/solver_true_error_hist.png)")
        emit()

    # ── Write report ─────────────────────────────────────────────────────
    out_path = os.path.join(GEN_DIR, 'solver_error_colorplot.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Report written to {out_path}")


if __name__ == '__main__':
    main()
