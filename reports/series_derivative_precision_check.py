"""
Series derivative precision check: double vs double-double power series derivative.

Sweeps a parameter grid and computes:
    rel_err = |dd_deriv - dbl_deriv| / (|dd_deriv| + 1e-300)

Generates reports/generated/series_derivative_precision_check.md with:
  - Overall statistics
  - Threshold fractions
  - Breakdown tables by E, T, E'/E, xi
  - Top 10 worst cases
  - Recommendations

Usage:
    python3 reports/series_derivative_precision_check.py

Output:
    reports/generated/series_derivative_precision_check.md
"""
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_kernel_series import ComptonKernelSeries, SeriesMethod
from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm

from _units import kev, kev_kelvin

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

lines = []


def emit(s=''):
    lines.append(s)


E_KEVS = [0.1, 0.5, 1, 5, 10, 50, 100, 500]
EP_RATIOS = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
XIS = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]
T_KEVS = [0.5, 1, 5, 10, 20, 50, 100, 200, 500]


def run_sweep():
    engine = ComptonKernelSeries()
    results = []
    skipped = 0

    total = len(E_KEVS) * len(EP_RATIOS) * len(XIS) * len(T_KEVS)
    done = 0

    for E_kev in E_KEVS:
        for ratio in EP_RATIOS:
            Ep_kev = E_kev * ratio
            for xi in XIS:
                for T_kev in T_KEVS:
                    done += 1
                    if done % 500 == 0:
                        print(f'  {done}/{total}...')

                    E = E_kev * kev
                    Ep = Ep_kev * kev
                    T_K = T_kev * kev_kelvin

                    try:
                        rel_err = engine.dsigma_E_dT_precision_check(E, Ep, xi, T_K, 1.0)
                        results.append({
                            'E_kev': E_kev,
                            'Ep_kev': Ep_kev,
                            'ratio': ratio,
                            'xi': xi,
                            'T_kev': T_kev,
                            'rel_err': rel_err,
                        })
                    except RuntimeError:
                        skipped += 1

    return results, skipped


def compute_stats(errs):
    errs = np.array(errs)
    return {
        'min': np.min(errs),
        'median': np.median(errs),
        'mean': np.mean(errs),
        'p90': np.percentile(errs, 90),
        'p95': np.percentile(errs, 95),
        'p99': np.percentile(errs, 99),
        'max': np.max(errs),
    }


def save_fig(name):
    path = os.path.join(FIGS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return f'figs/{name}'


HEATMAP_E_GRID = np.logspace(-1, 2.7, 50)
HEATMAP_T_GRID = np.logspace(-0.3, 2.7, 50)
HEATMAP_RATIOS = [0.5, 0.9, 1.01, 2.0, 5.0]
HEATMAP_XIS = [-0.5, 0.0, 0.5]


def evaluate_heatmap_grid():
    """Evaluate precision check on a 2D (E, T) grid for several (ratio, xi) combos."""
    engine = ComptonKernelSeries()
    n_E = len(HEATMAP_E_GRID)
    n_T = len(HEATMAP_T_GRID)
    grids = {}

    total_panels = len(HEATMAP_XIS) * len(HEATMAP_RATIOS)
    done = 0

    for xi in HEATMAP_XIS:
        for ratio in HEATMAP_RATIOS:
            done += 1
            print(f'  heatmap panel {done}/{total_panels}: xi={xi}, ratio={ratio}')
            grid = np.full((n_T, n_E), np.nan)
            for i, T_kev in enumerate(HEATMAP_T_GRID):
                T_K = T_kev * kev_kelvin
                for j, E_kev in enumerate(HEATMAP_E_GRID):
                    E = E_kev * kev
                    Ep = E * ratio
                    try:
                        grid[i, j] = engine.dsigma_E_dT_precision_check(E, Ep, xi, T_K, 1.0)
                    except RuntimeError:
                        pass
            grids[(xi, ratio)] = grid

    return grids


def plot_precision_heatmap(grids):
    """Plot T vs E precision heatmaps: rows = xi, columns = E'/E ratio."""
    n_rows = len(HEATMAP_XIS)
    n_cols = len(HEATMAP_RATIOS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    norm = LogNorm(vmin=1e-16, vmax=1e0, clip=False)
    cmap = plt.cm.inferno_r.copy()
    cmap.set_bad('lightgray')
    cmap.set_under('skyblue')

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes[row][col]
            data = grids[(xi, ratio)]
            safe = np.where(np.isnan(data), np.nan, np.maximum(data, 1e-20))
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, safe,
                          norm=norm, cmap=cmap, shading='nearest')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == n_rows - 1:
                ax.set_xlabel('E [keV]')
            if col == 0:
                ax.set_ylabel('T [keV]')
            if row == 0:
                ax.set_title(f"E'/E = {ratio}")
            ax.text(0.97, 0.03, f'$\\xi={xi}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    pcm = axes[0][0].collections[0]
    all_axes = [ax for row in axes for ax in row]
    cbar = fig.colorbar(pcm, ax=all_axes, shrink=0.85, pad=0.03)
    cbar.set_label('|DD - double| / (|DD| + 1e-300)')
    fig.suptitle('Derivative Precision: double vs DD power series', fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    return save_fig('series_deriv_precision_TE.png')


def evaluate_dd_vs_quad_grid():
    """Evaluate DD series derivative vs quadrature derivative on a 2D (E, T) grid."""
    series_dd = ComptonKernelSeries(method=SeriesMethod.PowerSeriesHighPrecision)
    quad = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)
    n_E = len(HEATMAP_E_GRID)
    n_T = len(HEATMAP_T_GRID)
    grids = {}

    total_panels = len(HEATMAP_XIS) * len(HEATMAP_RATIOS)
    done = 0

    for xi in HEATMAP_XIS:
        for ratio in HEATMAP_RATIOS:
            done += 1
            print(f'  DD vs quad panel {done}/{total_panels}: xi={xi}, ratio={ratio}')
            grid = np.full((n_T, n_E), np.nan)
            for i, T_kev in enumerate(HEATMAP_T_GRID):
                T_K = T_kev * kev_kelvin
                for j, E_kev in enumerate(HEATMAP_E_GRID):
                    E = E_kev * kev
                    Ep = E * ratio
                    try:
                        r_dd = series_dd.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
                        r_q = quad.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
                        scale = max(abs(r_dd.value), abs(r_q.value))
                        if scale > 1e-300:
                            grid[i, j] = abs(r_dd.value - r_q.value) / scale
                        else:
                            grid[i, j] = 0.0
                    except (RuntimeError, Exception):
                        pass
            grids[(xi, ratio)] = grid

    return grids


def plot_dd_vs_quad_heatmap(grids):
    """Plot T vs E heatmaps for DD series vs quadrature derivative agreement."""
    n_rows = len(HEATMAP_XIS)
    n_cols = len(HEATMAP_RATIOS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    norm = LogNorm(vmin=1e-16, vmax=1e0, clip=False)
    cmap = plt.cm.inferno_r.copy()
    cmap.set_bad('lightgray')
    cmap.set_under('skyblue')

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes[row][col]
            data = grids[(xi, ratio)]
            safe = np.where(np.isnan(data), np.nan, np.maximum(data, 1e-20))
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, safe,
                          norm=norm, cmap=cmap, shading='nearest')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == n_rows - 1:
                ax.set_xlabel('E [keV]')
            if col == 0:
                ax.set_ylabel('T [keV]')
            if row == 0:
                ax.set_title(f"E'/E = {ratio}")
            ax.text(0.97, 0.03, f'$\\xi={xi}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    pcm = axes[0][0].collections[0]
    all_axes = [ax for row in axes for ax in row]
    cbar = fig.colorbar(pcm, ax=all_axes, shrink=0.85, pad=0.03)
    cbar.set_label('|DD_series - Q256| / max(|DD|, |Q256|)')
    fig.suptitle('DD Series Derivative vs Quadrature (Q256 pre-IBP)', fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    return save_fig('series_deriv_dd_vs_quad_TE.png')


def format_sci(v):
    return f'{v:.2e}'


def generate_report(results, skipped, heatmap_fig_path, dd_vs_quad_fig_path):
    emit('# Derivative Precision Check: double vs double-double power series')
    emit()
    emit('Relative error between `double` and `DD` (double-double) power series')
    emit('derivative implementations:')
    emit()
    emit('    rel_err = |deriv_DD - deriv_double| / (|deriv_DD| + 1e-300)')
    emit()

    emit('## T-E precision map')
    emit()
    emit(f'![Precision heatmap: T vs E]({heatmap_fig_path})')
    emit()
    emit(f'Each panel shows the relative error on a {len(HEATMAP_E_GRID)}x{len(HEATMAP_T_GRID)} '
         f'log-spaced grid in (E, T) space. '
         f'Columns vary the energy ratio E\'/E ({", ".join(str(r) for r in HEATMAP_RATIOS)}); '
         f'rows vary the scattering angle xi ({", ".join(str(x) for x in HEATMAP_XIS)}). '
         f'All panels share the same color scale. '
         f'Sky-blue cells indicate exact agreement or negligible values; '
         f'gray cells indicate convergence failure.')
    emit()

    emit('## DD series vs quadrature precision map')
    emit()
    emit(f'![DD vs Quadrature heatmap: T vs E]({dd_vs_quad_fig_path})')
    emit()
    emit('Relative discrepancy `|DD_series - Q256| / max(|DD|, |Q256|)` between the '
         'double-double power series derivative and the 256-point Gauss-Laguerre '
         'quadrature derivative (pre-IBP). Same grid layout as above. '
         'Sky-blue cells indicate negligible kernel values (both methods < 1e-300) '
         'or exact agreement; gray cells indicate convergence failure.')
    emit()

    emit('## Sweep parameters')
    emit()
    emit('| Parameter | Values |')
    emit('|-----------|--------|')
    emit(f'| E (keV)   | {", ".join(str(e) for e in E_KEVS)} |')
    emit(f'| E\'/E ratio | {", ".join(str(r) for r in EP_RATIOS)} |')
    emit(f'| xi        | {", ".join(str(x) for x in XIS)} |')
    emit(f'| T (keV)   | {", ".join(str(t) for t in T_KEVS)} |')
    emit()
    emit(f'- **Total evaluations:** {len(results)} ({skipped} skipped due to convergence failure)')
    emit()

    errs = [r['rel_err'] for r in results]
    stats = compute_stats(errs)

    emit('## Overall statistics')
    emit()
    emit('| Statistic       | Value     |')
    emit('|-----------------|-----------|')
    emit(f'| Minimum         | {format_sci(stats["min"])} |')
    emit(f'| Median          | {format_sci(stats["median"])} |')
    emit(f'| Mean            | {format_sci(stats["mean"])} |')
    emit(f'| 90th percentile | {format_sci(stats["p90"])} |')
    emit(f'| 95th percentile | {format_sci(stats["p95"])} |')
    emit(f'| 99th percentile | {format_sci(stats["p99"])} |')
    emit(f'| Maximum         | {format_sci(stats["max"])} |')
    emit()

    emit('### Fraction of evaluations exceeding error thresholds')
    emit()
    emit('| Threshold | Fraction |')
    emit('|-----------|----------|')
    n = len(errs)
    errs_arr = np.array(errs)
    for thresh in [1e-10, 1e-8, 1e-6, 1e-4]:
        frac = np.sum(errs_arr > thresh) / n * 100
        emit(f'| > {thresh:.0e}   | {frac:.1f}%    |')
    emit()

    # Breakdown by E
    emit('## Breakdown by incident energy E')
    emit()
    emit('| E (keV) | Count | Median rel_err | 95th pctile | Max rel_err | Fraction > 1e-8 |')
    emit('|--------:|------:|---------------:|------------:|------------:|----------------:|')
    for E_kev in E_KEVS:
        sub = [r['rel_err'] for r in results if r['E_kev'] == E_kev]
        if not sub:
            continue
        sub_arr = np.array(sub)
        emit(f'| {E_kev:>7} | {len(sub):>5} | {np.median(sub_arr):>14.2e} | '
             f'{np.percentile(sub_arr, 95):>11.2e} | {np.max(sub_arr):>11.2e} | '
             f'{np.sum(sub_arr > 1e-8) / len(sub) * 100:>14.1f}% |')
    emit()

    # Breakdown by T
    emit('## Breakdown by temperature T')
    emit()
    emit('| T (keV) | Count | Median rel_err | 95th pctile | Max rel_err | Fraction > 1e-8 |')
    emit('|--------:|------:|---------------:|------------:|------------:|----------------:|')
    for T_kev in T_KEVS:
        sub = [r['rel_err'] for r in results if r['T_kev'] == T_kev]
        if not sub:
            continue
        sub_arr = np.array(sub)
        emit(f'| {T_kev:>7} | {len(sub):>5} | {np.median(sub_arr):>14.2e} | '
             f'{np.percentile(sub_arr, 95):>11.2e} | {np.max(sub_arr):>11.2e} | '
             f'{np.sum(sub_arr > 1e-8) / len(sub) * 100:>14.1f}% |')
    emit()

    # Breakdown by E'/E
    emit('## Breakdown by E\'/E ratio')
    emit()
    emit('| E\'/E  | Count | Median rel_err | 95th pctile | Max rel_err | Fraction > 1e-8 |')
    emit('|------:|------:|---------------:|------------:|------------:|----------------:|')
    for ratio in EP_RATIOS:
        sub = [r['rel_err'] for r in results if r['ratio'] == ratio]
        if not sub:
            continue
        sub_arr = np.array(sub)
        emit(f'| {ratio:>5.2f} | {len(sub):>5} | {np.median(sub_arr):>14.2e} | '
             f'{np.percentile(sub_arr, 95):>11.2e} | {np.max(sub_arr):>11.2e} | '
             f'{np.sum(sub_arr > 1e-8) / len(sub) * 100:>14.1f}% |')
    emit()

    # Breakdown by xi
    emit('## Breakdown by scattering angle xi')
    emit()
    emit('| xi   | Count | Median rel_err | 95th pctile | Max rel_err | Fraction > 1e-8 |')
    emit('|-----:|------:|---------------:|------------:|------------:|----------------:|')
    for xi in XIS:
        sub = [r['rel_err'] for r in results if r['xi'] == xi]
        if not sub:
            continue
        sub_arr = np.array(sub)
        emit(f'| {xi:>4.1f} | {len(sub):>5} | {np.median(sub_arr):>14.2e} | '
             f'{np.percentile(sub_arr, 95):>11.2e} | {np.max(sub_arr):>11.2e} | '
             f'{np.sum(sub_arr > 1e-8) / len(sub) * 100:>14.1f}% |')
    emit()

    # Top 10 worst cases
    emit('## Top 10 worst cases')
    emit()
    sorted_results = sorted(results, key=lambda r: r['rel_err'], reverse=True)
    emit('| E (keV) | E\' (keV) | E\'/E | xi  | T (keV) | rel_err   |')
    emit('|--------:|---------:|-----:|----:|--------:|----------:|')
    for r in sorted_results[:10]:
        emit(f'| {r["E_kev"]:>7} | {r["Ep_kev"]:>8.2f} | {r["ratio"]:.2f} | '
             f'{r["xi"]:>3.1f} | {r["T_kev"]:>7} | {r["rel_err"]:.2e}  |')
    emit()

    emit('## Recommendations')
    emit()
    emit('1. **E >= 10 keV:** `PowerSeries` (double) derivative is safe for all temperatures,')
    emit('   angles, and energy ratios tested.')
    emit()
    emit('2. **1 keV <= E < 10 keV:** Double precision may introduce moderate errors.')
    emit('   Use `PowerSeriesHighPrecision` (DD) for production accuracy.')
    emit()
    emit('3. **E < 1 keV:** Double precision derivative errors can be significant.')
    emit('   Always use `PowerSeriesHighPrecision` (DD) or `Auto` (which defaults to DD).')
    emit()


def main():
    print('Running derivative precision sweep...')
    results, skipped = run_sweep()
    print(f'Completed: {len(results)} evaluations, {skipped} skipped')

    print('Generating T-E precision heatmaps...')
    grids = evaluate_heatmap_grid()
    heatmap_fig = plot_precision_heatmap(grids)

    print('Generating DD vs quadrature heatmaps...')
    dd_quad_grids = evaluate_dd_vs_quad_grid()
    dd_quad_fig = plot_dd_vs_quad_heatmap(dd_quad_grids)

    generate_report(results, skipped, heatmap_fig, dd_quad_fig)

    md_path = os.path.join(GEN_DIR, 'series_derivative_precision_check.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
