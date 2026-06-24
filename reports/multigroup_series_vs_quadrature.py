"""
Series (Auto) vs Quadrature kernel comparison for multigroup matrices.

Compares the multigroup-multiangle Compton scattering matrix when computed
using ComptonKernelSeries(Auto) vs ComptonKernelQuadrature(64) as the
point-wise kernel backend.

Sections:
  1. Element-wise relative difference across group pairs
  2. Side-by-side heatmaps (angle-integrated)
  3. Difference heatmap
  4. Convergence: series agreement at multiple temperatures
  5. Temperature-derivative comparison

Usage:
    python3 reports/multigroup_series_vs_quadrature.py

Output:
    reports/generated/multigroup_series_vs_quadrature.md  (+ .png plots in figs/)
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

import _compton_multigroup as cm
import _compton_kernel_quadrature as cq
from _compton_kernel_solver import ComptonKernelSolver
from _units import kev, kev_kelvin

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

lines = []


def emit(s=''):
    lines.append(s)


def save_fig(name):
    path = os.path.join(FIGS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return f'figs/{name}'


BOUNDS_KEV = np.logspace(np.log10(0.1), np.log10(100.0), 41)
BOUNDS_ERG = (BOUNDS_KEV * kev).tolist()
G = len(BOUNDS_ERG) - 1
CENTERS_KEV = np.sqrt(BOUNDS_KEV[:-1] * BOUNDS_KEV[1:])

KERNEL_Q64 = cq.ComptonKernelQuadrature(64)
KERNEL_SOLVER = ComptonKernelSolver()

TICK_POS = np.arange(0, G, max(1, G // 6))
TICK_LABELS = [f'{CENTERS_KEV[i]:.1f}' for i in TICK_POS]


def make_mg(n=16):
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=BOUNDS_ERG,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=n, quad_order_Ep=n, xi_order=n)


def set_ticks(ax):
    ax.set_xticks(TICK_POS)
    ax.set_xticklabels(TICK_LABELS, rotation=45, fontsize=7)
    ax.set_yticks(TICK_POS)
    ax.set_yticklabels(TICK_LABELS, fontsize=7)
    ax.set_xlabel("E' (keV)")
    ax.set_ylabel('E (keV)')


# ─── Section 1: Element-wise comparison ──────────────────────────────────

def section_elementwise():
    emit('## 1. Element-wise Relative Difference')
    emit()
    emit('Angle-integrated 40-group matrix (0.1–100 keV), quadrature order 16, '
         'T = 10 keV.  Comparison of `ComptonKernelSeries(Auto)` vs '
         '`ComptonKernelQuadrature(64)` as the point-wise kernel backend.')
    emit()

    T = 10.0 * kev_kelvin
    mg = make_mg(16)

    S_quad = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
    S_series = mg.compute_sigma_matrix(KERNEL_SOLVER, T=T, Ne=1.0)

    mask = (np.abs(S_quad) > 1e-40) & (np.abs(S_series) > 1e-40)
    scale = np.maximum(np.abs(S_quad), 1e-300)
    rel_diff = np.where(mask, np.abs(S_series - S_quad) / scale, np.nan)

    valid = rel_diff[mask]
    emit(f'- Entries compared: {valid.size} / {G * G}')
    emit(f'- Median relative difference: {np.median(valid):.2e}')
    emit(f'- 95th percentile: {np.percentile(valid, 95):.2e}')
    emit(f'- Max relative difference: {np.max(valid):.2e}')
    emit()

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.logspace(-8, 0, 50)
    ax.hist(valid, bins=bins, edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Relative difference |series − quad| / |quad|')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of element-wise relative differences (T = 10 keV)')
    ax.axvline(np.median(valid), color='red', linestyle='--', label=f'median = {np.median(valid):.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = save_fig('svq_histogram.png')

    emit(f'![Relative difference histogram]({fig_path})')
    emit()


# ─── Section 2: Side-by-side heatmaps ───────────────────────────────────

def section_heatmaps():
    emit('## 2. Side-by-Side Heatmaps')
    emit()

    T = 10.0 * kev_kelvin
    mg = make_mg(16)

    S_quad = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
    S_series = mg.compute_sigma_matrix(KERNEL_SOLVER, T=T, Ne=1.0)

    S_q_pos = np.maximum(np.abs(S_quad), 1e-50)
    S_s_pos = np.maximum(np.abs(S_series), 1e-50)

    sig_q = S_q_pos[S_q_pos > 1e-50]
    sig_s = S_s_pos[S_s_pos > 1e-50]
    vmin = min(sig_q.min(), sig_s.min())
    vmax = max(S_q_pos.max(), S_s_pos.max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for ax, data, title in [(axes[0], S_q_pos, 'Quadrature (Q64)'),
                             (axes[1], S_s_pos, 'Series (Auto)')]:
        im = ax.imshow(data, norm=LogNorm(vmin=vmin, vmax=vmax),
                       aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        set_ticks(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    mask = (np.abs(S_quad) > 1e-40) & (np.abs(S_series) > 1e-40)
    scale = np.where(mask, np.abs(S_quad), 1.0)
    rel_diff = np.where(mask, np.abs(S_series - S_quad) / scale, 0.0)
    im = axes[2].imshow(rel_diff, aspect='auto', origin='lower', cmap='Reds',
                        vmin=0, vmax=min(0.01, rel_diff.max()) if rel_diff.max() > 0 else 0.01)
    axes[2].set_title('Relative difference')
    set_ticks(axes[2])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle('Series vs Quadrature — angle-integrated, T = 10 keV', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('svq_heatmaps.png')

    emit(f'![Side-by-side heatmaps]({fig_path})')
    emit()


# ─── Section 3: Agreement across temperatures ───────────────────────────

def section_temperature_sweep():
    emit('## 3. Agreement Across Temperatures')
    emit()
    emit('Median and max relative difference between series(Auto) and '
         'quadrature(Q64) multigroup matrices at various temperatures.')
    emit()

    T_kevs = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    mg = make_mg(16)

    medians = []
    maxes = []
    p95s = []

    emit('| T (keV) | Median rel diff | 95th pctl | Max rel diff |')
    emit('|---------|----------------|-----------|--------------|')

    for T_kev in T_kevs:
        T = T_kev * kev_kelvin

        S_quad = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
        S_series = mg.compute_sigma_matrix(KERNEL_SOLVER, T=T, Ne=1.0)

        mask = (np.abs(S_quad) > 1e-40) & (np.abs(S_series) > 1e-40)
        scale = np.maximum(np.abs(S_quad), 1e-300)
        rel_diff = np.abs(S_series[mask] - S_quad[mask]) / scale[mask]

        if rel_diff.size == 0:
            medians.append(np.nan)
            maxes.append(np.nan)
            p95s.append(np.nan)
            emit(f'| {T_kev} | -- | -- | -- |')
            continue

        med = np.median(rel_diff)
        p95 = np.percentile(rel_diff, 95)
        mx = np.max(rel_diff)
        medians.append(med)
        maxes.append(mx)
        p95s.append(p95)
        emit(f'| {T_kev} | {med:.2e} | {p95:.2e} | {mx:.2e} |')

    emit()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(T_kevs, medians, 'bo-', markersize=5, label='Median')
    ax.semilogy(T_kevs, p95s, 'g^-', markersize=5, label='95th percentile')
    ax.semilogy(T_kevs, maxes, 'rs-', markersize=5, label='Max')
    ax.set_xscale('log')
    ax.set_xlabel('T (keV)')
    ax.set_ylabel('Relative difference')
    ax.set_title('Series vs Quadrature agreement across temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = save_fig('svq_temp_sweep.png')

    emit(f'![Temperature sweep]({fig_path})')
    emit()


# ─── Section 4: Row-sum comparison ───────────────────────────────────────

def section_row_sums():
    emit('## 4. Row-Sum (Total Cross Section) Comparison')
    emit()
    emit('Total scattering cross section per group (row sums of the G×G matrix) '
         'at T = 10 keV, showing that series and quadrature agree on the '
         'physically most important quantity.')
    emit()

    T = 10.0 * kev_kelvin
    mg = make_mg(16)

    S_quad = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
    S_series = mg.compute_sigma_matrix(KERNEL_SOLVER, T=T, Ne=1.0)

    row_q = S_quad.sum(axis=1)
    row_s = S_series.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.semilogy(CENTERS_KEV, row_q, 'b-o', markersize=4, label='Quadrature (Q64)')
    ax1.semilogy(CENTERS_KEV, row_s, 'r--s', markersize=4, label='Series (Auto)')
    ax1.set_ylabel('Row sum (total σ)')
    ax1.set_title('Total scattering cross section per group (T = 10 keV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    rel = np.abs(row_s - row_q) / np.maximum(np.abs(row_q), 1e-300)
    ax2.semilogy(CENTERS_KEV, rel, 'ko-', markersize=4)
    ax2.set_xlabel('Group center energy (keV)')
    ax2.set_ylabel('Relative diff')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = save_fig('svq_row_sums.png')

    emit(f'![Row sums]({fig_path})')
    emit()

    emit(f'Max row-sum relative difference: {np.max(rel):.2e}')
    emit()


# ─── Section 5: Temperature derivative comparison ────────────────────────

def section_derivative():
    emit('## 5. Temperature Derivative Comparison')
    emit()
    emit('Relative difference of the `dsigma_dT` multigroup matrix '
         'between series(Auto) and quadrature(Q64) backends.')
    emit()

    T_kevs = [1.0, 10.0, 100.0]
    mg = make_mg(16)

    fig, axes = plt.subplots(1, len(T_kevs), figsize=(5.5 * len(T_kevs), 5))

    emit('| T (keV) | Median rel diff | Max rel diff |')
    emit('|---------|----------------|--------------|')

    for idx, T_kev in enumerate(T_kevs):
        T = T_kev * kev_kelvin

        dS_quad = mg.compute_dsigma_dT_matrix(KERNEL_Q64, T=T, Ne=1.0)
        dS_series = mg.compute_dsigma_dT_matrix(KERNEL_SOLVER, T=T, Ne=1.0)

        mask = (np.abs(dS_quad) > 1e-40) & (np.abs(dS_series) > 1e-40)
        scale = np.where(mask, np.abs(dS_quad), 1.0)
        rel_diff = np.where(mask, np.abs(dS_series - dS_quad) / scale, 0.0)

        ax = axes[idx]
        im = ax.imshow(rel_diff, aspect='auto', origin='lower', cmap='Reds',
                       vmin=0, vmax=min(0.01, rel_diff.max()) if rel_diff.max() > 0 else 0.01)
        ax.set_title(f'T = {T_kev} keV')
        set_ticks(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        valid = rel_diff[mask]
        if valid.size > 0:
            emit(f'| {T_kev} | {np.median(valid):.2e} | {np.max(valid):.2e} |')
        else:
            emit(f'| {T_kev} | -- | -- |')

    fig.suptitle('dsigma/dT relative difference (series vs quadrature)', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('svq_derivative.png')

    emit()
    emit(f'![Derivative comparison]({fig_path})')
    emit()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    emit('# Series vs Quadrature — Multigroup Kernel Comparison')
    emit()
    emit('Comparison of the multigroup Compton scattering matrix when evaluated '
         'using `ComptonKernelSeries(Auto)` vs `ComptonKernelQuadrature(64)` as '
         'the point-wise kernel backend.  Both use the same '
         '`ComptonMultigroupKernel` integration machinery; only the inner kernel '
         'call differs.')
    emit()

    section_elementwise()
    section_heatmaps()
    section_temperature_sweep()
    section_row_sums()
    section_derivative()

    md_path = os.path.join(GEN_DIR, 'multigroup_series_vs_quadrature.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
