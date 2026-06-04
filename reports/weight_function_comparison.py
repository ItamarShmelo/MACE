"""
Weight function comparison report.

Compares multigroup Compton scattering matrices produced with three
different energy-weighting functions: Planck, Uniform, and Wien.

Usage:
    python3 reports/weight_function_comparison.py

Output:
    reports/generated/weight_function_comparison.md  (+ .png plots in figs/)
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

import _compton_multigroup as cm
import _compton_kernel_quadrature as cq
from _units import kev, kev_kelvin, sigma_thomson

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


BOUNDS_KEV = np.logspace(np.log10(0.1), np.log10(100.0), 21)
BOUNDS_ERG = BOUNDS_KEV * kev
KERNEL = cq.ComptonKernelQuadrature(64)
QUAD_ORDER = 16

WEIGHT_FUNCTIONS = {
    'Planck': cm.PlanckWeightFunction(cap_x=25.0),
    'Uniform': cm.UniformWeightFunction(),
    'Wien': cm.WienWeightFunction(cap_x=25.0),
}


def _build_mg(wf):
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=BOUNDS_ERG.tolist(),
        weight_function=wf,
        quad_order_E=QUAD_ORDER,
        quad_order_Ep=QUAD_ORDER,
        quad_order_mu=QUAD_ORDER)


def _rel_diff(S, S_ref):
    """Element-wise relative difference, masking near-zero entries."""
    mask = np.abs(S_ref) > 1e-40
    rd = np.full_like(S, np.nan)
    rd[mask] = np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask])
    return rd


# ─── Section 1: Heatmap comparison ──────────────────────────────────────

def section_heatmaps():
    G = len(BOUNDS_ERG) - 1
    centers_kev = np.sqrt(BOUNDS_KEV[:-1] * BOUNDS_KEV[1:])
    tick_pos = np.arange(0, G, max(1, G // 6))
    tick_lbl = [f'{centers_kev[i]:.1f}' for i in tick_pos]

    T_kevs = [1.0, 10.0, 100.0]
    wf_names = list(WEIGHT_FUNCTIONS.keys())

    for T_kev in T_kevs:
        T = T_kev * kev_kelvin

        matrices = {}
        for name, wf in WEIGHT_FUNCTIONS.items():
            mg = _build_mg(wf)
            matrices[name] = mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)

        emit(f'### T = {T_kev} keV')
        emit()

        fig, axes = plt.subplots(1, len(wf_names), figsize=(5.5 * len(wf_names), 5))

        all_pos = [np.maximum(np.abs(matrices[n]), 1e-50) for n in wf_names]
        sig_vals = np.concatenate([a[a > 1e-50] for a in all_pos])
        vmin = sig_vals.min() if sig_vals.size else 1e-50
        vmax = max(a.max() for a in all_pos)

        for idx, name in enumerate(wf_names):
            ax = axes[idx]
            S_pos = np.maximum(np.abs(matrices[name]), 1e-50)
            im = ax.imshow(S_pos, norm=LogNorm(vmin=vmin, vmax=vmax),
                           aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(name)
            ax.set_xlabel("E' (keV)")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, rotation=45, fontsize=7)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_lbl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_ylabel('E (keV)')
        fig.suptitle(f'Multigroup $\\sigma$ matrix — T = {T_kev} keV', y=1.02)
        fig.tight_layout()
        fig_path = save_fig(f'wf_heatmap_T{T_kev:.0f}.png')
        emit(f'![Heatmap T={T_kev}]({fig_path})')
        emit()


# ─── Section 2: Relative difference maps ────────────────────────────────

def section_relative_diffs():
    G = len(BOUNDS_ERG) - 1
    centers_kev = np.sqrt(BOUNDS_KEV[:-1] * BOUNDS_KEV[1:])
    tick_pos = np.arange(0, G, max(1, G // 6))
    tick_lbl = [f'{centers_kev[i]:.1f}' for i in tick_pos]

    T_kevs = [1.0, 10.0, 100.0]
    alt_names = ['Uniform', 'Wien']

    for T_kev in T_kevs:
        T = T_kev * kev_kelvin

        mg_planck = _build_mg(WEIGHT_FUNCTIONS['Planck'])
        S_planck = mg_planck.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)

        fig, axes = plt.subplots(1, len(alt_names),
                                 figsize=(6 * len(alt_names), 5))

        for idx, name in enumerate(alt_names):
            mg = _build_mg(WEIGHT_FUNCTIONS[name])
            S_alt = mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)
            rd = _rel_diff(S_alt, S_planck)

            ax = axes[idx]
            finite = rd[np.isfinite(rd)]
            if finite.size == 0:
                continue
            vmin_map = max(np.nanmin(finite), 1e-14)
            vmax_map = max(np.nanmax(finite), vmin_map * 10)
            im = ax.imshow(rd, aspect='auto', origin='lower', cmap='hot_r',
                           norm=LogNorm(vmin=vmin_map, vmax=vmax_map))
            ax.set_title(f'{name} vs Planck')
            ax.set_xlabel("E' (keV)")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, rotation=45, fontsize=7)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_lbl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_ylabel('E (keV)')
        fig.suptitle(f'Relative difference vs Planck — T = {T_kev} keV',
                     y=1.02)
        fig.tight_layout()
        fig_path = save_fig(f'wf_reldiff_T{T_kev:.0f}.png')
        emit(f'![Rel diff T={T_kev}]({fig_path})')
        emit()

        emit(f'| Weight | Median rel diff | Max rel diff |')
        emit(f'|--------|----------------|-------------|')
        for name in alt_names:
            mg = _build_mg(WEIGHT_FUNCTIONS[name])
            S_alt = mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)
            rd = _rel_diff(S_alt, S_planck)
            finite = rd[np.isfinite(rd)]
            if finite.size > 0:
                emit(f'| {name} | {np.nanmedian(finite):.2e} '
                     f'| {np.nanmax(finite):.2e} |')
            else:
                emit(f'| {name} | -- | -- |')
        emit()


# ─── Section 3: Row sums (total cross section) ──────────────────────────

def section_row_sums():
    G = len(BOUNDS_ERG) - 1
    centers_kev = np.sqrt(BOUNDS_KEV[:-1] * BOUNDS_KEV[1:])

    T_kevs = [1.0, 10.0, 100.0]

    fig, axes = plt.subplots(1, len(T_kevs), figsize=(6 * len(T_kevs), 5))

    for t_idx, T_kev in enumerate(T_kevs):
        T = T_kev * kev_kelvin
        ax = axes[t_idx]

        for name, wf in WEIGHT_FUNCTIONS.items():
            mg = _build_mg(wf)
            S = mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)
            row_sums = S.sum(axis=1)
            ax.semilogy(centers_kev, row_sums, 'o-', markersize=4, label=name)

        ax.set_xlabel('E (keV)')
        ax.set_title(f'T = {T_kev} keV')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Total scattering cross section (cm$^2$)')
    fig.suptitle('Row sums $\\sum_{g\'} \\sigma(g{\\to}g\')$ by weight function',
                 y=1.02)
    fig.tight_layout()
    fig_path = save_fig('wf_row_sums.png')
    emit(f'![Row sums]({fig_path})')
    emit()

    emit('| T (keV) | Weight | Max row-sum rel diff vs Planck |')
    emit('|---------|--------|-------------------------------|')
    for T_kev in T_kevs:
        T = T_kev * kev_kelvin
        mg_planck = _build_mg(WEIGHT_FUNCTIONS['Planck'])
        S_planck = mg_planck.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)
        rs_planck = S_planck.sum(axis=1)

        for name in ['Uniform', 'Wien']:
            mg = _build_mg(WEIGHT_FUNCTIONS[name])
            S = mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)
            rs = S.sum(axis=1)
            mask = rs_planck > 1e-40
            if np.any(mask):
                mx = np.max(np.abs(rs[mask] - rs_planck[mask]) / rs_planck[mask])
                emit(f'| {T_kev} | {name} | {mx:.2e} |')
            else:
                emit(f'| {T_kev} | {name} | -- |')
    emit()


# ─── Section 4: Weight function shape ────────────────────────────────────

def section_weight_shapes():
    x = np.linspace(0.01, 25.0, 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    planck_w = x**3 / np.expm1(x)
    wien_w = x**3 * np.exp(-x)

    planck_w /= planck_w.max()
    wien_w /= wien_w.max()

    ax.plot(x, planck_w, '-', linewidth=2,
            label='Planck: $x^3/(e^x-1)$ (normalized)')
    ax.plot(x, wien_w, '-.', linewidth=2,
            label='Wien: $x^3 e^{-x}$ (normalized)')

    ax.set_xlabel('$x = E / (kT)$')
    ax.set_ylabel('$w(x)$ / max')
    ax.set_title('Weight function profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = save_fig('wf_shapes.png')
    emit(f'![Weight function shapes]({fig_path})')
    emit()


# ─── Section 5: Pomraning-style differential cross section ───────────────

def section_pomraning():
    alt_styles = {
        'Uniform': {'color': 'green', 'ls': '--', 'lw': 1.5},
        'Wien':    {'color': 'red',   'ls': '-.', 'lw': 1.5},
    }

    eb_kev = BOUNDS_KEV
    eb_erg = BOUNDS_ERG.tolist()
    ewid_kev = np.diff(eb_kev)
    ec_kev = np.sqrt(eb_kev[:-1] * eb_kev[1:])

    T_kevs = [1.0, 10.0, 100.0]

    for T_kev in T_kevs:
        T = T_kev * kev_kelvin

        matrices = {}
        for name, wf in WEIGHT_FUNCTIONS.items():
            mg = _build_mg(wf)
            S = mg.compute_sigma_matrix(KERNEL, T=T, Ne=1.0)
            matrices[name] = np.array(S)

        dsigma_planck = matrices['Planck'] / ewid_kev[np.newaxis, :] / sigma_thomson

        peak_group = np.argmax(dsigma_planck.max(axis=1))
        groups_to_plot = sorted({
            0,
            len(ec_kev) // 4,
            len(ec_kev) // 2,
            peak_group,
            len(ec_kev) - 1,
        })

        fig, axes = plt.subplots(2, len(groups_to_plot),
                                 figsize=(4.5 * len(groups_to_plot), 8),
                                 gridspec_kw={'height_ratios': [3, 1]})

        for col, g_in in enumerate(groups_to_plot):
            ax_top = axes[0, col]
            ax_bot = axes[1, col]

            dsig_p = matrices['Planck'][g_in, :] / ewid_kev / sigma_thomson

            ax_top.stairs(dsig_p, edges=eb_kev, color='blue', linewidth=2.0,
                          linestyle='-', label='Planck')

            for name, sty in alt_styles.items():
                dsig = matrices[name][g_in, :] / ewid_kev / sigma_thomson
                ax_top.stairs(dsig, edges=eb_kev, label=name,
                              color=sty['color'], linestyle=sty['ls'],
                              linewidth=sty['lw'])

                mask = dsig_p > 1e-50
                safe_p = np.where(mask, dsig_p, 1.0)
                ratio = np.where(mask, dsig / safe_p, np.nan)
                ax_bot.stairs(ratio, edges=eb_kev, color=sty['color'],
                              linestyle=sty['ls'], linewidth=sty['lw'],
                              label=name)

            ax_bot.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)

            ax_top.set_ylim(bottom=0)
            ax_top.set_title(
                f'$E_{{in}}$ = {ec_kev[g_in]:.1f} keV '
                f'($x$ = {ec_kev[g_in]/T_kev:.1f})',
                fontsize=9)
            ax_top.grid(True, alpha=0.3)
            ax_top.set_ylabel(
                r"$\sigma / (\sigma_T \Delta E')$ [1/keV]", fontsize=8)

            ax_bot.set_xlabel(r"$E'$ [keV]", fontsize=8)
            ax_bot.set_ylabel('ratio to Planck', fontsize=8)
            ax_bot.grid(True, alpha=0.3)

            finite_ratios = []
            for name in alt_styles:
                dsig = matrices[name][g_in, :] / ewid_kev / sigma_thomson
                mask = dsig_p > 1e-50
                safe_p = np.where(mask, dsig_p, 1.0)
                r = np.where(mask, dsig / safe_p, np.nan)
                finite_ratios.append(r[np.isfinite(r)])
            if any(f.size > 0 for f in finite_ratios):
                all_r = np.concatenate([f for f in finite_ratios if f.size])
                spread = max(np.abs(all_r - 1.0).max(), 0.02)
                ax_bot.set_ylim(1.0 - 1.3 * spread, 1.0 + 1.3 * spread)

        axes[0, 0].legend(fontsize=7, loc='upper right')
        axes[1, 0].legend(fontsize=7, loc='upper right')

        fig.suptitle(
            f'Pomraning-style differential cross section — T = {T_kev} keV '
            f'({len(eb_kev)-1} log-spaced groups)',
            y=1.02)
        fig.tight_layout()
        fig_path = save_fig(f'wf_pomraning_T{T_kev:.0f}.png')
        emit(f'![Pomraning T={T_kev}]({fig_path})')
        emit()


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    emit('# Weight Function Comparison Report')
    emit()
    emit('Comparison of multigroup Compton scattering matrices computed '
         'with three weight functions: **Planck** (capped, cap_x=25), '
         '**Uniform** (flat), and **Wien** ($x^3 e^{-x}$).')
    emit()
    emit(f'Group structure: {len(BOUNDS_ERG)-1} log-spaced groups from '
         f'{BOUNDS_KEV[0]:.1f} to {BOUNDS_KEV[-1]:.0f} keV.  '
         f'Quadrature order: {QUAD_ORDER}.')
    emit()

    emit('## 1. Weight Function Profiles')
    emit()
    section_weight_shapes()

    emit('## 2. Multigroup Matrix Heatmaps')
    emit()
    emit('Side-by-side log-scale heatmaps of the angle-integrated '
         '$\\sigma(g{\\to}g\')$ matrix at several temperatures.')
    emit()
    section_heatmaps()

    emit('## 3. Relative Differences vs Planck')
    emit()
    emit('Element-wise relative difference of each alternative weight '
         'function against the Planck baseline.')
    emit()
    section_relative_diffs()

    emit('## 4. Total Cross Sections (Row Sums)')
    emit()
    emit('The row sum $\\sum_{g\'} \\sigma(g{\\to}g\')$ gives the total '
         'scattering cross section out of each group.  This is the most '
         'transport-relevant quantity.')
    emit()
    section_row_sums()

    emit('## 5. Pomraning-Style Differential Cross Section')
    emit()
    emit('Differential scattering cross section '
         '$\\sigma(E{\\to}E\') / (\\sigma_T \\cdot \\Delta E\')$ '
         'for a fixed incoming group, computed on a fine energy grid.  '
         'This shows how the weight function reshapes the spectral '
         'profile of the multigroup kernel.')
    emit()
    section_pomraning()

    md_path = os.path.join(GEN_DIR, 'weight_function_comparison.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
