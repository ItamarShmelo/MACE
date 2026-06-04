"""
Angular PDF report for group-to-group Compton scattering transitions.

Computes the angular probability density function (PDF) for transitions
from group 4 to groups 3, 4, and 5 using a 20-group Planck-weighted
structure matching the reference paper specification.

Group structure:
    20 groups from 0.55 keV to 10.5 keV
    First group width: 0.45 keV, all others: 0.5 keV

Parameters:
    50 equal angular bins on [-1, 1]
    Quadrature order: N=32 (all axes)
    T = 0.345 keV
    Kernel: ComptonKernelSeries(Auto)

Usage:
    python3 reports/angular_pdf_report.py

Output:
    reports/generated/angular_pdf_report.md  (+ .png plots in figs/)
"""
import sys
import os
import math
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))

import _compton_multigroup as cm
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
    plt.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    return f'figs/{name}'


# ─── Parameters ───────────────────────────────────────────────────────────

NUM_ANGLE_BINS = 50
QUAD_ORDER = 32
T_KEVS = [0.345, 10.0]

BOUNDARIES_KEV = [0.55] + [1.0 + 0.5 * i for i in range(20)]
BOUNDARIES_ERG = [b * kev for b in BOUNDARIES_KEV]

TRANSITIONS = [
    (4, 3),
    (4, 4),
    (4, 5),
]


def group_label(g_1based):
    """Return a label string for a 1-based group index."""
    lo = BOUNDARIES_KEV[g_1based - 1]
    hi = BOUNDARIES_KEV[g_1based]
    return f"g{g_1based} [{lo:.2f}, {hi:.2f}] keV"


# ─── Computation ──────────────────────────────────────────────────────────

def compute_angular_pdf(T):
    kernel = ComptonKernelSolver()

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=BOUNDARIES_ERG,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=QUAD_ORDER,
        quad_order_Ep=QUAD_ORDER,
        quad_order_mu=QUAD_ORDER,
    )

    S_3d = mg.compute_sigma_matrix(
        kernel, num_angle_bins=NUM_ANGLE_BINS, T=T, Ne=1.0)

    delta_mu = 2.0 / NUM_ANGLE_BINS
    mu_centers = np.linspace(-1 + delta_mu / 2, 1 - delta_mu / 2, NUM_ANGLE_BINS)
    mu_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)

    return S_3d, mu_centers, mu_edges, delta_mu


# ─── Report ───────────────────────────────────────────────────────────────

def generate_temperature_section(T_kev, fig_name):
    """Generate angular PDF plots for a single temperature."""
    T = T_kev * kev_kelvin

    emit(f'## T = {T_kev} keV')
    emit()

    print(f"Computing multiangle sigma matrix for T = {T_kev} keV...")
    S_3d, mu_centers, mu_edges, delta_mu = compute_angular_pdf(T)
    print("Done.")

    # ── CMMC Monte Carlo reference ──────────────────────────────────────────
    mc_obj = None
    NUM_MC_ANGLE_BINS = None
    try:
        import _compton_matrix_mc as mc_mod
        NUM_MC_ANGLE_BINS = mc_mod.ComptonMatrixMC.NUM_ANGLE_BINS
        centers_erg = [math.sqrt(BOUNDARIES_ERG[i] * BOUNDARIES_ERG[i + 1])
                       for i in range(len(BOUNDARIES_ERG) - 1)]
        mc_obj = mc_mod.ComptonMatrixMC(
            energy_groups_centers=centers_erg,
            energy_groups_boundaries=BOUNDARIES_ERG,
            num_of_samples=500000,
            force_detailed_balance=False,
            seed=42)
        mc_obj.set_tables(temperature_grid=[T * 0.9, T, T * 1.1])
        print(f"CMMC loaded for T = {T_kev} keV.")
    except Exception as e:
        print(f"CMMC not available: {e}")

    x_max = 4.0
    bin_width = x_max / NUM_ANGLE_BINS
    bin_edges = np.linspace(0, x_max, NUM_ANGLE_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for idx, (g_from, g_to) in enumerate(TRANSITIONS):
        ax = axes[idx]

        g0_idx = g_from - 1
        g_idx = g_to - 1
        row = S_3d[g0_idx, g_idx, :]
        total = row.sum()

        if total > 0:
            pdf = row / (total * bin_width)
        else:
            pdf = np.zeros_like(row)

        ax.plot(bin_centers, pdf, 'b-', linewidth=1.5, label='Series')

        if mc_obj is not None:
            cdf_mc = np.array(mc_obj.get_angle_cdf(
                temperature=T, g0=g0_idx, g=g_idx))
            pdf_mc_bins = np.diff(cdf_mc)
            mc_bin_width = x_max / NUM_MC_ANGLE_BINS
            pdf_mc = pdf_mc_bins / mc_bin_width
            mc_centers = np.linspace(
                mc_bin_width / 2, x_max - mc_bin_width / 2, NUM_MC_ANGLE_BINS)
            ax.plot(mc_centers, pdf_mc, 'ro', markersize=4, label='CMMC')

        ax.set_xlabel('Angular bin')
        ax.set_ylabel(r'$p$')
        ax.set_title(f'Group {g_from} → {g_to}\n'
                     f'({BOUNDARIES_KEV[g_from-1]:.2f}–{BOUNDARIES_KEV[g_from]:.2f} '
                     f'→ {BOUNDARIES_KEV[g_to-1]:.2f}–{BOUNDARIES_KEV[g_to]:.2f} keV)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, x_max)
        ax.legend(fontsize=8)

        sigma_total = total
        peak_idx = np.argmax(pdf)
        emit(f'### Group {g_from} → {g_to}')
        emit()
        emit(f'- Incident: {group_label(g_from)}')
        emit(f'- Scattered: {group_label(g_to)}')
        emit(f'- Total cross section (sum over bins): {sigma_total:.6e} cm²')
        emit(f'- Peak PDF value: {pdf.max():.4f} '
             f'(x = {bin_centers[peak_idx]:.3f})')
        emit(f'- Corresponding μ ∈ [{mu_edges[peak_idx]:.3f}, '
             f'{mu_edges[peak_idx+1]:.3f}]')
        emit()

    fig.suptitle(f'Angular PDF — T = {T_kev} keV, Series kernel, N = {QUAD_ORDER}',
                 y=1.02)
    fig.tight_layout()
    fig_path = save_fig(fig_name)

    emit(f'![Angular PDF plots T={T_kev} keV]({fig_path})')
    emit()


def main():
    emit('# Angular PDF Report')
    emit()
    emit('Angular probability density functions for group-to-group Compton '
         'scattering transitions using the series kernel.')
    emit()
    emit('## Parameters')
    emit()
    emit(f'- **Group structure:** 20 Planck-weighted groups, '
         f'{BOUNDARIES_KEV[0]:.2f} keV to {BOUNDARIES_KEV[-1]:.2f} keV')
    emit(f'- **Temperatures:** {", ".join(f"{t} keV" for t in T_KEVS)}')
    emit(f'- **Angular bins:** {NUM_ANGLE_BINS} equal bins on [-1, 1]')
    emit(f'- **Quadrature order:** N = {QUAD_ORDER} (all axes)')
    emit(f'- **Kernel:** ComptonKernelSeries(Auto)')
    emit(f'- **Normalization:** PDF density on [0, 4]')
    emit()

    emit('## Group Boundaries')
    emit()
    emit('| Group | Range (keV) | Width (keV) |')
    emit('|-------|-------------|-------------|')
    for g in range(1, len(BOUNDARIES_KEV)):
        lo = BOUNDARIES_KEV[g - 1]
        hi = BOUNDARIES_KEV[g]
        emit(f'| {g} | [{lo:.2f}, {hi:.2f}] | {hi - lo:.2f} |')
    emit()

    for T_kev in T_KEVS:
        fig_name = f'angular_pdf_g4_T{T_kev:.3g}kev.png'
        generate_temperature_section(T_kev, fig_name)

    md_path = os.path.join(GEN_DIR, 'angular_pdf_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
