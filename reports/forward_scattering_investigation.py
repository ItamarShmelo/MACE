"""
Forward scattering (xi -> 1) divergence investigation.

Investigates the last-bin spike in the 4->4 angular PDF at T = 0.345 keV.
Produces diagnostic output and plots comparing kernel methods near xi = 1.
"""
import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))

import _compton_multigroup as cm
import _compton_kernel_series as cs
import _compton_kernel_quadrature as cq
from _units import kev, kev_kelvin

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

T_KEV = 0.345
T_K = T_KEV * kev_kelvin
NUM_ANGLE_BINS = 50
QUAD_ORDER = 32

BOUNDARIES_KEV = [0.55] + [1.0 + 0.5 * i for i in range(20)]
BOUNDARIES_ERG = [b * kev for b in BOUNDARIES_KEV]

E_CENTER = math.sqrt(BOUNDARIES_ERG[3] * BOUNDARIES_ERG[4])  # group 4 geometric center


def safe_eval(kernel, E, Ep, xi, T, Ne):
    """Evaluate kernel, returning (value, abs_error, rel_error) or NaN on failure."""
    try:
        r = kernel.sigma_E(E, Ep, xi, T, Ne)
        return r.value, r.estimated_abs_error, r.estimated_rel_error
    except Exception:
        return float('nan'), float('nan'), float('nan')


# =============================================================================
# STEP 1: Diagnostic dump at GL nodes of last bin
# =============================================================================
def step1_diagnostic_dump():
    print("=" * 80)
    print("STEP 1: Diagnostic dump at GL nodes of last angular bin [0.96, 1.0]")
    print("=" * 80)
    print(f"  E = E' = {E_CENTER / kev:.4f} keV (group 4 center)")
    print(f"  T = {T_KEV} keV")
    print()

    nodes, weights = cm.gauss_legendre_rule(QUAD_ORDER)

    mu_lo, mu_hi = 0.96, 1.0
    half_w = 0.5 * (mu_hi - mu_lo)
    mid = 0.5 * (mu_lo + mu_hi)
    xi_nodes = half_w * nodes + mid

    kernel_auto = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    kernel_asymp = cs.ComptonKernelSeries(cs.SeriesMethod.Asymptotic)
    kernel_ps = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeries)
    kernel_pshp = cs.ComptonKernelSeries(cs.SeriesMethod.PowerSeriesHighPrecision)
    kernel_quad = cq.ComptonKernelQuadrature(128, cq.QuadratureForm.PostIBP)

    print(f"{'i':>3} {'xi':>18} {'1-xi':>12} {'Auto':>14} {'Asymp':>14} "
          f"{'PS':>14} {'PSHP':>14} {'Quad128':>14} {'Auto_relerr':>12}")
    print("-" * 130)

    for i in range(len(xi_nodes)):
        xi = float(xi_nodes[i])
        one_minus_xi = 1.0 - xi

        v_auto, _, re_auto = safe_eval(kernel_auto, E_CENTER, E_CENTER, xi, T_K, 1.0)
        v_asymp, _, _ = safe_eval(kernel_asymp, E_CENTER, E_CENTER, xi, T_K, 1.0)
        v_ps, _, _ = safe_eval(kernel_ps, E_CENTER, E_CENTER, xi, T_K, 1.0)
        v_pshp, _, _ = safe_eval(kernel_pshp, E_CENTER, E_CENTER, xi, T_K, 1.0)
        v_quad, _, _ = safe_eval(kernel_quad, E_CENTER, E_CENTER, xi, T_K, 1.0)

        print(f"{i:3d} {xi:18.14f} {one_minus_xi:12.4e} "
              f"{v_auto:14.6e} {v_asymp:14.6e} {v_ps:14.6e} "
              f"{v_pshp:14.6e} {v_quad:14.6e} {re_auto:12.4e}")

    print()


# =============================================================================
# STEP 2: Method comparison plot -- sigma_E vs xi near forward scattering
# Also test with E != E' (group edges)
# =============================================================================
def step2_method_comparison():
    print("=" * 80)
    print("STEP 2: Method comparison plot")
    print("=" * 80)

    one_minus_xi = np.logspace(-1.3, -12, 200)  # from ~0.05 down to 1e-12
    xi_grid = 1.0 - one_minus_xi

    kernel_auto = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    kernel_asymp = cs.ComptonKernelSeries(cs.SeriesMethod.Asymptotic)
    kernel_quad128 = cq.ComptonKernelQuadrature(128, cq.QuadratureForm.PostIBP)

    # Test three (E, E') combinations within group 4
    E_lo = BOUNDARIES_ERG[3]   # 2.0 keV
    E_hi = BOUNDARIES_ERG[4]   # 2.5 keV
    E_mid = E_CENTER           # 2.236 keV

    cases = [
        (E_mid, E_mid, "E=E'=2.24 keV"),
        (E_lo, E_hi, "E=2.0, E'=2.5 keV"),
        (E_hi, E_lo, "E=2.5, E'=2.0 keV"),
    ]

    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 4*len(cases)), squeeze=False)

    for row, (E, Ep, label) in enumerate(cases):
        vals_auto = np.full(len(xi_grid), np.nan)
        vals_asymp = np.full(len(xi_grid), np.nan)
        vals_quad = np.full(len(xi_grid), np.nan)
        errs_auto = np.full(len(xi_grid), np.nan)

        for i, xi in enumerate(xi_grid):
            v, _, re = safe_eval(kernel_auto, E, Ep, float(xi), T_K, 1.0)
            vals_auto[i] = v
            errs_auto[i] = re
            v, _, _ = safe_eval(kernel_asymp, E, Ep, float(xi), T_K, 1.0)
            vals_asymp[i] = v
            v, _, _ = safe_eval(kernel_quad128, E, Ep, float(xi), T_K, 1.0)
            vals_quad[i] = v

        ax_val = axes[row, 0]
        ax_err = axes[row, 1]

        for name, vals, ls in [('Auto/Asymp', vals_auto, '-'),
                               ('Quad128', vals_quad, '--')]:
            valid = np.isfinite(vals) & (vals > 0)
            if valid.any():
                ax_val.plot(one_minus_xi[valid], vals[valid], ls, label=name, linewidth=1.5)

        ax_val.set_xscale('log')
        ax_val.set_yscale('log')
        ax_val.set_ylabel(r'$\Sigma_E$ [cm$^2$/erg]')
        ax_val.set_title(f'{label}')
        ax_val.legend(fontsize=9)
        ax_val.grid(True, alpha=0.3)
        ax_val.invert_xaxis()

        # Relative difference Auto vs Quad
        valid = (np.isfinite(vals_auto) & np.isfinite(vals_quad)
                 & (np.abs(vals_quad) > 0))
        rel_diff = np.abs(vals_auto[valid] - vals_quad[valid]) / np.abs(vals_quad[valid])
        ax_err.plot(one_minus_xi[valid], rel_diff, 'b-', label='|Auto-Quad|/|Quad|',
                    linewidth=1.5)
        valid_re = np.isfinite(errs_auto) & (errs_auto > 0)
        ax_err.plot(one_minus_xi[valid_re], errs_auto[valid_re], 'orange',
                    label='Auto self-reported', linewidth=1)
        ax_err.axhline(0.01, color='red', linestyle='--', alpha=0.7, label='1%')
        ax_err.set_xscale('log')
        ax_err.set_yscale('log')
        ax_err.set_ylabel('Relative error')
        ax_err.set_title(f'{label} -- error')
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3)
        ax_err.invert_xaxis()

    axes[-1, 0].set_xlabel(r'$1 - \xi$')
    axes[-1, 1].set_xlabel(r'$1 - \xi$')

    fig.suptitle(f'Step 2: Kernel value near xi=1, T = {T_KEV} keV', y=1.01)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fwd_scatter_step2_method_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# STEP 3: Quadrature (post-IBP) vs Series -- focus on where they diverge
# Test at group edges where E != E'
# =============================================================================
def step3_quadrature_check():
    print("=" * 80)
    print("STEP 3: Kernel behavior at group edges near xi=1")
    print("=" * 80)

    one_minus_xi = np.logspace(-1.3, -10, 150)
    xi_grid = 1.0 - one_minus_xi

    kernel_auto = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    kernel_q128 = cq.ComptonKernelQuadrature(128, cq.QuadratureForm.PostIBP)

    E_lo = BOUNDARIES_ERG[3]   # 2.0 keV
    E_hi = BOUNDARIES_ERG[4]   # 2.5 keV

    cases = [
        (E_CENTER, E_CENTER, "E=E'=center"),
        (E_lo, E_lo, "E=E'=2.0 keV (lower edge)"),
        (E_hi, E_hi, "E=E'=2.5 keV (upper edge)"),
        (E_lo, E_hi, "E=2.0, E'=2.5 keV"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (E, Ep, label) in enumerate(cases):
        ax = axes[idx]
        vals_auto = np.full(len(xi_grid), np.nan)
        vals_quad = np.full(len(xi_grid), np.nan)

        for i, xi in enumerate(xi_grid):
            v, _, _ = safe_eval(kernel_auto, E, Ep, float(xi), T_K, 1.0)
            vals_auto[i] = v
            v, _, _ = safe_eval(kernel_q128, E, Ep, float(xi), T_K, 1.0)
            vals_quad[i] = v

        valid_a = np.isfinite(vals_auto) & (vals_auto > 0)
        valid_q = np.isfinite(vals_quad) & (vals_quad > 0)

        if valid_a.any():
            ax.plot(one_minus_xi[valid_a], vals_auto[valid_a], 'b-',
                    label='Series(Auto)', linewidth=1.5)
        if valid_q.any():
            ax.plot(one_minus_xi[valid_q], vals_quad[valid_q], 'g--',
                    label='Quad128', linewidth=1.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$1 - \xi$')
        ax.set_ylabel(r'$\Sigma_E$')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    fig.suptitle(f'Step 3: Kernel at group edges near xi=1, T={T_KEV} keV', y=1.01)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fwd_scatter_step3_quadrature_check.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# STEP 4: Energy quadrature convergence for last bin
# =============================================================================
def step4_eps_sweep():
    print("=" * 80)
    print("STEP 4: Energy quadrature convergence for last angular bin")
    print("=" * 80)

    sub_boundaries = BOUNDARIES_ERG[2:6]  # groups 3,4,5
    kernel_auto = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    NUM_BINS = 32

    # CMMC reference
    cmmc_last_pdf = None
    try:
        import _compton_matrix_mc as mc_mod
        centers_erg = [math.sqrt(BOUNDARIES_ERG[i] * BOUNDARIES_ERG[i + 1])
                       for i in range(len(BOUNDARIES_ERG) - 1)]
        mc_obj = mc_mod.ComptonMatrixMC(
            energy_groups_centers=centers_erg,
            energy_groups_boundaries=BOUNDARIES_ERG,
            num_of_samples=2000000,
            force_detailed_balance=False,
            seed=42)
        mc_obj.set_tables(temperature_grid=[T_K * 0.9, T_K, T_K * 1.1])
        cdf_mc = np.array(mc_obj.get_angle_cdf(temperature=T_K, g0=3, g=3))
        pdf_mc_bins = np.diff(cdf_mc)
        x_max = 4.0
        mc_bin_width = x_max / mc_mod.ComptonMatrixMC.NUM_ANGLE_BINS
        mc_pdf = pdf_mc_bins / mc_bin_width
        cmmc_last_pdf = mc_pdf[-1]
        print(f"  CMMC last-bin PDF: {cmmc_last_pdf:.5f}")
    except Exception as e:
        print(f"  CMMC not available: {e}")

    # Sweep N_E with fixed N_mu=32
    quad_orders = [8, 12, 16, 24, 32, 48, 64, 80, 96]
    last_bin_pdf = []
    second_last_pdf = []

    for qE in quad_orders:
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=sub_boundaries,
            quad_order_E=qE,
            quad_order_Ep=qE,
            quad_order_mu=32,
        )
        S = mg.compute_sigma_matrix(kernel_auto, NUM_BINS, T_K, 1.0)
        row = S[1, 1, :]
        total = row.sum()
        bw = 4.0 / NUM_BINS
        pdf = row / (total * bw) if total > 0 else np.zeros(NUM_BINS)
        last_bin_pdf.append(pdf[-1])
        second_last_pdf.append(pdf[-2])
        print(f"  N_E={qE:3d}: PDF[-1]={pdf[-1]:.5f}  PDF[-2]={pdf[-2]:.5f}")

    last_bin_pdf = np.array(last_bin_pdf)
    second_last_pdf = np.array(second_last_pdf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(quad_orders, last_bin_pdf, 'b-o', linewidth=2, markersize=7,
             label='Last bin (mu in [0.9375, 1.0])')
    ax1.plot(quad_orders, second_last_pdf, 'g-s', linewidth=1.5, markersize=5,
             label='Second-to-last bin')
    if cmmc_last_pdf is not None:
        ax1.axhline(cmmc_last_pdf, color='red', linestyle='--', linewidth=2,
                    label=f'CMMC last bin = {cmmc_last_pdf:.4f}')
    ax1.set_xlabel(r'$N_E = N_{E\prime}$ (energy quadrature order)')
    ax1.set_ylabel('PDF')
    ax1.set_title(f'Last-bin PDF convergence vs energy quadrature\n'
                  f'Group 4→4, T={T_KEV} keV, {NUM_BINS} angle bins, N_mu=32')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.35)

    # Relative error vs CMMC
    if cmmc_last_pdf is not None:
        rel_err = (last_bin_pdf - cmmc_last_pdf) / cmmc_last_pdf
        ax2.semilogy(quad_orders, np.abs(rel_err), 'b-o', linewidth=2, markersize=7,
                     label='|Series - CMMC| / CMMC')
        ax2.axhline(0.01, color='red', linestyle='--', alpha=0.7, label='1% error')
        ax2.axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='5% error')
        ax2.set_xlabel(r'$N_E = N_{E\prime}$ (energy quadrature order)')
        ax2.set_ylabel('Relative error vs CMMC')
        ax2.set_title('Convergence rate')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fwd_scatter_step4_eps_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# STEP 5: Full angular PDF validation -- focused on group 4 transitions only
# Use SAME bin count as CMMC (32 bins) to enable fair comparison
# =============================================================================
def step5_validation_plot():
    print("=" * 80)
    print("STEP 5: Angular PDF with corrected energy quadrature")
    print("=" * 80)

    sub_boundaries = BOUNDARIES_ERG[2:6]
    kernel = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    NUM_BINS = 32

    configs = [
        (32, 32, 'N_E=32 (default)'),
        (80, 32, 'N_E=80 (converged)'),
    ]
    results = {}
    for qE, qmu, label in configs:
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=sub_boundaries,
            quad_order_E=qE,
            quad_order_Ep=qE,
            quad_order_mu=qmu,
        )
        print(f"  Computing: {label}...")
        S = mg.compute_sigma_matrix(kernel, NUM_BINS, T_K, 1.0)
        results[label] = S

    mc_pdf = None
    mc_centers = None
    try:
        import _compton_matrix_mc as mc_mod
        NUM_MC_BINS = mc_mod.ComptonMatrixMC.NUM_ANGLE_BINS
        centers_erg = [math.sqrt(BOUNDARIES_ERG[i] * BOUNDARIES_ERG[i + 1])
                       for i in range(len(BOUNDARIES_ERG) - 1)]
        mc_obj = mc_mod.ComptonMatrixMC(
            energy_groups_centers=centers_erg,
            energy_groups_boundaries=BOUNDARIES_ERG,
            num_of_samples=2000000,
            force_detailed_balance=False,
            seed=42)
        mc_obj.set_tables(temperature_grid=[T_K * 0.9, T_K, T_K * 1.1])
        cdf_mc = np.array(mc_obj.get_angle_cdf(temperature=T_K, g0=3, g=3))
        pdf_mc_bins = np.diff(cdf_mc)
        x_max = 4.0
        mc_bin_width = x_max / NUM_MC_BINS
        mc_pdf = pdf_mc_bins / mc_bin_width
        mc_centers = np.linspace(mc_bin_width / 2, x_max - mc_bin_width / 2, NUM_MC_BINS)
        print(f"  CMMC loaded ({NUM_MC_BINS} bins)")
    except Exception as e:
        print(f"  CMMC not available: {e}")

    x_max = 4.0
    bw = x_max / NUM_BINS
    bc = np.linspace(bw / 2, x_max - bw / 2, NUM_BINS)
    g_from, g_to = 1, 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'N_E=32 (default)': 'blue', 'N_E=80 (converged)': 'green'}
    for label, S in results.items():
        row = S[g_from, g_to, :]
        total = row.sum()
        pdf = row / (total * bw) if total > 0 else np.zeros(NUM_BINS)
        ax1.plot(bc, pdf, '-', color=colors[label], linewidth=1.5, label=label)
        print(f"  {label}: last bin PDF = {pdf[-1]:.5f}")

        zoom_start = 3.0
        mask = bc >= zoom_start
        ax2.plot(bc[mask], pdf[mask], '-o', color=colors[label],
                 linewidth=1.5, markersize=5, label=label)

    if mc_pdf is not None:
        ax1.plot(mc_centers, mc_pdf, 'ro', markersize=5, label='CMMC')
        mc_mask = mc_centers >= zoom_start
        ax2.plot(mc_centers[mc_mask], mc_pdf[mc_mask], 'ro', markersize=7, label='CMMC')

    ax1.set_xlabel('Angular bin')
    ax1.set_ylabel('PDF')
    ax1.set_title(f'Group 4\u21924, T={T_KEV} keV, {NUM_BINS} bins\nFull range')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, x_max)

    ax2.set_xlabel('Angular bin')
    ax2.set_ylabel('PDF')
    ax2.set_title(f'Zoom: last bins \u2014 fixed by increasing N_E')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(zoom_start, x_max)

    fig.suptitle(f'Step 5: Energy quadrature fix for last-bin spike, T={T_KEV} keV', y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fwd_scatter_step5_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', nargs='+', type=int, default=[1,2,3,4,5])
    args = parser.parse_args()

    if 1 in args.steps:
        step1_diagnostic_dump()
    if 2 in args.steps:
        step2_method_comparison()
    if 3 in args.steps:
        step3_quadrature_check()
    if 4 in args.steps:
        step4_eps_sweep()
    if 5 in args.steps:
        step5_validation_plot()
    print("\nAll requested steps complete.")
