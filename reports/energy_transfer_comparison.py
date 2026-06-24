"""
Energy-transfer redistribution comparison: deterministic kernel multiplier vs CMMC.

Compares the multigroup matrix computed with the EnergyTransferMultiplier
(Ep-E)/(Ec_gp - Ec_g) against CMMC's use_energy_redistribution mode.

Generates: reports/generated/energy_transfer_comparison.md
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CMMC', 'cpp_modules'))

import _compton_multigroup as cm
import _compton_multigroup_misc as cm_misc
from _compton_kernel_solver import ComptonKernelSolver
import _compton_matrix_mc as mc_mod
import _units as units

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'reports', 'generated')
FIG_DIR = os.path.join(REPORT_DIR, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)

T_KEV = 0.345
T_K = T_KEV * units.kev_kelvin
sigma_T = units.sigma_thomson
NUM_SAMPLES = int(10e6)
NUM_ANGLE_BINS_DET = 32
QUAD_ORDER = 16
QUAD_ORDER_ANGLE = 8

emax = 6.0
eb_kev = np.linspace(0.1, emax, 40)
eb = eb_kev * units.kev
ec = np.sqrt(eb[1:] * eb[:-1])
ewid = np.diff(eb)
Ng = len(ec)
ec_kev = ec / units.kev

report_lines = []

def R(line=""):
    report_lines.append(line)

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return name

# ═════════════════════════════════════════════════════════════════════════
# Computation
# ═════════════════════════════════════════════════════════════════════════

print(f"Energy grid: {Ng} groups, {eb_kev[0]:.2f} to {eb_kev[-1]:.2f} keV")

print(f"Computing CMMC standard ({NUM_SAMPLES/1e6:.0f}M samples)...")
mc_std = mc_mod.ComptonMatrixMC(
    energy_groups_centers=ec.tolist(),
    energy_groups_boundaries=eb.tolist(),
    num_of_samples=NUM_SAMPLES,
    force_detailed_balance=False,
    seed=42,
    use_energy_redistribution=False)
S_mc_std = np.array(mc_std.calculate_S_matrix(temperature=T_K))

print(f"Computing CMMC redistributed ({NUM_SAMPLES/1e6:.0f}M samples)...")
mc_redist = mc_mod.ComptonMatrixMC(
    energy_groups_centers=ec.tolist(),
    energy_groups_boundaries=eb.tolist(),
    num_of_samples=NUM_SAMPLES,
    force_detailed_balance=False,
    seed=42,
    use_energy_redistribution=True)
S_mc_redist = np.array(mc_redist.calculate_S_matrix(temperature=T_K))

print("Computing deterministic (standard)...")
kernel = ComptonKernelSolver()
mg = cm.ComptonMultigroupKernel(
    energy_group_boundaries=eb.tolist(),
    weight_function=cm.PlanckWeightFunction(cap_x=25.0),
    quad_order_E=QUAD_ORDER,
    quad_order_Ep=QUAD_ORDER,
    xi_order=QUAD_ORDER)

S_det_std = np.array(mg.compute_sigma_matrix(kernel, T=T_K, Ne=1.0))

print("Computing deterministic (energy-transfer multiplier)...")
et_mult = cm_misc.EnergyTransferMultiplier(
    energy_group_boundaries=eb.tolist(),
    energy_group_centers=ec.tolist())
S_det_et = np.array(mg.compute_sigma_matrix(kernel, T=T_K, Ne=1.0,
                                             multiplier=et_mult))

print("Computing deterministic multiangle (energy-transfer multiplier)...")
mg_angle = cm.ComptonMultigroupKernel(
    energy_group_boundaries=eb.tolist(),
    weight_function=cm.PlanckWeightFunction(cap_x=25.0),
    quad_order_E=QUAD_ORDER_ANGLE,
    quad_order_Ep=QUAD_ORDER_ANGLE,
    xi_order=QUAD_ORDER_ANGLE)
S_det_et_angle = np.array(mg_angle.compute_sigma_matrix(
    kernel, num_angle_bins=NUM_ANGLE_BINS_DET, T=T_K, Ne=1.0,
    multiplier=et_mult))

S_det_redist = S_det_et.copy()
row_sums_orig = np.sum(S_det_std, axis=1)
for g in range(Ng):
    off_diag_sum = np.sum(S_det_et[g, :]) - S_det_et[g, g]
    S_det_redist[g, g] = row_sums_orig[g] - off_diag_sum

# ═════════════════════════════════════════════════════════════════════════
# Report header
# ═════════════════════════════════════════════════════════════════════════

R("# Energy-Transfer Redistribution Comparison")
R()
R("Comparison of the multigroup Compton scattering matrix computed with the")
R("`EnergyTransferMultiplier` $f(E, E') = (E'-E)/(E_{c,g'} - E_{c,g})$ against")
R("CMMC's `use_energy_redistribution` Monte Carlo mode.")
R()
R(f"- **Temperature:** T = {T_KEV} keV")
R(f"- **Energy grid:** {Ng} uniform groups from {eb_kev[0]:.2f} to {eb_kev[-1]:.2f} keV")
R(f"- **Group centers:** geometric mean $\\sqrt{{E_{{lo}} \\cdot E_{{hi}}}}$")
R(f"- **Quadrature order:** {QUAD_ORDER} (angle-resolved: {QUAD_ORDER_ANGLE})")
R(f"- **CMMC samples:** {NUM_SAMPLES/1e6:.0f}M per run, seed=42")
R(f"- **Weight function:** Planck (cap_x=25)")
R()

# ═════════════════════════════════════════════════════════════════════════
# 1. Heatmap comparison
# ═════════════════════════════════════════════════════════════════════════

R("## 1. Matrix Heatmaps")
R()
R("Side-by-side log-scale heatmaps of the angle-integrated $\\sigma(g{\\to}g')$ matrix.")
R("Left column: standard mode.  Right column: energy-transfer redistributed.")
R("Top row: CMMC.  Bottom row: deterministic (Series kernel).")
R()

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
vmin = max(S_mc_std[S_mc_std > 0].min(), S_det_std[S_det_std > 0].min()) * 0.1
vmax = max(S_mc_std.max(), S_det_std.max())

for ax, data, title in [
    (axes[0, 0], S_mc_std,      "CMMC standard"),
    (axes[0, 1], S_mc_redist,   "CMMC redistributed"),
    (axes[1, 0], S_det_std,     "Det. standard"),
    (axes[1, 1], S_det_redist,  "Det. energy-transfer"),
]:
    masked = np.where(data > 0, data, vmin * 0.01)
    im = ax.pcolormesh(eb_kev, eb_kev, np.log10(masked), vmin=np.log10(vmin), vmax=np.log10(vmax),
                       cmap='viridis', shading='flat')
    ax.set_xlabel("$E'$ [keV]")
    ax.set_ylabel("$E$ [keV]")
    ax.set_title(title)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label="$\\log_{10}(\\sigma)$ [cm$^2$]")

fig.suptitle(f"Scattering matrix heatmaps, T={T_KEV} keV", y=1.01)
fig.tight_layout()
fname = save_fig(fig, "et_heatmaps.png")
R(f"![Heatmaps](figs/{fname})")
R()

# ═════════════════════════════════════════════════════════════════════════
# 2. Element-wise relative differences
# ═════════════════════════════════════════════════════════════════════════

R("## 2. Element-Wise Relative Differences (Det vs CMMC)")
R()
R("Relative difference $|\\sigma_\\mathrm{det} - \\sigma_\\mathrm{CMMC}| / \\sigma_\\mathrm{CMMC}$")
R("for matrix entries where $\\sigma_\\mathrm{CMMC} > 10^{-6} \\sigma_T$.")
R()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
thresh = 1e-6 * sigma_T

for ax, S_det, S_mc, title_tag in [
    (axes[0], S_det_std, S_mc_std, "Standard"),
    (axes[1], S_det_redist, S_mc_redist, "Redistributed"),
]:
    mask = S_mc > thresh
    reldiff = np.where(mask, np.abs(S_det - S_mc) / np.abs(S_mc), np.nan)
    im = ax.pcolormesh(eb_kev, eb_kev, reldiff, vmin=0, vmax=0.1,
                       cmap='hot_r', shading='flat')
    ax.set_xlabel("$E'$ [keV]")
    ax.set_ylabel("$E$ [keV]")
    ax.set_title(title_tag)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label="Relative difference")

fig.suptitle(f"Element-wise |Det - CMMC| / CMMC, T={T_KEV} keV", y=1.01)
fig.tight_layout()
fname = save_fig(fig, "et_reldiff.png")
R(f"![Relative differences](figs/{fname})")
R()

for label, S_det, S_mc in [
    ("Standard", S_det_std, S_mc_std),
    ("Redistributed", S_det_redist, S_mc_redist),
]:
    mask = S_mc > thresh
    rd = np.abs(S_det[mask] - S_mc[mask]) / np.abs(S_mc[mask])
    R(f"**{label}:** median rel diff = {np.median(rd):.2e}, "
      f"90th percentile = {np.percentile(rd, 90):.2e}, "
      f"max = {np.max(rd):.2e}")
R()

# ═════════════════════════════════════════════════════════════════════════
# 3. Row sums (total cross sections)
# ═════════════════════════════════════════════════════════════════════════

R("## 3. Total Cross Sections (Row Sums)")
R()
R("The row sum $\\sum_{g'} \\sigma(g{\\to}g')$ gives the total scattering cross section")
R("out of each group.  Energy redistribution preserves row sums by construction.")
R()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

rs_mc_std = S_mc_std.sum(axis=1) / sigma_T
rs_mc_redist = S_mc_redist.sum(axis=1) / sigma_T
rs_det_std = S_det_std.sum(axis=1) / sigma_T
rs_det_redist = S_det_redist.sum(axis=1) / sigma_T

ax = axes[0]
ax.stairs(rs_mc_std, edges=eb_kev, color='red', linewidth=2.0, label='CMMC std')
ax.stairs(rs_mc_redist, edges=eb_kev, color='orange', linewidth=1.5, linestyle=':', label='CMMC redist')
ax.stairs(rs_det_std, edges=eb_kev, color='blue', linewidth=1.0, linestyle='--', label='Det. std')
ax.stairs(rs_det_redist, edges=eb_kev, color='cyan', linewidth=1.0, linestyle='-.', label='Det. redist')
ax.set_xlabel("$E$ [keV]")
ax.set_ylabel(r"$\sum_{g'}\sigma(g{\to}g') / \sigma_T$")
ax.set_title("Row sums")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
mask_rs = rs_mc_std > 1e-8
ratio_rs_std = np.where(mask_rs, rs_det_std / rs_mc_std, np.nan)
ratio_rs_redist = np.where(mask_rs, rs_det_redist / rs_mc_redist, np.nan)
ax.stairs(ratio_rs_std, edges=eb_kev, color='blue', linewidth=1.2, label='Standard')
ax.stairs(ratio_rs_redist, edges=eb_kev, color='green', linewidth=1.2, linestyle='--', label='Redistributed')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel("$E$ [keV]")
ax.set_ylabel("Det / CMMC")
ax.set_title("Row-sum ratios")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.95, 1.05)

fig.suptitle(f"Total cross sections, T={T_KEV} keV", y=1.01)
fig.tight_layout()
fname = save_fig(fig, "et_row_sums.png")
R(f"![Row sums](figs/{fname})")
R()

R("| Mode | Max row-sum rel diff (Det/CMMC) |")
R("|------|--------------------------------|")
mask_rs_mc_std = S_mc_std.sum(axis=1) > thresh
rs_rd_std = np.abs(S_det_std.sum(axis=1)[mask_rs_mc_std] - S_mc_std.sum(axis=1)[mask_rs_mc_std]) / S_mc_std.sum(axis=1)[mask_rs_mc_std]
rs_rd_redist = np.abs(S_det_redist.sum(axis=1)[mask_rs_mc_std] - S_mc_redist.sum(axis=1)[mask_rs_mc_std]) / S_mc_redist.sum(axis=1)[mask_rs_mc_std]
R(f"| Standard | {rs_rd_std.max():.2e} |")
R(f"| Redistributed | {rs_rd_redist.max():.2e} |")
R()

row_sum_conservation = np.abs(S_det_redist.sum(axis=1) - S_det_std.sum(axis=1))
R(f"**Row-sum conservation check (deterministic):** "
  f"max |row_sum(redist) - row_sum(orig)| = {row_sum_conservation.max():.2e}")
R()

# ═════════════════════════════════════════════════════════════════════════
# 4. Pomraning-style differential cross sections
# ═════════════════════════════════════════════════════════════════════════

R("## 4. Pomraning-Style Differential Cross Sections")
R()
R("Differential scattering cross section $\\sigma(E{\\to}E')/(\\sigma_T \\cdot \\Delta E')$")
R("for selected incoming groups, comparing standard and redistributed matrices.")
R()

E_in_targets_kev = [1.5, 2.25, 3.5]
for E_in_kev in E_in_targets_kev:
    g_in = int(np.argmin(np.abs(ec - E_in_kev * units.kev)))

    dsigma_mc_std    = S_mc_std[g_in, :]    / (ewid / units.kev) / sigma_T
    dsigma_mc_redist = S_mc_redist[g_in, :] / (ewid / units.kev) / sigma_T
    dsigma_det_std   = S_det_std[g_in, :]   / (ewid / units.kev) / sigma_T
    dsigma_det_et    = S_det_redist[g_in, :] / (ewid / units.kev) / sigma_T

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.stairs(dsigma_mc_std, edges=eb_kev, color='red', linewidth=2.0, label='CMMC standard')
    ax.stairs(dsigma_det_std, edges=eb_kev, color='blue', linewidth=1.0, linestyle='--', label='Det. standard')
    ax.set_yscale('log')
    ax.set_xlabel(r"$E'$ [keV]")
    ax.set_ylabel(r"$d\sigma / dE'$ [$\sigma_T$ / keV]")
    ax.set_title("Standard")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ymax = max(dsigma_det_std.max(), dsigma_mc_std.max(), 1e-3)
    ax.set_ylim(ymax * 1e-5, ymax * 3)

    ax = axes[0, 1]
    ax.stairs(dsigma_mc_redist, edges=eb_kev, color='red', linewidth=2.0, label='CMMC redistributed')
    ax.stairs(dsigma_det_et, edges=eb_kev, color='blue', linewidth=1.0, linestyle='--', label='Det. energy-transfer')
    ax.set_yscale('log')
    ax.set_xlabel(r"$E'$ [keV]")
    ax.set_ylabel(r"$d\sigma / dE'$ [$\sigma_T$ / keV]")
    ax.set_title("Redistributed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ymax = max(dsigma_det_et.max(), dsigma_mc_redist.max(), 1e-3)
    ax.set_ylim(ymax * 1e-5, ymax * 3)

    ax = axes[1, 0]
    mask = dsigma_mc_std > 1e-6
    ratio = np.where(mask, dsigma_det_std / dsigma_mc_std, np.nan)
    ax.stairs(ratio, edges=eb_kev, color='black', linewidth=1.2)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r"$E'$ [keV]")
    ax.set_ylabel("Det / CMMC")
    ax.set_title("Ratio: standard")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.1)

    ax = axes[1, 1]
    mask = dsigma_mc_redist > 1e-6
    ratio = np.where(mask, dsigma_det_et / dsigma_mc_redist, np.nan)
    ax.stairs(ratio, edges=eb_kev, color='black', linewidth=1.2)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r"$E'$ [keV]")
    ax.set_ylabel("Det / CMMC")
    ax.set_title("Ratio: redistributed")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.1)

    fig.suptitle(
        f"$E_{{in}}$ = {ec_kev[g_in]:.2f} keV (group {g_in}), T = {T_KEV} keV",
        y=1.01)
    fig.tight_layout()
    figname = f"et_pomraning_g{g_in}.png"
    save_fig(fig, figname)

    R(f"### $E_{{in}}$ = {ec_kev[g_in]:.2f} keV (group {g_in})")
    R()
    R(f"![Pomraning g={g_in}](figs/{figname})")
    R()

# ═════════════════════════════════════════════════════════════════════════
# 5. Effect of redistribution on the matrix
# ═════════════════════════════════════════════════════════════════════════

R("## 5. Effect of Redistribution on Matrix Structure")
R()
R("Ratio of the redistributed matrix to the standard matrix, element-wise,")
R("showing how the energy-transfer multiplier reshapes the scattering kernel.")
R()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, S_redist, S_orig, title_tag in [
    (axes[0], S_mc_redist, S_mc_std, "CMMC"),
    (axes[1], S_det_redist, S_det_std, "Deterministic"),
]:
    mask = S_orig > thresh
    ratio_mat = np.where(mask, S_redist / S_orig, np.nan)
    im = ax.pcolormesh(eb_kev, eb_kev, ratio_mat, vmin=0.0, vmax=2.0,
                       cmap='RdBu_r', shading='flat')
    ax.set_xlabel("$E'$ [keV]")
    ax.set_ylabel("$E$ [keV]")
    ax.set_title(f"{title_tag}: redist / standard")
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label="Ratio")

fig.suptitle(f"Redistribution ratio, T={T_KEV} keV", y=1.01)
fig.tight_layout()
fname = save_fig(fig, "et_redist_ratio.png")
R(f"![Redistribution ratio](figs/{fname})")
R()
R("Values < 1 (blue) indicate reduced off-diagonal transfer; > 1 (red) indicate enhanced transfer.")
R("The diagonal absorbs the difference to preserve row sums.")
R()

# ═════════════════════════════════════════════════════════════════════════
# 6. Angle PDF comparison
# ═════════════════════════════════════════════════════════════════════════

dxi = 2.0 / NUM_ANGLE_BINS_DET
xi_edges = np.linspace(-1.0, 1.0, NUM_ANGLE_BINS_DET + 1)

def angle_pdf_section(T_kev, section_num, E_in_kev_angle=2.25, g_out_offsets=None):
    if g_out_offsets is None:
        g_out_offsets = [2, 4, 8]

    T_K_local = T_kev * units.kev_kelvin
    g_in = int(np.argmin(np.abs(ec - E_in_kev_angle * units.kev)))

    R(f"## {section_num}. Angular Distribution (PDF) Comparison — T = {T_kev} keV")
    R()
    R("Scattering angle probability density for selected $(g, g')$ pairs,")
    R("comparing the deterministic multiangle matrix (with energy-transfer multiplier)")
    R("against CMMC's redistributed angular histograms.")
    R()

    print(f"Computing CMMC angle tables (T={T_kev} keV)...")
    mc_angle = mc_mod.ComptonMatrixMC(
        energy_groups_centers=ec.tolist(),
        energy_groups_boundaries=eb.tolist(),
        num_of_samples=NUM_SAMPLES,
        force_detailed_balance=False,
        seed=42,
        use_energy_redistribution=True)
    mc_angle.set_tables(temperature_grid=[T_K_local * 0.99, T_K_local * 1.01])

    print(f"Computing deterministic multiangle (T={T_kev} keV)...")
    S_det_angle_local = np.array(mg_angle.compute_sigma_matrix(
        kernel, num_angle_bins=NUM_ANGLE_BINS_DET, T=T_K_local, Ne=1.0,
        multiplier=et_mult))

    for g_off in g_out_offsets:
        g_out = min(g_in + g_off, Ng - 1)
        if g_out == g_in:
            continue

        cdf_mc = np.array(mc_angle.get_angle_cdf(temperature=T_K_local, g0=g_in, g=g_out))
        pdf_mc = np.diff(cdf_mc)

        angle_row = S_det_angle_local[g_in, g_out, :]
        total_det = angle_row.sum()
        pdf_det = angle_row / total_det if total_det > 0 else np.zeros_like(angle_row)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.stairs(pdf_mc / dxi, edges=xi_edges, color='red', linewidth=2.0, label='CMMC redistributed')
        ax1.stairs(pdf_det / dxi, edges=xi_edges, color='blue', linewidth=1.0, linestyle='--', label='Det. energy-transfer')
        ax1.set_xlabel(r"$\xi = \cos\theta$")
        ax1.set_ylabel(r"PDF density [1/unit $\xi$]")
        ax1.set_title(
            f"g={g_in} ({ec_kev[g_in]:.2f} keV) "
            f"$\\to$ g'={g_out} ({ec_kev[g_out]:.2f} keV)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        mask_pdf = pdf_mc > 1e-8
        ratio_pdf = np.where(mask_pdf, pdf_det / pdf_mc, np.nan)
        ax2.stairs(ratio_pdf, edges=xi_edges, color='black', linewidth=1.2)
        ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_xlabel(r"$\xi = \cos\theta$")
        ax2.set_ylabel("Det / CMMC")
        ax2.set_title("PDF ratio")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.7, 1.3)

        fig.suptitle(f"Angular PDF, T={T_kev} keV", y=1.01)
        fig.tight_layout()
        tag = f"T{T_kev}".replace(".", "p")
        figname = f"et_angle_pdf_{tag}_g{g_in}_to_g{g_out}.png"
        save_fig(fig, figname)

        R(f"### g={g_in} ({ec_kev[g_in]:.2f} keV) $\\to$ g'={g_out} ({ec_kev[g_out]:.2f} keV)")
        R()
        R(f"![Angle PDF g{g_in} to g{g_out}](figs/{figname})")
        R()

angle_pdf_section(T_KEV, section_num=6)
angle_pdf_section(10.0, section_num=7)

# ═════════════════════════════════════════════════════════════════════════
# 8. Summary statistics
# ═════════════════════════════════════════════════════════════════════════

R("## 8. Summary")
R()
R("| Metric | Standard | Redistributed |")
R("|--------|----------|---------------|")

mask_all = S_mc_std > thresh
rd_std = np.abs(S_det_std[mask_all] - S_mc_std[mask_all]) / np.abs(S_mc_std[mask_all])
mask_all_r = S_mc_redist > thresh
rd_redist = np.abs(S_det_redist[mask_all_r] - S_mc_redist[mask_all_r]) / np.abs(S_mc_redist[mask_all_r])

R(f"| Median element rel diff | {np.median(rd_std):.2e} | {np.median(rd_redist):.2e} |")
R(f"| 90th pctile element rel diff | {np.percentile(rd_std, 90):.2e} | {np.percentile(rd_redist, 90):.2e} |")
R(f"| Max element rel diff | {np.max(rd_std):.2e} | {np.max(rd_redist):.2e} |")
R(f"| Max row-sum rel diff | {rs_rd_std.max():.2e} | {rs_rd_redist.max():.2e} |")
R(f"| Row-sum conservation (det) | - | {row_sum_conservation.max():.2e} |")
R()

# ═════════════════════════════════════════════════════════════════════════
# Write report
# ═════════════════════════════════════════════════════════════════════════

report_path = os.path.join(REPORT_DIR, "energy_transfer_comparison.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"\nReport written to: {report_path}")
print("Done.")
