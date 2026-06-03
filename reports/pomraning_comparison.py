"""
Pomraning-style plot: differential cross section vs outgoing photon energy.
Compares Series kernel to CMMC for group 4 incoming at T=0.345 keV.
"""
import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CMMC', 'cpp_modules'))

import _compton_multigroup as cm
import _compton_kernel_series as cs
import _compton_matrix_mc as mc_mod
import _units as units

T_KEV = 0.345
T_K = T_KEV * units.kev_kelvin
sigma_T = units.sigma_thomson

emax = 6.0
eb_kev = np.sort(np.unique(np.concatenate([
    np.linspace(0.1, emax, 80),
    np.linspace(1.5, 3.5, 40),
])))
eb = eb_kev * units.kev
ec = 0.5 * (eb[1:] + eb[:-1])
ewid = np.diff(eb)
Ng = len(ec)
print(f"Energy grid: {Ng} groups, {eb_kev[0]:.2f} to {eb_kev[-1]:.2f} keV, min dE={np.diff(eb_kev).min()*1000:.1f} eV, max dE={np.diff(eb_kev).max()*1000:.1f} eV")

# --- CMMC ---
print("Computing CMMC (10M samples)...")
mc_obj = mc_mod.ComptonMatrixMC(
    energy_groups_centers=ec,
    energy_groups_boundaries=eb,
    num_of_samples=int(10e6),
    force_detailed_balance=False,
    seed=42)
S_mc = np.array(mc_obj.calculate_S_matrix(temperature=T_K))
print(f"  CMMC done. Shape: {S_mc.shape}")

# --- Series kernel (angle-integrated) ---
print("Computing Series kernel (angle-integrated)...")
kernel = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
mg = cm.ComptonMultigroupKernel(
    energy_group_boundaries=eb.tolist(),
    quad_order_E=32,
    quad_order_Ep=32,
    quad_order_mu=32)
S_series = mg.compute_sigma_matrix(kernel, T=T_K, Ne=1.0)
S_series = np.array(S_series)
print(f"  Series done. Shape: {S_series.shape}")

# Find group closest to 2.25 keV
E_in_kev = 2.25
g_in = np.argmin(np.abs(ec - E_in_kev * units.kev))
print(f"  Incoming group: g={g_in}, E_center={ec[g_in]/units.kev:.3f} keV")

dsigma_mc = S_mc[g_in, :] / (ewid / units.kev) / sigma_T
dsigma_series = S_series[g_in, :] / (ewid / units.kev) / sigma_T

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax1, ax2, ax3 = axes

# Left: log scale
ax1.stairs(dsigma_mc, edges=eb_kev, color='red', linewidth=2.0, label='CMMC')
ax1.stairs(dsigma_series, edges=eb_kev, color='blue', linewidth=1.0, linestyle='--', label='Series')
ax1.set_yscale('log')
ax1.set_xlabel(r"Outgoing photon energy $E'$ [keV]")
ax1.set_ylabel(r"$\sigma(E \to E') / (\sigma_T \cdot \Delta E')$ [1/keV]")
ax1.set_title(f"T = {T_KEV} keV, $E_{{in}}$ = {ec[g_in]/units.kev:.2f} keV")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1.0, 4.0)
ax1.set_ylim(1e-4, dsigma_series.max() * 3)

# Middle: linear scale near peak
ax2.stairs(dsigma_mc, edges=eb_kev, color='red', linewidth=2.0, label='CMMC')
ax2.stairs(dsigma_series, edges=eb_kev, color='blue', linewidth=1.0, linestyle='--', label='Series')
ax2.set_xlabel(r"Outgoing photon energy $E'$ [keV]")
ax2.set_ylabel(r"$\sigma(E \to E') / (\sigma_T \cdot \Delta E')$ [1/keV]")
ax2.set_title("Linear scale (peak region)")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1.5, 3.5)
ax2.set_ylim(bottom=0)

# Right: ratio plot
mask = dsigma_mc > 0
ratio = np.where(mask, dsigma_series / dsigma_mc, np.nan)
ax3.stairs(ratio, edges=eb_kev, color='black', linewidth=1.2)
ax3.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax3.set_xlabel(r"Outgoing photon energy $E'$ [keV]")
ax3.set_ylabel("Series / CMMC")
ax3.set_title("Ratio (Series / CMMC)")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(1.5, 3.5)
ax3.set_ylim(0.99, 1.01)

fig.suptitle(r"Compton $\sigma / \sigma_T$ — Pomraning-style, T=0.345 keV (CMMC $E^3$ weight)", y=1.02)
fig.tight_layout()
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'reports', 'generated', 'figs', 'pomraning_T0.345kev_g4_E3weight.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")
