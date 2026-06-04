"""
Diagnose peak disagreement on the actual Pomraning uniform grid.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CMMC', 'cpp_modules'))

import _compton_multigroup as cm
import _compton_kernel_series as cs
import _compton_matrix_mc as mc_mod
import _units as units

T_KEV = 0.345
T_K = T_KEV * units.kev_kelvin

eb_kev = np.linspace(0.1, 6.0, 120)
eb = eb_kev * units.kev
ec = 0.5 * (eb[1:] + eb[:-1])
ewid = np.diff(eb)
Ng = len(ec)

print(f"Grid: {Ng} groups, dE = {np.diff(eb_kev)[0]*1000:.1f} eV")

# CMMC
print("CMMC (10M)...")
mc_obj = mc_mod.ComptonMatrixMC(
    energy_groups_centers=ec,
    energy_groups_boundaries=eb,
    num_of_samples=int(10e6),
    force_detailed_balance=False,
    seed=42)
S_mc = np.array(mc_obj.calculate_S_matrix(temperature=T_K))

# Series
print("Series (N_E=32)...")
kernel = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
mg = cm.ComptonMultigroupKernel(
    energy_group_boundaries=eb.tolist(),
    weight_function=cm.PlanckWeightFunction(cap_x=25.0),
    quad_order_E=32,
    quad_order_Ep=32,
    quad_order_mu=32)
S_s = np.array(mg.compute_sigma_matrix(kernel, T=T_K, Ne=1.0))

# Find incoming group
E_in_kev = 2.25
g_in = np.argmin(np.abs(ec - E_in_kev * units.kev))
print(f"Incoming: g={g_in}, E={ec[g_in]/units.kev:.4f} keV")

# Normalized differential cross section near peak
dsigma_mc = S_mc[g_in, :] / (ewid / units.kev) / units.sigma_thomson
dsigma_s = S_s[g_in, :] / (ewid / units.kev) / units.sigma_thomson

print(f"\n{'g':>4} {'E_out':>8} {'dσ_series':>12} {'dσ_CMMC':>12} {'ratio':>8} {'diff%':>8}")
for gp in range(max(0, g_in-5), min(Ng, g_in+6)):
    r = dsigma_s[gp] / dsigma_mc[gp] if dsigma_mc[gp] > 0 else float('nan')
    d = (dsigma_s[gp] - dsigma_mc[gp]) / dsigma_mc[gp] * 100 if dsigma_mc[gp] > 0 else float('nan')
    print(f"{gp:>4} {ec[gp]/units.kev:>8.4f} {dsigma_s[gp]:>12.4f} {dsigma_mc[gp]:>12.4f} {r:>8.4f} {d:>+8.2f}%")

print(f"\nPeak Series at g={np.argmax(dsigma_s[g_in-5:g_in+6])+g_in-5}, "
      f"Peak CMMC at g={np.argmax(dsigma_mc[g_in-5:g_in+6])+g_in-5}")
