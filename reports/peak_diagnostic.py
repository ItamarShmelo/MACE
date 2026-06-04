"""
Diagnose peak disagreement between Series and CMMC in Pomraning plot.
Compares S matrix values near the elastic peak for group g_in.
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

# Use a uniform grid near the peak to eliminate bin-width effects
E_center = 2.25 * units.kev
half_range = 0.5 * units.kev
N_groups = 20
eb = np.linspace(E_center - half_range, E_center + half_range, N_groups + 1)
ec = 0.5 * (eb[1:] + eb[:-1])
ewid = np.diff(eb)

print(f"Uniform grid: {N_groups} groups, {eb[0]/units.kev:.3f} to {eb[-1]/units.kev:.3f} keV")
print(f"Bin width: {ewid[0]/units.kev*1000:.1f} eV")
print()

# CMMC with 10M samples
print("CMMC (10M samples, E^3 weight)...")
mc_obj = mc_mod.ComptonMatrixMC(
    energy_groups_centers=ec,
    energy_groups_boundaries=eb,
    num_of_samples=int(10e6),
    force_detailed_balance=False,
    seed=42)
S_mc = np.array(mc_obj.calculate_S_matrix(temperature=T_K))

# Series at different quadrature orders
g_in = N_groups // 2  # middle group
print(f"\nIncoming group: g={g_in}, E={ec[g_in]/units.kev:.4f} keV")
print()

for NE in [16, 32, 64]:
    kernel = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=eb.tolist(),
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=NE,
        quad_order_Ep=NE,
        quad_order_mu=32)
    S_s = np.array(mg.compute_sigma_matrix(kernel, T=T_K, Ne=1.0))
    
    # Compare near diagonal
    print(f"--- N_E = {NE} ---")
    print(f"{'g_out':>5} {'E_out(keV)':>10} {'S_series':>12} {'S_cmmc':>12} {'ratio':>8}")
    for gp in range(max(0, g_in-3), min(N_groups, g_in+4)):
        ratio = S_s[g_in, gp] / S_mc[g_in, gp] if S_mc[g_in, gp] > 0 else float('nan')
        print(f"{gp:>5} {ec[gp]/units.kev:>10.4f} {S_s[g_in,gp]:>12.6e} {S_mc[g_in,gp]:>12.6e} {ratio:>8.4f}")
    print()
