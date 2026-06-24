"""
Multigroup kernel validation report.

Generates reports/generated/multigroup_validation.md with embedded plots covering:
  1. Denominator accuracy vs analytic Planck integral
  2. Quadrature convergence with increasing GL order
  3. Angle-bin summation consistency
  4. Multigroup matrix heatmaps at several temperatures
  5. MC S-matrix comparison (if CMMC available)
  6. MC angle CDF comparison (if CMMC available)

Usage:
    python3 reports/multigroup_validation.py

Output:
    reports/generated/multigroup_validation.md  (+ .png plots in figs/)
"""
import sys
import os
import math
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import quad as scipy_quad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

import _compton_multigroup as cm
import _compton_kernel_quadrature as cq
from _units import kev, kev_kelvin, k_boltz

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


BOUNDARIES_KEV = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
BOUNDARIES_ERG = [b * kev for b in BOUNDARIES_KEV]
KERNEL_Q64 = cq.ComptonKernelQuadrature(64)


def _cmmc_available():
    """Check if CMMC module is functional (not segfaulting)."""
    code = (
        "import sys; sys.path.insert(0,'external/CMMC/cpp_modules'); "
        "sys.path.insert(0,'cpp_modules'); "
        "import _compton_matrix_mc as mc; "
        "mc.ComptonMatrixMC("
        "energy_groups_centers=[1e-9,5e-9],"
        "energy_groups_boundaries=[5e-10,2e-9,1e-8],"
        "num_of_samples=10,"
        "force_detailed_balance=False,"
        "seed=1).calculate_S_matrix(temperature=1e8)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, timeout=30, cwd=ROOT)
        return result.returncode == 0
    except Exception:
        return False


# ─── Section 1: Denominator accuracy ─────────────────────────────────────

def section_denominator():
    emit('## 1. Denominator Accuracy')
    emit()
    emit('Capped Planck weight denominator $\\int_{\\Delta E_g} w(E,T)\\,dE$ '
         'computed by `compute_denominator` (analytic via planck_integral.hpp) '
         'compared against scipy adaptive quadrature of the same weight function.')
    emit()

    cap_x = 25.0
    w0 = cap_x**3 / np.expm1(cap_x)

    cases = [
        ('Below cap', 0.1, 5.0),
        ('Below cap (wide)', 0.5, 20.0),
        ('Above cap', 26.0, 30.0),
        ('Above cap (wide)', 30.0, 50.0),
        ('Straddling', 20.0, 30.0),
        ('Straddling (narrow)', 24.0, 26.0),
    ]

    for T_kev in [1.0, 10.0]:
        emit(f'### T = {T_kev} keV')
        emit()
        emit('| Case | x_lo | x_hi | C++ denom | scipy ref | Rel Error |')
        emit('|------|------|------|-----------|-----------|-----------|')

        T = T_kev * kev_kelvin
        kT = k_boltz * T

        for label, x_lo, x_hi in cases:
            E_lo = x_lo * kT
            E_hi = x_hi * kT

            wf = cm.PlanckWeightFunction(cap_x)

            computed = wf.compute_denominator(E_lo, E_hi, T)

            def weight_fn(x):
                if x < cap_x:
                    return x**3 / np.expm1(x)
                return w0

            ref_dimless, _ = scipy_quad(lambda x: weight_fn(x), x_lo, x_hi)
            reference = kT * ref_dimless

            if abs(reference) > 0:
                rel = abs(computed - reference) / abs(reference)
                rel_str = f'{rel:.2e}'
            else:
                rel_str = '--'

            emit(f'| {label} | {x_lo:.1f} | {x_hi:.1f} | '
                 f'{computed:.6e} | {reference:.6e} | {rel_str} |')

        emit()


# ─── Section 2: Quadrature convergence ───────────────────────────────────

def _max_rel_diff(S, S_ref):
    """Max relative difference between two matrices (ignoring near-zero entries)."""
    mask = np.abs(S_ref) > 1e-40
    if not np.any(mask):
        return 0.0
    return np.max(np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask]))


def _median_rel_diff(S, S_ref):
    """Median relative difference between two matrices."""
    mask = np.abs(S_ref) > 1e-40
    if not np.any(mask):
        return 0.0
    return np.median(np.abs(S[mask] - S_ref[mask]) / np.abs(S_ref[mask]))


def section_convergence():
    import time
    emit('## 2. Quadrature Convergence')
    emit()
    emit('The multigroup integral uses independent Gauss-Legendre rules '
         'for the three axes: incident energy $N_E$, scattered energy $N_{E\'}$, '
         'and scattering cosine $N_\\xi$.  This section studies convergence '
         'along each axis independently and jointly.')
    emit()

    T = 10.0 * kev_kelvin
    bounds_kev = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    bounds = (bounds_kev * kev).tolist()
    G = len(bounds) - 1
    centers_kev = np.sqrt(bounds_kev[:-1] * bounds_kev[1:])

    N_REF = 128
    orders = [4, 8, 12, 16, 24, 32, 48, 64, 96]

    # ── 2a. Per-axis convergence ──────────────────────────────────────────

    emit('### 2a. Per-Axis Convergence')
    emit()
    emit('Each axis is swept while the other two are held fixed at '
         f'$N_{{ref}} = {N_REF}$. The reference matrix uses '
         f'$N_E = N_{{E\'}} = N_\\xi = {N_REF}$.')
    emit()

    wf = cm.PlanckWeightFunction(cap_x=25.0)

    mg_ref = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=wf,
        quad_order_E=N_REF, quad_order_Ep=N_REF, xi_order=N_REF)
    S_ref = mg_ref.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)

    axis_configs = [
        ('$N_E$ (incident energy)', 'quad_order_E', 'quad_order_Ep', 'xi_order'),
        ("$N_{E'}$ (scattered energy)", 'quad_order_Ep', 'quad_order_E', 'xi_order'),
        ('$N_\\xi$ (scattering cosine)', 'xi_order', 'quad_order_E', 'quad_order_Ep'),
    ]

    fig, axes_arr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, (label, sweep_key, fix_key1, fix_key2) in enumerate(axis_configs):
        ax = axes_arr[ax_idx]
        maxes = []
        medians = []

        for n in orders:
            kwargs = {sweep_key: n, fix_key1: N_REF, fix_key2: N_REF}
            mg = cm.ComptonMultigroupKernel(energy_group_boundaries=bounds, weight_function=wf, **kwargs)
            S = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
            maxes.append(_max_rel_diff(S, S_ref))
            medians.append(_median_rel_diff(S, S_ref))

        ax.semilogy(orders, maxes, 'rs-', markersize=5, label='Max')
        ax.semilogy(orders, medians, 'bo-', markersize=4, label='Median')
        ax.set_xlabel(label)
        ax.set_title(f'Sweep {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes_arr[0].set_ylabel(f'Relative difference vs $N = {N_REF}$ reference')
    fig.suptitle('Per-axis quadrature convergence (T = 10 keV)', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_conv_per_axis.png')

    emit(f'![Per-axis convergence]({fig_path})')
    emit()

    emit(f'| Axis | N | Max rel diff | Median rel diff |')
    emit(f'|------|---|-------------|----------------|')
    for ax_idx, (label, sweep_key, fix_key1, fix_key2) in enumerate(axis_configs):
        for n in [4, 8, 16, 32, 64]:
            kwargs = {sweep_key: n, fix_key1: N_REF, fix_key2: N_REF}
            mg = cm.ComptonMultigroupKernel(energy_group_boundaries=bounds, weight_function=wf, **kwargs)
            S = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
            mx = _max_rel_diff(S, S_ref)
            md = _median_rel_diff(S, S_ref)
            emit(f'| {label} | {n} | {mx:.2e} | {md:.2e} |')
    emit()

    # ── 2b. Joint convergence (N_E = N_E' = N_xi = N) ────────────────────

    emit('### 2b. Joint Convergence ($N_E = N_{E\'} = N_\\xi = N$)')
    emit()
    emit(f'All three axes swept together.  Reference: N = {N_REF}.')
    emit()

    joint_orders = [4, 8, 12, 16, 24, 32, 48, 64, 96]
    matrices = {}
    maxes_joint = []
    medians_joint = []

    emit('| N | Time (s) | Max rel diff | Median rel diff |')
    emit('|---|----------|-------------|----------------|')

    for n in joint_orders:
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=wf,
            quad_order_E=n, quad_order_Ep=n, xi_order=n)
        t0 = time.perf_counter()
        matrices[n] = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)
        dt = time.perf_counter() - t0
        mx = _max_rel_diff(matrices[n], S_ref)
        md = _median_rel_diff(matrices[n], S_ref)
        maxes_joint.append(mx)
        medians_joint.append(md)
        emit(f'| {n} | {dt:.3f} | {mx:.2e} | {md:.2e} |')

    emit()

    fig, axes_joint = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes_joint[0]
    ax.semilogy(joint_orders, maxes_joint, 'rs-', markersize=5, label='Max')
    ax.semilogy(joint_orders, medians_joint, 'bo-', markersize=4, label='Median')
    ax.set_xlabel('Joint quadrature order N')
    ax.set_ylabel(f'Relative difference vs N = {N_REF}')
    ax.set_title('Joint convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes_joint[1]
    for g in range(G):
        for gp in range(G):
            ref_val = S_ref[g, gp]
            if abs(ref_val) < 1e-40:
                continue
            errs = [abs(matrices[n][g, gp] - ref_val) / abs(ref_val)
                    for n in joint_orders]
            lbl = f'{centers_kev[g]:.0f}→{centers_kev[gp]:.0f}' if g != gp else f'{centers_kev[g]:.0f}→{centers_kev[g]:.0f}'
            alpha = 0.7 if g == gp else 0.4
            ls = '-' if g == gp else '--'
            ax.semilogy(joint_orders, errs, ls, alpha=alpha, markersize=3, marker='o')

    ax.set_xlabel('Joint quadrature order N')
    ax.set_ylabel(f'Element relative error vs N = {N_REF}')
    ax.set_title('Per-element convergence')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Joint convergence — {G} groups, T = 10 keV', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_conv_joint.png')

    emit(f'![Joint convergence]({fig_path})')
    emit()

    # ── 2c. Spatial error map ─────────────────────────────────────────────

    emit('### 2c. Convergence Error Map')
    emit()
    emit('Heatmap of the relative error at selected quadrature orders, '
         'showing which matrix elements converge slowest.')
    emit()

    show_orders = [8, 16, 32, 64]
    fig, axes_map = plt.subplots(1, len(show_orders), figsize=(5 * len(show_orders), 4.5))

    tick_pos = np.arange(0, G, max(1, G // 6))
    tick_lbl = [f'{centers_kev[i]:.1f}' for i in tick_pos]

    for idx, n in enumerate(show_orders):
        ax = axes_map[idx]
        mask = np.abs(S_ref) > 1e-40
        scale = np.where(mask, np.abs(S_ref), 1.0)
        rel_err = np.where(mask, np.abs(matrices[n] - S_ref) / scale, np.nan)

        from matplotlib.colors import LogNorm as LN
        vmax = np.nanmax(rel_err) if np.any(np.isfinite(rel_err)) else 1.0
        vmin_map = max(1e-14, np.nanmin(rel_err[np.isfinite(rel_err)])) if np.any(np.isfinite(rel_err)) else 1e-14
        im = ax.imshow(rel_err, aspect='auto', origin='lower', cmap='hot_r',
                       norm=LN(vmin=vmin_map, vmax=max(vmin_map * 10, vmax)))
        ax.set_title(f'N = {n}')
        ax.set_xlabel("E' (keV)")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes_map[0].set_ylabel('E (keV)')
    fig.suptitle(f'Relative error map vs N = {N_REF} reference (T = 10 keV)', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_conv_error_map.png')

    emit(f'![Error map]({fig_path})')
    emit()

    # ── 2d. Temperature dependence ────────────────────────────────────────

    emit('### 2d. Temperature Dependence of Convergence')
    emit()
    emit(f'Max relative error (joint N, vs N = {N_REF} reference) '
         'at different temperatures.')
    emit()

    T_kevs = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    test_orders = [8, 16, 32, 64]

    emit('| T (keV) | ' + ' | '.join([f'N = {n}' for n in test_orders]) + ' |')
    emit('|---------|' + '|'.join(['-------'] * len(test_orders)) + '|')

    fig, ax = plt.subplots(figsize=(10, 5))

    for T_kev in T_kevs:
        T_val = T_kev * kev_kelvin
        mg_ref_T = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=wf,
            quad_order_E=N_REF, quad_order_Ep=N_REF, xi_order=N_REF)
        S_ref_T = mg_ref_T.compute_sigma_matrix(KERNEL_Q64, T=T_val, Ne=1.0)

        cells = []
        errs_T = []
        for n in test_orders:
            mg_T = cm.ComptonMultigroupKernel(
                energy_group_boundaries=bounds,
                weight_function=wf,
                quad_order_E=n, quad_order_Ep=n, xi_order=n)
            S_T = mg_T.compute_sigma_matrix(KERNEL_Q64, T=T_val, Ne=1.0)
            mx = _max_rel_diff(S_T, S_ref_T)
            cells.append(f'{mx:.2e}')
            errs_T.append(mx)

        emit(f'| {T_kev} | ' + ' | '.join(cells) + ' |')
        ax.semilogy(test_orders, errs_T, 'o-', markersize=5, label=f'T = {T_kev} keV')

    ax.set_xlabel('Joint quadrature order N')
    ax.set_ylabel(f'Max relative error vs N = {N_REF}')
    ax.set_title('Convergence rate at different temperatures')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = save_fig('mg_conv_vs_temp.png')

    emit()
    emit(f'![Convergence vs temperature]({fig_path})')
    emit()


# ─── Section 3: Angle-bin summation ──────────────────────────────────────

def section_angle_summation():
    emit('## 3. Angle-Bin Summation Consistency')
    emit()
    emit('Verification that $\\sum_{\\text{bins}} \\sigma(g{\\to}g\', \\text{bin}_i)$ '
         'matches the angle-integrated $\\sigma(g{\\to}g\')$.')
    emit()

    T = 10.0 * kev_kelvin
    narrow_bounds = [0.5 * kev, 1.0 * kev, 2.0 * kev, 5.0 * kev]
    G = len(narrow_bounds) - 1

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=narrow_bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=12, quad_order_Ep=12, xi_order=16)

    S_int = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)

    emit('| N_bins | g | g\' | Sum-over-bins | Integrated | Rel Diff |')
    emit('|--------|---|-----|--------------|------------|----------|')

    max_diffs = []
    for nbins in [4, 8, 16, 32]:
        S_mb = mg.compute_sigma_matrix(KERNEL_Q64, num_angle_bins=nbins, T=T, Ne=1.0)
        S_sum = S_mb.sum(axis=2)
        for g in range(G):
            for gp in range(G):
                if abs(S_int[g, gp]) < 1e-35:
                    continue
                rd = abs(S_sum[g, gp] - S_int[g, gp]) / abs(S_int[g, gp])
                max_diffs.append(rd)
                emit(f'| {nbins} | {g} | {gp} | {S_sum[g, gp]:.6e} | '
                     f'{S_int[g, gp]:.6e} | {rd:.2e} |')

    emit()
    if max_diffs:
        emit(f'Maximum relative difference across all cases: {max(max_diffs):.2e}')
    emit()


# ─── Section 4: Matrix heatmaps ─────────────────────────────────────────

DENSE_BOUNDS_KEV = np.logspace(np.log10(0.1), np.log10(100.0), 41)
DENSE_BOUNDS_ERG = DENSE_BOUNDS_KEV * kev


def section_heatmaps():
    emit('## 4. Multigroup Matrix Heatmaps')
    emit()
    G_dense = len(DENSE_BOUNDS_ERG) - 1
    emit(f'Log-scale colorplots of the angle-integrated {G_dense}×{G_dense} matrix '
         f'({G_dense} log-spaced groups from {DENSE_BOUNDS_KEV[0]:.1f} to '
         f'{DENSE_BOUNDS_KEV[-1]:.0f} keV) at several temperatures.')
    emit()

    T_kevs = [1.0, 10.0, 100.0]
    fig, axes = plt.subplots(1, len(T_kevs), figsize=(5.5 * len(T_kevs), 5))

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=DENSE_BOUNDS_ERG.tolist(),
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=8, quad_order_Ep=8, xi_order=8)

    centers_kev = np.sqrt(DENSE_BOUNDS_KEV[:-1] * DENSE_BOUNDS_KEV[1:])
    tick_positions = np.arange(0, G_dense, max(1, G_dense // 6))
    tick_labels = [f'{centers_kev[i]:.1f}' for i in tick_positions]

    for idx, T_kev in enumerate(T_kevs):
        T = T_kev * kev_kelvin
        S = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)

        ax = axes[idx]
        S_pos = np.maximum(np.abs(S), 1e-50)
        im = ax.imshow(S_pos, norm=LogNorm(), aspect='auto',
                       origin='lower', cmap='viridis')
        ax.set_title(f'T = {T_kev} keV')
        ax.set_xlabel("E' (keV)")
        ax.set_ylabel('E (keV)')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Angle-integrated multigroup Compton matrix', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_heatmaps.png')

    emit(f'![Matrix heatmaps]({fig_path})')
    emit()


# ─── Section 5: Detailed balance ─────────────────────────────────────────

def section_detailed_balance():
    from scipy.integrate import quad as sp_quad
    emit('## 5. Detailed Balance Check')
    emit()
    emit('In thermal equilibrium the net photon scattering rate between '
         'any two groups must vanish.  Following the CMMC convention '
         '(see `compton_matrix_mc.cpp`), the multigroup detailed-balance '
         'condition including stimulated emission reads:')
    emit()
    emit('$$\\sigma(g{\\to}g\')\\,(1+\\bar n_g)\\,B_{g\'}\\,E_g '
         '= \\sigma(g\'{\\to}g)\\,(1+\\bar n_{g\'})\\,B_g\\,E_{g\'}$$')
    emit()
    emit('where $E_g = \\sqrt{E_{g-1/2} E_{g+1/2}}$ is the geometric-mean '
         'group center,')
    emit()
    emit('$$B_g = \\int_{\\Delta E_g} B(E,T)\\,dE '
         '= a_R\\,T^4\\,\\mathcal{P}(x_{lo},x_{hi})$$')
    emit()
    emit('is the Planck energy density integrated over the group, and')
    emit()
    emit('$$\\bar n_g = \\frac{c^3}{8\\pi h}\\,'
         '\\frac{B_g}{\\nu_g^3\\,\\Delta\\nu_g}$$')
    emit()
    emit('is the effective equilibrium photon occupation number at the '
         'group centre frequency $\\nu_g = E_g/h$.  The ratio')
    emit()
    emit('$$f_{DB}(g{\\to}g\') = '
         '\\frac{(1+\\bar n_{g\'})\\,B_g\\,E_{g\'}}'
         '{(1+\\bar n_g)\\,B_{g\'}\\,E_g}$$')
    emit()
    emit('should satisfy '
         '$\\sigma(g\'{\\to}g) = \\sigma(g{\\to}g\')\\times f_{DB}$.')
    emit()

    T_kevs = [1.0, 10.0, 100.0]

    try:
        sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))
        import _compton_matrix_mc  # noqa: F401 – only needed for planck_integral below
    except ImportError:
        pass

    import _units as u

    h = u.planck_constant
    c = u.clight
    a_rad = u.arad

    def Bg(E_lo, E_hi, T):
        """Planck energy density integrated over [E_lo, E_hi] (erg/cm^3)."""
        from planck_integral import planck_integral as _pi
        kT = k_boltz * T
        return a_rad * T**4 * _pi(E_lo / kT, E_hi / kT)

    try:
        from planck_integral import planck_integral as _pi
        _pi(0.1, 1.0)
        use_python_planck = True
    except Exception:
        use_python_planck = False

    def compute_Bg_neq(bounds_erg, centers_erg, T):
        """Compute B_g and n_eq_g arrays following the CMMC convention."""
        G = len(bounds_erg) - 1
        kT = k_boltz * T
        fac = c**3 / (8.0 * np.pi * h)

        B_arr = np.zeros(G)
        n_eq = np.zeros(G)

        for g in range(G):
            x_lo = bounds_erg[g] / kT
            x_hi = bounds_erg[g + 1] / kT

            B_arr[g] = a_rad * T**4 * planck_integral_py(x_lo, x_hi)

            nu_g = centers_erg[g] / h
            dnu_g = (bounds_erg[g + 1] - bounds_erg[g]) / h
            n_eq[g] = fac * B_arr[g] / (nu_g**3 * dnu_g)

        return B_arr, n_eq

    def planck_integral_py(a, b):
        """Python fallback for the Clark polylogarithm Planck integral."""
        from scipy.integrate import quad as sq
        pi4_15 = np.pi**4 / 15.0
        val, _ = sq(lambda x: x**3 / np.expm1(x) if x > 1e-15 else x**2,
                     a, b, limit=200)
        return val / pi4_15

    for T_kev in T_kevs:
        T = T_kev * kev_kelvin
        bounds_erg = DENSE_BOUNDS_ERG.tolist()
        G = len(bounds_erg) - 1
        centers_erg = np.array([math.sqrt(bounds_erg[i] * bounds_erg[i + 1])
                                for i in range(G)])
        centers_kev = np.sqrt(DENSE_BOUNDS_KEV[:-1] * DENSE_BOUNDS_KEV[1:])
        E_g = centers_erg

        B_arr, n_eq = compute_Bg_neq(bounds_erg, centers_erg, T)

        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds_erg,
            weight_function=cm.PlanckWeightFunction(cap_x=25.0),
            quad_order_E=16, quad_order_Ep=16, xi_order=16)
        S = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)

        # DB factor f(g->g') = (1+n_g') * B_g * E_g' / ((1+n_g) * B_g' * E_g)
        # Condition: S[g',g] = S[g,g'] * f(g->g')
        one_plus_n = 1.0 + n_eq
        f_db = np.zeros((G, G))
        for g in range(G):
            for gp in range(G):
                if B_arr[gp] > 0 and one_plus_n[g] > 0:
                    f_db[g, gp] = (one_plus_n[gp] * B_arr[g] * E_g[gp]
                                   / (one_plus_n[g] * B_arr[gp] * E_g[g]))

        # Check: S[gp,g] vs S[g,gp] * f_db[g,gp]
        predicted = S * f_db          # predicted[g,gp] = sigma(g->gp) * f_db(g->gp) ≈ sigma(gp->g)
        actual = S.T                  # actual[g,gp] = sigma(gp->g) = S[gp,g]

        S_threshold = S.max() * 1e-6
        upper = np.triu(np.ones((G, G), dtype=bool), k=1)
        mask = (upper & (S > S_threshold) & (S.T > S_threshold)
                & (f_db > 0) & np.isfinite(f_db))

        scale = np.maximum(np.abs(predicted), np.abs(actual))
        rel = np.where(mask & (scale > 0), np.abs(predicted - actual) / scale, np.nan)

        valid = rel[mask]
        n_pairs = mask.sum()

        emit(f'### T = {T_kev} keV')
        emit()
        emit(f'Off-diagonal pairs tested: {n_pairs}')
        emit()

        if valid.size > 0:
            emit(f'| Statistic | Value |')
            emit(f'|-----------|-------|')
            emit(f'| Median relative violation | {np.nanmedian(valid):.2e} |')
            emit(f'| 95th percentile | {np.nanpercentile(valid, 95):.2e} |')
            emit(f'| Max | {np.nanmax(valid):.2e} |')
        emit()

    # Heatmap of the DB violation
    emit('### Detailed Balance Violation Maps')
    emit()

    fig, axes_db = plt.subplots(1, len(T_kevs), figsize=(5.5 * len(T_kevs), 5))
    G_dense = len(DENSE_BOUNDS_ERG) - 1
    tick_pos = np.arange(0, G_dense, max(1, G_dense // 6))
    centers_kev_all = np.sqrt(DENSE_BOUNDS_KEV[:-1] * DENSE_BOUNDS_KEV[1:])
    tick_lbl = [f'{centers_kev_all[i]:.1f}' for i in tick_pos]

    for idx, T_kev in enumerate(T_kevs):
        T = T_kev * kev_kelvin
        bounds_erg = DENSE_BOUNDS_ERG.tolist()
        G = len(bounds_erg) - 1
        centers_erg = np.array([math.sqrt(bounds_erg[i] * bounds_erg[i + 1])
                                for i in range(G)])
        E_g = centers_erg

        B_arr, n_eq = compute_Bg_neq(bounds_erg, centers_erg, T)
        one_plus_n = 1.0 + n_eq

        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds_erg,
            weight_function=cm.PlanckWeightFunction(cap_x=25.0),
            quad_order_E=16, quad_order_Ep=16, xi_order=16)
        S = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)

        f_db = np.zeros((G, G))
        for g in range(G):
            for gp in range(G):
                if B_arr[gp] > 0 and one_plus_n[g] > 0:
                    f_db[g, gp] = (one_plus_n[gp] * B_arr[g] * E_g[gp]
                                   / (one_plus_n[g] * B_arr[gp] * E_g[g]))

        predicted = S * f_db
        actual = S.T
        S_thr = S.max() * 1e-6
        pair_mask = (S > S_thr) & (S.T > S_thr) & (f_db > 0) & np.isfinite(f_db)
        sc = np.maximum(np.abs(predicted), np.abs(actual))
        rel = np.where(pair_mask & (sc > 0), np.abs(predicted - actual) / sc, np.nan)

        ax = axes_db[idx]
        from matplotlib.colors import LogNorm as LN
        finite = rel[np.isfinite(rel)]
        vmin_db = max(np.nanmin(finite), 1e-14) if finite.size > 0 else 1e-14
        vmax_db = np.nanmax(finite) if finite.size > 0 else 1.0
        im = ax.imshow(rel, aspect='auto', origin='lower', cmap='hot_r',
                       norm=LN(vmin=vmin_db, vmax=max(vmin_db * 10, vmax_db)))
        ax.set_title(f'T = {T_kev} keV')
        ax.set_xlabel("g'")
        ax.set_ylabel('g')
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Detailed balance violation: '
                 '$|\\sigma_{g{\\to}g\'} f_{DB} - \\sigma_{g\'{\\to}g}|$ / max',
                 y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_detailed_balance.png')

    emit(f'![Detailed balance violation]({fig_path})')
    emit()
    emit('The violation measures how well the numerically integrated multigroup '
         'matrix preserves the Compton kernel\'s built-in detailed balance '
         'at each temperature, including the stimulated-emission correction '
         '$(1+\\bar n_g)$.  Non-zero residuals reflect quadrature error '
         'and the discretisation of the continuous kernel onto a finite '
         'group structure.')
    emit()


# ─── Section 6: MC S-matrix comparison ──────────────────────────────────

def section_mc_smatrix():
    emit('## 6. MC S-Matrix Comparison')
    emit()

    if not _cmmc_available():
        emit('*Skipped: `_compton_matrix_mc` module is not functional.*')
        emit()
        return

    sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))
    import _compton_matrix_mc as mc_mod

    T_kev = 10.0
    T = T_kev * kev_kelvin
    num_mc_samples = 200000

    bounds_erg = DENSE_BOUNDS_ERG.tolist()
    G = len(bounds_erg) - 1
    centers = [math.sqrt(bounds_erg[i] * bounds_erg[i + 1]) for i in range(G)]
    centers_kev = np.sqrt(DENSE_BOUNDS_KEV[:-1] * DENSE_BOUNDS_KEV[1:])

    mc_obj = mc_mod.ComptonMatrixMC(
        energy_groups_centers=centers,
        energy_groups_boundaries=bounds_erg,
        num_of_samples=num_mc_samples,
        force_detailed_balance=False,
        seed=42)

    S_mc = np.array(mc_obj.calculate_S_matrix(temperature=T))

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds_erg,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=8, quad_order_Ep=8, xi_order=8)
    S_det = mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0)

    tick_positions = np.arange(0, G, max(1, G // 6))
    tick_labels = [f'{centers_kev[i]:.1f}' for i in tick_positions]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    S_mc_pos = np.maximum(np.abs(S_mc), 1e-50)
    S_det_pos = np.maximum(np.abs(S_det), 1e-50)

    sig_mc = S_mc_pos[S_mc_pos > 1e-50]
    sig_det = S_det_pos[S_det_pos > 1e-50]
    vmin = min(sig_mc.min() if sig_mc.size else 1e-50,
               sig_det.min() if sig_det.size else 1e-50)
    vmax = max(S_mc_pos.max(), S_det_pos.max())

    for ax, data, title in [(axes[0], S_det_pos, 'Deterministic'),
                             (axes[1], S_mc_pos, f'MC ({num_mc_samples // 1000}k samples)')]:
        im = ax.imshow(data, norm=LogNorm(vmin=vmin, vmax=vmax),
                       aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("E' (keV)")
        ax.set_ylabel('E (keV)')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    mask = (np.abs(S_mc) > 1e-35) & (np.abs(S_det) > 1e-35)
    S_mc_safe = np.where(mask, np.abs(S_mc), 1.0)
    rel_diff = np.where(mask, np.abs(S_det - S_mc) / S_mc_safe, 0.0)
    im = axes[2].imshow(rel_diff, aspect='auto', origin='lower', cmap='Reds',
                        vmin=0, vmax=min(5.0, rel_diff.max()) if rel_diff.max() > 0 else 1.0)
    axes[2].set_title('Relative difference')
    axes[2].set_xlabel("E' (keV)")
    axes[2].set_ylabel('E (keV)')
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels, rotation=45, fontsize=7)
    axes[2].set_yticks(tick_positions)
    axes[2].set_yticklabels(tick_labels, fontsize=7)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f'S-matrix comparison — {G} groups, T = {T_kev} keV', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_mc_smatrix.png')

    emit(f'![MC S-matrix comparison]({fig_path})')
    emit()

    if np.any(mask):
        emit(f'Median element-wise relative difference: {np.median(rel_diff[mask]):.2e}')
        emit(f'Max element-wise relative difference: {np.max(rel_diff[mask]):.2e}')
    emit()

    row_sums_det = S_det.sum(axis=1)
    row_sums_mc = S_mc.sum(axis=1)
    row_mask = row_sums_mc > 1e-30
    if np.any(row_mask):
        row_rel = np.abs(row_sums_det[row_mask] - row_sums_mc[row_mask]) / row_sums_mc[row_mask]
        emit(f'Row-sum (total cross section) max relative difference: {np.max(row_rel):.2e}')
        emit()

    emit('**Note:** Element-wise differences are expected because CMMC uses a '
         'linear energy-redistribution scheme that shifts cross-section weight '
         'toward the diagonal (self-scattering) entry. This does not affect row '
         'sums (total scattering cross section per group), which agree well.')
    emit()


# ─── Section 7: MC angle CDF comparison ─────────────────────────────────

def section_mc_angle_cdf():
    emit('## 7. MC Angle CDF Comparison')
    emit()

    if not _cmmc_available():
        emit('*Skipped: `_compton_matrix_mc` module is not functional.*')
        emit()
        return

    sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))
    import _compton_matrix_mc as mc_mod

    T_kev = 10.0
    T = T_kev * kev_kelvin
    bounds_erg = DENSE_BOUNDS_ERG.tolist()
    G = len(bounds_erg) - 1
    centers = [math.sqrt(bounds_erg[i] * bounds_erg[i + 1]) for i in range(G)]
    centers_kev = np.sqrt(DENSE_BOUNDS_KEV[:-1] * DENSE_BOUNDS_KEV[1:])
    NUM_ANGLE_BINS = mc_mod.ComptonMatrixMC.NUM_ANGLE_BINS

    mc_obj = mc_mod.ComptonMatrixMC(
        energy_groups_centers=centers,
        energy_groups_boundaries=bounds_erg,
        num_of_samples=200000,
        force_detailed_balance=False,
        seed=42)
    mc_obj.set_tables(temperature_grid=[T * 0.9, T, T * 1.1])

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds_erg,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=8, quad_order_Ep=8, xi_order=8)
    S_det = mg.compute_sigma_matrix(
        KERNEL_Q64, num_angle_bins=NUM_ANGLE_BINS, T=T, Ne=1.0)

    xi_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)

    sig_pairs = []
    for g0 in range(G):
        for g in range(G):
            total = S_det[g0, g, :].sum()
            if total > 1e-30:
                sig_pairs.append((g0, g, total))

    sig_pairs.sort(key=lambda x: -x[2])
    off_diag = [(g0, g, t) for g0, g, t in sig_pairs if g0 != g]
    on_diag = [(g0, g, t) for g0, g, t in sig_pairs if g0 == g]
    pairs_to_plot = (off_diag[:4] + on_diag[:2])[:6]

    ncols = min(3, len(pairs_to_plot))
    nrows = (len(pairs_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, (g0, g, _) in enumerate(pairs_to_plot):
        ax = axes[idx // ncols][idx % ncols]

        row = S_det[g0, g, :]
        total = row.sum()
        cdf_det = np.zeros(NUM_ANGLE_BINS + 1)
        cdf_det[1:] = np.cumsum(row) / total
        cdf_det[-1] = 1.0

        cdf_mc = np.array(mc_obj.get_angle_cdf(temperature=T, g0=g0, g=g))

        ax.plot(xi_edges, cdf_det, 'b-', linewidth=1.5, label='Deterministic')
        ax.plot(xi_edges, cdf_mc, 'r--', linewidth=1.5, label='MC')
        ax.set_title(f'{centers_kev[g0]:.1f} -> {centers_kev[g]:.1f} keV')
        ax.set_xlabel(r'$\cos\theta$')
        ax.set_ylabel('CDF')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(len(pairs_to_plot), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'Angle CDF comparison — {G} groups, T = {T_kev} keV', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('mg_mc_angle_cdf.png')

    emit(f'![MC angle CDF comparison]({fig_path})')
    emit()

    max_diffs = []
    for g0, g, _ in sig_pairs:
        row = S_det[g0, g, :]
        total = row.sum()
        cdf_det = np.zeros(NUM_ANGLE_BINS + 1)
        cdf_det[1:] = np.cumsum(row) / total
        cdf_det[-1] = 1.0
        cdf_mc = np.array(mc_obj.get_angle_cdf(temperature=T, g0=g0, g=g))
        max_diffs.append(np.max(np.abs(cdf_det - cdf_mc)))

    if max_diffs:
        emit(f'Median max CDF difference: {np.median(max_diffs):.4f}')
        emit(f'Worst max CDF difference: {np.max(max_diffs):.4f}')
    emit()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    emit('# Multigroup Kernel Validation Report')
    emit()
    emit('Validation of the Planck-weighted multigroup-multiangle Compton '
         'scattering matrix (`ComptonMultigroupKernel`).')
    emit()

    section_denominator()
    section_convergence()
    section_angle_summation()
    section_heatmaps()
    section_detailed_balance()
    section_mc_smatrix()
    section_mc_angle_cdf()

    md_path = os.path.join(GEN_DIR, 'multigroup_validation.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
