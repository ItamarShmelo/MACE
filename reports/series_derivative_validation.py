"""
Series derivative validation report: validates series dsigma_E_dT implementation.

Generates reports/generated/series_derivative_validation.md with embedded plots covering:
  1. Finite-difference comparison (Richardson-extrapolated)
  2. Series vs quadrature derivative agreement
  3. Method comparison (PowerSeries, DD, Asymptotic, Auto)
  4. Small-T stability
  5. Derivative spectral profiles
  6. Angular distribution

Usage:
    python3 reports/series_derivative_validation.py

Output:
    reports/generated/series_derivative_validation.md  (+ .png plots in figs/)
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

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from _compton_kernel_series import ComptonKernelSeries, SeriesMethod

from _units import kev, kev_kelvin, me_c2, k_boltz

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


NICE_POINTS = [
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.5, 0.5, 1.0),
    (1.0, 2.0, -0.5, 1.0),
    (10.0, 8.0, 0.3, 5.0),
    (50.0, 45.0, 0.0, 20.0),
    (5.0, 5.0, 0.0, 10.0),
    (5.0, 3.0, 0.9, 10.0),
]


# ─── Section 1: Finite-Difference Comparison ──────────────────────────────

def section_fd_comparison():
    emit('## 1. Finite-Difference Comparison')
    emit()
    emit('Richardson-extrapolated centered FD of series `sigma_E` vs analytic series `dsigma_E_dT`.')
    emit()

    engine = ComptonKernelSeries()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    h_fracs = np.logspace(-6, -1, 30)

    for idx, (E_kev, Ep_kev, xi, T_kev) in enumerate(NICE_POINTS[:4]):
        E = E_kev * kev
        Ep = Ep_kev * kev
        T_K = T_kev * kev_kelvin

        sig = engine.sigma_E(E, Ep, xi, T_K, 1.0)
        if sig.estimated_rel_error > 1e-6:
            continue

        analytic = engine.dsigma_E_dT(E, Ep, xi, T_K, 1.0).value
        if abs(analytic) < 1e-300:
            continue

        rel_errs = []
        for hf in h_fracs:
            h = hf * T_K
            vp = engine.sigma_E(E, Ep, xi, T_K + h, 1.0).value
            vm = engine.sigma_E(E, Ep, xi, T_K - h, 1.0).value
            fd = (vp - vm) / (2.0 * h)
            rel_errs.append(abs(fd - analytic) / abs(analytic))

        label = f'E={E_kev}, E\'={Ep_kev}, xi={xi}, T={T_kev}'
        ax.loglog(h_fracs, rel_errs, '-o', markersize=3, label=label)

    ax.set_xlabel('h / T_K')
    ax.set_ylabel('|FD - analytic| / |analytic|')
    ax.set_title('Series Derivative: FD Error vs Step Size')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig_path = save_fig('series_deriv_fd_error_vs_h.png')

    emit(f'![FD Error vs h]({fig_path})')
    emit()

    emit('| E (keV) | E\' (keV) | xi | T (keV) | analytic | fd_rich | rel_err |')
    emit('|---------|-----------|------|---------|----------|---------|---------|')

    for E_kev, Ep_kev, xi, T_kev in NICE_POINTS:
        E = E_kev * kev
        Ep = Ep_kev * kev
        T_K = T_kev * kev_kelvin

        sig = engine.sigma_E(E, Ep, xi, T_K, 1.0)
        if sig.estimated_rel_error > 1e-6:
            emit(f'| {E_kev} | {Ep_kev} | {xi} | {T_kev} | -- | -- | skipped |')
            continue

        analytic = engine.dsigma_E_dT(E, Ep, xi, T_K, 1.0).value
        h = 1e-4 * T_K
        fd = lambda step: (engine.sigma_E(E, Ep, xi, T_K + step, 1.0).value
                           - engine.sigma_E(E, Ep, xi, T_K - step, 1.0).value) / (2.0 * step)
        fd_rich = (4.0 * fd(h / 2.0) - fd(h)) / 3.0

        rel = abs(analytic - fd_rich) / (abs(fd_rich) + 1e-300)
        emit(f'| {E_kev} | {Ep_kev} | {xi} | {T_kev} | '
             f'{analytic:.6e} | {fd_rich:.6e} | {rel:.2e} |')

    emit()


# ─── Section 2: Series vs Quadrature Agreement ────────────────────────────

def section_series_vs_quadrature():
    emit('## 2. Series vs Quadrature Derivative Agreement')
    emit()
    emit('Error-aware comparison: agreement within C * (series_error + quad_error), C=10.')
    emit()

    series = ComptonKernelSeries()
    quad = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)
    safety = 10.0

    emit('| E (keV) | E\' (keV) | xi | T (keV) | series | quadrature | diff | bound | pass? |')
    emit('|---------|-----------|------|---------|--------|------------|------|-------|-------|')

    for E_kev, Ep_kev, xi, T_kev in NICE_POINTS:
        E = E_kev * kev
        Ep = Ep_kev * kev
        T_K = T_kev * kev_kelvin

        r_s = series.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
        r_q = quad.dsigma_E_dT(E, Ep, xi, T_K, 1.0)

        diff = abs(r_s.value - r_q.value)
        bound = safety * (r_s.estimated_abs_error + r_q.estimated_abs_error)
        scale = max(abs(r_s.value), abs(r_q.value))
        ok = diff < bound or (scale > 1e-300 and diff / scale < 1e-4)

        emit(f'| {E_kev} | {Ep_kev} | {xi} | {T_kev} | '
             f'{r_s.value:.6e} | {r_q.value:.6e} | {diff:.2e} | '
             f'{bound:.2e} | {"PASS" if ok else "FAIL"} |')

    emit()


# ─── Section 3: Method Comparison ─────────────────────────────────────────

def section_method_comparison():
    emit('## 3. Method Comparison')
    emit()
    emit('Derivative values from each series method at the same parameter points.')
    emit()

    methods = [
        ('Auto', SeriesMethod.Auto),
        ('PowerSeries', SeriesMethod.PowerSeries),
        ('PowerSeriesHP', SeriesMethod.PowerSeriesHighPrecision),
        ('Asymptotic', SeriesMethod.Asymptotic),
    ]

    emit('| E (keV) | E\' (keV) | xi | T (keV) | ' +
         ' | '.join(name for name, _ in methods) + ' |')
    emit('|---------|-----------|------|---------|' +
         '|'.join('--------' for _ in methods) + '|')

    for E_kev, Ep_kev, xi, T_kev in NICE_POINTS:
        E = E_kev * kev
        Ep = Ep_kev * kev
        T_K = T_kev * kev_kelvin

        vals = []
        for name, method in methods:
            eng = ComptonKernelSeries(method=method)
            try:
                r = eng.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
                vals.append(f'{r.value:.6e}')
            except RuntimeError:
                vals.append('FAIL')

        emit(f'| {E_kev} | {Ep_kev} | {xi} | {T_kev} | ' +
             ' | '.join(vals) + ' |')

    emit()


# ─── Section 4: Small-T Stability ─────────────────────────────────────────

def section_small_T():
    emit('## 4. Small-T Stability')
    emit()
    emit('| T (keV) | dsigma/dT | rel_error | finite? |')
    emit('|---------|-----------|-----------|---------|')

    engine = ComptonKernelSeries()
    E = 1.0 * kev
    Ep = 1.0 * kev

    for T_kev in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        T_K = T_kev * kev_kelvin
        r = engine.dsigma_E_dT(E, Ep, 0.0, T_K, 1.0)
        finite = np.isfinite(r.value) and np.isfinite(r.estimated_rel_error)
        emit(f'| {T_kev} | {r.value:.6e} | '
             f'{r.estimated_rel_error:.2e} | {"yes" if finite else "NO"} |')

    emit()


# ─── Section 5: Derivative Spectral Profiles ─────────────────────────────

PROFILE_CONFIGS = [
    {'E_kev': 1.0,  'xi': 0.0,  'T_kevs': [0.5, 1.0, 5.0],    'Ep_range': (0.3, 3.0)},
    {'E_kev': 10.0, 'xi': 0.0,  'T_kevs': [1.0, 5.0, 20.0],   'Ep_range': (2.0, 30.0)},
    {'E_kev': 1.0,  'xi': 0.5,  'T_kevs': [0.5, 1.0, 5.0],    'Ep_range': (0.2, 4.0)},
    {'E_kev': 5.0,  'xi': -0.5, 'T_kevs': [1.0, 5.0, 20.0],   'Ep_range': (1.0, 20.0)},
]


def section_spectral_profiles():
    emit('## 5. Derivative Spectral Profiles')
    emit()
    emit('Spectral shape of series Sigma_E (top) and series dSigma_E/dT (bottom) '
         'as a function of scattered energy E\' at fixed incident energy, angle, and '
         'several temperatures.')
    emit()

    engine = ComptonKernelSeries()

    for cfg in PROFILE_CONFIGS:
        E_kev = cfg['E_kev']
        xi = cfg['xi']
        Ep_lo, Ep_hi = cfg['Ep_range']
        E = E_kev * kev
        Ep_arr = np.linspace(Ep_lo, Ep_hi, 400) * kev

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        for T_kev in cfg['T_kevs']:
            T_K = T_kev * kev_kelvin
            label = f'T = {T_kev} keV'

            sig_vals, dsig_vals = engine.sigma_E_vec(E, Ep_arr, xi, T_K, 1.0)
            dsig_v, dsig_e = engine.dsigma_E_dT_vec(E, Ep_arr, xi, T_K, 1.0)

            ax1.semilogy(Ep_arr / kev, np.abs(sig_vals), label=label)
            ax2.plot(Ep_arr / kev, dsig_v, label=label)

        ax1.set_ylabel('|Sigma_E|')
        ax1.set_title(f'E = {E_kev} keV, xi = {xi}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("E' (keV)")
        ax2.set_ylabel('dSigma_E / dT')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f'series_deriv_profile_E{E_kev}_xi{xi}.png'
        fig_path = save_fig(fname)
        emit(f'![Derivative profile E={E_kev}, xi={xi}]({fig_path})')
        emit()


# ─── Section 6: Angular Distribution ─────────────────────────────────────

ANGULAR_CONFIGS = [
    {'E_kev': 1.0,  'Ep_kev': 1.0,  'T_kevs': [0.5, 1.0, 5.0]},
    {'E_kev': 1.0,  'Ep_kev': 0.8,  'T_kevs': [0.5, 1.0, 5.0]},
    {'E_kev': 10.0, 'Ep_kev': 10.0, 'T_kevs': [1.0, 5.0, 20.0]},
    {'E_kev': 10.0, 'Ep_kev': 8.0,  'T_kevs': [1.0, 5.0, 20.0]},
]


def section_angular_distribution():
    emit('## 6. Angular Distribution of the Derivative')
    emit()
    emit('Series Sigma_E (top) and dSigma_E/dT (bottom) as a function '
         'of scattering angle xi = cos(theta) at fixed energies and several '
         'temperatures.')
    emit()

    xi_arr = np.linspace(-0.99, 0.99, 300)

    for cfg in ANGULAR_CONFIGS:
        E_kev = cfg['E_kev']
        Ep_kev = cfg['Ep_kev']
        E = E_kev * kev
        Ep = Ep_kev * kev

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        for T_kev in cfg['T_kevs']:
            T_K = T_kev * kev_kelvin
            label = f'T = {T_kev} keV'

            engine = ComptonKernelSeries()

            sig_vals = []
            dsig_vals = []
            for xi in xi_arr:
                sig_vals.append(engine.sigma_E(E, Ep, xi, T_K, 1.0).value)
                dsig_vals.append(engine.dsigma_E_dT(E, Ep, xi, T_K, 1.0).value)

            ax1.semilogy(xi_arr, np.abs(sig_vals), label=label)
            ax2.plot(xi_arr, dsig_vals, label=label)

        ax1.set_ylabel('|Sigma_E|')
        ax1.set_title(f"E = {E_kev} keV, E' = {Ep_kev} keV")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('xi = cos(theta)')
        ax2.set_ylabel('dSigma_E / dT')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f'series_deriv_angular_E{E_kev}_Ep{Ep_kev}.png'
        fig_path = save_fig(fname)
        emit(f"![Angular distribution E={E_kev}, E'={Ep_kev}]({fig_path})")
        emit()


# ─── Section 7: Series Auto vs Quadrature Colorplots ─────────────────────

HEATMAP_E_GRID = np.logspace(-1, 2.7, 50)
HEATMAP_T_GRID = np.logspace(-0.3, 2.7, 50)
HEATMAP_RATIOS = [0.5, 0.9, 1.01, 2.0, 5.0]
HEATMAP_XIS = [-0.5, 0.0, 0.5]


ASYMP_TAU_ALPHA_THRESHOLD = 0.025
GAMMA_DOUBLE_PRECISION_SAFE = 0.02

# Method codes for the selection map: 0=Asymptotic, 1=PowerSeries, 2=PowerSeriesHP, nan=failed
METHOD_ASYMPTOTIC = 0
METHOD_POWER = 1
METHOD_POWER_HP = 2


def auto_method_select(E_kev, ratio, xi, T_kev):
    """Replicate the C++ Auto dispatch logic in Python."""
    E = E_kev * kev
    Ep = E * ratio
    gamma = E / me_c2
    gamma_p = Ep / me_c2
    tau = T_kev * kev_kelvin * k_boltz / me_c2

    a = 1.0 - xi
    dg = gamma_p - gamma
    q2 = dg * dg + 2.0 * gamma * gamma_p * a
    omega2 = (1.0 + xi) / a

    gg_a = gamma * gamma_p * a
    factor1 = 1.0 + gg_a / 2.0
    factor2 = 1.0 + (dg * dg) / (2.0 * gg_a)
    Delta = np.sqrt(factor1 * factor2)
    lambda_plus = dg / 2.0 + Delta
    if lambda_plus < 1.0:
        lambda_plus = 1.0

    rho_plus = lambda_plus + gamma
    rho_minus = lambda_plus - gamma_p
    alpha_plus = 1.0 / np.sqrt(rho_plus**2 + omega2)
    alpha_minus = 1.0 / np.sqrt(rho_minus**2 + omega2)

    tau_alpha_max = tau * max(alpha_plus, alpha_minus)

    if tau_alpha_max < ASYMP_TAU_ALPHA_THRESHOLD:
        return METHOD_ASYMPTOTIC
    elif min(gamma, gamma_p) >= GAMMA_DOUBLE_PRECISION_SAFE:
        return METHOD_POWER
    else:
        return METHOD_POWER_HP


def section_auto_vs_quad_colorplot():
    emit('## 7. Series Auto vs Quadrature Error Map')
    emit()
    emit('Relative discrepancy `|series_auto - Q256| / max(|auto|, |Q256|)` between '
         'the series Auto derivative and the 256-point Gauss-Laguerre quadrature '
         'derivative (pre-IBP) over the (E, T) plane.')
    emit()

    series = ComptonKernelSeries(method=SeriesMethod.Auto)
    quad = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)

    n_E = len(HEATMAP_E_GRID)
    n_T = len(HEATMAP_T_GRID)
    n_rows = len(HEATMAP_XIS)
    n_cols = len(HEATMAP_RATIOS)

    error_grids = {}
    method_grids = {}
    value_grids = {}
    all_rel_errs = []
    total_panels = n_rows * n_cols
    done = 0
    n_failed = 0

    for xi in HEATMAP_XIS:
        for ratio in HEATMAP_RATIOS:
            done += 1
            print(f'  auto vs quad panel {done}/{total_panels}: xi={xi}, ratio={ratio}')
            err_grid = np.full((n_T, n_E), np.nan)
            meth_grid = np.full((n_T, n_E), np.nan)
            val_grid = np.full((n_T, n_E), np.nan)
            for i, T_kev in enumerate(HEATMAP_T_GRID):
                T_K = T_kev * kev_kelvin
                for j, E_kev in enumerate(HEATMAP_E_GRID):
                    E = E_kev * kev
                    Ep = E * ratio
                    try:
                        r_s = series.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
                        r_q = quad.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
                        val_grid[i, j] = r_s.value
                        scale = max(abs(r_s.value), abs(r_q.value))
                        if scale > 1e-300:
                            rel = abs(r_s.value - r_q.value) / scale
                            err_grid[i, j] = rel
                            all_rel_errs.append(rel)
                        else:
                            err_grid[i, j] = 0.0
                        meth_grid[i, j] = auto_method_select(E_kev, ratio, xi, T_kev)
                    except (RuntimeError, Exception):
                        n_failed += 1
            error_grids[(xi, ratio)] = err_grid
            method_grids[(xi, ratio)] = meth_grid
            value_grids[(xi, ratio)] = val_grid

    # ── Error colorplot ──────────────────────────────────────────────────
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
            data = error_grids[(xi, ratio)]
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
    all_axes = [ax for row_axes in axes for ax in row_axes]
    cbar = fig.colorbar(pcm, ax=all_axes, shrink=0.85, pad=0.03)
    cbar.set_label('|series_auto - Q256| / max(|auto|, |Q256|)')
    fig.suptitle('Series Auto Derivative vs Quadrature (Q256 pre-IBP)', fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    err_fig_path = save_fig('series_deriv_auto_vs_quad_TE.png')

    emit(f'![Series Auto vs Q256]({err_fig_path})')
    emit()
    emit(f'Each panel shows the relative error on a {n_E}x{n_T} log-spaced grid '
         f'in (E, T) space. Columns vary E\'/E ({", ".join(str(r) for r in HEATMAP_RATIOS)}); '
         f'rows vary xi ({", ".join(str(x) for x in HEATMAP_XIS)}). '
         f'Sky-blue cells indicate negligible kernel values (both methods < 1e-300) '
         f'or exact agreement; gray cells indicate convergence failure.')
    emit()

    # ── Derivative value colorplot (per-panel normalization) ──────────────
    val_cmap = plt.cm.viridis.copy()
    val_cmap.set_bad('lightgray')

    fig_v, axes_v = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes_v[row][col]
            data = value_grids[(xi, ratio)]
            safe = np.where(np.isfinite(data), np.abs(data), np.nan)
            safe = np.where(safe > 0, safe, np.nan)
            finite = safe[np.isfinite(safe)]
            if len(finite) > 0:
                pmin, pmax = np.min(finite), np.max(finite)
                pnorm = LogNorm(vmin=pmin, vmax=pmax)
            else:
                pnorm = LogNorm(vmin=1e-30, vmax=1e-15)
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, safe,
                          norm=pnorm, cmap=val_cmap, shading='nearest')
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
            if len(finite) > 0:
                ax.text(0.03, 0.97,
                        f'{pmin:.0e}\n  to\n{pmax:.0e}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=6, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fig_v.suptitle(r'Series Auto: $|d\Sigma_E/dT|$ Value Map (per-panel scale)',
                   fontsize=13, y=1.01)
    fig_v.subplots_adjust(wspace=0.05, hspace=0.15)
    val_fig_path = save_fig('series_deriv_auto_value_TE.png')

    emit('### Derivative value map')
    emit()
    emit(f'![Series Auto derivative value map]({val_fig_path})')
    emit()
    emit(f'Absolute value of `dsigma_E_dT` from series Auto on the same '
         f'{n_E}x{n_T} (E, T) grid. Each panel is independently '
         f'scaled to its own [min, max] range (annotated top-left). '
         f'Gray cells indicate convergence failure or exactly zero values.')
    emit()

    # ── Statistics ────────────────────────────────────────────────────────
    n_total = n_E * n_T * n_rows * n_cols
    errs_arr = np.array(all_rel_errs) if all_rel_errs else np.array([0.0])

    emit('### Summary statistics')
    emit()
    emit(f'| Metric | Value |')
    emit(f'|--------|-------|')
    emit(f'| Grid points evaluated | {n_total - n_failed} / {n_total} |')
    emit(f'| Failed (exception) | {n_failed} |')
    emit(f'| Points with valid comparison | {len(all_rel_errs)} |')
    emit()

    if len(all_rel_errs) > 0:
        emit(f'| Statistic | Value |')
        emit(f'|-----------|-------|')
        emit(f'| Minimum | {np.min(errs_arr):.2e} |')
        emit(f'| Median | {np.median(errs_arr):.2e} |')
        emit(f'| Mean | {np.mean(errs_arr):.2e} |')
        emit(f'| 90th percentile | {np.percentile(errs_arr, 90):.2e} |')
        emit(f'| 95th percentile | {np.percentile(errs_arr, 95):.2e} |')
        emit(f'| 99th percentile | {np.percentile(errs_arr, 99):.2e} |')
        emit(f'| Maximum | {np.max(errs_arr):.2e} |')
        emit()

        pct_1e8 = 100.0 * np.sum(errs_arr < 1e-8) / len(errs_arr)
        pct_1e6 = 100.0 * np.sum(errs_arr < 1e-6) / len(errs_arr)
        pct_1e4 = 100.0 * np.sum(errs_arr < 1e-4) / len(errs_arr)
        emit(f'| Points with error < 1e-8 | {pct_1e8:.1f}% |')
        emit(f'| Points with error < 1e-6 | {pct_1e6:.1f}% |')
        emit(f'| Points with error < 1e-4 | {pct_1e4:.1f}% |')
        emit()

    # ── Method selection colorplot ────────────────────────────────────────
    from matplotlib.colors import ListedColormap, BoundaryNorm

    method_cmap = ListedColormap(['#2196F3', '#4CAF50', '#FF9800'])
    method_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], method_cmap.N)
    method_labels = ['Asymptotic', 'PowerSeries', 'PowerSeriesHP']

    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                               sharex=True, sharey=True, squeeze=False)

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes2[row][col]
            data = method_grids[(xi, ratio)]
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, data,
                          cmap=method_cmap, norm=method_norm, shading='nearest')
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

    all_axes2 = [ax for row_axes in axes2 for ax in row_axes]
    cbar2 = fig2.colorbar(
        plt.cm.ScalarMappable(cmap=method_cmap, norm=method_norm),
        ax=all_axes2, shrink=0.85, pad=0.03, ticks=[0, 1, 2])
    cbar2.ax.set_yticklabels(method_labels)
    fig2.suptitle('Auto Method Selection Map', fontsize=13, y=1.01)
    fig2.subplots_adjust(wspace=0.05, hspace=0.15)
    meth_fig_path = save_fig('series_deriv_method_selection_TE.png')

    emit('### Auto method selection map')
    emit()
    emit(f'![Method selection map]({meth_fig_path})')
    emit()
    emit('Blue = Asymptotic, Green = PowerSeries (double), Orange = PowerSeriesHP (DD). '
         'The method is selected by the Auto dispatch logic based on '
         '`tau * max(alpha+, alpha-) < 0.025` (Asymptotic) and '
         '`min(gamma, gamma\') >= 0.02` (PowerSeries vs HP).')
    emit()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    emit('# Series Derivative Validation Report')
    emit()
    emit('Validation of series `dsigma_E_dT` (temperature derivative of the Compton kernel).')
    emit()

    section_fd_comparison()
    section_series_vs_quadrature()
    section_method_comparison()
    section_small_T()
    section_spectral_profiles()
    section_angular_distribution()
    section_auto_vs_quad_colorplot()

    md_path = os.path.join(GEN_DIR, 'series_derivative_validation.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
