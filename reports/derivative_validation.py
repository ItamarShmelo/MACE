"""
Derivative validation report: validates dsigma_E_dtau implementation.

Generates reports/generated/derivative_validation.md with embedded plots covering:
  1. Gauss-Laguerre convergence (NL=64/128/256) for both forms
  2. Finite-difference comparison (Richardson-extrapolated)
  3. Pre-IBP vs Post-IBP derivative agreement across temperature
  4. Kappa ratio validation (C++ vs scipy)
  5. Small-tau stability

Usage:
    python3 reports/derivative_validation.py

Output:
    reports/generated/derivative_validation.md  (+ .png plots in figs/)
"""
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_kernel_quadrature import (
    ComptonKernelQuadrature, QuadratureForm,
    scaled_K1, scaled_K2, kappa_ratio,
)

ME_C2 = 9.109383713928e-28 * (2.99792458e10)**2
KEV = 1.602176634e-9

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


# ─── Section 1: GL Convergence ────────────────────────────────────────────

def section_gl_convergence():
    emit('## 1. Derivative Gauss-Laguerre Convergence')
    emit()
    emit('Relative difference between NL and NL/2 (Richardson error proxy).')
    emit()

    for form_name, form_enum in [('Pre-IBP', QuadratureForm.PreIBP),
                                  ('Post-IBP', QuadratureForm.PostIBP)]:
        emit(f'### {form_name}')
        emit()
        emit('| E (keV) | E\' (keV) | xi | T (keV) | NL=64 | NL=128 | NL=256 |')
        emit('|---------|-----------|------|---------|-------|--------|--------|')

        for E_kev, Ep_kev, xi, tau_kev in NICE_POINTS:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2

            vals = []
            for NL in [64, 128, 256]:
                eng = ComptonKernelQuadrature(NL, form_enum)
                r = eng.dsigma_E_dtau(E, Ep, xi, tau, 1.0)
                vals.append(r.estimated_rel_error)

            emit(f'| {E_kev} | {Ep_kev} | {xi} | {tau_kev} | '
                 f'{vals[0]:.2e} | {vals[1]:.2e} | {vals[2]:.2e} |')

        emit()


# ─── Section 2: Finite-Difference Comparison ──────────────────────────────

def section_fd_comparison():
    emit('## 2. Finite-Difference Comparison')
    emit()
    emit('Richardson-extrapolated centered FD vs analytic derivative (pre-IBP, NL=256).')
    emit()

    engine = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    h_fracs = np.logspace(-6, -1, 30)

    for idx, (E_kev, Ep_kev, xi, tau_kev) in enumerate(NICE_POINTS[:4]):
        E = E_kev * KEV
        Ep = Ep_kev * KEV
        tau = tau_kev * KEV / ME_C2

        sig = engine.sigma_E(E, Ep, xi, tau, 1.0)
        if sig.estimated_rel_error > 1e-6:
            continue

        analytic = engine.dsigma_E_dtau(E, Ep, xi, tau, 1.0).value
        if abs(analytic) < 1e-300:
            continue

        rel_errs = []
        for hf in h_fracs:
            h = hf * tau
            vp = engine.sigma_E(E, Ep, xi, tau + h, 1.0).value
            vm = engine.sigma_E(E, Ep, xi, tau - h, 1.0).value
            fd = (vp - vm) / (2.0 * h)
            rel_errs.append(abs(fd - analytic) / abs(analytic))

        label = f'E={E_kev}, E\'={Ep_kev}, xi={xi}, T={tau_kev}'
        ax.loglog(h_fracs, rel_errs, '-o', markersize=3, label=label)

    ax.set_xlabel('h / tau')
    ax.set_ylabel('|FD - analytic| / |analytic|')
    ax.set_title('FD Error vs Step Size (pre-IBP)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig_path = save_fig('deriv_fd_error_vs_h.png')

    emit(f'![FD Error vs h]({fig_path})')
    emit()

    emit('| E (keV) | E\' (keV) | xi | T (keV) | analytic | fd_rich | rel_err |')
    emit('|---------|-----------|------|---------|----------|---------|---------|')

    for E_kev, Ep_kev, xi, tau_kev in NICE_POINTS:
        E = E_kev * KEV
        Ep = Ep_kev * KEV
        tau = tau_kev * KEV / ME_C2

        sig = engine.sigma_E(E, Ep, xi, tau, 1.0)
        if sig.estimated_rel_error > 1e-6:
            emit(f'| {E_kev} | {Ep_kev} | {xi} | {tau_kev} | -- | -- | skipped |')
            continue

        analytic = engine.dsigma_E_dtau(E, Ep, xi, tau, 1.0).value
        h = 1e-4 * tau
        fd_h = lambda step: (engine.sigma_E(E, Ep, xi, tau + step, 1.0).value
                             - engine.sigma_E(E, Ep, xi, tau - step, 1.0).value) / (2.0 * step)
        fd_rich = (4.0 * fd_h(h / 2.0) - fd_h(h)) / 3.0

        rel = abs(analytic - fd_rich) / (abs(fd_rich) + 1e-300)
        emit(f'| {E_kev} | {Ep_kev} | {xi} | {tau_kev} | '
             f'{analytic:.6e} | {fd_rich:.6e} | {rel:.2e} |')

    emit()


# ─── Section 3: Pre vs Post IBP Agreement ─────────────────────────────────

def section_pre_post_agreement():
    emit('## 3. Pre-IBP vs Post-IBP Derivative Agreement')
    emit()

    tau_kevs = np.logspace(-1, 2, 30)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    E_kev, Ep_kev, xi = 5.0, 3.0, 0.5

    rel_diffs = []
    for tau_kev in tau_kevs:
        E = E_kev * KEV
        Ep = Ep_kev * KEV
        tau = tau_kev * KEV / ME_C2

        eng_pre = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)
        eng_post = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)

        r_pre = eng_pre.dsigma_E_dtau(E, Ep, xi, tau, 1.0)
        r_post = eng_post.dsigma_E_dtau(E, Ep, xi, tau, 1.0)

        scale = max(abs(r_pre.value), abs(r_post.value))
        if scale < 1e-300:
            rel_diffs.append(np.nan)
        else:
            rel_diffs.append(abs(r_pre.value - r_post.value) / scale)

    ax.semilogy(tau_kevs, rel_diffs, 'b-o', markersize=3)
    ax.set_xlabel('T (keV)')
    ax.set_ylabel('|pre - post| / max(|pre|, |post|)')
    ax.set_title(f'Pre vs Post IBP Derivative (E={E_kev}, E\'={Ep_kev}, xi={xi})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    fig_path = save_fig('deriv_pre_post_agreement.png')

    emit(f'![Pre vs Post IBP]({fig_path})')
    emit()


# ─── Section 4: Kappa Ratio Validation ────────────────────────────────────

def section_kappa_validation():
    emit('## 4. Kappa Ratio Validation')
    emit()

    from scipy.special import kve

    xs = np.logspace(-1, 3, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cpp_k1 = [scaled_K1(x) for x in xs]
    scipy_k1 = [float(kve(1, x)) for x in xs]
    rel_k1 = [abs(c - s) / abs(s) for c, s in zip(cpp_k1, scipy_k1)]

    ax1.loglog(xs, rel_k1, 'b-')
    ax1.set_xlabel('x')
    ax1.set_ylabel('|C++ - scipy| / |scipy|')
    ax1.set_title('scaled_K1 relative error')
    ax1.axhline(1e-15, color='gray', linestyle='--', alpha=0.5, label='machine eps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    taus = np.logspace(-4, 2, 100)
    kappas = [kappa_ratio(t) for t in taus]
    cold_asym = np.ones_like(taus)
    hot_asym = 1.0 / (2.0 * taus)

    ax2.loglog(taus, kappas, 'b-', label='kappa(tau)')
    ax2.loglog(taus, cold_asym, 'r--', alpha=0.5, label='cold limit: 1')
    ax2.loglog(taus, hot_asym, 'g--', alpha=0.5, label='hot limit: 1/(2*tau)')
    ax2.set_xlabel('tau')
    ax2.set_ylabel('kappa')
    ax2.set_title('Kappa ratio asymptotics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = save_fig('deriv_kappa_validation.png')

    emit(f'![Kappa Validation]({fig_path})')
    emit()


# ─── Section 5: Small-tau Stability ───────────────────────────────────────

def section_small_tau():
    emit('## 5. Small-tau Stability')
    emit()
    emit('| T (keV) | tau | dsigma/dtau (pre) | rel_error | finite? |')
    emit('|---------|-----|-------------------|-----------|---------|')

    E = 1.0 * KEV
    Ep = 1.0 * KEV
    xi = 0.0

    for tau_kev in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        tau = tau_kev * KEV / ME_C2
        eng = ComptonKernelQuadrature(128, QuadratureForm.PreIBP)
        r = eng.dsigma_E_dtau(E, Ep, xi, tau, 1.0)
        finite = np.isfinite(r.value) and np.isfinite(r.estimated_rel_error)
        emit(f'| {tau_kev} | {tau:.4e} | {r.value:.6e} | '
             f'{r.estimated_rel_error:.2e} | {"yes" if finite else "NO"} |')

    emit()


# ─── Section 6: Derivative Spectral Profiles ─────────────────────────────

PROFILE_CONFIGS = [
    {'E_kev': 1.0,  'xi': 0.0,  'tau_kevs': [0.5, 1.0, 5.0],    'Ep_range': (0.3, 3.0)},
    {'E_kev': 10.0, 'xi': 0.0,  'tau_kevs': [1.0, 5.0, 20.0],   'Ep_range': (2.0, 30.0)},
    {'E_kev': 1.0,  'xi': 0.5,  'tau_kevs': [0.5, 1.0, 5.0],    'Ep_range': (0.2, 4.0)},
    {'E_kev': 5.0,  'xi': -0.5, 'tau_kevs': [1.0, 5.0, 20.0],   'Ep_range': (1.0, 20.0)},
]


def section_derivative_profiles():
    emit('## 6. Derivative Spectral Profiles')
    emit()
    emit('Spectral shape of $\\Sigma_E$ (top) and $\\partial\\Sigma_E/\\partial\\tau$ (bottom) '
         'as a function of scattered energy $E\'$ at fixed incident energy, angle, and '
         'several temperatures.  Pre-IBP, NL=256.')
    emit()

    engine = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)

    for cfg in PROFILE_CONFIGS:
        E_kev = cfg['E_kev']
        xi = cfg['xi']
        Ep_lo, Ep_hi = cfg['Ep_range']
        E = E_kev * KEV
        Ep_arr = np.linspace(Ep_lo, Ep_hi, 800) * KEV

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        for tau_kev in cfg['tau_kevs']:
            tau = tau_kev * KEV / ME_C2
            label = f'T = {tau_kev} keV'

            sig_vals = []
            dsig_vals = []
            for Ep in Ep_arr:
                sig_vals.append(engine.sigma_E(E, Ep, xi, tau, 1.0).value)
                dsig_vals.append(engine.dsigma_E_dtau(E, Ep, xi, tau, 1.0).value)

            ax1.semilogy(Ep_arr / KEV, np.abs(sig_vals), label=label)
            ax2.plot(Ep_arr / KEV, dsig_vals, label=label)

        ax1.set_ylabel('|$\\Sigma_E$|')
        ax1.set_title(f'E = {E_kev} keV, $\\xi$ = {xi}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("E' (keV)")
        ax2.set_ylabel('$\\partial\\Sigma_E / \\partial\\tau$')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f'deriv_profile_E{E_kev}_xi{xi}.png'
        fig_path = save_fig(fname)
        emit(f'![Derivative profile E={E_kev}, xi={xi}]({fig_path})')
        emit()


# ─── Section 7: Angular Distribution ─────────────────────────────────────

ANGULAR_CONFIGS = [
    {'E_kev': 1.0,  'Ep_kev': 1.0,  'tau_kevs': [0.5, 1.0, 5.0]},
    {'E_kev': 1.0,  'Ep_kev': 0.8,  'tau_kevs': [0.5, 1.0, 5.0]},
    {'E_kev': 10.0, 'Ep_kev': 10.0, 'tau_kevs': [1.0, 5.0, 20.0]},
    {'E_kev': 10.0, 'Ep_kev': 8.0,  'tau_kevs': [1.0, 5.0, 20.0]},
]


def section_angular_distribution():
    emit('## 7. Angular Distribution of the Derivative')
    emit()
    emit('$\\Sigma_E$ (top) and $\\partial\\Sigma_E/\\partial\\tau$ (bottom) as a function '
         'of scattering angle $\\xi = \\cos\\theta$ at fixed energies and several '
         'temperatures.  Pre-IBP, NL=256.')
    emit()

    engine = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)
    xi_arr = np.linspace(-0.99, 0.99, 600)

    for cfg in ANGULAR_CONFIGS:
        E_kev = cfg['E_kev']
        Ep_kev = cfg['Ep_kev']
        E = E_kev * KEV
        Ep = Ep_kev * KEV

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        for tau_kev in cfg['tau_kevs']:
            tau = tau_kev * KEV / ME_C2
            label = f'T = {tau_kev} keV'

            sig_vals = []
            dsig_vals = []
            for xi in xi_arr:
                sig_vals.append(engine.sigma_E(E, Ep, xi, tau, 1.0).value)
                dsig_vals.append(engine.dsigma_E_dtau(E, Ep, xi, tau, 1.0).value)

            ax1.semilogy(xi_arr, np.abs(sig_vals), label=label)
            ax2.plot(xi_arr, dsig_vals, label=label)

        ax1.set_ylabel('|$\\Sigma_E$|')
        ax1.set_title(f"E = {E_kev} keV, E' = {Ep_kev} keV")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('$\\xi = \\cos\\theta$')
        ax2.set_ylabel('$\\partial\\Sigma_E / \\partial\\tau$')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f'deriv_angular_E{E_kev}_Ep{Ep_kev}.png'
        fig_path = save_fig(fname)
        emit(f"![Angular distribution E={E_kev}, E'={Ep_kev}]({fig_path})")
        emit()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    emit('# Derivative Validation Report')
    emit()
    emit('Validation of `dsigma_E_dtau` (temperature derivative of the Compton kernel).')
    emit()

    section_gl_convergence()
    section_fd_comparison()
    section_pre_post_agreement()
    section_kappa_validation()
    section_small_tau()
    section_derivative_profiles()
    section_angular_distribution()

    md_path = os.path.join(GEN_DIR, 'derivative_validation.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
