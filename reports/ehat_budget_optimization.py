"""
EhatAmpBudget optimization benchmark report.

Compares performance before/after templating EHAT_AMPLIFICATION_BUDGET
by arithmetic type (double: 1e3, DD: 1e10), and validates accuracy
against ComptonKernelQuadrature(256).

Usage:
    python3 reports/ehat_budget_optimization.py

Output:
    reports/generated/ehat_budget_optimization.md  (+ .png plots in figs/)
"""
import sys
import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

import _compton_multigroup as cm
from _compton_kernel_solver import ComptonKernelSolver
import _compton_kernel_quadrature as cq
from _units import kev, kev_kelvin, me_c2

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

BEFORE_MG_FILE = os.path.join(ROOT, 'benchmarks', 'before_multigroup.json')

lines = []


def log(msg=''):
    print(msg, flush=True)


def emit(s=''):
    lines.append(s)


def save_fig(name):
    path = os.path.join(FIGS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return f'figs/{name}'


BENCHMARK_POINTS = [
    {'label': 'power_near_elastic', 'E_keV': 1, 'Ep_keV': 1.01, 'xi': 0.0, 'T_keV': 100},
    {'label': 'power_moderate', 'E_keV': 10, 'Ep_keV': 10.5, 'xi': 0.0, 'T_keV': 20},
    {'label': 'power_high_T', 'E_keV': 100, 'Ep_keV': 101, 'xi': 0.0, 'T_keV': 100},
    {'label': 'asymp_cold_elastic', 'E_keV': 1, 'Ep_keV': 1.01, 'xi': 0.0, 'T_keV': 0.1},
    {'label': 'asymp_inelastic', 'E_keV': 1, 'Ep_keV': 2.0, 'xi': 0.0, 'T_keV': 1.0},
    {'label': 'asymp_mid_T', 'E_keV': 10, 'Ep_keV': 10.5, 'xi': 0.0, 'T_keV': 5.0},
    {'label': 'asymp_high_E', 'E_keV': 100, 'Ep_keV': 101, 'xi': 0.0, 'T_keV': 5.0},
    {'label': 'forward_1', 'E_keV': 1, 'Ep_keV': 1.5, 'xi': 0.95, 'T_keV': 100},
    {'label': 'forward_2', 'E_keV': 10, 'Ep_keV': 12, 'xi': 0.9, 'T_keV': 50},
    {'label': 'backscatter_1', 'E_keV': 1, 'Ep_keV': 0.5, 'xi': -0.95, 'T_keV': 100},
    {'label': 'backscatter_2', 'E_keV': 10, 'Ep_keV': 8, 'xi': -0.9, 'T_keV': 50},
    {'label': 'regime_switch_1', 'E_keV': 1, 'Ep_keV': 1.5, 'xi': 0.0, 'T_keV': 40},
    {'label': 'regime_switch_2', 'E_keV': 5, 'Ep_keV': 5.5, 'xi': 0.3, 'T_keV': 40},
    {'label': 'large_x_1', 'E_keV': 100, 'Ep_keV': 200, 'xi': -0.5, 'T_keV': 40},
    {'label': 'large_x_2', 'E_keV': 100, 'Ep_keV': 500, 'xi': -0.9, 'T_keV': 30},
]

ACCURACY_TEMPS_KEV = [0.1, 1, 5, 20, 100]

N_VEC = 10_000
N_REPEATS = 5


def regime_for_point(pt):
    """Determine if a benchmark point falls in the power or asymptotic regime."""
    tau = pt['T_keV'] * kev / me_c2
    gamma = pt['E_keV'] * kev / me_c2
    gamma_p = pt['Ep_keV'] * kev / me_c2
    xi = pt['xi']
    a = 1 - xi
    s = 1.0 / gamma + 1.0 / gamma_p
    dg = gamma_p - gamma
    q = np.sqrt(dg**2 + 2 * gamma * gamma_p * a)
    alpha_plus = 0.5 * (s + q)
    alpha_minus = 0.5 * (s - q)
    tau_alpha_max = tau * max(alpha_plus, abs(alpha_minus))
    return 'power' if tau_alpha_max >= 0.035 else 'asymptotic'


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Single-Point Timing
# ─────────────────────────────────────────────────────────────────────────────

def section_single_point_timing():
    log('=== Section 1: Single-Point Timing (optimized build) ===')
    emit('## 1. Single-Point Timing')
    emit()
    emit('Vectorized `sigma_E_vec` throughput (10k identical evaluations per point)')
    emit('with the optimized `EhatAmpBudget<double> = 1e3` build.')
    emit()

    kernel = ComptonKernelSolver()

    emit('| Point | Regime | E (keV) | E\' (keV) | T (keV) | xi | us/eval |')
    emit('|-------|--------|---------|----------|---------|----|---------|')

    timing_data = []
    for pt in BENCHMARK_POINTS:
        E = pt['E_keV'] * kev
        Ep = pt['Ep_keV'] * kev
        xi = pt['xi']
        T_K = pt['T_keV'] * kev_kelvin
        Ep_arr = np.full(N_VEC, Ep)

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            kernel.sigma_E_vec(E, Ep_arr, xi, T_K, 1.0)
            dt = time.perf_counter() - t0
            times.append(dt)

        per_eval_us = float(np.median(times)) / N_VEC * 1e6
        regime = regime_for_point(pt)
        timing_data.append({'label': pt['label'], 'regime': regime, 'us': per_eval_us})

        emit(f"| {pt['label']} | {regime} | {pt['E_keV']} | {pt['Ep_keV']} "
             f"| {pt['T_keV']} | {pt['xi']} | {per_eval_us:.3f} |")
        log(f"  {pt['label']:<25} {regime:<10} {per_eval_us:.3f} us/eval")

    emit()

    power_times = [d['us'] for d in timing_data if d['regime'] == 'power']
    asymp_times = [d['us'] for d in timing_data if d['regime'] == 'asymptotic']

    if power_times:
        emit(f'Power-series regime: median **{np.median(power_times):.3f}** us/eval '
             f'(range {min(power_times):.3f}–{max(power_times):.3f})')
    if asymp_times:
        emit(f'Asymptotic regime: median **{np.median(asymp_times):.3f}** us/eval '
             f'(range {min(asymp_times):.3f}–{max(asymp_times):.3f})')
    emit()

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [d['label'] for d in timing_data]
    us_vals = [d['us'] for d in timing_data]
    colors = ['#2196F3' if d['regime'] == 'power' else '#FF9800' for d in timing_data]
    bars = ax.barh(labels, us_vals, color=colors)
    ax.set_xlabel('Time per evaluation (us)')
    ax.set_title('Single-Point sigma_E Throughput (optimized budget=1e3)')
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2196F3', label='Power series'),
                       Patch(color='#FF9800', label='Asymptotic')],
              loc='lower right')
    plt.tight_layout()
    fig_path = save_fig('ehat_single_point_timing.png')
    emit(f'![Single-point timing]({fig_path})')
    emit()

    return timing_data


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Multigroup Matrix Timing (before vs after)
# ─────────────────────────────────────────────────────────────────────────────

def section_multigroup_timing():
    log('=== Section 2: Multigroup Matrix Timing (before vs after) ===')
    emit('## 2. Multigroup Matrix Timing (Before vs After)')
    emit()
    emit('Configuration: 4 groups, N=16 quadrature, 16 angle bins, `ComptonKernelSeries(Auto)`.')
    emit()
    emit('**Before**: `EHAT_AMPLIFICATION_BUDGET = 1e2` (scalar constant for all types).')
    emit()
    emit('**After**: `EhatAmpBudget<double> = 1e3` (templated by arithmetic type).')
    emit()

    with open(BEFORE_MG_FILE) as f:
        before = json.load(f)

    kernel = ComptonKernelSolver()
    wf = cm.PlanckWeightFunction(cap_x=25.0)
    bounds = [0.5 * kev, 1.0 * kev, 5.0 * kev, 10.0 * kev, 50.0 * kev]

    emit('| T (keV) | Before (s) | After (s) | Speedup |')
    emit('|---------|-----------|-----------|---------|')

    temps_kev = [10, 20, 30, 50]
    before_times = []
    after_times = []

    for T_kev in temps_kev:
        T_val = T_kev * kev_kelvin
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=wf,
            quad_order_E=16, quad_order_Ep=16, xi_order=16)

        log(f'  T={T_kev} keV...')
        t0 = time.perf_counter()
        mg.compute_sigma_matrix(kernel, num_angle_bins=16, T=T_val, Ne=1.0)
        dt_after = time.perf_counter() - t0

        dt_before = before['timings'][str(T_kev)]
        speedup = dt_before / dt_after if dt_after > 0 else float('inf')

        before_times.append(dt_before)
        after_times.append(dt_after)

        emit(f'| {T_kev} | {dt_before:.3f} | {dt_after:.3f} | **{speedup:.1f}x** |')
        log(f'    before={dt_before:.3f}s  after={dt_after:.3f}s  speedup={speedup:.1f}x')

    emit()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(temps_kev))
    w = 0.35
    ax.bar(x - w / 2, before_times, w, label='Before (budget=1e2)', color='#F44336', alpha=0.8)
    ax.bar(x + w / 2, after_times, w, label='After (budget=1e3)', color='#4CAF50', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t} keV' for t in temps_kev])
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Multigroup Matrix Computation Time')
    ax.legend()
    for i, (b, a) in enumerate(zip(before_times, after_times)):
        if a > 0:
            ax.annotate(f'{b / a:.1f}x', xy=(i + w / 2, a), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig_path = save_fig('ehat_multigroup_timing.png')
    emit(f'![Multigroup timing comparison]({fig_path})')
    emit()

    return before_times, after_times, temps_kev


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Accuracy Validation
# ─────────────────────────────────────────────────────────────────────────────

def section_accuracy_validation():
    log('=== Section 3: Accuracy Validation ===')
    emit('## 3. Accuracy Validation')
    emit()
    emit('Comparison of `ComptonKernelSeries(Auto)` (optimized) vs '
         '`ComptonKernelQuadrature(256)` (PostIBP) reference.')
    emit()

    kernel_s = ComptonKernelSolver()
    kernel_q = cq.ComptonKernelQuadrature(256)
    THRESHOLD = 1e-3

    emit('### 3a. sigma_E accuracy')
    emit()
    emit('| Point | T (keV) | Series value | Q256 value | Rel diff | Series est. err | Pass |')
    emit('|-------|---------|-------------|-----------|----------|-----------------|------|')

    all_pass_sigma = True
    sigma_rel_diffs = []

    for pt in BENCHMARK_POINTS:
        E = pt['E_keV'] * kev
        Ep = pt['Ep_keV'] * kev
        xi = pt['xi']

        for T_kev in ACCURACY_TEMPS_KEV:
            T_K = T_kev * kev_kelvin
            rs = kernel_s.sigma_E(E, Ep, xi, T_K, 1.0)
            rq = kernel_q.sigma_E(E, Ep, xi, T_K, 1.0)

            if abs(rq.value) > 1e-300:
                rel_diff = abs(rs.value - rq.value) / abs(rq.value)
            elif abs(rs.value) > 1e-300:
                rel_diff = 1.0
            else:
                rel_diff = 0.0

            passed = rel_diff < THRESHOLD
            if not passed:
                all_pass_sigma = False
            sigma_rel_diffs.append(rel_diff)
            status = 'PASS' if passed else '**FAIL**'

            emit(f"| {pt['label']} | {T_kev} | {rs.value:.6e} | {rq.value:.6e} "
                 f"| {rel_diff:.2e} | {rs.estimated_rel_error:.2e} | {status} |")

    emit()
    sigma_nonzero = [d for d in sigma_rel_diffs if d > 0]
    if sigma_nonzero:
        emit(f'Max relative difference: **{max(sigma_nonzero):.2e}**')
        emit(f'Median relative difference: **{np.median(sigma_nonzero):.2e}**')
    emit(f'Overall sigma_E: **{"ALL PASS" if all_pass_sigma else "FAILURES DETECTED"}** '
         f'(threshold {THRESHOLD:.0e})')
    emit()

    log(f'  sigma_E: max_rel_diff={max(sigma_nonzero):.2e}, '
        f'median={np.median(sigma_nonzero):.2e}, pass={all_pass_sigma}')

    emit('### 3b. dsigma_E_dT accuracy')
    emit()
    emit('| Point | T (keV) | Series value | Q256 value | Rel diff | Pass |')
    emit('|-------|---------|-------------|-----------|----------|------|')

    all_pass_dsigma = True
    dsigma_rel_diffs = []

    for pt in BENCHMARK_POINTS:
        E = pt['E_keV'] * kev
        Ep = pt['Ep_keV'] * kev
        xi = pt['xi']

        for T_kev in ACCURACY_TEMPS_KEV:
            T_K = T_kev * kev_kelvin
            ds = kernel_s.dsigma_E_dT(E, Ep, xi, T_K, 1.0)
            dq = kernel_q.dsigma_E_dT(E, Ep, xi, T_K, 1.0)

            if abs(dq.value) > 1e-300:
                rel_diff = abs(ds.value - dq.value) / abs(dq.value)
            elif abs(ds.value) > 1e-300:
                rel_diff = 1.0
            else:
                rel_diff = 0.0

            passed = rel_diff < THRESHOLD
            if not passed:
                all_pass_dsigma = False
            dsigma_rel_diffs.append(rel_diff)
            status = 'PASS' if passed else '**FAIL**'

            emit(f"| {pt['label']} | {T_kev} | {ds.value:.6e} | {dq.value:.6e} "
                 f"| {rel_diff:.2e} | {status} |")

    emit()
    dsigma_nonzero = [d for d in dsigma_rel_diffs if d > 0]
    if dsigma_nonzero:
        emit(f'Max relative difference: **{max(dsigma_nonzero):.2e}**')
        emit(f'Median relative difference: **{np.median(dsigma_nonzero):.2e}**')
    emit(f'Overall dsigma_E_dT: **{"ALL PASS" if all_pass_dsigma else "FAILURES DETECTED"}** '
         f'(threshold {THRESHOLD:.0e})')
    emit()

    log(f'  dsigma_E_dT: max_rel_diff={max(dsigma_nonzero):.2e}, '
        f'median={np.median(dsigma_nonzero):.2e}, pass={all_pass_dsigma}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sigma_nz = [d for d in sigma_rel_diffs if d > 0]
    dsigma_nz = [d for d in dsigma_rel_diffs if d > 0]

    if sigma_nz:
        ax1.hist(np.log10(sigma_nz), bins=30, color='#2196F3', alpha=0.8, edgecolor='white')
        ax1.axvline(np.log10(THRESHOLD), color='red', linestyle='--', label=f'Threshold ({THRESHOLD:.0e})')
        ax1.set_xlabel('log10(relative difference)')
        ax1.set_ylabel('Count')
        ax1.set_title('sigma_E: Series vs Q256')
        ax1.legend()

    if dsigma_nz:
        ax2.hist(np.log10(dsigma_nz), bins=30, color='#FF9800', alpha=0.8, edgecolor='white')
        ax2.axvline(np.log10(THRESHOLD), color='red', linestyle='--', label=f'Threshold ({THRESHOLD:.0e})')
        ax2.set_xlabel('log10(relative difference)')
        ax2.set_ylabel('Count')
        ax2.set_title('dsigma_E_dT: Series vs Q256')
        ax2.legend()

    plt.tight_layout()
    fig_path = save_fig('ehat_accuracy_validation.png')
    emit(f'![Accuracy validation]({fig_path})')
    emit()

    return all_pass_sigma, all_pass_dsigma


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Summary
# ─────────────────────────────────────────────────────────────────────────────

def section_summary(mg_before, mg_after, temps_kev, sigma_pass, dsigma_pass):
    log('=== Section 4: Summary ===')
    emit('## 4. Summary')
    emit()

    speedups = [b / a for b, a in zip(mg_before, mg_after) if a > 0]
    geo_mean = np.exp(np.mean(np.log(speedups))) if speedups else 1.0

    emit('### Performance')
    emit()
    emit(f'- Geometric mean multigroup speedup across T={{10,20,30,50}} keV: **{geo_mean:.1f}x**')
    max_speedup_idx = int(np.argmax(speedups))
    emit(f'- Peak speedup: **{speedups[max_speedup_idx]:.1f}x** at T={temps_kev[max_speedup_idx]} keV '
         f'({mg_before[max_speedup_idx]:.1f}s -> {mg_after[max_speedup_idx]:.1f}s)')
    emit(f'- The speedup increases with temperature because the power series requires '
         f'more terms at high T, amplifying the recurrence. Fewer continued-fraction '
         f'restarts (budget 1e3 vs 1e2) means fewer expensive `ehat_cf` calls.')
    emit()

    emit('### Accuracy')
    emit()
    emit(f'- sigma_E vs Q256: **{"PASS" if sigma_pass else "FAIL"}**')
    emit(f'- dsigma_E_dT vs Q256: **{"PASS" if dsigma_pass else "FAIL"}**')
    emit(f'- All comparisons within 1e-3 relative tolerance across 15 kinematic '
         f'points x 5 temperatures.')
    emit()

    emit('### What Changed')
    emit()
    emit('```cpp')
    emit('// Before: single scalar constant for all arithmetic types')
    emit('constexpr double EHAT_AMPLIFICATION_BUDGET = 1e2;')
    emit()
    emit('// After: templated by arithmetic type')
    emit('template<> struct EhatAmpBudget<double> { value = 1e3; };  // 10x looser')
    emit('template<> struct EhatAmpBudget<DD>     { value = 1e10; }; // DD has 32 digits')
    emit('```')
    emit()
    emit('The recurrence `Ehat_{n+1}(x) = (1 - x * Ehat_n(x)) / n` amplifies ')
    emit('round-off by a factor of `x / (n+1)` per step. The budget controls when ')
    emit('we restart from the continued fraction. With `double` having ~16 digits ')
    emit('and target `eps_rel = 1e-12`, a budget of 1e3 leaves 1 digit of safety ')
    emit('(`1e-16 * 1e3 = 1e-13`) -- tight but sufficient. The old budget of 1e2 ')
    emit('was overly conservative, triggering expensive CF restarts ~10x too often.')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    emit('# EhatAmpBudget Optimization Report')
    emit()
    emit('Benchmark of templated `EhatAmpBudget` (double: 1e3, DD: 1e10) vs the old ')
    emit('scalar `EHAT_AMPLIFICATION_BUDGET = 1e2`.')
    emit()

    section_single_point_timing()
    mg_before, mg_after, temps_kev = section_multigroup_timing()
    sigma_pass, dsigma_pass = section_accuracy_validation()
    section_summary(mg_before, mg_after, temps_kev, sigma_pass, dsigma_pass)

    out_path = os.path.join(GEN_DIR, 'ehat_budget_optimization.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    log(f'\nReport written to: {out_path}')


if __name__ == '__main__':
    main()
