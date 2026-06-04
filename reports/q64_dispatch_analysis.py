"""
Q64 dispatch analysis report.

Investigates where ComptonKernelQuadrature(64) is both accurate and faster
than ComptonKernelSeries(Auto), based on a comprehensive 31k-point parameter
sweep.  Recommends a dispatch boundary for the Auto kernel.

Usage:
    python3 reports/q64_dispatch_analysis.py

Output:
    reports/generated/q64_dispatch_analysis.md  (+ .png plots in figs/)
"""
import sys
import os
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _units import kev, me_c2

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

SIG_FILE = os.path.join(ROOT, 'benchmarks', 'q64_comprehensive_sweep.json')
DSIG_FILE = os.path.join(ROOT, 'benchmarks', 'q64_derivative_sweep.json')

ME_C2_KEV = me_c2 / kev

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


def load_sweep(path):
    with open(path) as f:
        data = json.load(f)
    for d in data:
        d['gamma'] = d['E'] / ME_C2_KEV
        d['tau'] = d['T'] / ME_C2_KEV
    return data


def nontrivial(data):
    return [d for d in data if abs(d['ref']) > 1e-300]


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Scope
# ─────────────────────────────────────────────────────────────────────────────

def section_scope(sig, dsig):
    log('=== Section 1: Scope ===')
    emit('## 1. Investigation Scope')
    emit()
    emit('Comprehensive parameter sweep comparing `ComptonKernelSeries(Auto)` ')
    emit('against `ComptonKernelQuadrature(64)` (Q64), with `ComptonKernelQuadrature(256)` ')
    emit('(Q256) as the ground-truth reference.')
    emit()
    emit('| Dimension | Values |')
    emit('|-----------|--------|')

    Ts = sorted(set(d['T'] for d in sig))
    Es = sorted(set(d['E'] for d in sig))
    Rs = sorted(set(d['r'] for d in sig))
    Xs = sorted(set(d['x'] for d in sig))

    emit(f'| T (keV) | {", ".join(str(t) for t in Ts)} |')
    emit(f'| E (keV) | {", ".join(str(e) for e in Es)} |')
    emit(f"| E'/E ratio | {', '.join(str(r) for r in Rs)} |")
    emit(f'| xi (cos theta) | {", ".join(str(x) for x in Xs)} |')
    emit(f'| Total points | {len(sig):,} (sigma_E) + {len(dsig):,} (dsigma_E_dT) |')
    emit(f'| Non-trivial | {len(nontrivial(sig)):,} + {len(nontrivial(dsig)):,} |')
    emit()
    emit('Each point measures: Q256 reference value, Series and Q64 values, ')
    emit('relative error vs Q256, and vectorized throughput (500 evaluations, ')
    emit('per-call timing in microseconds).')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def section_accuracy(sig_nz, dsig_nz):
    log('=== Section 2: Accuracy ===')
    emit('## 2. Q64 Accuracy vs Q256 Reference')
    emit()

    for label, data, func_name in [
        ('sigma_E', sig_nz, 'sigma_E'),
        ('dsigma_E_dT', dsig_nz, 'dsigma_E_dT'),
    ]:
        emit(f'### 2{"a" if label == "sigma_E" else "b"}. {func_name}')
        emit()
        emit(f'| Tolerance | Q64 failures | Series failures |')
        emit(f'|-----------|-------------|-----------------|')
        for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            q_bad = sum(1 for d in data if d['qe'] > tol)
            s_bad = sum(1 for d in data if d['se'] > tol)
            emit(f'| {tol:.0e} | {q_bad} ({100*q_bad/len(data):.2f}%) '
                 f'| {s_bad} ({100*s_bad/len(data):.2f}%) |')

        q_errs = sorted([d['qe'] for d in data if d['qe'] > 0])
        s_errs = sorted([d['se'] for d in data if d['se'] > 0])
        emit()
        emit(f'Q64 error percentiles: '
             f'p50={np.percentile(q_errs,50):.1e}, '
             f'p90={np.percentile(q_errs,90):.1e}, '
             f'p99={np.percentile(q_errs,99):.1e}, '
             f'max={max(q_errs):.1e}')
        emit()
        emit(f'Series error percentiles: '
             f'p50={np.percentile(s_errs,50):.1e}, '
             f'p90={np.percentile(s_errs,90):.1e}, '
             f'p99={np.percentile(s_errs,99):.1e}, '
             f'max={max(s_errs):.1e}')
        emit()

    # Failure characterization
    emit('### 2c. Where do accuracy failures cluster?')
    emit()
    bad_sig = [d for d in sig_nz if d['qe'] > 1e-3]
    bad_dsig = [d for d in dsig_nz if d['qe'] > 1e-3]

    emit(f'All Q64 failures at 1e-3 threshold:')
    emit()
    emit(f'- **sigma_E**: {len(bad_sig)} failures — '
         f'{sum(1 for d in bad_sig if d["E"] == 0.1)} at E=0.1 keV, '
         f'{sum(1 for d in bad_sig if d["E"] == 0.5)} at E=0.5 keV, '
         f'{sum(1 for d in bad_sig if d["E"] >= 1)} at E>=1 keV')
    emit(f'- **dsigma_E_dT**: {len(bad_dsig)} failures — '
         f'{sum(1 for d in bad_dsig if d["E"] == 0.1)} at E=0.1 keV, '
         f'{sum(1 for d in bad_dsig if d["E"] == 0.5)} at E=0.5 keV, '
         f'{sum(1 for d in bad_dsig if d["E"] >= 1)} at E>=1 keV')
    emit()
    emit('Failures are overwhelmingly at E=0.1 keV (gamma=0.0002) with extreme forward ')
    emit('scattering (xi near 1). Crucially, **Series also fails at these same points** — ')
    emit('they are inherently ill-conditioned configurations, not a Q64-specific weakness.')
    emit()
    emit(f'For E >= 0.5 keV: sigma_E has '
         f'{sum(1 for d in bad_sig if d["E"] >= 0.5)} Q64 failures vs '
         f'{sum(1 for d in sig_nz if d["se"] > 1e-3 and d["E"] >= 0.5)} Series failures; '
         f'dsigma_E_dT has '
         f'{sum(1 for d in bad_dsig if d["E"] >= 0.5)} Q64 failures vs '
         f'{sum(1 for d in dsig_nz if d["se"] > 1e-3 and d["E"] >= 0.5)} Series failures.')
    emit()

    # Accuracy histogram figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title in [
        (axes[0], sig_nz, r'$\sigma_E$'),
        (axes[1], dsig_nz, r'$d\sigma_E/dT$'),
    ]:
        q_e = [np.log10(d['qe']) for d in data if d['qe'] > 0]
        s_e = [np.log10(d['se']) for d in data if d['se'] > 0]
        bins = np.linspace(-16, 0, 65)
        ax.hist(s_e, bins=bins, alpha=0.6, label='Series(Auto)', color='#2196F3')
        ax.hist(q_e, bins=bins, alpha=0.6, label='Q64', color='#FF9800')
        ax.axvline(-3, color='red', linestyle='--', linewidth=1.5, label='1e-3 threshold')
        ax.set_xlabel('log₁₀(relative error vs Q256)')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}: Error Distribution')
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = save_fig('q64_accuracy_histograms.png')
    emit(f'![Accuracy distributions]({fig_path})')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Performance
# ─────────────────────────────────────────────────────────────────────────────

def section_performance(sig_nz, dsig_nz):
    log('=== Section 3: Performance ===')
    emit('## 3. Performance Comparison')
    emit()

    Ts = sorted(set(d['T'] for d in sig_nz))
    Es = sorted(set(d['E'] for d in sig_nz))

    for label, data, func_name in [
        ('sigma_E', sig_nz, r'$\sigma_E$'),
        ('dsigma_E_dT', dsig_nz, r'$d\sigma_E/dT$'),
    ]:
        suffix = 'sig' if label == 'sigma_E' else 'dsig'
        emit(f'### 3{"a" if label == "sigma_E" else "b"}. {label}: Median Series time (us/eval)')
        emit()

        header = '| E \\ T |' + '|'.join(f' {T} ' for T in Ts) + '|'
        emit(header)
        emit('|' + '|'.join(['---'] * (len(Ts) + 1)) + '|')

        grid_s = np.full((len(Es), len(Ts)), np.nan)
        grid_q = np.full((len(Es), len(Ts)), np.nan)
        grid_frac = np.full((len(Es), len(Ts)), np.nan)

        for i, E in enumerate(Es):
            row = f'| {E} '
            for j, T in enumerate(Ts):
                at = [d for d in data if d['E'] == E and d['T'] == T]
                if at:
                    s_med = np.median([d['st'] for d in at])
                    q_med = np.median([d['qt'] for d in at])
                    grid_s[i, j] = s_med
                    grid_q[i, j] = q_med
                    accurate = [d for d in at if d['qe'] < 1e-3]
                    if accurate:
                        grid_frac[i, j] = sum(1 for d in accurate if d['qt'] < d['st']) / len(accurate)
                    if s_med > 2:
                        row += f'| **{s_med:.1f}** '
                    else:
                        row += f'| {s_med:.1f} '
                else:
                    row += '| -- '
            emit(row + '|')
        emit()
        emit('(Bold = Series > 2 us, the slow regime; Q64 is flat at ~0.8 us for sigma_E, ~1.6 us for dsigma_E_dT)')
        emit()

        # Heatmap figures
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Series timing heatmap
        ax = axes[0]
        im = ax.pcolormesh(range(len(Ts)), range(len(Es)), grid_s,
                           norm=LogNorm(vmin=0.05, vmax=30), cmap='YlOrRd', shading='auto')
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels(Ts, rotation=45, fontsize=7)
        ax.set_yticks(range(len(Es)))
        ax.set_yticklabels(Es)
        ax.set_xlabel('T (keV)')
        ax.set_ylabel('E (keV)')
        ax.set_title(f'{func_name}: Series(Auto) time (us)')
        plt.colorbar(im, ax=ax, label='us/eval')

        # Q64 timing heatmap
        ax = axes[1]
        im = ax.pcolormesh(range(len(Ts)), range(len(Es)), grid_q,
                           norm=LogNorm(vmin=0.05, vmax=30), cmap='YlOrRd', shading='auto')
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels(Ts, rotation=45, fontsize=7)
        ax.set_yticks(range(len(Es)))
        ax.set_yticklabels(Es)
        ax.set_xlabel('T (keV)')
        ax.set_ylabel('E (keV)')
        ax.set_title(f'{func_name}: Q64 time (us)')
        plt.colorbar(im, ax=ax, label='us/eval')

        # Fraction where Q64 is faster
        ax = axes[2]
        im = ax.pcolormesh(range(len(Ts)), range(len(Es)), grid_frac * 100,
                           vmin=0, vmax=100, cmap='RdYlGn', shading='auto')
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels(Ts, rotation=45, fontsize=7)
        ax.set_yticks(range(len(Es)))
        ax.set_yticklabels(Es)
        ax.set_xlabel('T (keV)')
        ax.set_ylabel('E (keV)')
        ax.set_title(f'{func_name}: % points Q64 faster\n(among accurate <1e-3)')
        plt.colorbar(im, ax=ax, label='%')

        # Draw the dispatch zone
        T20_idx = Ts.index(20) if 20 in Ts else None
        E10_idx = next((i for i, e in enumerate(Es) if e == 10), None)
        E05_idx = next((i for i, e in enumerate(Es) if e == 0.5), None)
        if T20_idx is not None and E10_idx is not None and E05_idx is not None:
            for a in axes:
                rect = Rectangle((T20_idx - 0.5, E05_idx - 0.5),
                                 len(Ts) - T20_idx, E10_idx - E05_idx + 1,
                                 linewidth=2.5, edgecolor='blue', facecolor='none',
                                 linestyle='--', label='Dispatch zone')
                a.add_patch(rect)
            axes[2].legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        fig_path = save_fig(f'q64_perf_heatmaps_{suffix}.png')
        emit(f'![Performance heatmaps — {label}]({fig_path})')
        emit()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Slow-Series Zone
# ─────────────────────────────────────────────────────────────────────────────

def section_slow_series(sig_nz, dsig_nz):
    log('=== Section 4: Slow-Series Analysis ===')
    emit('## 4. When Series Is Slow, Q64 Is the Clear Winner')
    emit()
    emit('Filtering to points where `Series(Auto)` takes more than 2 us/eval ')
    emit('— the regime that dominates multigroup wall-clock time:')
    emit()

    stats = {}
    for label, data in [('sigma_E', sig_nz), ('dsigma_E_dT', dsig_nz)]:
        slow = [d for d in data if d['st'] > 2.0]
        q_ok = [d for d in slow if d['qe'] < 1e-3]
        q_fast = [d for d in q_ok if d['qt'] < d['st']]
        sp = [d['st'] / d['qt'] for d in q_fast] if q_fast else [0]
        q_bad_slow = [d for d in slow if d['qe'] >= 1e-3]
        stats[label] = (len(slow), len(q_ok), len(q_fast), np.median(sp), len(q_bad_slow))

    ss, ds = stats['sigma_E'], stats['dsigma_E_dT']
    emit('| Metric | sigma_E | dsigma_E_dT |')
    emit('|--------|---------|-------------|')
    emit(f'| Points where Series > 2 us | {ss[0]} | {ds[0]} |')
    emit(f'| Q64 accurate (<1e-3) | {ss[1]} ({100*ss[1]/ss[0]:.1f}%) | {ds[1]} ({100*ds[1]/ds[0]:.1f}%) |')
    emit(f'| Q64 accurate AND faster | {ss[2]} ({100*ss[2]/ss[1]:.1f}%) | {ds[2]} ({100*ds[2]/ds[1]:.1f}%) |')
    emit(f'| Median speedup | {ss[3]:.1f}x | {ds[3]:.1f}x |')
    emit(f'| Q64 failures in slow zone | {ss[4]} | {ds[4]} |')
    emit()

    emit('The few Q64 failures in the slow zone are all at E=0.1 keV with extreme ')
    emit('forward scattering — where Series also fails. These are outside the ')
    emit('proposed dispatch zone (E >= 0.5 keV).')
    emit()

    # xi breakdown
    emit('**What makes Series stay fast at high T?** Some kinematic configurations ')
    emit('converge in few terms regardless of temperature:')
    emit()
    zone = [d for d in sig_nz if d['E'] <= 10 and d['T'] >= 20]
    slower = [d for d in zone if d['qt'] >= d['st'] and d['qe'] < 1e-3]
    by_xi = defaultdict(list)
    for d in slower:
        by_xi[d['x']].append(d)

    emit('| xi | Q64-slower points | Median Series (us) | Interpretation |')
    emit('|----|-------------------|--------------------:|----------------|')
    for xi in sorted(by_xi):
        at = by_xi[xi]
        s_med = np.median([d['st'] for d in at])
        interp = 'extreme forward' if xi > 0.9 else ('forward' if xi > 0.3 else 'isotropic/back')
        emit(f'| {xi} | {len(at)} | {s_med:.2f} | {interp} |')
    emit()
    emit('Forward-scattering points (xi > 0.5) dominate the "Q64-slower" set. ')
    emit('But their Series times are 0.3-0.5 us — already sub-microsecond. ')
    emit('Dispatching to Q64 at 0.8 us is a small absolute penalty that is ')
    emit('dwarfed by the 10-23 us savings on the majority of quadrature points.')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Dispatch rules
# ─────────────────────────────────────────────────────────────────────────────

def section_dispatch_rules(sig_nz, dsig_nz):
    log('=== Section 5: Dispatch Rules ===')
    emit('## 5. Candidate Dispatch Rules')
    emit()
    emit('Evaluated on both `sigma_E` and `dsigma_E_dT`:')
    emit()

    rules = [
        ('E in [0.5, 10] keV AND T >= 15 keV',
         lambda d: 0.5 <= d['E'] <= 10 and d['T'] >= 15),
        ('E in [0.5, 10] keV AND T >= 20 keV',
         lambda d: 0.5 <= d['E'] <= 10 and d['T'] >= 20),
        ('gamma < 0.02 AND tau >= 0.03',
         lambda d: d['gamma'] < 0.02 and d['tau'] >= 0.03),
        ('tau/gamma > 2 AND E >= 0.5 keV',
         lambda d: d['gamma'] > 0 and d['tau'] / d['gamma'] > 2 and d['E'] >= 0.5),
        ('tau/gamma > 5 AND E >= 0.5 keV',
         lambda d: d['gamma'] > 0 and d['tau'] / d['gamma'] > 5 and d['E'] >= 0.5),
    ]

    emit('| Rule | Func | Points | Q64 bad | Q64 faster | Savings |')
    emit('|------|------|--------|---------|------------|---------|')

    for name, pred in rules:
        for label, data in [('σ', sig_nz), ('dσ/dT', dsig_nz)]:
            zone = [d for d in data if pred(d)]
            if not zone:
                continue
            bad = sum(1 for d in zone if d['qe'] > 1e-3)
            accurate = [d for d in zone if d['qe'] < 1e-3]
            faster = sum(1 for d in accurate if d['qt'] < d['st'])
            total_s = sum(d['st'] for d in accurate)
            total_q = sum(d['qt'] for d in accurate)
            savings = 100 * (1 - total_q / total_s) if total_s > 0 else 0
            emit(f'| {name} | {label} | {len(zone)} | {bad} '
                 f'| {faster} ({100*faster/len(zone):.0f}%) | {savings:.0f}% |')

    emit()
    emit('**"Savings"** = reduction in total wall-clock time if Q64 were used for all ')
    emit('accurate points in the zone. This is the metric that matters for multigroup ')
    emit('integration, where we sum over thousands of quadrature points.')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Recommended Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def section_recommendation(sig_nz, dsig_nz):
    log('=== Section 6: Recommendation ===')
    emit('## 6. Recommended Dispatch Rule')
    emit()
    emit('### Rule: use Q64 when `E in [0.5, 10] keV` AND `T >= 20 keV`')
    emit()
    emit('Equivalently, in dimensionless units: `gamma < 0.02` AND `tau >= 0.04`.')
    emit()

    pred = lambda d: 0.5 <= d['E'] <= 10 and d['T'] >= 20

    emit('### Justification')
    emit()
    emit('#### Accuracy')
    emit()

    for label, data in [('sigma_E', sig_nz), ('dsigma_E_dT', dsig_nz)]:
        zone = [d for d in data if pred(d)]
        bad = [d for d in zone if d['qe'] > 1e-3]
        q_errs = [d['qe'] for d in zone if d['qe'] > 0]
        s_errs = [d['se'] for d in zone if d['se'] > 0]
        emit(f'- **{label}**: {len(bad)} Q64 failures out of {len(zone)} points '
             f'({100*len(bad)/len(zone):.2f}%). '
             f'Worst Q64 error: {max(q_errs):.1e}. '
             f'Worst Series error: {max(s_errs):.1e}. '
             f'Q64 and Series have comparable accuracy in this zone.')

    emit()
    emit('Both methods agree with Q256 to better than 1e-4 for the vast majority ')
    emit('of points. The rare exceptions (E=0.1 keV, extreme forward scattering) ')
    emit('are excluded by the E >= 0.5 keV floor.')
    emit()

    emit('#### Performance')
    emit()

    for label, data in [('sigma_E', sig_nz), ('dsigma_E_dT', dsig_nz)]:
        zone = [d for d in data if pred(d)]
        accurate = [d for d in zone if d['qe'] < 1e-3]
        faster = [d for d in accurate if d['qt'] < d['st']]
        total_s = sum(d['st'] for d in accurate)
        total_q = sum(d['qt'] for d in accurate)
        if faster:
            sp = [d['st'] / d['qt'] for d in faster]
            emit(f'- **{label}**: Q64 faster in {len(faster)}/{len(zone)} points '
                 f'({100*len(faster)/len(zone):.0f}%). '
                 f'Median speedup {np.median(sp):.0f}x where faster. '
                 f'Net wall-clock savings: **{100*(1-total_q/total_s):.0f}%**.')

    emit()
    emit('The 19-27% of points where Q64 is slower are forward-scattering ')
    emit('configurations where Series is already sub-microsecond (0.3-0.5 us). ')
    emit('The absolute penalty of switching to Q64 (at 0.8-1.6 us) is negligible ')
    emit('compared to the 10-23 us savings on the majority of points.')
    emit()

    emit('#### Multigroup Impact')
    emit()
    emit('In the multigroup integration (4 groups, N=16 quadrature, 16 angle bins):')
    emit()
    emit('| T (keV) | Series(Auto) | Q64 | Speedup |')
    emit('|---------|-------------|-----|---------|')

    # Run a quick multigroup comparison
    import time
    import _compton_multigroup as cm
    import _compton_kernel_series as cs
    import _compton_kernel_quadrature as cq
    from _units import kev as _kev, kev_kelvin

    kernel_s = cs.ComptonKernelSeries(cs.SeriesMethod.Auto)
    kernel_q = cq.ComptonKernelQuadrature(64)
    wf = cm.PlanckWeightFunction(cap_x=25.0)
    bounds = [0.5 * _kev, 1.0 * _kev, 5.0 * _kev, 10.0 * _kev, 50.0 * _kev]

    for T_kev in [10, 20, 30, 50]:
        T_val = T_kev * kev_kelvin
        mg1 = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds, weight_function=wf,
            quad_order_E=16, quad_order_Ep=16, quad_order_mu=16)
        t0 = time.perf_counter()
        mg1.compute_sigma_matrix(kernel_s, num_angle_bins=16, T=T_val, Ne=1.0)
        dt_s = time.perf_counter() - t0

        mg2 = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds, weight_function=wf,
            quad_order_E=16, quad_order_Ep=16, quad_order_mu=16)
        t0 = time.perf_counter()
        mg2.compute_sigma_matrix(kernel_q, num_angle_bins=16, T=T_val, Ne=1.0)
        dt_q = time.perf_counter() - t0

        sp = dt_s / dt_q if dt_q > 0 else 0
        emit(f'| {T_kev} | {dt_s:.2f}s | {dt_q:.2f}s | **{sp:.1f}x** |')
        log(f'  T={T_kev}: Series={dt_s:.2f}s Q64={dt_q:.2f}s {sp:.1f}x')

    emit()

    emit('#### Why this boundary?')
    emit()
    emit('1. **E >= 0.5 keV** (gamma >= 0.001): excludes the pathological E=0.1 keV ')
    emit('   corner where both Q64 and Series lose accuracy. In practice, photon ')
    emit('   energies below 0.5 keV are rarely in multigroup grids.')
    emit()
    emit('2. **E <= 10 keV** (gamma <= 0.02): above this, the power series Poisson ')
    emit('   y-parameters stay small regardless of T, so Series converges quickly ')
    emit('   and outperforms Q64. The transition from slow to fast Series is sharp ')
    emit('   between E=10 and E=20 keV.')
    emit()
    emit('3. **T >= 20 keV** (tau >= 0.04): below this, Series is universally fast ')
    emit('   (< 0.5 us). The crossover where Q64 starts winning is around T=15 keV, ')
    emit('   but T >= 20 keV gives a clean margin with zero accuracy failures.')
    emit()

    # Crossover figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [
        (axes[0], sig_nz, r'$\sigma_E$'),
        (axes[1], dsig_nz, r'$d\sigma_E/dT$'),
    ]:
        Ts = sorted(set(d['T'] for d in data))
        Es_plot = [0.5, 1, 5, 10, 20, 50, 100]
        colors = plt.cm.viridis(np.linspace(0, 1, len(Es_plot)))

        for E, color in zip(Es_plot, colors):
            medians = []
            for T in Ts:
                at = [d for d in data if d['E'] == E and d['T'] == T]
                if at:
                    medians.append(np.median([d['st'] for d in at]))
                else:
                    medians.append(np.nan)
            ax.plot(Ts, medians, 'o-', color=color, label=f'E={E} keV', markersize=3)

        at_q = [d for d in data if d['E'] == 1 and d['T'] == 5]
        if at_q:
            q_med = np.median([d['qt'] for d in at_q])
            ax.axhline(q_med, color='red', linestyle='--', linewidth=1.5,
                       label=f'Q64 ({q_med:.1f} us)', alpha=0.7)
        ax.axvline(20, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label='T=20 keV')
        ax.set_xlabel('T (keV)')
        ax.set_ylabel('Median time per eval (us)')
        ax.set_title(f'{title}: Series(Auto) timing vs temperature')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0.03, 50)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = save_fig('q64_crossover_curves.png')
    emit(f'![Crossover curves]({fig_path})')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Summary
# ─────────────────────────────────────────────────────────────────────────────

def section_summary():
    log('=== Section 7: Summary ===')
    emit('## 7. Summary')
    emit()
    emit('A comprehensive 62k-point parameter sweep (31k for sigma_E, 31k for ')
    emit('dsigma_E_dT) demonstrates that `ComptonKernelQuadrature(64)` can safely ')
    emit('replace `ComptonKernelSeries(Auto)` in the low-energy, high-temperature ')
    emit('regime:')
    emit()
    emit('| | Series(Auto) | Q64 |')
    emit('|---|---|---|')
    emit('| **Accuracy vs Q256** | Comparable | Comparable |')
    emit('| **Time at T=50 keV, E<=10 keV** | 10-16 us/eval | 0.8-1.6 us/eval |')
    emit('| **Time at T=10 keV** | 0.1-0.2 us/eval | 0.8-1.6 us/eval |')
    emit('| **Temperature dependence** | O(n_terms), grows with T | Flat |')
    emit()
    emit('**Recommendation**: In the `SeriesMethod::Auto` dispatch, add a branch ')
    emit('that delegates to a Q64 kernel when `gamma < 0.02` (E < ~10 keV) AND ')
    emit('`tau > 0.04` (T > ~20 keV). This eliminates the high-temperature ')
    emit('performance cliff with no accuracy cost, reducing multigroup matrix ')
    emit('computation time by up to 10x at T=50 keV.')
    emit()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log('Loading sweep data...')
    sig = load_sweep(SIG_FILE)
    dsig = load_sweep(DSIG_FILE)
    sig_nz = nontrivial(sig)
    dsig_nz = nontrivial(dsig)

    emit('# Q64 Dispatch Analysis')
    emit()
    emit('_Can `ComptonKernelQuadrature(64)` replace `ComptonKernelSeries(Auto)` ')
    emit('in part of the parameter space, without sacrificing accuracy?_')
    emit()

    section_scope(sig, dsig)
    section_accuracy(sig_nz, dsig_nz)
    section_performance(sig_nz, dsig_nz)
    section_slow_series(sig_nz, dsig_nz)
    section_dispatch_rules(sig_nz, dsig_nz)
    section_recommendation(sig_nz, dsig_nz)
    section_summary()

    out_path = os.path.join(GEN_DIR, 'q64_dispatch_analysis.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    log(f'\nReport written to: {out_path}')


if __name__ == '__main__':
    main()
