"""
Part 1 – Deterministic vs Monte-Carlo sigma-matrix comparison.

Sweeps 3 weight functions x 6 energy grids x 25 temperatures.
MC uses 10 seeds at 5 M samples each; results are reported with
mean +/- 2 sigma error bars.

Usage:
    python3 reports/mega_validation/part1_sigma_comparison.py [--plot-tier standard] [--no-cache]

Output:
    reports/generated/mega_val_part1_sigma.md
    reports/generated/figs/mega_val_p1/*.png
    reports/generated/cache/mega_val_p1/*.npz
"""
import os
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'cpp_modules'))

from common import (
    GRIDS, TEMPS_KEV, WEIGHT_SPECS, MC_SEEDS, MC_SAMPLES,
    KERNEL, KEV, KEV_KELVIN, SIGMA_T,
    make_det, make_mc, run_mc_ensemble,
    mixed_error, error_stats, row_sums,
    cache_key, save_checkpoint, load_checkpoint,
    TimingLog, progress, emit, save_fig, write_report,
    base_arg_parser, FailureTracker, is_representative, make_weight,
)

PART = 1
GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs', 'mega_val_p1')
CACHE_DIR = os.path.join(GEN_DIR, 'cache', 'mega_val_p1')
REPORT_FILE = 'mega_val_part1_sigma.md'
ABS_FLOOR = SIGMA_T * 1e-10

# ── plotting helpers ──────────────────────────────────────────────────────

def plot_pomraning(S_det, S_mc_mean, S_mc_2sig, centers_kev, T_kev,
                   grid_tag, wname):
    """Pomraning-style profiles for the top-5 source groups."""
    G = S_det.shape[0]
    rs = np.abs(S_det).sum(axis=1)
    top5 = np.argsort(rs)[-5:][::-1]
    fig, axes = plt.subplots(len(top5), 1, figsize=(8, 3 * len(top5)),
                             sharex=True, squeeze=False)
    for idx, g in enumerate(top5):
        ax = axes[idx, 0]
        ax.plot(centers_kev, S_det[g] / SIGMA_T, 'k-', lw=1.2, label='det')
        ax.errorbar(centers_kev, S_mc_mean[g] / SIGMA_T,
                    yerr=S_mc_2sig[g] / SIGMA_T,
                    fmt='o', ms=3, capsize=2, color='C0', label='MC±2σ')
        ax.set_ylabel(f'σ(g={g}→g\') / σ_T')
        ax.legend(fontsize=7)
        ax.set_yscale('symlog', linthresh=1e-8)
    axes[-1, 0].set_xlabel('E\' (keV)')
    axes[-1, 0].set_xscale('log')
    fig.suptitle(f'{grid_tag}  T={T_kev:.4g} keV  {wname}', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_error_heatmap(err, centers_kev, T_kev, grid_tag, wname):
    """Element-wise relative-error heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = max(err[err > 0].min(), 1e-6) if np.any(err > 0) else 1e-6
    im = ax.pcolormesh(centers_kev, centers_kev, err,
                       norm=LogNorm(vmin=vmin, vmax=max(err.max(), 1e-1)),
                       cmap='inferno')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E\' (keV)')
    ax.set_ylabel('E (keV)')
    ax.set_title(f'|det−MC|/max(|det|,floor)  {grid_tag} T={T_kev:.4g} {wname}',
                 fontsize=9)
    fig.colorbar(im, ax=ax, label='relative error')
    fig.tight_layout()
    return fig


# ── main loop ─────────────────────────────────────────────────────────────

def main():
    args = base_arg_parser('Part 1: sigma comparison').parse_args()
    plot_tier = args.plot_tier
    use_cache = not args.no_cache

    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    total_cases = len(GRIDS) * len(TEMPS_KEV) * len(WEIGHT_SPECS)
    tracker = FailureTracker(total_cases)
    tlog = TimingLog()
    lines: list[str] = []
    summary_rows: list[dict] = []
    t_start = time.time()

    emit(lines, '# Part 1 – Sigma Matrix: Deterministic vs Monte-Carlo')
    emit(lines)
    emit(lines, '## Configuration')
    emit(lines)
    emit(lines, f'- Temperatures: {len(TEMPS_KEV)} points, '
         f'{TEMPS_KEV[0]:.1e} – {TEMPS_KEV[-1]:.1e} keV')
    emit(lines, f'- MC seeds: {len(MC_SEEDS)}, samples/seed: {MC_SAMPLES:,}')
    emit(lines, f'- Plot tier: {plot_tier}')
    emit(lines, f'- Det config: base_order=24, peak_max_depth=5, tol=1e-3')
    emit(lines)

    # grid table
    emit(lines, '### Energy grids')
    emit(lines)
    emit(lines, '| Tag | Groups | Name |')
    emit(lines, '|-----|--------|------|')
    for g in GRIDS:
        ngrp = len(g['bounds_kev']) - 1
        emit(lines, f"| {g['tag']} | {ngrp} | {g['name']} |")
    emit(lines)

    # ── per-weight sections ───────────────────────────────────────────────

    case_num = 0
    for wname, wf_factory in WEIGHT_SPECS:
        emit(lines, f'## Weight: {wname}')
        emit(lines)
        emit(lines, '| Grid | T (keV) | max err | mean err | p95 err '
             '| det (s) | MC (s) |')
        emit(lines, '|------|---------|---------|----------|---------|'
             '---------|--------|')

        for grid in GRIDS:
            gtag = grid['tag']
            bkev = grid['bounds_kev']
            G = len(bkev) - 1
            centers_kev = np.sqrt(bkev[:-1] * bkev[1:])

            for T_kev in TEMPS_KEV:
                case_num += 1
                T_K = float(T_kev * KEV_KELVIN)
                label = f'{gtag} T={T_kev:.4g} {wname}'
                ck = cache_key(CACHE_DIR, gtag, T_kev, wname)

                cached = load_checkpoint(ck) if use_cache else None
                if cached is not None:
                    S_det = cached['S_det']
                    mc_mean = cached['mc_mean']
                    mc_2sig = cached['mc_2sig']
                    dt_det = float(cached['dt_det'])
                    dt_mc = float(cached['dt_mc'])
                    progress(PART, f'[{case_num}/{total_cases}] {label} [cached]')
                else:
                    progress(PART, f'[{case_num}/{total_cases}] {label}')
                    try:
                        wf = wf_factory()

                        with tlog('det'):
                            t0 = time.perf_counter()
                            det_obj = make_det(bkev, wf)
                            S_det = np.array(det_obj.compute_sigma_matrix(
                                KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
                            dt_det = time.perf_counter() - t0

                        with tlog('mc'):
                            t0 = time.perf_counter()
                            mc_mean, mc_std, mc_2sig, _ = run_mc_ensemble(
                                bkev, wf, T_K, MC_SAMPLES, MC_SEEDS, 'sigma')
                            dt_mc = time.perf_counter() - t0

                        save_checkpoint(ck,
                                        S_det=S_det, mc_mean=mc_mean,
                                        mc_2sig=mc_2sig,
                                        dt_det=np.float64(dt_det),
                                        dt_mc=np.float64(dt_mc))

                    except Exception as exc:
                        tracker.record(gtag, T_kev, wname, exc)
                        traceback.print_exc()
                        continue

                err, _ = mixed_error(mc_mean, S_det, ABS_FLOOR)
                mx, mn, md, p95 = error_stats(err)
                summary_rows.append(dict(
                    grid=gtag, T_kev=T_kev, weight=wname,
                    max_err=mx, mean_err=mn, p95_err=p95,
                    dt_det=dt_det, dt_mc=dt_mc))

                emit(lines,
                     f'| {gtag} | {T_kev:.4g} | {mx:.3e} | {mn:.3e} '
                     f'| {p95:.3e} | {dt_det:.1f} | {dt_mc:.1f} |')

                # per-case plots
                if is_representative(gtag, T_kev, plot_tier):
                    tag = f'p1_{gtag}_T{T_kev:.4g}_{wname}'
                    fig = plot_pomraning(S_det, mc_mean, mc_2sig,
                                        centers_kev, T_kev, gtag, wname)
                    fp = save_fig(FIGS_DIR, GEN_DIR, f'{tag}_pomraning.png')
                    emit(lines, f'![Pomraning {label}]({fp})')
                    emit(lines)

                    fig = plot_error_heatmap(err, centers_kev, T_kev,
                                            gtag, wname)
                    fp = save_fig(FIGS_DIR, GEN_DIR, f'{tag}_errheat.png')
                    emit(lines, f'![Error heatmap {label}]({fp})')
                    emit(lines)

        emit(lines)

    # ── summary heatmaps ──────────────────────────────────────────────────

    emit(lines, '## Summary')
    emit(lines)

    for wname, _ in WEIGHT_SPECS:
        rows = [r for r in summary_rows if r['weight'] == wname]
        if not rows:
            continue
        grid_tags = sorted(set(r['grid'] for r in rows))
        temps = sorted(set(r['T_kev'] for r in rows))
        mat = np.full((len(grid_tags), len(temps)), np.nan)
        for r in rows:
            i = grid_tags.index(r['grid'])
            j = temps.index(r['T_kev'])
            mat[i, j] = r['p95_err']

        fig, ax = plt.subplots(figsize=(10, 4))
        vmin = np.nanmin(mat[mat > 0]) if np.any(mat > 0) else 1e-6
        im = ax.pcolormesh(range(len(temps)), range(len(grid_tags)), mat,
                           norm=LogNorm(vmin=max(vmin, 1e-6), vmax=1.0),
                           cmap='RdYlGn_r')
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([f'{t:.3g}' for t in temps], rotation=60, fontsize=7)
        ax.set_yticks(range(len(grid_tags)))
        ax.set_yticklabels(grid_tags, fontsize=8)
        ax.set_xlabel('T (keV)')
        ax.set_title(f'p95 relative error — {wname}')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fp = save_fig(FIGS_DIR, GEN_DIR, f'p1_summary_{wname}.png')
        emit(lines, f'![Summary {wname}]({fp})')
        emit(lines)

    # ── timing summary ────────────────────────────────────────────────────

    emit(lines, '## Timing')
    emit(lines)
    emit(lines, '*All times under 4-process concurrent load; '
         'for isolated timings run this part individually.*')
    emit(lines)
    total_det = sum(r['dt_det'] for r in summary_rows)
    total_mc = sum(r['dt_mc'] for r in summary_rows)
    emit(lines, f'- Total det time: {total_det/3600:.2f} h')
    emit(lines, f'- Total MC time:  {total_mc/3600:.2f} h')
    emit(lines)

    # ── failures ──────────────────────────────────────────────────────────

    tracker.emit_section(lines)

    elapsed = time.time() - t_start
    emit(lines, f'\n*Generated in {elapsed:.0f}s ({elapsed/3600:.2f}h).*')

    write_report(GEN_DIR, REPORT_FILE, lines)
    sys.exit(tracker.exit_code)


if __name__ == '__main__':
    main()
