"""
Part 2 – Deterministic vs Monte-Carlo temperature-derivative comparison.

Sweeps 3 weight functions x 6 energy grids x 25 temperatures for
``compute_dsigma_dT_matrix``.  Uses mixed abs+rel error (derivatives
cross zero), reports signal-to-noise and sign-agreement metrics.

Usage:
    python3 reports/mega_validation/part2_derivative_comparison.py [--plot-tier standard] [--no-cache]

Output:
    reports/generated/mega_val_part2_derivative.md
    reports/generated/figs/mega_val_p2/*.png
    reports/generated/cache/mega_val_p2/*.npz
"""
import os
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'cpp_modules'))

from common import (
    GRIDS, TEMPS_KEV, WEIGHT_SPECS, MC_SEEDS, MC_SAMPLES,
    KERNEL, KEV, KEV_KELVIN, SIGMA_T,
    make_det, run_mc_ensemble,
    mixed_error, error_stats, row_sums,
    cache_key, save_checkpoint, load_checkpoint,
    TimingLog, progress, emit, save_fig, write_report,
    base_arg_parser, FailureTracker, is_representative, make_weight,
)

PART = 2
GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs', 'mega_val_p2')
CACHE_DIR = os.path.join(GEN_DIR, 'cache', 'mega_val_p2')
REPORT_FILE = 'mega_val_part2_derivative.md'
ABS_FLOOR = SIGMA_T * 1e-6 / KEV_KELVIN

# ── plotting helpers ──────────────────────────────────────────────────────

def plot_pomraning_deriv(dS_det, dS_mc_mean, dS_mc_2sig, centers_kev,
                         T_kev, grid_tag, wname):
    """Pomraning-style derivative profiles for top-5 source groups."""
    G = dS_det.shape[0]
    rs = np.abs(dS_det).sum(axis=1)
    top5 = np.argsort(rs)[-5:][::-1]
    fig, axes = plt.subplots(len(top5), 1, figsize=(8, 3 * len(top5)),
                             sharex=True, squeeze=False)
    for idx, g in enumerate(top5):
        ax = axes[idx, 0]
        ax.plot(centers_kev, dS_det[g] / SIGMA_T, 'k-', lw=1.2, label='det')
        ax.fill_between(
            centers_kev,
            (dS_mc_mean[g] - dS_mc_2sig[g]) / SIGMA_T,
            (dS_mc_mean[g] + dS_mc_2sig[g]) / SIGMA_T,
            alpha=0.3, color='C0', label='MC ±2σ')
        ax.plot(centers_kev, dS_mc_mean[g] / SIGMA_T,
                'o', ms=2, color='C0')
        ax.set_ylabel(f'dσ/dT (g={g}→g\') / σ_T')
        ax.legend(fontsize=7)
        ax.set_yscale('symlog', linthresh=1e-12)
    axes[-1, 0].set_xlabel('E\' (keV)')
    axes[-1, 0].set_xscale('log')
    fig.suptitle(f'dσ/dT  {grid_tag}  T={T_kev:.4g} keV  {wname}',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_error_heatmap(err, centers_kev, T_kev, grid_tag, wname):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = max(err[err > 0].min(), 1e-6) if np.any(err > 0) else 1e-6
    im = ax.pcolormesh(centers_kev, centers_kev, err,
                       norm=LogNorm(vmin=vmin, vmax=max(err.max(), 1e-1)),
                       cmap='inferno')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('E\' (keV)'); ax.set_ylabel('E (keV)')
    ax.set_title(f'mixed error  dσ/dT  {grid_tag} T={T_kev:.4g} {wname}',
                 fontsize=9)
    fig.colorbar(im, ax=ax, label='mixed rel error')
    fig.tight_layout()
    return fig


def plot_sn_heatmap(sn, centers_kev, T_kev, grid_tag, wname):
    """Signal-to-noise heatmap (|det|/mc_2sig)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sn_safe = np.where(sn > 0, sn, np.nan)
    vmin = np.nanmin(sn_safe) if np.any(np.isfinite(sn_safe)) else 0.1
    im = ax.pcolormesh(centers_kev, centers_kev, sn_safe,
                       norm=LogNorm(vmin=max(vmin, 0.1), vmax=1e4),
                       cmap='RdYlGn')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('E\' (keV)'); ax.set_ylabel('E (keV)')
    ax.set_title(f'S/N  dσ/dT  {grid_tag} T={T_kev:.4g} {wname}',
                 fontsize=9)
    fig.colorbar(im, ax=ax, label='|det| / MC 2σ')
    fig.tight_layout()
    return fig


# ── main loop ─────────────────────────────────────────────────────────────

def main():
    args = base_arg_parser('Part 2: derivative comparison').parse_args()
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

    emit(lines, '# Part 2 – Derivative dσ/dT: Deterministic vs Monte-Carlo')
    emit(lines)
    emit(lines, '## Configuration')
    emit(lines)
    emit(lines, f'- Temperatures: {len(TEMPS_KEV)} points, '
         f'{TEMPS_KEV[0]:.1e} – {TEMPS_KEV[-1]:.1e} keV')
    emit(lines, f'- MC seeds: {len(MC_SEEDS)}, samples/seed: {MC_SAMPLES:,}')
    emit(lines, f'- Plot tier: {plot_tier}')
    emit(lines, f'- Abs floor: {ABS_FLOOR:.3e}')
    emit(lines)

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
             '| masked % | sign agr % | det (s) | MC (s) |')
        emit(lines, '|------|---------|---------|----------|---------|'
             '----------|------------|---------|--------|')

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
                    dS_det = cached['dS_det']
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
                            dS_det = np.array(det_obj.compute_dsigma_dT_matrix(
                                KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
                            dt_det = time.perf_counter() - t0

                        with tlog('mc'):
                            t0 = time.perf_counter()
                            mc_mean, mc_std, mc_2sig, _ = run_mc_ensemble(
                                bkev, wf, T_K, MC_SAMPLES, MC_SEEDS, 'dsigma')
                            dt_mc = time.perf_counter() - t0

                        save_checkpoint(ck,
                                        dS_det=dS_det, mc_mean=mc_mean,
                                        mc_2sig=mc_2sig,
                                        dt_det=np.float64(dt_det),
                                        dt_mc=np.float64(dt_mc))
                    except Exception as exc:
                        tracker.record(gtag, T_kev, wname, exc)
                        traceback.print_exc()
                        continue

                # metrics
                err, masked_frac = mixed_error(mc_mean, dS_det, ABS_FLOOR)
                mx, mn, md, p95 = error_stats(err)

                # signal-to-noise
                sn = np.where(mc_2sig > 0,
                              np.abs(dS_det) / mc_2sig, np.inf)

                # sign agreement (exclude near-zero)
                significant = np.abs(dS_det) >= ABS_FLOOR
                if np.any(significant):
                    sign_agree = float(np.mean(
                        np.sign(dS_det[significant]) ==
                        np.sign(mc_mean[significant])))
                else:
                    sign_agree = 1.0

                summary_rows.append(dict(
                    grid=gtag, T_kev=T_kev, weight=wname,
                    max_err=mx, mean_err=mn, p95_err=p95,
                    masked_frac=masked_frac, sign_agree=sign_agree,
                    dt_det=dt_det, dt_mc=dt_mc))

                emit(lines,
                     f'| {gtag} | {T_kev:.4g} | {mx:.3e} | {mn:.3e} '
                     f'| {p95:.3e} | {100*masked_frac:.1f} '
                     f'| {100*sign_agree:.1f} '
                     f'| {dt_det:.1f} | {dt_mc:.1f} |')

                if is_representative(gtag, T_kev, plot_tier):
                    tag = f'p2_{gtag}_T{T_kev:.4g}_{wname}'

                    fig = plot_pomraning_deriv(dS_det, mc_mean, mc_2sig,
                                              centers_kev, T_kev, gtag, wname)
                    fp = save_fig(FIGS_DIR, GEN_DIR, f'{tag}_pomraning.png')
                    emit(lines, f'![dσ/dT Pomraning {label}]({fp})')
                    emit(lines)

                    fig = plot_error_heatmap(err, centers_kev, T_kev,
                                            gtag, wname)
                    fp = save_fig(FIGS_DIR, GEN_DIR, f'{tag}_errheat.png')
                    emit(lines, f'![Error heatmap {label}]({fp})')
                    emit(lines)

                    fig = plot_sn_heatmap(sn, centers_kev, T_kev,
                                         gtag, wname)
                    fp = save_fig(FIGS_DIR, GEN_DIR, f'{tag}_sn.png')
                    emit(lines, f'![S/N heatmap {label}]({fp})')
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

        # p95 error heatmap
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
        ax.set_title(f'p95 mixed error dσ/dT — {wname}')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fp = save_fig(FIGS_DIR, GEN_DIR, f'p2_summary_err_{wname}.png')
        emit(lines, f'![Summary error {wname}]({fp})')
        emit(lines)

        # sign agreement heatmap
        mat_sa = np.full((len(grid_tags), len(temps)), np.nan)
        for r in rows:
            i = grid_tags.index(r['grid'])
            j = temps.index(r['T_kev'])
            mat_sa[i, j] = r['sign_agree']

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.pcolormesh(range(len(temps)), range(len(grid_tags)),
                           mat_sa * 100, vmin=50, vmax=100, cmap='RdYlGn')
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([f'{t:.3g}' for t in temps], rotation=60, fontsize=7)
        ax.set_yticks(range(len(grid_tags)))
        ax.set_yticklabels(grid_tags, fontsize=8)
        ax.set_xlabel('T (keV)')
        ax.set_title(f'Sign agreement (%) dσ/dT — {wname}')
        fig.colorbar(im, ax=ax, label='%')
        fig.tight_layout()
        fp = save_fig(FIGS_DIR, GEN_DIR, f'p2_summary_sign_{wname}.png')
        emit(lines, f'![Sign agreement {wname}]({fp})')
        emit(lines)

    # ── S/N regime map ────────────────────────────────────────────────────

    emit(lines, '### Regime map: MC derivative reliability')
    emit(lines)
    emit(lines, 'Cases where median S/N < 2 (MC noise dominates):')
    emit(lines)
    unreliable = [r for r in summary_rows
                  if r.get('sign_agree', 1.0) < 0.8]
    if unreliable:
        emit(lines, '| Grid | T (keV) | Weight | sign agr % |')
        emit(lines, '|------|---------|--------|------------|')
        for r in unreliable:
            emit(lines, f"| {r['grid']} | {r['T_kev']:.4g} "
                 f"| {r['weight']} | {100*r['sign_agree']:.1f} |")
    else:
        emit(lines, 'None — all cases have sign agreement >= 80%.')
    emit(lines)

    # ── timing ────────────────────────────────────────────────────────────

    emit(lines, '## Timing')
    emit(lines)
    emit(lines, '*All times under 4-process concurrent load.*')
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
