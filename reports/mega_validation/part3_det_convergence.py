"""
Part 3 – Deterministic quadrature-parameter convergence analysis.

Sweeps base_order, peak_max_depth, mu_order, tail_order, far_order
one at a time, measuring row-sum relative error against a high-quality
reference.  Covers both sigma and dsigma/dT, all 3 weight functions,
3 representative grids, and 10 temperatures.

Usage:
    python3 reports/mega_validation/part3_det_convergence.py [--plot-tier standard] [--no-cache]

Output:
    reports/generated/mega_val_part3_det_convergence.md
    reports/generated/figs/mega_val_p3/*.png
    reports/generated/cache/mega_val_p3/*.npz
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

import _compton_multigroup as cm
from common import (
    GRIDS, WEIGHT_SPECS, KERNEL, KEV, KEV_KELVIN, SIGMA_T,
    make_det, mixed_error, error_stats, row_sums,
    cache_key, save_checkpoint, load_checkpoint,
    TimingLog, progress, emit, save_fig, write_report,
    base_arg_parser, FailureTracker, make_weight,
)

PART = 3
GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs', 'mega_val_p3')
CACHE_DIR = os.path.join(GEN_DIR, 'cache', 'mega_val_p3')
REPORT_FILE = 'mega_val_part3_det_convergence.md'

# representative subset
CONV_GRID_TAGS = ['G32log', 'G24nu', 'G28hyb']
CONV_GRIDS = [g for g in GRIDS if g['tag'] in CONV_GRID_TAGS]
CONV_TEMPS_KEV = np.geomspace(1e-5, 1e4, 10)

# reference config (highest quality)
REF_CONFIG = dict(
    base_order=32,
    peak_max_depth=5,
    integration_tolerance=1e-4,
    cutoff_ratio=1e-10,
    tail_order=32,
    far_order=32,
    mu_order=32,
)

# production defaults (held constant when sweeping one parameter)
PROD_DEFAULTS = dict(
    base_order=24,
    peak_max_depth=5,
    integration_tolerance=1e-3,
    cutoff_ratio=1e-8,
    tail_order=24,
    far_order=24,
    mu_order=24,
)

# sweep definitions
SWEEPS = [
    {
        'param': 'base_order',
        'values': [4, 6, 8, 12, 16, 20, 24, 32],
        'label': 'base order',
    },
    {
        'param': 'peak_max_depth',
        'values': [0, 1, 2, 3, 4, 5],
        'label': 'peak depth',
    },
    {
        'param': 'mu_order',
        'values': [4, 6, 8, 12, 16, 24, 32],
        'label': 'μ order',
    },
    {
        'param': 'tail_order',
        'values': [4, 6, 8, 12, 16, 24, 32],
        'label': 'tail order',
    },
    {
        'param': 'far_order',
        'values': [4, 6, 8, 12, 16, 24, 32],
        'label': 'far order',
    },
]

ABS_FLOOR_SIGMA = SIGMA_T * 1e-10
ABS_FLOOR_DERIV = SIGMA_T * 1e-6 / KEV_KELVIN


def _make_config(**overrides):
    """Build MGIntegrationConfig from PROD_DEFAULTS with overrides."""
    kw = {**PROD_DEFAULTS, **overrides}
    return cm.MGIntegrationConfig(
        base_order=kw['base_order'],
        peak_max_depth=kw['peak_max_depth'],
        integration_tolerance=kw['integration_tolerance'],
        cutoff_ratio=kw['cutoff_ratio'],
        tail_order=kw['tail_order'],
        far_order=kw['far_order'],
        mu_order=kw['mu_order'],
    )


def _compute_pair(bkev, wf, config, T_K):
    """Return (S_sigma, S_dsigma) for a given config."""
    G = len(bkev) - 1
    det = make_det(bkev, wf, config)
    S = np.array(det.compute_sigma_matrix(
        KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
    dS = np.array(det.compute_dsigma_dT_matrix(
        KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
    return S, dS


# ── main ──────────────────────────────────────────────────────────────────

def main():
    args = base_arg_parser('Part 3: det convergence').parse_args()
    plot_tier = args.plot_tier
    use_cache = not args.no_cache

    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    total_sweep_pts = sum(len(s['values']) for s in SWEEPS)
    total_cases = (len(CONV_GRIDS) * len(CONV_TEMPS_KEV) * len(WEIGHT_SPECS)
                   * (1 + total_sweep_pts))  # +1 for reference
    tracker = FailureTracker(total_cases)
    tlog = TimingLog()
    lines: list[str] = []
    t_start = time.time()

    emit(lines, '# Part 3 – Deterministic Quadrature Convergence')
    emit(lines)
    emit(lines, '## Configuration')
    emit(lines)
    emit(lines, f'- Grids: {", ".join(CONV_GRID_TAGS)}')
    emit(lines, f'- Temperatures: {len(CONV_TEMPS_KEV)} points, '
         f'{CONV_TEMPS_KEV[0]:.1e} – {CONV_TEMPS_KEV[-1]:.1e} keV')
    emit(lines, f'- Weight functions: {", ".join(w[0] for w in WEIGHT_SPECS)}')
    emit(lines)
    emit(lines, '### Reference config')
    emit(lines)
    for k, v in REF_CONFIG.items():
        emit(lines, f'- `{k}` = {v}')
    emit(lines)
    emit(lines, '### Production defaults (held when sweeping)')
    emit(lines)
    for k, v in PROD_DEFAULTS.items():
        emit(lines, f'- `{k}` = {v}')
    emit(lines)

    # ── Phase 1: compute references ───────────────────────────────────────

    emit(lines, '## Phase 1: Reference computations')
    emit(lines)
    ref_config = _make_config(**REF_CONFIG)
    refs: dict[tuple, dict] = {}
    case_num = 0

    for grid in CONV_GRIDS:
        gtag = grid['tag']
        bkev = grid['bounds_kev']
        for T_kev in CONV_TEMPS_KEV:
            T_K = float(T_kev * KEV_KELVIN)
            for wname, wf_factory in WEIGHT_SPECS:
                case_num += 1
                label = f'ref {gtag} T={T_kev:.4g} {wname}'
                ck = cache_key(CACHE_DIR, gtag, T_kev, wname, '_ref')
                cached = load_checkpoint(ck) if use_cache else None

                if cached is not None:
                    refs[(gtag, T_kev, wname)] = {
                        'S': cached['S'], 'dS': cached['dS']}
                    progress(PART, f'[{case_num}] {label} [cached]')
                    continue

                progress(PART, f'[{case_num}] {label}')
                try:
                    wf = wf_factory()
                    with tlog('ref'):
                        S, dS = _compute_pair(bkev, wf, ref_config, T_K)
                    refs[(gtag, T_kev, wname)] = {'S': S, 'dS': dS}
                    save_checkpoint(ck, S=S, dS=dS)
                except Exception as exc:
                    tracker.record(gtag, T_kev, wname, exc)
                    traceback.print_exc()

    emit(lines, f'Computed {len(refs)} reference matrices.')
    emit(lines)

    # ── Phase 2: parameter sweeps ─────────────────────────────────────────

    # convergence_data[sweep_param][wname][(gtag, T_kev)] = list of
    #   (param_value, rs_err_sigma, rs_err_dsigma, wall_time)
    convergence_data: dict[str, dict[str, dict[tuple, list]]] = {}

    for sweep in SWEEPS:
        param = sweep['param']
        values = sweep['values']
        plabel = sweep['label']
        convergence_data[param] = {}

        emit(lines, f'## Sweep: {plabel} (`{param}`)')
        emit(lines)
        emit(lines, f'Values: {values}')
        emit(lines)

        for wname, wf_factory in WEIGHT_SPECS:
            convergence_data[param][wname] = {}

            emit(lines, f'### {plabel} — {wname}')
            emit(lines)
            emit(lines, '| Grid | T (keV) | value | σ row-sum err '
                 '| dσ/dT row-sum err | time (s) |')
            emit(lines, '|------|---------|-------|'
                 '----------------|-------------------|----------|')

            for grid in CONV_GRIDS:
                gtag = grid['tag']
                bkev = grid['bounds_kev']

                for T_kev in CONV_TEMPS_KEV:
                    T_K = float(T_kev * KEV_KELVIN)
                    ref = refs.get((gtag, T_kev, wname))
                    if ref is None:
                        continue

                    rs_ref_sigma = row_sums(ref['S'])
                    rs_ref_dsigma = row_sums(ref['dS'])
                    curve: list[tuple] = []

                    for val in values:
                        case_num += 1
                        label = f'{param}={val} {gtag} T={T_kev:.4g} {wname}'
                        ck = cache_key(CACHE_DIR, gtag, T_kev, wname,
                                       f'_{param}{val}')
                        cached = load_checkpoint(ck) if use_cache else None

                        if cached is not None:
                            S = cached['S']
                            dS = cached['dS']
                            dt = float(cached['dt'])
                            progress(PART, f'[{case_num}] {label} [cached]')
                        else:
                            progress(PART, f'[{case_num}] {label}')
                            try:
                                wf = wf_factory()
                                config = _make_config(**{param: val})
                                with tlog('sweep'):
                                    t0 = time.perf_counter()
                                    S, dS = _compute_pair(
                                        bkev, wf, config, T_K)
                                    dt = time.perf_counter() - t0
                                save_checkpoint(ck, S=S, dS=dS,
                                                dt=np.float64(dt))
                            except Exception as exc:
                                tracker.record(gtag, T_kev, wname, exc)
                                traceback.print_exc()
                                continue

                        # row-sum error vs reference
                        rs_s = row_sums(S)
                        rs_ds = row_sums(dS)
                        mask_s = np.abs(rs_ref_sigma) > ABS_FLOOR_SIGMA
                        mask_d = np.abs(rs_ref_dsigma) > ABS_FLOOR_DERIV

                        if np.any(mask_s):
                            err_s = float(np.max(np.abs(
                                rs_s[mask_s] / rs_ref_sigma[mask_s] - 1)))
                        else:
                            err_s = 0.0
                        if np.any(mask_d):
                            err_d = float(np.max(np.abs(
                                rs_ds[mask_d] / rs_ref_dsigma[mask_d] - 1)))
                        else:
                            err_d = 0.0

                        curve.append((val, err_s, err_d, dt))
                        emit(lines,
                             f'| {gtag} | {T_kev:.4g} | {val} '
                             f'| {err_s:.3e} | {err_d:.3e} | {dt:.2f} |')

                    convergence_data[param][wname][(gtag, T_kev)] = curve

            emit(lines)

        # ── convergence plots for this sweep ──────────────────────────────

        if plot_tier == 'smoke':
            continue

        for wname, _ in WEIGHT_SPECS:
            fig_s, ax_s = plt.subplots(figsize=(8, 5))
            fig_d, ax_d = plt.subplots(figsize=(8, 5))

            for (gtag, T_kev), curve in convergence_data[param].get(
                    wname, {}).items():
                if not curve:
                    continue
                vals = [c[0] for c in curve]
                errs_s = [max(c[1], 1e-16) for c in curve]
                errs_d = [max(c[2], 1e-16) for c in curve]
                lbl = f'{gtag} T={T_kev:.3g}'
                ax_s.plot(vals, errs_s, '.-', label=lbl, ms=4, lw=0.8)
                ax_d.plot(vals, errs_d, '.-', label=lbl, ms=4, lw=0.8)

            for ax, title, kind in [
                (ax_s, f'σ row-sum error vs {plabel} — {wname}', 'sigma'),
                (ax_d, f'dσ/dT row-sum error vs {plabel} — {wname}', 'dsigma'),
            ]:
                ax.set_xlabel(plabel)
                ax.set_ylabel('max |row-sum err vs ref|')
                ax.set_yscale('log')
                if param != 'peak_max_depth':
                    ax.set_xscale('log')
                ax.set_title(title, fontsize=10)
                ax.legend(fontsize=6, ncol=2, loc='best')
                ax.grid(True, alpha=0.3)

            fig_s.tight_layout()
            fig_d.tight_layout()
            fp = save_fig(FIGS_DIR, GEN_DIR,
                          f'p3_conv_{param}_{wname}_sigma.png')
            emit(lines, f'![{plabel} σ {wname}]({fp})')
            plt.close(fig_s)
            fp = save_fig(FIGS_DIR, GEN_DIR,
                          f'p3_conv_{param}_{wname}_dsigma.png')
            emit(lines, f'![{plabel} dσ/dT {wname}]({fp})')
            plt.close(fig_d)
            emit(lines)

    # ── cross-parameter interaction (base_order x peak_depth) ─────────────

    emit(lines, '## Cross-parameter: base_order × peak_depth')
    emit(lines)

    bo_vals = [4, 8, 12, 16, 24, 32]
    pd_vals = [1, 2, 3, 4, 5, 6, 7]

    # pick one representative case: G32log, median T, wien
    repr_grid = CONV_GRIDS[0]
    repr_T = float(CONV_TEMPS_KEV[len(CONV_TEMPS_KEV) // 2])
    repr_wname = 'wien'
    ref = refs.get((repr_grid['tag'], repr_T, repr_wname))
    if ref is not None:
        rs_ref = row_sums(ref['S'])
        mat_cross = np.full((len(bo_vals), len(pd_vals)), np.nan)

        for i, bo in enumerate(bo_vals):
            for j, pd_val in enumerate(pd_vals):
                ck = cache_key(CACHE_DIR, repr_grid['tag'], repr_T,
                               repr_wname, f'_cross_bo{bo}_pd{pd_val}')
                cached = load_checkpoint(ck) if use_cache else None
                if cached is not None:
                    S = cached['S']
                else:
                    try:
                        wf = make_weight(repr_wname)
                        config = _make_config(base_order=bo,
                                              peak_max_depth=pd_val)
                        S, _ = _compute_pair(
                            repr_grid['bounds_kev'], wf, config,
                            float(repr_T * KEV_KELVIN))
                        save_checkpoint(ck, S=S)
                    except Exception:
                        continue

                rs = row_sums(S)
                mask = np.abs(rs_ref) > ABS_FLOOR_SIGMA
                if np.any(mask):
                    mat_cross[i, j] = float(np.max(
                        np.abs(rs[mask] / rs_ref[mask] - 1)))

        fig, ax = plt.subplots(figsize=(7, 5))
        vmin = np.nanmin(mat_cross[mat_cross > 0]) if np.any(
            mat_cross > 0) else 1e-6
        im = ax.pcolormesh(pd_vals, bo_vals, mat_cross,
                           norm=LogNorm(vmin=max(vmin, 1e-8), vmax=1.0),
                           cmap='viridis')
        ax.set_xlabel('peak_max_depth')
        ax.set_ylabel('base_order')
        ax.set_title(
            f'σ row-sum error  {repr_grid["tag"]} T={repr_T:.3g} {repr_wname}',
            fontsize=10)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fp = save_fig(FIGS_DIR, GEN_DIR, 'p3_cross_bo_pd.png')
        emit(lines, f'![Cross-parameter]({fp})')
        emit(lines)

    # ── accuracy recommendations ──────────────────────────────────────────

    emit(lines, '## Accuracy Recommendations')
    emit(lines)
    emit(lines, 'Minimum parameter values to achieve target '
         'row-sum accuracy (worst case across grids/temps/weights):')
    emit(lines)
    targets = [0.01, 0.001, 0.0001]
    emit(lines, '| Parameter | 1% | 0.1% | 0.01% |')
    emit(lines, '|-----------|-----|------|-------|')

    for sweep in SWEEPS:
        param = sweep['param']
        cells = []
        for target in targets:
            best_val = '—'
            for wname in [w[0] for w in WEIGHT_SPECS]:
                for key, curve in convergence_data.get(param, {}).get(
                        wname, {}).items():
                    for val, err_s, err_d, _ in curve:
                        if err_s <= target and err_d <= target:
                            if best_val == '—' or val < int(best_val):
                                best_val = str(val)
                            break
            cells.append(best_val)
        emit(lines, f'| `{param}` | {" | ".join(cells)} |')
    emit(lines)

    # ── timing & Pareto ───────────────────────────────────────────────────

    emit(lines, '## Timing vs Accuracy (Pareto)')
    emit(lines)
    for sweep in SWEEPS:
        param = sweep['param']
        plabel = sweep['label']
        fig, ax = plt.subplots(figsize=(7, 5))
        for wname in [w[0] for w in WEIGHT_SPECS]:
            for (gtag, T_kev), curve in convergence_data.get(param, {}).get(
                    wname, {}).items():
                if not curve:
                    continue
                times = [c[3] for c in curve]
                errs = [max(c[1], 1e-16) for c in curve]
                ax.plot(times, errs, '.-', ms=4, lw=0.8,
                        label=f'{gtag} T={T_kev:.3g} {wname}')
        ax.set_xlabel('wall time (s)')
        ax.set_ylabel('max row-sum error')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f'Pareto: {plabel}', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fp = save_fig(FIGS_DIR, GEN_DIR, f'p3_pareto_{param}.png')
        emit(lines, f'![Pareto {plabel}]({fp})')
        emit(lines)

    # ── failures ──────────────────────────────────────────────────────────

    tracker.emit_section(lines)

    elapsed = time.time() - t_start
    emit(lines, f'\n*Generated in {elapsed:.0f}s ({elapsed/3600:.2f}h).*')

    write_report(GEN_DIR, REPORT_FILE, lines)
    sys.exit(tracker.exit_code)


if __name__ == '__main__':
    main()
