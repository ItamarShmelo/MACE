"""
Part 4 – MC convergence, grid sensitivity, and scaling analysis.

Phase A: MC sample-count convergence (25 k … 50 M) with 10 seeds each.
Phase B: Grid sensitivity — row-sum profiles and total cross section
         compared across all 6 grids.
Phase C: Scaling analysis — det and MC wall-time vs grid size / samples.

Usage:
    python3 reports/mega_validation/part4_mc_convergence.py [--plot-tier standard] [--no-cache]

Output:
    reports/generated/mega_val_part4_mc_convergence.md
    reports/generated/figs/mega_val_p4/*.png
    reports/generated/cache/mega_val_p4/*.npz
"""
import os
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'cpp_modules'))

import _compton_multigroup as cm
from common import (
    GRIDS, WEIGHT_SPECS, MC_SEEDS, KERNEL,
    KEV, KEV_KELVIN, SIGMA_T,
    make_det, make_mc, run_mc_ensemble,
    mixed_error, error_stats, row_sums,
    cache_key, save_checkpoint, load_checkpoint,
    TimingLog, progress, emit, save_fig, write_report,
    base_arg_parser, FailureTracker, make_weight,
)

PART = 4
GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs', 'mega_val_p4')
CACHE_DIR = os.path.join(GEN_DIR, 'cache', 'mega_val_p4')
REPORT_FILE = 'mega_val_part4_mc_convergence.md'

ABS_FLOOR_SIGMA = SIGMA_T * 1e-10

# Phase A config
MC_SAMPLE_SWEEP = [
    25_000, 50_000, 100_000, 250_000, 500_000,
    1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000,
]
PHASE_A_GRID_TAGS = ['G16log', 'G32log', 'G24nu']
PHASE_A_GRIDS = [g for g in GRIDS if g['tag'] in PHASE_A_GRID_TAGS]
PHASE_A_TEMPS_KEV = np.geomspace(1e-4, 1e3, 8)
PHASE_A_WNAME = 'wien'

# Phase B config
PHASE_B_TEMPS_KEV = np.geomspace(1e-5, 1e4, 10)
PHASE_B_WNAME = 'wien'

# Phase C config
PHASE_C_TEMPS_KEV = np.geomspace(1e-3, 1e3, 5)


# ── Phase A: MC sample-count convergence ──────────────────────────────────

def phase_a(lines, use_cache, plot_tier, tracker, tlog):
    emit(lines, '## Phase A: MC Sample-Count Convergence')
    emit(lines)
    emit(lines, f'- Sample counts: {MC_SAMPLE_SWEEP}')
    emit(lines, f'- Seeds per count: {len(MC_SEEDS)}')
    emit(lines, f'- Grids: {", ".join(PHASE_A_GRID_TAGS)}')
    emit(lines, f'- Temperatures: {len(PHASE_A_TEMPS_KEV)} points')
    emit(lines, f'- Weight: {PHASE_A_WNAME}')
    emit(lines)

    wf_factory = dict(WEIGHT_SPECS)[PHASE_A_WNAME]

    # compute deterministic references first
    det_refs: dict[tuple, np.ndarray] = {}
    for grid in PHASE_A_GRIDS:
        gtag = grid['tag']
        bkev = grid['bounds_kev']
        G = len(bkev) - 1
        for T_kev in PHASE_A_TEMPS_KEV:
            T_K = float(T_kev * KEV_KELVIN)
            ck = cache_key(CACHE_DIR, gtag, T_kev, PHASE_A_WNAME, '_detref')
            cached = load_checkpoint(ck) if use_cache else None
            if cached is not None:
                det_refs[(gtag, T_kev)] = cached['S']
                continue
            try:
                wf = wf_factory()
                det = make_det(bkev, wf)
                S = np.array(det.compute_sigma_matrix(
                    KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
                det_refs[(gtag, T_kev)] = S
                save_checkpoint(ck, S=S)
            except Exception as exc:
                tracker.record(gtag, T_kev, PHASE_A_WNAME, exc)
                traceback.print_exc()

    # sweep sample counts
    # cv_data[(gtag, T_kev)] = [(N, cv_sigma, cv_dsigma, bias_sigma, dt)]
    cv_data: dict[tuple, list] = {}

    case_num = 0
    total_mc_cases = (len(PHASE_A_GRIDS) * len(PHASE_A_TEMPS_KEV)
                      * len(MC_SAMPLE_SWEEP))

    for grid in PHASE_A_GRIDS:
        gtag = grid['tag']
        bkev = grid['bounds_kev']
        G = len(bkev) - 1

        for T_kev in PHASE_A_TEMPS_KEV:
            T_K = float(T_kev * KEV_KELVIN)
            ref_S = det_refs.get((gtag, T_kev))
            curve: list[tuple] = []

            for N in MC_SAMPLE_SWEEP:
                case_num += 1
                label = f'A {gtag} T={T_kev:.4g} N={N}'
                ck = cache_key(CACHE_DIR, gtag, T_kev, PHASE_A_WNAME,
                               f'_mcN{N}')
                cached = load_checkpoint(ck) if use_cache else None

                if cached is not None:
                    mc_stack = cached['stack']
                    dt = float(cached['dt'])
                    progress(PART, f'[{case_num}/{total_mc_cases}] '
                             f'{label} [cached]')
                else:
                    progress(PART, f'[{case_num}/{total_mc_cases}] {label}')
                    try:
                        wf = wf_factory()
                        t0 = time.perf_counter()
                        _, _, _, mc_stack = run_mc_ensemble(
                            bkev, wf, T_K, N, MC_SEEDS, 'sigma')
                        dt = time.perf_counter() - t0
                        save_checkpoint(ck, stack=mc_stack,
                                        dt=np.float64(dt))
                    except Exception as exc:
                        tracker.record(gtag, T_kev, PHASE_A_WNAME, exc)
                        traceback.print_exc()
                        continue

                mc_mean = mc_stack.mean(axis=0)
                rs_per_seed = mc_stack.sum(axis=-1)  # (n_seeds, G)
                rs_mean = rs_per_seed.mean(axis=0)
                rs_std = rs_per_seed.std(axis=0, ddof=1) if len(MC_SEEDS) > 1 \
                    else np.zeros_like(rs_mean)

                mask = np.abs(rs_mean) > ABS_FLOOR_SIGMA
                if np.any(mask):
                    cv = float(np.mean(np.abs(rs_std[mask] / rs_mean[mask])))
                else:
                    cv = 0.0

                if ref_S is not None:
                    rs_ref = row_sums(ref_S)
                    mask_ref = np.abs(rs_ref) > ABS_FLOOR_SIGMA
                    if np.any(mask_ref):
                        bias = float(np.max(np.abs(
                            rs_mean[mask_ref] / rs_ref[mask_ref] - 1)))
                    else:
                        bias = 0.0
                else:
                    bias = np.nan

                curve.append((N, cv, bias, dt))

            cv_data[(gtag, T_kev)] = curve

    # ── CV vs N plots ─────────────────────────────────────────────────────

    emit(lines, '### CV vs sample count')
    emit(lines)

    fig, ax = plt.subplots(figsize=(9, 6))
    for (gtag, T_kev), curve in cv_data.items():
        if not curve:
            continue
        Ns = [c[0] for c in curve]
        cvs = [max(c[1], 1e-16) for c in curve]
        ax.plot(Ns, cvs, '.-', ms=4, lw=0.8,
                label=f'{gtag} T={T_kev:.3g}')

    N_ref = np.array(MC_SAMPLE_SWEEP, dtype=float)
    cv0 = 0.5
    ax.plot(N_ref, cv0 / np.sqrt(N_ref / N_ref[0]),
            'k--', lw=1, alpha=0.5, label='∝ 1/√N')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('MC samples per seed')
    ax.set_ylabel('CV of row sums')
    ax.set_title('MC convergence: coefficient of variation')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = save_fig(FIGS_DIR, GEN_DIR, 'p4_cv_vs_N.png')
    emit(lines, f'![CV vs N]({fp})')
    emit(lines)

    # ── Bias vs N plots ───────────────────────────────────────────────────

    emit(lines, '### Bias vs sample count')
    emit(lines)

    fig, ax = plt.subplots(figsize=(9, 6))
    for (gtag, T_kev), curve in cv_data.items():
        if not curve:
            continue
        Ns = [c[0] for c in curve]
        biases = [max(c[2], 1e-16) for c in curve]
        ax.plot(Ns, biases, '.-', ms=4, lw=0.8,
                label=f'{gtag} T={T_kev:.3g}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('MC samples per seed')
    ax.set_ylabel('max |MC_mean/det - 1|')
    ax.set_title('MC bias vs det reference')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = save_fig(FIGS_DIR, GEN_DIR, 'p4_bias_vs_N.png')
    emit(lines, f'![Bias vs N]({fp})')
    emit(lines)

    # ── cost-accuracy Pareto ──────────────────────────────────────────────

    emit(lines, '### Cost-accuracy Pareto')
    emit(lines)

    fig, ax = plt.subplots(figsize=(9, 6))
    for (gtag, T_kev), curve in cv_data.items():
        if not curve:
            continue
        dts = [c[3] for c in curve]
        cvs = [max(c[1], 1e-16) for c in curve]
        ax.plot(dts, cvs, '.-', ms=4, lw=0.8,
                label=f'{gtag} T={T_kev:.3g}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('wall time (s)')
    ax.set_ylabel('CV of row sums')
    ax.set_title('MC cost-accuracy Pareto')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = save_fig(FIGS_DIR, GEN_DIR, 'p4_pareto.png')
    emit(lines, f'![Pareto]({fp})')
    emit(lines)

    # 1/sqrt(N) fit
    emit(lines, '### Power-law fit')
    emit(lines)
    emit(lines, '| Grid | T (keV) | fitted exponent | expected −0.5 |')
    emit(lines, '|------|---------|-----------------|---------------|')
    for (gtag, T_kev), curve in cv_data.items():
        if len(curve) < 3:
            continue
        Ns = np.array([c[0] for c in curve], dtype=float)
        cvs = np.array([c[1] for c in curve])
        mask = cvs > 0
        if np.sum(mask) < 2:
            continue
        coeffs = np.polyfit(np.log10(Ns[mask]), np.log10(cvs[mask]), 1)
        emit(lines, f'| {gtag} | {T_kev:.4g} | {coeffs[0]:.3f} | −0.500 |')
    emit(lines)


# ── Phase B: Grid sensitivity ────────────────────────────────────────────

def phase_b(lines, use_cache, plot_tier, tracker, tlog):
    emit(lines, '## Phase B: Grid Sensitivity Study')
    emit(lines)
    emit(lines, f'- All 6 grids at {len(PHASE_B_TEMPS_KEV)} temperatures')
    emit(lines, f'- Weight: {PHASE_B_WNAME}')
    emit(lines)

    wf_factory = dict(WEIGHT_SPECS)[PHASE_B_WNAME]

    emit(lines, '### Total cross section (sum of row sums) vs temperature')
    emit(lines)
    emit(lines, '| Grid | Groups | ' + ' | '.join(
        f'T={t:.3g}' for t in PHASE_B_TEMPS_KEV) + ' |')
    emit(lines, '|------|--------|' + '|'.join(
        '---------' for _ in PHASE_B_TEMPS_KEV) + '|')

    grid_rs: dict[str, dict[float, np.ndarray]] = {}

    for grid in GRIDS:
        gtag = grid['tag']
        bkev = grid['bounds_kev']
        G = len(bkev) - 1
        grid_rs[gtag] = {}
        cells = [f'{gtag}', str(G)]

        for T_kev in PHASE_B_TEMPS_KEV:
            T_K = float(T_kev * KEV_KELVIN)
            ck = cache_key(CACHE_DIR, gtag, T_kev, PHASE_B_WNAME, '_gridB')
            cached = load_checkpoint(ck) if use_cache else None

            if cached is not None:
                S = cached['S']
            else:
                try:
                    wf = wf_factory()
                    det = make_det(bkev, wf)
                    S = np.array(det.compute_sigma_matrix(
                        KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
                    save_checkpoint(ck, S=S)
                except Exception as exc:
                    tracker.record(gtag, T_kev, PHASE_B_WNAME, exc)
                    traceback.print_exc()
                    S = np.zeros((G, G))

            rs = row_sums(S)
            grid_rs[gtag][T_kev] = rs
            total_xs = float(rs.sum())
            cells.append(f'{total_xs/SIGMA_T:.4g}')

        emit(lines, '| ' + ' | '.join(cells) + ' |')
    emit(lines)
    emit(lines, '*(values in units of σ_T)*')
    emit(lines)

    # row-sum overlay plots
    if plot_tier != 'smoke':
        emit(lines, '### Row-sum profiles')
        emit(lines)
        for T_kev in PHASE_B_TEMPS_KEV:
            fig, ax = plt.subplots(figsize=(8, 5))
            for grid in GRIDS:
                gtag = grid['tag']
                bkev = grid['bounds_kev']
                centers = np.sqrt(bkev[:-1] * bkev[1:])
                rs = grid_rs.get(gtag, {}).get(T_kev)
                if rs is None:
                    continue
                ax.plot(centers, rs / SIGMA_T, '.-', ms=3, lw=0.8,
                        label=f'{gtag} ({len(bkev)-1}g)')
            ax.set_xscale('log')
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_xlabel('E (keV)')
            ax.set_ylabel('row sum / σ_T')
            ax.set_title(f'Row-sum profiles — T = {T_kev:.4g} keV')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fp = save_fig(FIGS_DIR, GEN_DIR,
                          f'p4_gridsens_T{T_kev:.4g}.png')
            emit(lines, f'![Grid sensitivity T={T_kev:.4g}]({fp})')
            emit(lines)

    # resolution metric
    emit(lines, '### Resolution metric: min(group width) / kT')
    emit(lines)
    emit(lines, '| Grid | ' + ' | '.join(
        f'T={t:.3g}' for t in PHASE_B_TEMPS_KEV) + ' |')
    emit(lines, '|------|' + '|'.join(
        '---------' for _ in PHASE_B_TEMPS_KEV) + '|')
    for grid in GRIDS:
        gtag = grid['tag']
        bkev = grid['bounds_kev']
        min_width = float(np.min(np.diff(bkev)))
        cells = [gtag]
        for T_kev in PHASE_B_TEMPS_KEV:
            cells.append(f'{min_width / T_kev:.3g}')
        emit(lines, '| ' + ' | '.join(cells) + ' |')
    emit(lines)


# ── Phase C: Scaling analysis ─────────────────────────────────────────────

def phase_c(lines, use_cache, plot_tier, tracker, tlog):
    emit(lines, '## Phase C: Scaling Analysis')
    emit(lines)
    emit(lines, '*Timing runs executed sequentially within Part 4.*')
    emit(lines)

    wf_factory = dict(WEIGHT_SPECS)['wien']

    # det time vs groups
    emit(lines, '### Det time vs number of groups')
    emit(lines)
    grids_sorted = sorted(GRIDS, key=lambda g: len(g['bounds_kev']))
    det_times: list[tuple] = []

    for grid in grids_sorted:
        gtag = grid['tag']
        bkev = grid['bounds_kev']
        G = len(bkev) - 1
        times_at_temps = []

        for T_kev in PHASE_C_TEMPS_KEV:
            T_K = float(T_kev * KEV_KELVIN)
            try:
                wf = wf_factory()
                det = make_det(bkev, wf)
                t0 = time.perf_counter()
                det.compute_sigma_matrix(KERNEL, T=T_K, Ne=1.0)
                dt = time.perf_counter() - t0
                times_at_temps.append(dt)
            except Exception as exc:
                tracker.record(gtag, T_kev, 'wien', exc)
                traceback.print_exc()

        if times_at_temps:
            avg_t = float(np.mean(times_at_temps))
            det_times.append((G, avg_t))
            progress(PART, f'Scaling: det {gtag} G={G} avg={avg_t:.2f}s')

    fig, ax = plt.subplots(figsize=(7, 5))
    gs = [d[0] for d in det_times]
    ts = [d[1] for d in det_times]
    ax.plot(gs, ts, 'o-', ms=6, lw=1.5)
    if len(gs) >= 2:
        coeffs = np.polyfit(np.log10(gs), np.log10(ts), 1)
        g_fit = np.linspace(min(gs), max(gs), 50)
        ax.plot(g_fit, 10**np.polyval(coeffs, np.log10(g_fit)),
                'k--', lw=0.8, label=f'fit: ∝ G^{coeffs[0]:.2f}')
        ax.legend()
    ax.set_xlabel('number of groups')
    ax.set_ylabel('wall time (s)')
    ax.set_title('Det compute time vs grid size')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = save_fig(FIGS_DIR, GEN_DIR, 'p4_det_scaling.png')
    emit(lines, f'![Det scaling]({fp})')
    emit(lines)

    emit(lines, '| Groups | avg time (s) |')
    emit(lines, '|--------|-------------|')
    for G, t in det_times:
        emit(lines, f'| {G} | {t:.3f} |')
    emit(lines)

    # MC time vs samples
    emit(lines, '### MC time vs sample count')
    emit(lines)
    test_grid = [g for g in GRIDS if g['tag'] == 'G32log'][0]
    bkev = test_grid['bounds_kev']
    T_K = float(1.0 * KEV_KELVIN)
    mc_times: list[tuple] = []

    for N in MC_SAMPLE_SWEEP:
        try:
            wf = wf_factory()
            mc_obj = make_mc(bkev, wf, N, seed=42)
            t0 = time.perf_counter()
            mc_obj.compute_sigma_matrix(T=T_K, Ne=1.0)
            dt = time.perf_counter() - t0
            mc_times.append((N, dt))
            progress(PART, f'Scaling: MC N={N} dt={dt:.2f}s')
        except Exception as exc:
            tracker.record('G32log', 1.0, 'wien', exc)
            traceback.print_exc()

    fig, ax = plt.subplots(figsize=(7, 5))
    Ns = [m[0] for m in mc_times]
    ts = [m[1] for m in mc_times]
    ax.plot(Ns, ts, 'o-', ms=6, lw=1.5)
    if len(Ns) >= 2:
        coeffs = np.polyfit(np.log10(Ns), np.log10(ts), 1)
        N_fit = np.linspace(min(Ns), max(Ns), 50)
        ax.plot(N_fit, 10**np.polyval(coeffs, np.log10(N_fit)),
                'k--', lw=0.8, label=f'fit: ∝ N^{coeffs[0]:.2f}')
        ax.legend()
    ax.set_xlabel('MC samples')
    ax.set_ylabel('wall time (s)')
    ax.set_title('MC compute time vs sample count')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = save_fig(FIGS_DIR, GEN_DIR, 'p4_mc_scaling.png')
    emit(lines, f'![MC scaling]({fp})')
    emit(lines)

    emit(lines, '| Samples | time (s) |')
    emit(lines, '|---------|----------|')
    for N, t in mc_times:
        emit(lines, f'| {N:,} | {t:.3f} |')
    emit(lines)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    args = base_arg_parser('Part 4: MC convergence + grid study').parse_args()
    plot_tier = args.plot_tier
    use_cache = not args.no_cache

    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    total_cases = (len(PHASE_A_GRIDS) * len(PHASE_A_TEMPS_KEV)
                   * len(MC_SAMPLE_SWEEP)
                   + len(GRIDS) * len(PHASE_B_TEMPS_KEV) + 50)
    tracker = FailureTracker(total_cases)
    tlog = TimingLog()
    lines: list[str] = []
    t_start = time.time()

    emit(lines, '# Part 4 – MC Convergence, Grid Sensitivity, Scaling')
    emit(lines)

    phase_a(lines, use_cache, plot_tier, tracker, tlog)
    phase_b(lines, use_cache, plot_tier, tracker, tlog)
    phase_c(lines, use_cache, plot_tier, tracker, tlog)

    tracker.emit_section(lines)

    elapsed = time.time() - t_start
    emit(lines, f'\n*Generated in {elapsed:.0f}s ({elapsed/3600:.2f}h).*')

    write_report(GEN_DIR, REPORT_FILE, lines)
    sys.exit(tracker.exit_code)


if __name__ == '__main__':
    main()
