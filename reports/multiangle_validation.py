"""
Multiangle deterministic-vs-MC validation report.

Compares the full 3D multiangle sigma matrix S(g, g', xi_bin) from the
deterministic integrator against an internal Monte Carlo ensemble across
6 temperatures and 3 energy grids (18 combinations, 8 angle bins each).

Supports three execution modes:
  --worker N   Run a single task (for individual SLURM jobs)
  --collect    Aggregate worker results and generate report + plots
  (no args)    Run everything in one process (local mode)

Usage (distributed):
    bash reports/run_multiangle_validation.sh

Usage (single process):
    OMP_NUM_THREADS=16 python3.12 reports/multiangle_validation.py
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cpp_modules"))

import _compton_multigroup as cm
from _compton_differential_cross_section import ComptonKernelSolver
from _units import kev, kev_kelvin

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KERNEL = ComptonKernelSolver()

TEMPERATURES_KEV = [0.1, 1.0, 10.0, 100.0]

GRIDS = {
    "coarse_4g": np.array([1, 5, 10, 50], dtype=float),
    "medium_10g": np.logspace(np.log10(0.5), np.log10(100), 11),
}

MC_SAMPLES = 1_000_000
MC_SEEDS = [42, 137, 271, 577, 1009]
NUM_ANGLE_BINS = 8

ROW_SUM_SIGMA_TOL = 3.0

RESULTS_DIR = os.path.join(ROOT, "reports", "generated", "multiangle_results")


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

def build_task_list():
    """Build flat task list: 6 temps x 3 grids = 18 tasks."""
    tasks = []
    for T_kev in TEMPERATURES_KEV:
        for grid_name in GRIDS:
            tasks.append((T_kev, grid_name))
    return tasks


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def get_det_config(T_kev):
    """Select integration config appropriate for the temperature."""
    if T_kev < 0.1:
        return cm.MGIntegrationConfig(
            base_order=24,
            peak_max_depth=5,
            xi_order=48,
            integration_tolerance=1e-4,
            cutoff_ratio=1e-10,
            cold_temperature_order=24,
        )
    return cm.MGIntegrationConfig.warm_flat()


def make_det(bounds_kev, config):
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=(bounds_kev * kev).tolist(),
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=config,
    )


def make_mc(bounds_kev, num_samples, seed):
    return cm.ComptonMonteCarloKernel(
        energy_group_boundaries=(bounds_kev * kev).tolist(),
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        config=cm.MCIntegrationConfig(
            num_samples=num_samples,
            seed=seed,
            discard_out_of_grid=True,
        ),
    )


# ---------------------------------------------------------------------------
# MC ensemble
# ---------------------------------------------------------------------------

def run_mc_ensemble_3d(bounds_kev, T_K, seeds, num_angle_bins):
    """Run angle-binned MC with multiple seeds, return (mean, std, stack)."""
    G = len(bounds_kev) - 1
    runs = []
    for seed in seeds:
        mc_obj = make_mc(bounds_kev, MC_SAMPLES, seed)
        S = np.array(mc_obj.compute_sigma_matrix(
            num_angle_bins=num_angle_bins, T=T_K, Ne=1.0))
        runs.append(S.reshape(G, G, num_angle_bins))
    stack = np.stack(runs)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1), stack


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compare_3d(S_det_3d, S_det_2d, mc_mean_3d, mc_std_3d):
    """Compare multiangle det vs MC ensemble.

    Returns dict with angle-summed row-sum z-scores, per-bin row-sum
    z-scores, element-wise 3D z-score statistics, and angle-integrated
    consistency.
    """
    G = S_det_3d.shape[0]
    N = S_det_3d.shape[2]

    # --- angle-summed row sums ---
    det_2d_from_3d = S_det_3d.sum(axis=2)
    mc_2d = mc_mean_3d.sum(axis=2)
    mc_2d_std = np.sqrt((mc_std_3d ** 2).sum(axis=2))

    rs_det = det_2d_from_3d.sum(axis=1)
    rs_mc = mc_2d.sum(axis=1)
    rs_mc_std = np.sqrt((mc_2d_std ** 2).sum(axis=1))
    rs_sigma = np.where(rs_mc_std > 0,
                        np.abs(rs_det - rs_mc) / rs_mc_std, 0.0)

    # --- per-bin row sums ---
    bin_rs_max_sigma = np.zeros(N)
    bin_rs_med_sigma = np.zeros(N)
    for a in range(N):
        det_rs_a = S_det_3d[:, :, a].sum(axis=1)
        mc_rs_a = mc_mean_3d[:, :, a].sum(axis=1)
        mc_rs_std_a = np.sqrt((mc_std_3d[:, :, a] ** 2).sum(axis=1))
        sigma_a = np.where(mc_rs_std_a > 0,
                           np.abs(det_rs_a - mc_rs_a) / mc_rs_std_a, 0.0)
        bin_rs_max_sigma[a] = float(np.max(sigma_a))
        bin_rs_med_sigma[a] = float(np.median(sigma_a))

    # --- element-wise 3D z-scores ---
    elem_sigma = np.where(mc_std_3d > 0,
                          np.abs(S_det_3d - mc_mean_3d) / mc_std_3d, 0.0)

    # --- angle-integrated consistency ---
    denom = np.maximum(np.abs(S_det_2d), 1e-300)
    consistency = float(np.max(np.abs(det_2d_from_3d - S_det_2d) / denom))

    neg_rows = int(np.sum(rs_det < 0))

    return {
        "G": G,
        "rs_max_sigma": float(np.max(rs_sigma)),
        "rs_median_sigma": float(np.median(rs_sigma)),
        "elem_max_sigma": float(np.max(elem_sigma)),
        "elem_median_sigma": float(np.median(elem_sigma)),
        "elem_p95_sigma": float(np.percentile(elem_sigma, 95)),
        "elem_within_3sig": float(np.mean(elem_sigma < 3.0)),
        "bin_rs_max_sigma": bin_rs_max_sigma.tolist(),
        "bin_rs_med_sigma": bin_rs_med_sigma.tolist(),
        "consistency": consistency,
        "neg_rows": neg_rows,
    }


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_one(T_kev, grid_name, bounds_kev):
    """Compute multiangle det and MC matrices, return comparison metrics."""
    T_K = T_kev * kev_kelvin
    config = get_det_config(T_kev)
    G = len(bounds_kev) - 1

    t0 = time.perf_counter()
    det_obj = make_det(bounds_kev, config)
    S_det_3d = np.array(det_obj.compute_sigma_matrix(
        kernel=KERNEL, num_angle_bins=NUM_ANGLE_BINS,
        T=T_K, Ne=1.0)).reshape(G, G, NUM_ANGLE_BINS)
    S_det_2d = np.array(det_obj.compute_sigma_matrix(
        kernel=KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
    dt_det = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_mean_3d, mc_std_3d, _ = run_mc_ensemble_3d(
        bounds_kev, T_K, MC_SEEDS, NUM_ANGLE_BINS)
    dt_mc = time.perf_counter() - t0

    metrics = compare_3d(S_det_3d, S_det_2d, mc_mean_3d, mc_std_3d)
    metrics["T_kev"] = T_kev
    metrics["grid"] = grid_name
    metrics["dt_det"] = dt_det
    metrics["dt_mc"] = dt_mc
    metrics["S_det_3d"] = S_det_3d
    metrics["mc_mean_3d"] = mc_mean_3d
    metrics["mc_std_3d"] = mc_std_3d
    metrics["bounds_kev"] = bounds_kev
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def group_centers_kev(bounds_kev):
    return np.sqrt(bounds_kev[:-1] * bounds_kev[1:])


def plot_xi_profiles(results_by_temp, figs_dir):
    """Plot xi profiles: for each temperature, pick the finest grid available
    and show det vs MC for representative (g_in, g_out) pairs."""
    xi_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)
    xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])

    paths = []
    for T_kev, r in sorted(results_by_temp.items()):
        S_det_3d = r["S_det_3d"]
        mc_mean_3d = r["mc_mean_3d"]
        mc_std_3d = r["mc_std_3d"]
        bounds_kev = r["bounds_kev"]
        centers = group_centers_kev(bounds_kev)
        G = len(centers)

        g_in_indices = [0, G // 2, G - 1] if G > 2 else list(range(G))

        fig, axes = plt.subplots(1, len(g_in_indices),
                                 figsize=(5 * len(g_in_indices), 4))
        if len(g_in_indices) == 1:
            axes = [axes]
        fig.suptitle(
            f"xi profile: det vs MC ({len(MC_SEEDS)} seeds) "
            f"— T = {T_kev:.3g} keV, {r['grid']}", fontsize=13)

        for ax, gi in zip(axes, g_in_indices):
            gp = int(np.argmax(np.abs(S_det_3d[gi, :, :]).sum(axis=-1)))
            det_xi = S_det_3d[gi, gp, :]
            mc_xi = mc_mean_3d[gi, gp, :]
            mc_err = mc_std_3d[gi, gp, :]

            ax.step(xi_centers, det_xi, where="mid", label="Det",
                    linewidth=1.5)
            ax.errorbar(xi_centers, mc_xi, yerr=mc_err, fmt="o", ms=3,
                        capsize=2, label="MC mean±1σ", linewidth=1.0)
            ax.set_xlabel(r"$\xi = \cos\theta$")
            ax.set_ylabel(r"$\sigma(g \to g', \xi)$ [cm$^2$]")
            ax.set_title(f"g={gi}->g'={gp}\n"
                         f"({centers[gi]:.2g}->{centers[gp]:.2g} keV)")
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        fig.tight_layout()
        fname = f"ma_xi_profile_T{T_kev:.3g}keV.png"
        fig.savefig(os.path.join(figs_dir, fname), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        paths.append((T_kev, fname))

    return paths


def plot_angle_heatmaps(results_by_temp, figs_dir):
    """Plot angle-bin heatmaps for selected temperatures."""
    selected_temps = [1.0, 10.0, 100.0]
    paths = []

    for T_kev in selected_temps:
        if T_kev not in results_by_temp:
            continue
        r = results_by_temp[T_kev]
        S_det_3d = r["S_det_3d"]
        bounds_kev = r["bounds_kev"]
        G = S_det_3d.shape[0]

        bins_to_show = [0, NUM_ANGLE_BINS // 2, NUM_ANGLE_BINS - 1]
        xi_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)

        fig, axes = plt.subplots(1, len(bins_to_show),
                                 figsize=(5 * len(bins_to_show), 4))
        fig.suptitle(
            f"Angle-bin heatmaps — T = {T_kev:.3g} keV, {r['grid']}",
            fontsize=13)

        for ax, a in zip(axes, bins_to_show):
            xi_lo = xi_edges[a]
            xi_hi = xi_edges[a + 1]
            mat = S_det_3d[:, :, a]
            vmax = np.max(np.abs(mat))
            if vmax == 0:
                vmax = 1.0
            im = ax.imshow(mat, origin="lower", aspect="equal",
                           cmap="viridis")
            ax.set_xlabel("g' (outgoing)")
            ax.set_ylabel("g (incoming)")
            ax.set_title(f"bin {a}: ξ∈[{xi_lo:.2f},{xi_hi:.2f}]")
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.tight_layout()
        fname = f"ma_heatmap_T{T_kev:.3g}keV.png"
        fig.savefig(os.path.join(figs_dir, fname), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        paths.append((T_kev, fname))

    return paths


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------

def emit_header(lines):
    lines.append("# Multiangle Validation Report")
    lines.append("")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Temperatures: {TEMPERATURES_KEV} keV")
    lines.append(f"- Grids: {list(GRIDS.keys())}")
    lines.append(f"- Angle bins: {NUM_ANGLE_BINS}")
    lines.append(f"- MC samples per seed: {MC_SAMPLES:,}")
    lines.append(f"- MC seeds: {len(MC_SEEDS)}")
    lines.append(f"- Weight function: Planck (cap_x=25)")
    lines.append("")


def emit_sigma_table(lines, results):
    lines.append("## Multiangle Sigma: Det vs MC Ensemble")
    lines.append("")
    lines.append(f"MC: {len(MC_SEEDS)} seeds x {MC_SAMPLES:,} samples, "
                 f"{NUM_ANGLE_BINS} angle bins. "
                 "Errors in units of MC standard deviation.")
    lines.append("")
    lines.append("| T (keV) | Grid | G | RS max-z | RS med-z | "
                 "Elem p95-z | %<3σ | Consistency | "
                 "Det (s) | MC (s) | Pass |")
    lines.append("|---------|------|---|----------|----------|"
                 "-----------|------|-------------|"
                 "---------|--------|------|")

    for r in results:
        rs_pass = r["rs_max_sigma"] < ROW_SUM_SIGMA_TOL
        el_pass = r["elem_within_3sig"] > 0.99
        passed = rs_pass and el_pass and r["neg_rows"] == 0
        tag = "PASS" if passed else "FAIL"
        lines.append(
            f"| {r['T_kev']:8.3g} | {r['grid']:12s} | {r['G']:2d} "
            f"| {r['rs_max_sigma']:8.2f} | {r['rs_median_sigma']:8.2f} "
            f"| {r['elem_p95_sigma']:9.2f} "
            f"| {100*r['elem_within_3sig']:5.1f} "
            f"| {r['consistency']:11.2e} "
            f"| {r['dt_det']:7.1f} | {r['dt_mc']:6.1f} | {tag:4s} |"
        )
    lines.append("")


def emit_perbin_table(lines, results):
    lines.append("## Per-Bin Row-Sum Z-Scores (max across groups)")
    lines.append("")
    xi_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)
    header_bins = " | ".join(
        f"[{xi_edges[a]:.2f},{xi_edges[a+1]:.2f}]"
        for a in range(NUM_ANGLE_BINS))
    lines.append(f"| T (keV) | Grid | {header_bins} |")
    sep_bins = " | ".join("---" for _ in range(NUM_ANGLE_BINS))
    lines.append(f"|---------|------|{sep_bins}|")

    for r in results:
        bin_vals = " | ".join(f"{v:5.2f}" for v in r["bin_rs_max_sigma"])
        lines.append(f"| {r['T_kev']:8.3g} | {r['grid']:12s} | {bin_vals} |")
    lines.append("")


def emit_verdict(lines, results):
    sigma_pass = all(
        r["rs_max_sigma"] < ROW_SUM_SIGMA_TOL
        and r["elem_within_3sig"] > 0.99
        and r["neg_rows"] == 0
        for r in results
    )

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Multiangle sigma: {'PASS' if sigma_pass else 'FAIL'} "
                 f"(row-sum < {ROW_SUM_SIGMA_TOL:.0f}σ, >99% elements <3σ)")
    lines.append(f"- **Overall: {'PASS' if sigma_pass else 'FAIL'}**")
    lines.append("")
    return sigma_pass


# ---------------------------------------------------------------------------
# Worker mode  (--worker N)
# ---------------------------------------------------------------------------

def run_worker(task_idx):
    """Execute a single task and save results to disk."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tasks = build_task_list()
    if task_idx >= len(tasks):
        print(f"ERROR: task_idx {task_idx} out of range (0..{len(tasks)-1})")
        sys.exit(1)

    T_kev, grid_name = tasks[task_idx]
    bounds_kev = GRIDS[grid_name]
    print(f"[worker {task_idx}] T={T_kev} keV  grid={grid_name}")

    metrics = run_one(T_kev, grid_name, bounds_kev)

    out_path = os.path.join(RESULTS_DIR, f"task_{task_idx:03d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"  done in {metrics['dt_det'] + metrics['dt_mc']:.1f}s -> {out_path}")


# ---------------------------------------------------------------------------
# Collect mode  (--collect)
# ---------------------------------------------------------------------------

def collect_results():
    """Load all worker results, generate plots, and write report."""
    gen_dir = os.path.join(ROOT, "reports", "generated")
    figs_dir = os.path.join(gen_dir, "multiangle_figs")
    os.makedirs(figs_dir, exist_ok=True)

    tasks = build_task_list()
    results = []
    missing = []

    for idx in range(len(tasks)):
        path = os.path.join(RESULTS_DIR, f"task_{idx:03d}.pkl")
        if not os.path.exists(path):
            missing.append(idx)
            continue
        with open(path, "rb") as f:
            r = pickle.load(f)
        results.append(r)

    if missing:
        print(f"WARNING: missing results for tasks {missing}")

    print(f"Loaded: {len(results)} multiangle tasks")

    results_by_temp = {}
    for r in results:
        T_kev = r["T_kev"]
        grid = r["grid"]
        G = r["G"]
        if T_kev not in results_by_temp or results_by_temp[T_kev]["G"] < G:
            results_by_temp[T_kev] = r

    xi_paths = plot_xi_profiles(results_by_temp, figs_dir)
    print(f"  {len(xi_paths)} xi profile figures saved.")
    hm_paths = plot_angle_heatmaps(results_by_temp, figs_dir)
    print(f"  {len(hm_paths)} heatmap figures saved.")

    report_text, overall = build_report(results, xi_paths, hm_paths)

    out_path = os.path.join(gen_dir, "multiangle_validation.md")
    with open(out_path, "w") as f:
        f.write(report_text)

    print()
    print(report_text)
    print(f"Report written to {out_path}")
    sys.exit(0 if overall else 1)


def build_report(results, xi_paths, hm_paths):
    """Assemble markdown report from collected data."""
    lines = []
    emit_header(lines)
    emit_sigma_table(lines, results)
    emit_perbin_table(lines, results)

    lines.append("## Angular (ξ) Profiles")
    lines.append("")
    lines.append(f"{NUM_ANGLE_BINS} angle bins. For each temperature, the "
                 "finest available grid is used. Each panel shows the "
                 "outgoing group with largest total cross-section for a "
                 "given incoming group.")
    lines.append("")
    for T_kev, fname in xi_paths:
        lines.append(f"### T = {T_kev:.3g} keV")
        lines.append("")
        lines.append(f"![xi profile T={T_kev:.3g} keV](multiangle_figs/{fname})")
        lines.append("")

    lines.append("## Angle-Bin Heatmaps")
    lines.append("")
    lines.append("Deterministic S(g, g') for selected angle bins "
                 "(backward, mid, forward).")
    lines.append("")
    for T_kev, fname in hm_paths:
        lines.append(f"### T = {T_kev:.3g} keV")
        lines.append("")
        lines.append(f"![heatmap T={T_kev:.3g} keV](multiangle_figs/{fname})")
        lines.append("")

    overall = emit_verdict(lines, results)

    dt_det = sum(r.get("dt_det", 0) for r in results)
    dt_mc = sum(r.get("dt_mc", 0) for r in results)
    lines.append("## Timing")
    lines.append("")
    lines.append(f"  Deterministic total: {dt_det:.1f} s")
    lines.append(f"  Monte Carlo total:   {dt_mc:.1f} s")
    lines.append("")
    return "\n".join(lines), overall


# ---------------------------------------------------------------------------
# Monolithic mode  (no args)
# ---------------------------------------------------------------------------

def run_monolithic():
    """Single-process mode for local testing."""
    t_start = time.perf_counter()

    gen_dir = os.path.join(ROOT, "reports", "generated")
    figs_dir = os.path.join(gen_dir, "multiangle_figs")
    os.makedirs(figs_dir, exist_ok=True)

    print("=" * 72)
    print("  Multiangle Validation  (monolithic)")
    print(f"  OMP: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print("=" * 72, flush=True)

    results = []
    total = len(TEMPERATURES_KEV) * len(GRIDS)
    idx = 0
    for T_kev in TEMPERATURES_KEV:
        for grid_name, bounds_kev in GRIDS.items():
            idx += 1
            print(f"[{idx:2d}/{total}] T={T_kev:8.3g} keV, "
                  f"grid={grid_name} ... ", end="", flush=True)
            try:
                m = run_one(T_kev, grid_name, bounds_kev)
                results.append(m)
                print(f"rs={m['rs_max_sigma']:.1f}σ, "
                      f"p95={m['elem_p95_sigma']:.1f}σ, "
                      f"<3σ={100*m['elem_within_3sig']:.0f}%, "
                      f"det={m['dt_det']:.1f}s, mc={m['dt_mc']:.1f}s")
            except Exception as e:
                print(f"EXCEPTION: {e}")
                results.append({
                    "T_kev": T_kev, "grid": grid_name,
                    "G": len(bounds_kev) - 1,
                    "rs_max_sigma": 999.0, "rs_median_sigma": 999.0,
                    "elem_max_sigma": 999.0, "elem_median_sigma": 999.0,
                    "elem_p95_sigma": 999.0, "elem_within_3sig": 0.0,
                    "bin_rs_max_sigma": [999.0] * NUM_ANGLE_BINS,
                    "bin_rs_med_sigma": [999.0] * NUM_ANGLE_BINS,
                    "consistency": 999.0, "neg_rows": -1,
                    "dt_det": 0.0, "dt_mc": 0.0,
                })

    results_by_temp = {}
    for r in results:
        T_kev = r["T_kev"]
        G = r["G"]
        if T_kev not in results_by_temp or results_by_temp[T_kev]["G"] < G:
            results_by_temp[T_kev] = r

    xi_paths = plot_xi_profiles(results_by_temp, figs_dir)
    print(f"  {len(xi_paths)} xi profile figures saved.", flush=True)
    hm_paths = plot_angle_heatmaps(results_by_temp, figs_dir)
    print(f"  {len(hm_paths)} heatmap figures saved.", flush=True)

    dt_total = time.perf_counter() - t_start
    report_text, overall = build_report(results, xi_paths, hm_paths)

    out_path = os.path.join(gen_dir, "multiangle_validation.md")
    with open(out_path, "w") as f:
        f.write(report_text)

    print()
    print(report_text)
    print(f"\nTotal wall-clock: {dt_total:.1f}s ({dt_total/60:.1f} min)")
    print(f"Report written to {out_path}")
    sys.exit(0 if overall else 1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multiangle det-vs-MC validation report")
    parser.add_argument(
        "--worker", type=int, metavar="IDX",
        help="Run a single task (0..17)")
    parser.add_argument(
        "--collect", action="store_true",
        help="Collect worker results and generate report")
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args.worker)
    elif args.collect:
        collect_results()
    else:
        run_monolithic()


if __name__ == "__main__":
    main()
