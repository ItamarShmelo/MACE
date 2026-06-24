"""
Multigroup solver validation report.

Exercises the refactored ComptonKernelSolver dispatch through full multigroup
integration, comparing deterministic matrices against internal Monte Carlo
reference across 6 temperatures and 3 energy grids (18 combinations).

Supports three execution modes:
  --worker N   Run a single task (for SLURM array jobs)
  --collect    Aggregate worker results and generate report + plots
  (no args)    Run everything in one process (local / single-job mode)

Usage (distributed):
    # launch.sh submits array workers + dependent collect job
    bash reports/run_multigroup_validation.sh

Usage (single process):
    OMP_NUM_THREADS=16 python3.12 reports/multigroup_solver_validation.py
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
from _units import kev, kev_kelvin, sigma_thomson

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KERNEL = ComptonKernelSolver()

TEMPERATURES_KEV = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]

GRIDS = {
    "coarse_4g": np.array([1, 5, 10, 50], dtype=float),
    "medium_10g": np.logspace(np.log10(0.5), np.log10(100), 11),
    "fine_24g": np.logspace(np.log10(0.1), np.log10(200), 25),
}

MC_SAMPLES = 5_000_000
MC_SEEDS = [42, 137, 271, 577, 1009, 1543, 2027, 2741, 3571, 4219]
NUM_ANGLE_BINS = 8

DERIV_TEMPS_KEV = [1.0, 10.0, 100.0]
DERIV_GRID = "coarse_4g"

ROW_SUM_SIGMA_TOL = 3.0
ELEM_SIGMA_TOL = 3.0
DERIV_ROW_SUM_SIGMA_TOL = 3.0

PROFILE_GRID = "medium_10g"
MU_PROFILE_GRID = "coarse_4g"
MU_PROFILE_TEMPS_KEV = [0.1, 10.0, 100.0]

RESULTS_DIR = os.path.join(ROOT, "reports", "generated", "results")


def build_task_list():
    """Build flat task list for SLURM array dispatch.

    Returns list of (task_type, T_kev, grid_name) tuples.
    Index ranges: 0..17 sigma, 18..20 deriv, 21..23 mu.
    """
    tasks = []
    for T_kev in TEMPERATURES_KEV:
        for grid_name in GRIDS:
            tasks.append(("sigma", T_kev, grid_name))
    for T_kev in DERIV_TEMPS_KEV:
        tasks.append(("deriv", T_kev, DERIV_GRID))
    for T_kev in MU_PROFILE_TEMPS_KEV:
        tasks.append(("mu", T_kev, MU_PROFILE_GRID))
    return tasks


def get_det_config(T_kev):
    """Select integration config appropriate for the temperature."""
    if T_kev < 0.1:
        return cm.MGIntegrationConfig(
            base_order=48,
            peak_max_depth=7,
            mu_order=96,
            integration_tolerance=1e-6,
            cutoff_ratio=1e-10,
            cold_temperature_order=48,
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
# Comparison metrics
# ---------------------------------------------------------------------------

def run_mc_ensemble(bounds_kev, T_K, seeds, compute="sigma"):
    """Run MC with multiple seeds and return (mean, std, stack)."""
    G = len(bounds_kev) - 1
    runs = []
    for seed in seeds:
        mc_obj = make_mc(bounds_kev, MC_SAMPLES, seed)
        if compute == "sigma":
            S = np.array(mc_obj.compute_sigma_matrix(T=T_K, Ne=1.0))
        else:
            S = np.array(mc_obj.compute_dsigma_dT_matrix(T=T_K, Ne=1.0))
        runs.append(S.reshape(G, G))
    stack = np.stack(runs)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1), stack


def run_mc_ensemble_3d(bounds_kev, T_K, seeds, num_angle_bins):
    """Run angle-binned MC with multiple seeds."""
    G = len(bounds_kev) - 1
    runs = []
    for seed in seeds:
        mc_obj = make_mc(bounds_kev, MC_SAMPLES, seed)
        S = np.array(mc_obj.compute_sigma_matrix(
            num_angle_bins=num_angle_bins, T=T_K, Ne=1.0))
        runs.append(S.reshape(G, G, num_angle_bins))
    stack = np.stack(runs)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1), stack


def compare_matrices(S_det, mc_mean, mc_std):
    """Compare det vs MC ensemble using z-scores (number of sigma)."""
    G = S_det.shape[0]

    rs_det = S_det.sum(axis=-1)
    rs_mc_mean = mc_mean.sum(axis=-1)
    rs_mc_std = np.sqrt((mc_std ** 2).sum(axis=-1))
    rs_sigma = np.where(rs_mc_std > 0,
                        np.abs(rs_det - rs_mc_mean) / rs_mc_std,
                        0.0)

    elem_sigma = np.where(mc_std > 0,
                          np.abs(S_det - mc_mean) / mc_std,
                          0.0)

    rs_rel = np.abs(rs_det - rs_mc_mean) / np.maximum(np.abs(rs_mc_mean), 1e-300)
    neg_rows = int(np.sum(rs_det < 0))

    return {
        "G": G,
        "rs_max_sigma": float(np.max(rs_sigma)),
        "rs_median_sigma": float(np.median(rs_sigma)),
        "rs_max_rel": float(np.max(rs_rel)),
        "elem_max_sigma": float(np.max(elem_sigma)),
        "elem_median_sigma": float(np.median(elem_sigma)),
        "elem_p95_sigma": float(np.percentile(elem_sigma, 95)),
        "elem_within_3sig": float(np.mean(elem_sigma < 3.0)),
        "neg_rows": neg_rows,
    }


# ---------------------------------------------------------------------------
# Single combination runner
# ---------------------------------------------------------------------------

def run_one(T_kev, grid_name, bounds_kev, keep_matrices=False):
    """Compute det and MC ensemble, return comparison metrics and timing."""
    T_K = T_kev * kev_kelvin
    config = get_det_config(T_kev)
    G = len(bounds_kev) - 1

    t0 = time.perf_counter()
    det_obj = make_det(bounds_kev, config)
    S_det = np.array(det_obj.compute_sigma_matrix(
        kernel=KERNEL, T=T_K, Ne=1.0)).reshape(G, G)
    dt_det = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_mean, mc_std, _ = run_mc_ensemble(bounds_kev, T_K, MC_SEEDS)
    dt_mc = time.perf_counter() - t0

    metrics = compare_matrices(S_det, mc_mean, mc_std)
    metrics["dt_det"] = dt_det
    metrics["dt_mc"] = dt_mc
    if keep_matrices:
        metrics["S_det"] = S_det
        metrics["S_mc"] = mc_mean
        metrics["S_mc_std"] = mc_std
        metrics["bounds_kev"] = bounds_kev
    return metrics


def run_derivative(T_kev, bounds_kev):
    """Compute derivative matrices and compare against MC ensemble."""
    T_K = T_kev * kev_kelvin
    config = get_det_config(T_kev)
    G = len(bounds_kev) - 1

    det_obj = make_det(bounds_kev, config)
    dS_det = np.array(det_obj.compute_dsigma_dT_matrix(
        kernel=KERNEL, T=T_K, Ne=1.0)).reshape(G, G)

    mc_mean, mc_std, _ = run_mc_ensemble(bounds_kev, T_K, MC_SEEDS, compute="dsigma")

    rs_det = dS_det.sum(axis=-1)
    rs_mc_mean = mc_mean.sum(axis=-1)
    rs_mc_std = np.sqrt((mc_std ** 2).sum(axis=-1))
    rs_sigma = np.where(rs_mc_std > 0,
                        np.abs(rs_det - rs_mc_mean) / rs_mc_std,
                        0.0)

    return {
        "T_kev": T_kev,
        "rs_max_sigma": float(np.max(rs_sigma)),
        "rs_median_sigma": float(np.median(rs_sigma)),
    }


# ---------------------------------------------------------------------------
# Profile plots
# ---------------------------------------------------------------------------

def group_centers_kev(bounds_kev):
    return np.sqrt(bounds_kev[:-1] * bounds_kev[1:])


def plot_ep_profiles(results_with_matrices, figs_dir):
    """Plot E' scattering profiles (rows of the sigma matrix) for each T.

    For each temperature, shows det vs MC for 3 representative incoming
    groups on the PROFILE_GRID.
    """
    paths = []
    for r in results_with_matrices:
        T_kev = r["T_kev"]
        S_det = r["S_det"]
        S_mc = r["S_mc"]
        S_mc_std = r["S_mc_std"]
        bounds = r["bounds_kev"]
        centers = group_centers_kev(bounds)
        G = len(centers)

        g_indices = [0, G // 2, G - 1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"E' profile: det vs MC ({len(MC_SEEDS)} seeds) — T = {T_kev:.3g} keV, {PROFILE_GRID}",
                     fontsize=13)

        for ax, gi in zip(axes, g_indices):
            x = np.arange(G)
            width = 0.35
            det_row = S_det[gi, :]
            mc_row = S_mc[gi, :]
            mc_err = S_mc_std[gi, :]

            ax.bar(x - width / 2, det_row, width, label="Det", alpha=0.8)
            ax.bar(x + width / 2, mc_row, width, label="MC mean", alpha=0.8)
            ax.errorbar(x + width / 2, mc_row, yerr=mc_err, fmt="none",
                        ecolor="black", capsize=2, linewidth=0.8)
            ax.set_xlabel("outgoing group g'")
            ax.set_ylabel(r"$\sigma(g \to g')$ [cm$^2$]")
            ax.set_title(f"g={gi} (E={centers[gi]:.2g} keV)")
            ax.set_xticks(x)
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        fig.tight_layout()
        fname = f"ep_profile_T{T_kev:.3g}keV.png"
        path = os.path.join(figs_dir, fname)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append((T_kev, fname))

    return paths


def plot_mu_profiles(figs_dir):
    """Plot angular (mu) scattering profiles for representative cases.

    Computes multiangle matrices on the MU_PROFILE_GRID and plots
    the angular distribution for selected (g, g') pairs.
    """
    bounds_kev = GRIDS[MU_PROFILE_GRID]
    G = len(bounds_kev) - 1
    centers = group_centers_kev(bounds_kev)
    mu_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    paths = []
    for T_kev in MU_PROFILE_TEMPS_KEV:
        T_K = T_kev * kev_kelvin
        config = get_det_config(T_kev)

        det_obj = make_det(bounds_kev, config)
        S_det_3d = np.array(det_obj.compute_sigma_matrix(
            kernel=KERNEL, num_angle_bins=NUM_ANGLE_BINS, T=T_K, Ne=1.0
        )).reshape(G, G, NUM_ANGLE_BINS)

        mc_mean_3d, mc_std_3d, _ = run_mc_ensemble_3d(
            bounds_kev, T_K, MC_SEEDS, NUM_ANGLE_BINS)

        g_in_indices = [0, G // 2, G - 1] if G > 2 else list(range(G))

        fig, axes = plt.subplots(1, len(g_in_indices), figsize=(5 * len(g_in_indices), 4))
        if len(g_in_indices) == 1:
            axes = [axes]
        fig.suptitle(f"mu profile: det vs MC ({len(MC_SEEDS)} seeds) — T = {T_kev:.3g} keV, {MU_PROFILE_GRID}",
                     fontsize=13)

        for ax, gi in zip(axes, g_in_indices):
            gp = int(np.argmax(np.abs(S_det_3d[gi, :, :]).sum(axis=-1)))
            det_mu = S_det_3d[gi, gp, :]
            mc_mu = mc_mean_3d[gi, gp, :]
            mc_err = mc_std_3d[gi, gp, :]

            ax.step(mu_centers, det_mu, where="mid", label="Det", linewidth=1.5)
            ax.errorbar(mu_centers, mc_mu, yerr=mc_err, fmt="o", ms=3,
                        capsize=2, label=f"MC mean±1σ", linewidth=1.0)
            ax.set_xlabel(r"$\mu = \cos\theta$")
            ax.set_ylabel(r"$\sigma(g \to g', \mu)$ [cm$^2$]")
            ax.set_title(f"g={gi}->g'={gp}\n"
                         f"({centers[gi]:.2g}->{centers[gp]:.2g} keV)")
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        fig.tight_layout()
        fname = f"mu_profile_T{T_kev:.3g}keV.png"
        path = os.path.join(figs_dir, fname)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append((T_kev, fname))

    return paths


def plot_mu_profiles_from_data(mu_data_list, figs_dir):
    """Plot mu profiles from pre-computed worker data (collect mode)."""
    mu_edges = np.linspace(-1, 1, NUM_ANGLE_BINS + 1)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    paths = []
    for data in mu_data_list:
        T_kev = data["T_kev"]
        S_det_3d = data["S_det_3d"]
        mc_mean_3d = data["mc_mean_3d"]
        mc_std_3d = data["mc_std_3d"]
        bounds_kev = data["bounds_kev"]
        centers = group_centers_kev(bounds_kev)
        G = len(centers)

        g_in_indices = [0, G // 2, G - 1] if G > 2 else list(range(G))

        fig, axes = plt.subplots(1, len(g_in_indices),
                                 figsize=(5 * len(g_in_indices), 4))
        if len(g_in_indices) == 1:
            axes = [axes]
        fig.suptitle(
            f"mu profile: det vs MC ({len(MC_SEEDS)} seeds) "
            f"— T = {T_kev:.3g} keV", fontsize=13)

        for ax, gi in zip(axes, g_in_indices):
            gp = int(np.argmax(np.abs(S_det_3d[gi, :, :]).sum(axis=-1)))
            det_mu = S_det_3d[gi, gp, :]
            mc_mu = mc_mean_3d[gi, gp, :]
            mc_err = mc_std_3d[gi, gp, :]

            ax.step(mu_centers, det_mu, where="mid", label="Det", linewidth=1.5)
            ax.errorbar(mu_centers, mc_mu, yerr=mc_err, fmt="o", ms=3,
                        capsize=2, label="MC mean±1σ", linewidth=1.0)
            ax.set_xlabel(r"$\mu = \cos\theta$")
            ax.set_ylabel(r"$\sigma(g \to g', \mu)$ [cm$^2$]")
            ax.set_title(f"g={gi}->g'={gp}\n"
                         f"({centers[gi]:.2g}->{centers[gp]:.2g} keV)")
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        fig.tight_layout()
        fname = f"mu_profile_T{T_kev:.3g}keV.png"
        path = os.path.join(figs_dir, fname)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append((T_kev, fname))

    return paths


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------

def emit_header(lines):
    lines.append("# Multigroup Solver Validation Report")
    lines.append("")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    lines.append("")
    lines.append("## Solver Configuration")
    lines.append("")
    lines.append(f"- `ASYMP_TAU_ALPHA_THRESHOLD`: 0.035")
    lines.append(f"- `power_series_self_tol`: 1e-7")
    lines.append(f"- `asymp_self_tol`: 1e-7")
    lines.append(f"- Dispatch: error-driven cascade (A1 -> A2 -> P1 -> P2 -> P3)")
    lines.append("")
    lines.append("## Test Matrix")
    lines.append("")
    lines.append(f"- Temperatures: {TEMPERATURES_KEV} keV")
    lines.append(f"- Grids: {list(GRIDS.keys())}")
    lines.append(f"- MC samples per seed: {MC_SAMPLES:,}")
    lines.append(f"- MC seeds: {len(MC_SEEDS)}")
    lines.append(f"- Weight function: Planck (cap_x=25)")
    lines.append("")


def emit_sigma_table(lines, results):
    lines.append("## Sigma Matrix: Det vs MC Ensemble")
    lines.append("")
    lines.append(f"MC: {len(MC_SEEDS)} seeds x {MC_SAMPLES:,} samples. "
                 "Errors reported in units of MC standard deviation (sigma).")
    lines.append("")
    lines.append("| T (keV) | Grid | G | RS max-sigma | RS rel | "
                 "Elem max-sigma | Elem p95 | %<3sig | Det (s) | MC (s) | Pass |")
    lines.append("|---------|------|---|--------------|--------|"
                 "----------------|----------|--------|---------|--------|------|")
    for r in results:
        rs_pass = r["rs_max_sigma"] < ROW_SUM_SIGMA_TOL
        el_pass = r["elem_within_3sig"] > 0.99
        passed = rs_pass and r["neg_rows"] == 0
        tag = "PASS" if passed else "FAIL"
        lines.append(
            f"| {r['T_kev']:8.3g} | {r['grid']:12s} | {r['G']:2d} "
            f"| {r['rs_max_sigma']:12.2f} | {r['rs_max_rel']:6.1e} "
            f"| {r['elem_max_sigma']:14.1f} | {r['elem_p95_sigma']:8.2f} "
            f"| {100*r['elem_within_3sig']:5.1f}% "
            f"| {r['dt_det']:7.1f} | {r['dt_mc']:6.1f} | {tag:4s} |"
        )
    lines.append("")


def emit_deriv_table(lines, deriv_results):
    lines.append("## Derivative Matrix: Det vs MC Ensemble")
    lines.append("")
    lines.append(f"Grid: `{DERIV_GRID}`, temperatures: {DERIV_TEMPS_KEV} keV, "
                 f"{len(MC_SEEDS)} MC seeds")
    lines.append("")
    lines.append("| T (keV) | RS max-sigma | RS med-sigma | Pass |")
    lines.append("|---------|--------------|--------------|------|")
    for r in deriv_results:
        passed = r["rs_max_sigma"] < DERIV_ROW_SUM_SIGMA_TOL
        tag = "PASS" if passed else "FAIL"
        lines.append(
            f"| {r['T_kev']:8.3g} "
            f"| {r['rs_max_sigma']:12.2f} | {r['rs_median_sigma']:12.2f} "
            f"| {tag:4s} |"
        )
    lines.append("")


def emit_verdict(lines, results, deriv_results):
    sigma_pass = all(
        r["rs_max_sigma"] < ROW_SUM_SIGMA_TOL
        and r["neg_rows"] == 0
        for r in results
    )
    deriv_pass = all(
        r["rs_max_sigma"] < DERIV_ROW_SUM_SIGMA_TOL
        for r in deriv_results
    )
    overall = sigma_pass and deriv_pass

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Sigma:      {'PASS' if sigma_pass else 'FAIL'} "
                 f"(row-sum < {ROW_SUM_SIGMA_TOL:.0f} sigma)")
    lines.append(f"- Derivative: {'PASS' if deriv_pass else 'FAIL'} "
                 f"(row-sum < {DERIV_ROW_SUM_SIGMA_TOL:.0f} sigma)")
    lines.append(f"- **Overall:  {'PASS' if overall else 'FAIL'}**")
    lines.append("")
    return overall


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

    task_type, T_kev, grid_name = tasks[task_idx]
    bounds_kev = GRIDS[grid_name]
    t0 = time.perf_counter()
    print(f"[worker {task_idx}] {task_type}  T={T_kev} keV  grid={grid_name}")

    result = {"task_type": task_type, "task_idx": task_idx,
              "T_kev": T_kev, "grid_name": grid_name}

    if task_type == "sigma":
        keep = (grid_name == PROFILE_GRID)
        m = run_one(T_kev, grid_name, bounds_kev, keep_matrices=keep)
        m["T_kev"] = T_kev
        m["grid"] = grid_name
        result["metrics"] = m

    elif task_type == "deriv":
        dr = run_derivative(T_kev, bounds_kev)
        result["metrics"] = dr

    elif task_type == "mu":
        T_K = T_kev * kev_kelvin
        config = get_det_config(T_kev)
        G = len(bounds_kev) - 1
        det_obj = make_det(bounds_kev, config)
        S_det_3d = np.array(det_obj.compute_sigma_matrix(
            kernel=KERNEL, num_angle_bins=NUM_ANGLE_BINS, T=T_K, Ne=1.0
        )).reshape(G, G, NUM_ANGLE_BINS)
        mc_mean_3d, mc_std_3d, _ = run_mc_ensemble_3d(
            bounds_kev, T_K, MC_SEEDS, NUM_ANGLE_BINS)
        result["S_det_3d"] = S_det_3d
        result["mc_mean_3d"] = mc_mean_3d
        result["mc_std_3d"] = mc_std_3d
        result["bounds_kev"] = bounds_kev

    dt = time.perf_counter() - t0
    result["wall_time"] = dt

    out_path = os.path.join(RESULTS_DIR, f"task_{task_idx:03d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  done in {dt:.1f}s -> {out_path}")


# ---------------------------------------------------------------------------
# Collect mode  (--collect)
# ---------------------------------------------------------------------------

def build_report(sigma_results, profile_results, deriv_results,
                 ep_paths, mu_paths):
    """Assemble markdown report text from collected data."""
    lines = []
    emit_header(lines)
    emit_sigma_table(lines, sigma_results)

    lines.append("## E' Profiles (energy transfer spectra)")
    lines.append("")
    lines.append(f"Grid: `{PROFILE_GRID}`. "
                 f"Each panel shows one incoming group; bars compare det vs MC mean "
                 f"({len(MC_SEEDS)} seeds). Error bars show ±1 MC standard deviation.")
    lines.append("")
    for T_kev, fname in ep_paths:
        lines.append(f"### T = {T_kev:.3g} keV")
        lines.append("")
        lines.append(f"![E' profile T={T_kev:.3g} keV](figs/{fname})")
        lines.append("")

    lines.append("## mu Profiles (angular distributions)")
    lines.append("")
    lines.append(f"Grid: `{MU_PROFILE_GRID}`, {NUM_ANGLE_BINS} angle bins. "
                 "For each incoming group, the outgoing group with largest "
                 "cross-section is selected.")
    lines.append("")
    for T_kev, fname in mu_paths:
        lines.append(f"### T = {T_kev:.3g} keV")
        lines.append("")
        lines.append(f"![mu profile T={T_kev:.3g} keV](figs/{fname})")
        lines.append("")

    emit_deriv_table(lines, deriv_results)
    overall = emit_verdict(lines, sigma_results, deriv_results)

    dt_det = sum(r.get("dt_det", 0) for r in sigma_results)
    dt_mc = sum(r.get("dt_mc", 0) for r in sigma_results)
    lines.append("## Timing")
    lines.append("")
    lines.append(f"  Deterministic total: {dt_det:.1f} s")
    lines.append(f"  Monte Carlo total:   {dt_mc:.1f} s")
    lines.append("")
    return "\n".join(lines), overall


def collect_results():
    """Load all worker results, generate plots, and write report."""
    gen_dir = os.path.join(ROOT, "reports", "generated")
    figs_dir = os.path.join(gen_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    tasks = build_task_list()
    sigma_results, profile_results, deriv_results, mu_data = [], [], [], []
    missing = []

    for idx in range(len(tasks)):
        path = os.path.join(RESULTS_DIR, f"task_{idx:03d}.pkl")
        if not os.path.exists(path):
            missing.append(idx)
            continue
        with open(path, "rb") as f:
            r = pickle.load(f)
        if r["task_type"] == "sigma":
            sigma_results.append(r["metrics"])
            if r["grid_name"] == PROFILE_GRID:
                profile_results.append(r["metrics"])
        elif r["task_type"] == "deriv":
            deriv_results.append(r["metrics"])
        elif r["task_type"] == "mu":
            mu_data.append(r)

    if missing:
        print(f"WARNING: missing results for tasks {missing}")

    print(f"Loaded: {len(sigma_results)} sigma, {len(deriv_results)} deriv, "
          f"{len(mu_data)} mu")

    ep_paths = plot_ep_profiles(profile_results, figs_dir)
    print(f"  {len(ep_paths)} E' profile figures saved.")
    mu_paths = plot_mu_profiles_from_data(mu_data, figs_dir)
    print(f"  {len(mu_paths)} mu profile figures saved.")

    report_text, overall = build_report(
        sigma_results, profile_results, deriv_results, ep_paths, mu_paths)

    out_path = os.path.join(gen_dir, "multigroup_solver_validation.md")
    with open(out_path, "w") as f:
        f.write(report_text)

    print()
    print(report_text)
    print(f"Report written to {out_path}")
    sys.exit(0 if overall else 1)


# ---------------------------------------------------------------------------
# Monolithic mode  (no args, runs everything in one process)
# ---------------------------------------------------------------------------

def run_monolithic():
    """Original single-process mode for local testing."""
    t_start = time.perf_counter()

    gen_dir = os.path.join(ROOT, "reports", "generated")
    figs_dir = os.path.join(gen_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    print("=" * 72)
    print("  Multigroup Solver Validation  (monolithic)")
    print(f"  MC workers: {MC_WORKERS}  OMP: "
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print("=" * 72, flush=True)

    results, profile_results = [], []
    total = len(TEMPERATURES_KEV) * len(GRIDS)
    idx = 0
    for T_kev in TEMPERATURES_KEV:
        for grid_name, bounds_kev in GRIDS.items():
            idx += 1
            keep = (grid_name == PROFILE_GRID)
            print(f"[{idx:2d}/{total}] T={T_kev:8.3g} keV, grid={grid_name} ... ",
                  end="", flush=True)
            try:
                m = run_one(T_kev, grid_name, bounds_kev, keep_matrices=keep)
                m["T_kev"] = T_kev
                m["grid"] = grid_name
                results.append(m)
                if keep:
                    profile_results.append(m)
                print(f"rs={m['rs_max_sigma']:.1f}σ, "
                      f"elem_max={m['elem_max_sigma']:.1f}σ, "
                      f"<3σ={100*m['elem_within_3sig']:.0f}%, "
                      f"det={m['dt_det']:.1f}s, mc={m['dt_mc']:.1f}s")
            except Exception as e:
                print(f"EXCEPTION: {e}")
                results.append({
                    "T_kev": T_kev, "grid": grid_name,
                    "G": len(bounds_kev) - 1,
                    "rs_max_sigma": 999.0, "rs_median_sigma": 999.0,
                    "rs_max_rel": 999.0,
                    "elem_max_sigma": 999.0, "elem_median_sigma": 999.0,
                    "elem_p95_sigma": 999.0, "elem_within_3sig": 0.0,
                    "neg_rows": -1, "dt_det": 0.0, "dt_mc": 0.0,
                })

    print(flush=True)
    ep_paths = plot_ep_profiles(profile_results, figs_dir)
    print(f"  {len(ep_paths)} E' profile figures saved.", flush=True)
    mu_paths = plot_mu_profiles(figs_dir)
    print(f"  {len(mu_paths)} mu profile figures saved.", flush=True)

    deriv_results = []
    deriv_bounds = GRIDS[DERIV_GRID]
    for T_kev in DERIV_TEMPS_KEV:
        print(f"[deriv] T={T_kev:8.3g} keV ... ", end="", flush=True)
        try:
            dr = run_derivative(T_kev, deriv_bounds)
            deriv_results.append(dr)
            print(f"rs={dr['rs_max_sigma']:.1f}σ")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            deriv_results.append({
                "T_kev": T_kev,
                "rs_max_sigma": 999.0, "rs_median_sigma": 999.0,
            })

    dt_total = time.perf_counter() - t_start
    report_text, overall = build_report(
        results, profile_results, deriv_results, ep_paths, mu_paths)

    out_path = os.path.join(gen_dir, "multigroup_solver_validation.md")
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
        description="Multigroup solver validation report")
    parser.add_argument(
        "--worker", type=int, metavar="IDX",
        help="Run a single SLURM array task (0..23)")
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
