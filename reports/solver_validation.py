"""
Solver validation report: demonstrates ComptonKernelSolver effectiveness and robustness.

Generates reports/generated/solver_validation.md with embedded plots covering:
  1. Regime coverage map (method selection vs tau and E'/E)
  2. Accuracy vs Q256 reference
  3. Method selection statistics
  4. Edge-case gallery
  5. Out-of-calibration-domain behavior
  6. Timing comparison
  7. Non-negativity check

Cascade: asymptotic -> power series -> quadrature (NL=256)

Usage:
    python3 reports/solver_validation.py

Output:
    reports/generated/solver_validation.md  (+ .png plots in figs/)
"""
import sys
import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'python'))

from _compton_kernel_solver import ComptonKernelSolver, SolverMethod
from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm

ME_C2 = 9.109383713928e-28 * (2.99792458e10)**2
KEV = 1.602176634e-9

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

solver = ComptonKernelSolver()
quad256 = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)

lines = []


def emit(s=""):
    lines.append(s)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Regime coverage map
# ═══════════════════════════════════════════════════════════════════════════════

def section_regime_map():
    emit("## 1. Regime Coverage Map")
    emit()
    emit("Method selected by the solver across (T, E'/E) space at fixed E=10 keV, xi=0.")
    emit()

    T_vals = np.logspace(-1, 2.5, 40)  # 0.1 to ~316 keV
    ratio_vals = np.logspace(-0.5, 0.7, 30)  # 0.32 to 5.0

    method_map = np.zeros((len(T_vals), len(ratio_vals)), dtype=int)
    E = 10.0 * KEV

    for i, T_keV in enumerate(T_vals):
        tau = T_keV * KEV / ME_C2
        for j, ratio in enumerate(ratio_vals):
            Ep = E * ratio
            try:
                r = solver.sigma_E(E, Ep, 0.0, tau, 1.0)
                method_map[i, j] = r.method_used.value
            except Exception:
                method_map[i, j] = -1

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap('Set1', 3)
    im = ax.pcolormesh(ratio_vals, T_vals, method_map, cmap=cmap, vmin=0, vmax=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("E'/E ratio")
    ax.set_ylabel("T [keV]")
    ax.set_title("Solver Method Selection (E=10 keV, xi=0)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Asymptotic', 'Power', 'Quadrature'])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'solver_regime_map.png'), dpi=120)
    plt.close(fig)

    emit("![Regime coverage map](figs/solver_regime_map.png)")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Accuracy vs Q256 reference
# ═══════════════════════════════════════════════════════════════════════════════

def section_accuracy():
    emit("## 2. Accuracy vs Q256 Reference")
    emit()

    T_vals = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    E_vals = [1.0, 10.0, 50.0, 100.0]
    ratios = [1.01, 1.05, 1.1, 1.5, 2.0, 3.0, 5.0]
    xi_vals = [-0.5, 0.0, 0.5]

    rel_errors = []
    methods_used = []

    for T_keV in T_vals:
        tau = T_keV * KEV / ME_C2
        for E_keV in E_vals:
            E = E_keV * KEV
            for ratio in ratios:
                Ep = E * ratio
                for xi in xi_vals:
                    try:
                        sr = solver.sigma_E(E, Ep, xi, tau, 1.0)
                        qr = quad256.sigma_E(E, Ep, xi, tau, 1.0)
                    except Exception:
                        continue

                    if abs(qr.value) < 1e-300:
                        continue

                    rel_diff = abs(sr.value - qr.value) / abs(qr.value)
                    rel_errors.append(rel_diff)
                    methods_used.append(sr.method_used.value)

    rel_errors = np.array(rel_errors)
    methods_used = np.array(methods_used)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    method_names = ['Asymptotic', 'Power', 'Quadrature']
    for idx, (ax, name) in enumerate(zip(axes, method_names)):
        mask = methods_used == idx
        if mask.sum() > 0:
            data = rel_errors[mask]
            data_clipped = np.clip(data, 1e-16, None)
            ax.hist(np.log10(data_clipped), bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(np.log10(1e-8), color='red', linestyle='--', label='1e-8 target')
            ax.set_title(f"{name} (n={mask.sum()})")
            ax.set_xlabel("log10(|solver - Q256| / |Q256|)")
            ax.legend()
        else:
            ax.set_title(f"{name} (n=0)")
    fig.suptitle("Solver vs Q256 Relative Discrepancy by Method")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'solver_accuracy_hist.png'), dpi=120)
    plt.close(fig)

    emit("![Accuracy histogram](figs/solver_accuracy_hist.png)")
    emit()

    n_below_1e8 = (rel_errors < 1e-8).sum()
    n_below_1e6 = (rel_errors < 1e-6).sum()
    emit(f"- Total points evaluated: {len(rel_errors)}")
    emit(f"- Points with |solver - Q256|/|Q256| < 1e-8: {n_below_1e8} ({100*n_below_1e8/len(rel_errors):.1f}%)")
    emit(f"- Points with |solver - Q256|/|Q256| < 1e-6: {n_below_1e6} ({100*n_below_1e6/len(rel_errors):.1f}%)")
    emit(f"- Maximum discrepancy: {rel_errors.max():.2e}")
    emit(f"- Median discrepancy: {np.median(rel_errors):.2e}")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Fallback statistics
# ═══════════════════════════════════════════════════════════════════════════════

def section_fallback_stats():
    emit("## 3. Method Selection Statistics")
    emit()

    T_vals = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200]
    E_vals = [0.1, 1.0, 10.0, 50.0, 100.0]
    ratios = [1.001, 1.01, 1.05, 1.1, 1.5, 2.0, 3.0, 5.0]
    xi_vals = [-0.5, 0.0, 0.5]

    counts = {'Asymptotic': 0, 'PowerSeries': 0, 'Quadrature': 0}
    n_target_met = 0
    n_clamped = 0
    n_total = 0

    for T_keV in T_vals:
        tau = T_keV * KEV / ME_C2
        for E_keV in E_vals:
            E = E_keV * KEV
            for ratio in ratios:
                Ep = E * ratio
                for xi in xi_vals:
                    try:
                        r = solver.sigma_E(E, Ep, xi, tau, 1.0)
                    except Exception:
                        continue
                    n_total += 1
                    counts[r.method_used.name] += 1
                    if r.target_met:
                        n_target_met += 1
                    if r.clamped:
                        n_clamped += 1

    emit("| Metric | Count | Percentage |")
    emit("|--------|-------|------------|")
    emit(f"| Total points | {n_total} | 100% |")
    for name, count in counts.items():
        emit(f"| Method: {name} | {count} | {100*count/n_total:.1f}% |")
    emit(f"| target_met=True | {n_target_met} | {100*n_target_met/n_total:.1f}% |")
    emit(f"| clamped=True | {n_clamped} | {100*n_clamped/n_total:.1f}% |")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Edge-case gallery
# ═══════════════════════════════════════════════════════════════════════════════

def section_edge_cases():
    emit("## 4. Edge-Case Gallery")
    emit()

    cases = [
        ("xi=0.999, E=10, E'=15, T=10", 10.0, 15.0, 0.999, 10.0),
        ("xi=-0.999, E=10, E'=15, T=10", 10.0, 15.0, -0.999, 10.0),
        ("Near-elastic: E=10, E'=10.001, T=20", 10.0, 10.001, 0.0, 20.0),
        ("High T: E=100, E'=120, T=500", 100.0, 120.0, 0.0, 500.0),
        ("sigma0 underflow: E=1, E'=100, T=0.01", 1.0, 100.0, 0.0, 0.01),
        ("Cancellation: E=1, E'=1.01, xi=0.5, T=100", 1.0, 1.01, 0.5, 100.0),
    ]

    emit("| Case | Value | rel_error | Method | target_met | clamped | tau_alpha_max |")
    emit("|------|-------|-----------|--------|------------|---------|---------------|")

    for label, E_keV, Ep_keV, xi, T_keV in cases:
        E = E_keV * KEV; Ep = Ep_keV * KEV; tau = T_keV * KEV / ME_C2
        try:
            r = solver.sigma_E(E, Ep, xi, tau, 1.0)
            emit(f"| {label} | {r.value:.3e} | {r.estimated_rel_error:.2e} | "
                 f"{r.method_used.name} | {r.target_met} | {r.clamped} | {r.tau_alpha_max:.4f} |")
        except Exception as e:
            emit(f"| {label} | ERROR: {e} | - | - | - | - | - |")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Out-of-calibration-domain behavior
# ═══════════════════════════════════════════════════════════════════════════════

def section_out_of_domain():
    emit("## 5. Out-of-Calibration-Domain Behavior")
    emit()
    emit("Points beyond the calibration grid (tau > 1.0, xi > 0.9, E'/E > 10).")
    emit()

    cases = [
        ("tau=2.0 (T=1022 keV)", 10.0, 15.0, 0.0, 1022.0),
        ("xi=0.999", 10.0, 15.0, 0.999, 10.0),
        ("E'/E=50", 1.0, 50.0, 0.0, 200.0),
        ("E'/E=0.01", 100.0, 1.0, 0.0, 200.0),
    ]

    emit("| Case | Value | rel_error | Method | target_met |")
    emit("|------|-------|-----------|--------|------------|")
    for label, E_keV, Ep_keV, xi, T_keV in cases:
        E = E_keV * KEV; Ep = Ep_keV * KEV; tau = T_keV * KEV / ME_C2
        try:
            r = solver.sigma_E(E, Ep, xi, tau, 1.0)
            emit(f"| {label} | {r.value:.3e} | {r.estimated_rel_error:.2e} | "
                 f"{r.method_used.name} | {r.target_met} |")
        except Exception as e:
            emit(f"| {label} | ERROR | - | - | - |")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Timing comparison
# ═══════════════════════════════════════════════════════════════════════════════

def section_timing():
    emit("## 6. Timing Comparison")
    emit()

    E = 10.0 * KEV
    Ep_arr = np.linspace(5.0, 20.0, 200) * KEV
    xi = 0.0
    n_repeats = 5

    timings = {}

    for T_keV, label in [(1.0, "T=1keV (asymptotic)"), (50.0, "T=50keV (mixed)"), (100.0, "T=100keV (power/quad)")]:
        tau = T_keV * KEV / ME_C2

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            solver.sigma_E_vec(E, Ep_arr, xi, tau, 1.0)
        t_solver = (time.perf_counter() - t0) / n_repeats

        from _compton_kernel_series import ComptonKernelSeries, SeriesMethod
        series = ComptonKernelSeries(SeriesMethod.Auto)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            series.sigma_E_vec(E, Ep_arr, xi, tau, 1.0)
        t_series = (time.perf_counter() - t0) / n_repeats

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for Ep in Ep_arr:
                quad256.sigma_E(E, float(Ep), xi, tau, 1.0)
        t_quad = (time.perf_counter() - t0) / n_repeats

        timings[label] = (t_solver * 1000, t_series * 1000, t_quad * 1000)

    emit("| Regime | Solver (ms) | Series-only (ms) | Q256 (ms) | Solver/Series ratio |")
    emit("|--------|-------------|------------------|-----------|---------------------|")
    for label, (ts, tser, tq) in timings.items():
        ratio = ts / tser if tser > 0 else float('inf')
        emit(f"| {label} | {ts:.1f} | {tser:.1f} | {tq:.1f} | {ratio:.1f}x |")
    emit()
    emit("(200 E' points per measurement, averaged over 5 repeats)")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Non-negativity check
# ═══════════════════════════════════════════════════════════════════════════════

def section_non_negativity():
    emit("## 7. Non-Negativity Check")
    emit()

    T_vals = [0.1, 1, 5, 10, 20, 50, 100]
    E_vals = [1.0, 10.0, 100.0]
    ratios = [1.01, 1.1, 1.5, 2.0, 3.0, 5.0]
    xi_vals = [-0.5, 0.0, 0.5]

    n_total = 0
    n_negative = 0
    n_clamped = 0

    for T_keV in T_vals:
        tau = T_keV * KEV / ME_C2
        for E_keV in E_vals:
            E = E_keV * KEV
            for ratio in ratios:
                Ep = E * ratio
                for xi in xi_vals:
                    try:
                        r = solver.sigma_E(E, Ep, xi, tau, 1.0)
                    except Exception:
                        continue
                    n_total += 1
                    if r.value < 0:
                        n_negative += 1
                    if r.clamped:
                        n_clamped += 1

    emit(f"- Total points checked: {n_total}")
    emit(f"- Negative values (after clamping): {n_negative}")
    emit(f"- Points where clamping was applied: {n_clamped}")
    if n_negative == 0:
        emit(f"- **Result: All values are non-negative.**")
    emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    emit("# Solver Validation Report")
    emit()
    emit("Demonstrates ComptonKernelSolver effectiveness and robustness.")
    emit(f"Target relative tolerance: 1e-8")
    emit()

    section_regime_map()
    section_accuracy()
    section_fallback_stats()
    section_edge_cases()
    section_out_of_domain()
    section_timing()
    section_non_negativity()

    out_path = os.path.join(GEN_DIR, 'solver_validation.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Report written to {out_path}")


if __name__ == '__main__':
    main()
