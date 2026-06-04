"""
Multigroup kernel performance report.

Benchmarks ComptonMultigroupKernel computation time as a function of:
  1. Gauss-Legendre quadrature order (4–256)
  2. Number of energy groups (5–80)
  3. Number of angle bins (1–64)
  4. Kernel backend: quadrature vs series(auto)

Usage:
    python3 reports/multigroup_performance.py

Output:
    reports/generated/multigroup_performance.md  (+ .png plots in figs/)
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

import _compton_multigroup as cm
import _compton_kernel_quadrature as cq
from _compton_kernel_solver import ComptonKernelSolver
from _units import kev, kev_kelvin

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


KERNEL_Q64 = cq.ComptonKernelQuadrature(64)
KERNEL_SOLVER = ComptonKernelSolver()
T = 10.0 * kev_kelvin


def make_bounds(n_groups, E_lo_kev=0.1, E_hi_kev=100.0):
    return np.logspace(np.log10(E_lo_kev), np.log10(E_hi_kev), n_groups + 1) * kev


def bench(func, repeats=1):
    """Run func and return wall-clock seconds (best of repeats)."""
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


# ─── Section 1: Scaling with quadrature order ────────────────────────────

def section_quad_order():
    emit('## 1. Scaling with Quadrature Order')
    emit()
    emit('Angle-integrated 20-group matrix (0.1–100 keV), quadrature kernel Q64.')
    emit()

    bounds = make_bounds(20).tolist()
    orders = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    times = []

    emit('| N | Time (s) | N³ | Time / N³ (μs) |')
    emit('|---|----------|-----|----------------|')

    for n in orders:
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=cm.PlanckWeightFunction(cap_x=25.0),
            quad_order_E=n, quad_order_Ep=n, quad_order_mu=n)
        dt = bench(lambda: mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0))
        times.append(dt)
        n3 = n ** 3
        per_n3 = dt / n3 * 1e6
        emit(f'| {n} | {dt:.3f} | {n3} | {per_n3:.2f} |')

    emit()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(orders, times, 'bo-', markersize=5)
    ns = np.array(orders, dtype=float)
    ax1.loglog(ns, times[3] * (ns / orders[3]) ** 3, 'r--', alpha=0.5, label='$\\propto N^3$')
    ax1.set_xlabel('Quadrature order N')
    ax1.set_ylabel('Wall time (s)')
    ax1.set_title('Absolute time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    per_n3 = [t / (n ** 3) for t, n in zip(times, orders)]
    ax2.semilogx(orders, [p * 1e6 for p in per_n3], 'go-', markersize=5)
    ax2.set_xlabel('Quadrature order N')
    ax2.set_ylabel('Time / N³ (μs)')
    ax2.set_title('Cost per quadrature point')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Scaling with quadrature order (20 groups)', y=1.02)
    fig.tight_layout()
    fig_path = save_fig('perf_quad_order.png')

    emit(f'![Quadrature order scaling]({fig_path})')
    emit()


# ─── Section 2: Scaling with number of groups ────────────────────────────

def section_num_groups():
    emit('## 2. Scaling with Number of Groups')
    emit()
    emit('Angle-integrated matrix, quadrature order 8, quadrature kernel Q64.')
    emit()

    group_counts = [5, 10, 20, 30, 40, 60, 80]
    times = []

    emit('| G | Time (s) | G² | Time / G² (μs) |')
    emit('|---|----------|-----|----------------|')

    for g in group_counts:
        bounds = make_bounds(g).tolist()
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=cm.PlanckWeightFunction(cap_x=25.0),
            quad_order_E=8, quad_order_Ep=8, quad_order_mu=8)
        dt = bench(lambda: mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0))
        times.append(dt)
        g2 = g * g
        per_g2 = dt / g2 * 1e6
        emit(f'| {g} | {dt:.3f} | {g2} | {per_g2:.1f} |')

    emit()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(group_counts, times, 'bo-', markersize=5)
    gs = np.array(group_counts, dtype=float)
    ax.loglog(gs, times[2] * (gs / group_counts[2]) ** 2, 'r--', alpha=0.5, label='$\\propto G^2$')
    ax.set_xlabel('Number of groups G')
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Scaling with group count (N=8)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = save_fig('perf_num_groups.png')

    emit(f'![Group count scaling]({fig_path})')
    emit()


# ─── Section 3: Scaling with angle bins ──────────────────────────────────

def section_angle_bins():
    emit('## 3. Scaling with Number of Angle Bins')
    emit()
    emit('20-group matrix, quadrature order 8, quadrature kernel Q64.')
    emit()

    bounds = make_bounds(20).tolist()
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds,
        weight_function=cm.PlanckWeightFunction(cap_x=25.0),
        quad_order_E=8, quad_order_Ep=8, quad_order_mu=8)

    bin_counts = [1, 2, 4, 8, 16, 32, 64]
    times_bins = []
    times_integ = []

    dt_int = bench(lambda: mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0))

    emit('| N_bins | Time (s) | Angle-integrated (s) | Ratio |')
    emit('|--------|----------|---------------------|-------|')

    for nb in bin_counts:
        dt = bench(lambda: mg.compute_sigma_matrix(KERNEL_Q64, num_angle_bins=nb, T=T, Ne=1.0))
        times_bins.append(dt)
        emit(f'| {nb} | {dt:.3f} | {dt_int:.3f} | {dt / dt_int:.2f} |')

    emit()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(bin_counts, times_bins, 'bo-', markersize=5, label='Multiangle')
    ax.axhline(dt_int, color='r', linestyle='--', alpha=0.7, label='Angle-integrated')
    ns = np.array(bin_counts, dtype=float)
    ax.loglog(ns, times_bins[0] * ns, 'g--', alpha=0.4, label='$\\propto N_{bins}$')
    ax.set_xlabel('Number of angle bins')
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Scaling with angle bins (20 groups, N=8)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = save_fig('perf_angle_bins.png')

    emit(f'![Angle bins scaling]({fig_path})')
    emit()


# ─── Section 4: Quadrature vs Series kernel ──────────────────────────────

def section_kernel_comparison():
    emit('## 4. Quadrature vs Series (Auto) Kernel Timing')
    emit()
    emit('Angle-integrated matrix at various group counts and quadrature orders.')
    emit()

    configs = [
        (10, 8),
        (20, 8),
        (40, 8),
        (20, 16),
        (20, 32),
    ]

    emit('| G | N | Quadrature (s) | Series Auto (s) | Speedup |')
    emit('|---|---|----------------|-----------------|---------|')

    quad_times = []
    series_times = []
    labels = []

    for g, n in configs:
        bounds = make_bounds(g).tolist()
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=bounds,
            weight_function=cm.PlanckWeightFunction(cap_x=25.0),
            quad_order_E=n, quad_order_Ep=n, quad_order_mu=n)

        dt_q = bench(lambda: mg.compute_sigma_matrix(KERNEL_Q64, T=T, Ne=1.0))
        dt_s = bench(lambda: mg.compute_sigma_matrix(KERNEL_SOLVER, T=T, Ne=1.0))
        speedup = dt_q / dt_s if dt_s > 0 else float('inf')

        quad_times.append(dt_q)
        series_times.append(dt_s)
        labels.append(f'G={g},N={n}')

        emit(f'| {g} | {n} | {dt_q:.3f} | {dt_s:.3f} | {speedup:.2f}x |')

    emit()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width / 2, quad_times, width, label='Quadrature (Q64)', color='steelblue')
    ax.bar(x + width / 2, series_times, width, label='Series (Auto)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Kernel backend comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig_path = save_fig('perf_kernel_comparison.png')

    emit(f'![Kernel comparison]({fig_path})')
    emit()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    emit('# Multigroup Kernel Performance Report')
    emit()
    emit('Benchmarks of `ComptonMultigroupKernel` computation time at T = 10 keV.')
    emit()

    section_quad_order()
    section_num_groups()
    section_angle_bins()
    section_kernel_comparison()

    md_path = os.path.join(GEN_DIR, 'multigroup_performance.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Report written to {md_path}')


if __name__ == '__main__':
    main()
