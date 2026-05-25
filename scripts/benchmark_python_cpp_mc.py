#!/usr/bin/env python3
"""
Benchmark: Pure Python vs C++ pybind11 vs CMMC Monte Carlo.

Produces two separate benchmarks:
  1. Pointwise kernel:  Python sigma_E vs C++ sigma_E
  2. Multigroup S-matrix:  Python+dblquad vs C+++dblquad vs CMMC MC
     with comparison plots for each Pomraning case.

Usage:
    python3 scripts/benchmark_python_cpp_mc.py
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import dblquad
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, 'reports', 'generated')
FIGS_DIR = os.path.join(REPORTS_DIR, 'figs')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'python'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from pycompton.compton_kernel_quadrature import sigma_E as py_sigma_E

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9
KEV_KELVIN = KEV / 1.380649e-16
XI_EPS = 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Pointwise kernel benchmark
# ═══════════════════════════════════════════════════════════════════════════════

POINTWISE_CASES = [
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.5, 0.5, 1.0),
    (1.0, 2.0, -0.5, 1.0),
    (10.0, 8.0, 0.3, 5.0),
    (10.0, 12.0, -0.3, 5.0),
    (50.0, 45.0, 0.0, 20.0),
    (50.0, 55.0, 0.7, 20.0),
    (0.1, 0.08, 0.0, 0.1),
    (0.1, 0.12, -0.8, 0.1),
    (5.0, 5.0, 0.0, 10.0),
    (5.0, 3.0, 0.9, 10.0),
]


def run_pointwise_benchmark(n_repeats=20):
    print("=" * 72)
    print("PART 1: POINTWISE KERNEL BENCHMARK")
    print("=" * 72)
    print(f"  {len(POINTWISE_CASES)} test points x {n_repeats} repeats "
          f"= {len(POINTWISE_CASES) * n_repeats} evaluations per method")
    print()

    cpp_engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)

    # C++ pybind11
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for E_kev, Ep_kev, xi, tau_kev in POINTWISE_CASES:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2
            cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
    cpp_time = time.perf_counter() - t0

    # Python fixed Gauss-Laguerre
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for E_kev, Ep_kev, xi, tau_kev in POINTWISE_CASES:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2
            py_sigma_E(E, Ep, xi, tau, 1.0, NL=128, method="fixed")
    py_fixed_time = time.perf_counter() - t0

    # Python adaptive quad
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for E_kev, Ep_kev, xi, tau_kev in POINTWISE_CASES:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2
            py_sigma_E(E, Ep, xi, tau, 1.0, method="adaptive")
    py_adaptive_time = time.perf_counter() - t0

    n_total = len(POINTWISE_CASES) * n_repeats

    print(f"  C++ pybind11:           {cpp_time:8.4f}s  "
          f"({cpp_time/n_total*1e6:8.1f} us/call)")
    print(f"  Python fixed GL:        {py_fixed_time:8.4f}s  "
          f"({py_fixed_time/n_total*1e6:8.1f} us/call)")
    print(f"  Python adaptive quad:   {py_adaptive_time:8.4f}s  "
          f"({py_adaptive_time/n_total*1e6:8.1f} us/call)")
    print()
    print(f"  Python(fixed)/C++ slowdown:    {py_fixed_time/cpp_time:8.1f}x")
    print(f"  Python(adaptive)/C++ slowdown: {py_adaptive_time/cpp_time:8.1f}x")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Multigroup S-matrix benchmark with Pomraning comparison plots
# ═══════════════════════════════════════════════════════════════════════════════

MBARN = 1e-3 * 1e-24  # millibarn in cm^2

POMRANING_CASES = [
    dict(name="Pomraning_1keV_low",  T_kev=1.0,  emax=75.0,
         ein_kev=[5.0, 10.0, 20.0, 40.0, 60.0],
         ylim=[1, 1e4],
         ref_dir="Pomeraning_1kev_low"),
    dict(name="Pomraning_1keV_high", T_kev=1.0,  emax=340.0,
         ein_kev=[80.0, 120.0, 200.0, 300.0],
         ylim=[1e-2, 1e2],
         ref_dir="Pomeraning_1kev_high"),
    dict(name="Pomraning_20keV_low", T_kev=20.0, emax=140.0,
         ein_kev=[5.0, 10.0, 20.0, 40.0, 60.0],
         ylim=[1e-1, 1e3],
         ref_dir="Pomeraning_20kev_low"),
    dict(name="Pomraning_20keV_high", T_kev=20.0, emax=440.0,
         ein_kev=[80.0, 120.0, 200.0, 300.0],
         ylim=[1e-2, 1e2],
         ref_dir="Pomeraning_20kev_high"),
    dict(name="Pomraning_100keV", T_kev=100.0, emax=500.0,
         ein_kev=[30.0, 75.0, 150.0, 350.0],
         ylim=[1e-3, 1e2],
         ref_dir=None),
]


def make_energy_bins(emax_kev, n_bins=40):
    """Build a moderately dense energy bin grid (in erg)."""
    eb_kev = np.array(sorted(set(
        list(np.linspace(0.01, emax_kev, n_bins))
        + list(np.geomspace(0.01, emax_kev, n_bins // 2))
    )))
    return eb_kev * KEV


def integrate_bin_cpp(engine, E_in, E_lo, E_hi, tau):
    def integrand(Ep, xi):
        return engine.sigma_E(E_in, Ep, xi, tau, 1.0).value
    val, err = dblquad(integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                       lambda xi: E_lo, lambda xi: E_hi,
                       epsabs=1e-35, epsrel=1e-2)
    return 2.0 * np.pi * val


def integrate_bin_py(E_in, E_lo, E_hi, tau, NL=128):
    def integrand(Ep, xi):
        val, _, _ = py_sigma_E(E_in, Ep, xi, tau, 1.0, NL=NL, method="fixed")
        return val
    val, err = dblquad(integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                       lambda xi: E_lo, lambda xi: E_hi,
                       epsabs=1e-35, epsrel=1e-2)
    return 2.0 * np.pi * val


def get_cmmc_matrix(T_kev, eb_erg, num_samples=500000):
    try:
        from _compton_matrix_mc import ComptonMatrixMC
    except ImportError:
        print("  WARNING: CMMC not available, skipping MC benchmark")
        return None

    T_kelvin = T_kev * KEV_KELVIN
    ec = 0.5 * (eb_erg[:-1] + eb_erg[1:])

    compton_engine = ComptonMatrixMC(
        energy_groups_centers=ec.tolist(),
        energy_groups_boundaries=eb_erg.tolist(),
        num_of_samples=num_samples,
        force_detailed_balance=False,
        seed=42,
    )
    S_mat = np.array(compton_engine.calculate_S_matrix(temperature=T_kelvin))
    return S_mat


def load_pomraning_reference(ref_dir):
    """Load Pomraning reference data from text files."""
    ref_base = os.path.join(ROOT, 'external', 'CMMC', 'examples', ref_dir)
    if not os.path.isdir(ref_base):
        return {}
    data = {}
    for fn in os.listdir(ref_base):
        if fn.endswith('.txt'):
            e_kev = float(fn.replace('kev.txt', ''))
            data[e_kev] = np.loadtxt(os.path.join(ref_base, fn), delimiter=',')
    return data


def compute_row_cpp(engine, E_in, eb_erg, tau):
    num_groups = len(eb_erg) - 1
    row = np.zeros(num_groups)
    for gp in range(num_groups):
        row[gp] = integrate_bin_cpp(engine, E_in, eb_erg[gp], eb_erg[gp + 1],
                                    tau)
    return row


def compute_row_py(E_in, eb_erg, tau, NL=128):
    num_groups = len(eb_erg) - 1
    row = np.zeros(num_groups)
    for gp in range(num_groups):
        row[gp] = integrate_bin_py(E_in, eb_erg[gp], eb_erg[gp + 1], tau, NL)
    return row


def run_multigroup_benchmark():
    print("=" * 72)
    print("PART 2: MULTIGROUP S-MATRIX BENCHMARK + POMRANING PLOTS")
    print("=" * 72)
    print()

    cpp_engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

    for case in POMRANING_CASES:
        name = case["name"]
        T_kev = case["T_kev"]
        emax = case["emax"]
        ein_kev = case["ein_kev"]
        ylim = case["ylim"]
        ref_dir = case.get("ref_dir")
        tau = T_kev * KEV / ME_C2

        eb_erg = make_energy_bins(emax, n_bins=40)
        eb_kev_arr = eb_erg / KEV
        ec_erg = 0.5 * (eb_erg[:-1] + eb_erg[1:])
        ewid_kev = np.diff(eb_erg) / KEV
        num_groups = len(eb_erg) - 1

        print(f"--- {name} (T={T_kev}keV, {num_groups} groups, "
              f"{len(ein_kev)} E_in) ---")

        ref_data = load_pomraning_reference(ref_dir) if ref_dir else {}

        # CMMC Monte Carlo (on the same bin grid)
        t0 = time.perf_counter()
        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=200000)
        mc_time = time.perf_counter() - t0
        if S_mc is not None:
            print(f"  CMMC MC:          {mc_time:8.2f}s")

        # Compute rows for each E_in and time them
        cpp_times = []
        py_times = []

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(ein_kev), 10)))

        for idx, e0_kev in enumerate(ein_kev):
            g = np.argmin(np.abs(ec_erg - e0_kev * KEV))
            E_in = ec_erg[g]
            e_label = f"{ec_erg[g] / KEV:.4g}"
            color = colors[idx]

            # C++
            t0 = time.perf_counter()
            row_cpp = compute_row_cpp(cpp_engine, E_in, eb_erg, tau)
            dt = time.perf_counter() - t0
            cpp_times.append(dt)

            # Python
            t0 = time.perf_counter()
            row_py = compute_row_py(E_in, eb_erg, tau)
            dt = time.perf_counter() - t0
            py_times.append(dt)

            # Plot: C++ as solid stair
            sigma_cpp = row_cpp / ewid_kev / MBARN
            ax.stairs(sigma_cpp, edges=eb_kev_arr, color=color,
                      linewidth=1.8,
                      label=f"$E_{{\\mathrm{{in}}}}$={e_label} keV (C++)")

            # Plot: Python as dashed stair
            sigma_py = row_py / ewid_kev / MBARN
            ax.stairs(sigma_py, edges=eb_kev_arr, color=color,
                      linewidth=1.2, linestyle='--')

            # Plot: MC as dotted stair
            if S_mc is not None:
                sigma_mc = S_mc[g, :] / ewid_kev / MBARN
                ax.stairs(sigma_mc, edges=eb_kev_arr, color=color,
                          linewidth=1.0, linestyle=':')

        # Plot Pomraning reference data
        if ref_data:
            ref_plotted = False
            for e_ref, pts in sorted(ref_data.items()):
                lbl = "Pomraning (book)" if not ref_plotted else None
                ax.plot(pts[:, 0], pts[:, 1], 'o', markersize=4, color='k',
                        label=lbl)
                ref_plotted = True

        ax.set_yscale('log')
        ax.set_ylim(ylim)
        ax.set_xlim([0.0, emax])
        ax.grid(True, which='both', alpha=0.3)
        ax.set_title(f"{name}  (T = {T_kev} keV)")
        ax.set_xlabel("final photon energy $E$ [keV]")
        ax.set_ylabel("$\\sigma(E)$ [mbarn/keV]")

        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], color='gray', linewidth=1.8, linestyle='-',
                   label='C++ quadrature'),
            Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--',
                   label='Python quadrature'),
            Line2D([0], [0], color='gray', linewidth=1.0, linestyle=':',
                   label='CMMC Monte Carlo'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=9)

        fig.tight_layout()
        for ext in ('png', 'pdf'):
            path = os.path.join(FIGS_DIR, f"{name}.{ext}")
            fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved plots to reports/generated/figs/{name}.png/pdf")

        # Timing summary
        total_cpp = sum(cpp_times)
        total_py = sum(py_times)
        print(f"  C++ total:   {total_cpp:.2f}s")
        print(f"  Python total: {total_py:.2f}s")
        print(f"  Python/C++ slowdown: {total_py / total_cpp:.1f}x")
        if S_mc is not None:
            print(f"  CMMC MC time: {mc_time:.2f}s")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pointwise_benchmark()
    run_multigroup_benchmark()
