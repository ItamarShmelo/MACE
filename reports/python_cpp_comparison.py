"""
Python vs C++ vs CMMC Monte Carlo comparison report.

Generates a markdown report with embedded plots covering:
  1. Pointwise kernel agreement  (Python fixed-GL / adaptive vs C++)
  2. Pomraning multigroup S-matrix comparison plots
  3. Timing comparison across all three implementations

Usage:
    python3 reports/python_cpp_comparison.py

Output:
    reports/generated/python_cpp_comparison.md  (+ .png plots)
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
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'python'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from pycompton.compton_kernel_quadrature import sigma_E as py_sigma_E

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9
KEV_KELVIN = KEV / 1.380649e-16
XI_EPS = 1e-10
MBARN = 1e-3 * 1e-24

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Pointwise kernel agreement
# ═══════════════════════════════════════════════════════════════════════════════

POINTWISE_CASES = [
    # (E_kev, Ep_kev, xi, T_kev)
    (1.0,   1.0,   0.0,   1.0),
    (1.0,   0.5,   0.5,   1.0),
    (1.0,   2.0,  -0.5,   1.0),
    (10.0,  8.0,   0.3,   5.0),
    (10.0,  12.0, -0.3,   5.0),
    (50.0,  45.0,  0.0,  20.0),
    (50.0,  55.0,  0.7,  20.0),
    (0.1,   0.08,  0.0,   0.1),
    (0.1,   0.12, -0.8,   0.1),
    (5.0,   5.0,   0.0,  10.0),
    (5.0,   3.0,   0.9,  10.0),
    (100.0, 80.0,  0.0, 100.0),
    (100.0, 120.0, 0.5, 100.0),
    (200.0, 150.0,-0.3, 100.0),
]


def section_pointwise(report):
    report.append("## 1. Pointwise Kernel Agreement\n")
    report.append("Compares individual sigma_E evaluations from the pure Python implementation")
    report.append("(both fixed Gauss-Laguerre and adaptive `scipy.integrate.quad`) against")
    report.append("the C++ pybind11 implementation.\n")

    cpp_engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)

    report.append("| # | E [keV] | E' [keV] | xi | T [keV] | C++ | Py fixed | Py adaptive | fixed rel diff | adaptive rel diff |")
    report.append("|---|---|---|---|---|---|---|---|---|---|")

    fixed_diffs = []
    adaptive_diffs = []
    case_labels = []

    for i, (E_kev, Ep_kev, xi, T_kev) in enumerate(POINTWISE_CASES):
        E = E_kev * KEV
        Ep = Ep_kev * KEV
        tau = T_kev * KEV / ME_C2

        cpp_r = cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
        cpp_val = cpp_r.value

        py_fixed, _, _ = py_sigma_E(E, Ep, xi, tau, 1.0, NL=128, method="fixed")
        py_adapt, _, _ = py_sigma_E(E, Ep, xi, tau, 1.0, method="adaptive")

        scale = max(abs(cpp_val), 1e-300)
        d_fixed = abs(py_fixed - cpp_val) / scale if scale > 1e-300 else 0
        d_adapt = abs(py_adapt - cpp_val) / scale if scale > 1e-300 else 0

        fixed_diffs.append(max(d_fixed, 1e-16))
        adaptive_diffs.append(max(d_adapt, 1e-16))
        case_labels.append(f"{i+1}")

        report.append(
            f"| {i+1} | {E_kev} | {Ep_kev} | {xi} | {T_kev} | "
            f"{cpp_val:.4e} | {py_fixed:.4e} | {py_adapt:.4e} | "
            f"{d_fixed:.2e} | {d_adapt:.2e} |"
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(POINTWISE_CASES))
    w = 0.35
    ax.bar(x - w/2, fixed_diffs, w, label='Fixed GL', color='steelblue',
           edgecolor='k', linewidth=0.5)
    ax.bar(x + w/2, adaptive_diffs, w, label='Adaptive quad', color='coral',
           edgecolor='k', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=8)
    ax.set_xlabel("Test case #")
    ax.set_ylabel("Relative difference vs C++")
    ax.set_title("Python vs C++ Pointwise Agreement")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    plot_name = "pointwise_agreement.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Pointwise Agreement](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Timing comparison
# ═══════════════════════════════════════════════════════════════════════════════

def section_timing(report):
    report.append("## 2. Pointwise Timing Comparison\n")
    report.append("Measures single sigma_E evaluation time for each method.\n")

    n_repeats = 20
    cpp_engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)

    def bench(func):
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for E_kev, Ep_kev, xi, T_kev in POINTWISE_CASES:
                E = E_kev * KEV
                Ep = Ep_kev * KEV
                tau = T_kev * KEV / ME_C2
                func(E, Ep, xi, tau)
        return time.perf_counter() - t0

    cpp_time = bench(lambda E, Ep, xi, tau: cpp_engine.sigma_E(E, Ep, xi, tau, 1.0))
    py_fixed_time = bench(lambda E, Ep, xi, tau: py_sigma_E(E, Ep, xi, tau, 1.0, NL=128, method="fixed"))
    py_adapt_time = bench(lambda E, Ep, xi, tau: py_sigma_E(E, Ep, xi, tau, 1.0, method="adaptive"))

    n_total = len(POINTWISE_CASES) * n_repeats

    report.append("| Method | Total [s] | Per call [us] | Slowdown vs C++ |")
    report.append("|---|---|---|---|")
    report.append(f"| C++ pybind11 | {cpp_time:.4f} | {cpp_time/n_total*1e6:.1f} | 1.0x |")
    report.append(f"| Python fixed GL | {py_fixed_time:.4f} | {py_fixed_time/n_total*1e6:.1f} | {py_fixed_time/cpp_time:.1f}x |")
    report.append(f"| Python adaptive quad | {py_adapt_time:.4f} | {py_adapt_time/n_total*1e6:.1f} | {py_adapt_time/cpp_time:.1f}x |")

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ['C++ pybind11', 'Python\nfixed GL', 'Python\nadaptive quad']
    times_us = [cpp_time/n_total*1e6, py_fixed_time/n_total*1e6, py_adapt_time/n_total*1e6]
    colors = ['#2ecc71', 'steelblue', 'coral']
    bars = ax.bar(methods, times_us, color=colors, edgecolor='k', linewidth=0.5)
    for bar, t in zip(bars, times_us):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{t:.1f} us", ha='center', fontsize=9)
    ax.set_ylabel("Time per evaluation [us]")
    ax.set_title("Pointwise sigma_E Timing (NL=128)")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    plot_name = "pointwise_timing.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Pointwise Timing](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Pomraning multigroup comparison
# ═══════════════════════════════════════════════════════════════════════════════

POMRANING_CASES = [
    dict(name="Pomraning_1keV_low",  T_kev=1.0,  emax=75.0,
         ein_kev=[5.0, 10.0, 20.0, 40.0, 60.0],
         ylim=[1, 1e4], ref_dir="Pomeraning_1kev_low"),
    dict(name="Pomraning_1keV_high", T_kev=1.0,  emax=340.0,
         ein_kev=[80.0, 120.0, 200.0, 300.0],
         ylim=[1e-2, 1e2], ref_dir="Pomeraning_1kev_high"),
    dict(name="Pomraning_20keV_low", T_kev=20.0, emax=140.0,
         ein_kev=[5.0, 10.0, 20.0, 40.0, 60.0],
         ylim=[1e-1, 1e3], ref_dir="Pomeraning_20kev_low"),
    dict(name="Pomraning_20keV_high", T_kev=20.0, emax=440.0,
         ein_kev=[80.0, 120.0, 200.0, 300.0],
         ylim=[1e-2, 1e2], ref_dir="Pomeraning_20kev_high"),
    dict(name="Pomraning_100keV", T_kev=100.0, emax=500.0,
         ein_kev=[30.0, 75.0, 150.0, 350.0],
         ylim=[1e-3, 1e2], ref_dir=None),
]


def make_energy_bins(emax_kev, n_bins=40):
    eb_kev = np.array(sorted(set(
        list(np.linspace(0.01, emax_kev, n_bins))
        + list(np.geomspace(0.01, emax_kev, n_bins // 2))
    )))
    return eb_kev * KEV


def integrate_bin_cpp(engine, E_in, E_lo, E_hi, tau):
    def integrand(Ep, xi):
        return engine.sigma_E(E_in, Ep, xi, tau, 1.0).value
    val, _ = dblquad(integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                     lambda xi: E_lo, lambda xi: E_hi,
                     epsabs=1e-35, epsrel=1e-2)
    return 2.0 * np.pi * val


def integrate_bin_py(E_in, E_lo, E_hi, tau, NL=128):
    def integrand(Ep, xi):
        val, _, _ = py_sigma_E(E_in, Ep, xi, tau, 1.0, NL=NL, method="fixed")
        return val
    val, _ = dblquad(integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                     lambda xi: E_lo, lambda xi: E_hi,
                     epsabs=1e-35, epsrel=1e-2)
    return 2.0 * np.pi * val


def get_cmmc_matrix(T_kev, eb_erg, num_samples=200000):
    try:
        from _compton_matrix_mc import ComptonMatrixMC
    except ImportError:
        return None
    T_kelvin = T_kev * KEV_KELVIN
    ec = 0.5 * (eb_erg[:-1] + eb_erg[1:])
    compton_engine = ComptonMatrixMC(
        energy_groups_centers=ec.tolist(),
        energy_groups_boundaries=eb_erg.tolist(),
        num_of_samples=num_samples,
        force_detailed_balance=False, seed=42,
    )
    return np.array(compton_engine.calculate_S_matrix(temperature=T_kelvin))


def load_pomraning_reference(ref_dir):
    if ref_dir is None:
        return {}
    ref_base = os.path.join(ROOT, 'external', 'CMMC', 'examples', ref_dir)
    if not os.path.isdir(ref_base):
        return {}
    data = {}
    for fn in os.listdir(ref_base):
        if fn.endswith('.txt'):
            e_kev = float(fn.replace('kev.txt', ''))
            data[e_kev] = np.loadtxt(os.path.join(ref_base, fn), delimiter=',')
    return data


def section_pomraning(report):
    report.append("## 3. Pomraning Multigroup Comparison\n")
    report.append("For each Pomraning case, the multigroup S-matrix is computed using:")
    report.append("- **C++ quadrature** (solid lines): pybind11 kernel + scipy `dblquad`")
    report.append("- **Python quadrature** (dashed lines): pure numpy/scipy kernel + `dblquad`")
    report.append("- **CMMC Monte Carlo** (dotted lines): external MC reference")
    report.append("- **Pomraning (book)** (black circles): digitized reference data\n")

    cpp_engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

    all_cpp_times = []
    all_py_times = []
    all_mc_times = []
    case_names = []

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

        print(f"    {name} ({num_groups} groups, {len(ein_kev)} E_in)...")

        ref_data = load_pomraning_reference(ref_dir)

        t0 = time.perf_counter()
        S_mc = get_cmmc_matrix(T_kev, eb_erg)
        mc_time = time.perf_counter() - t0

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(ein_kev), 10)))

        cpp_time_total = 0
        py_time_total = 0

        for idx, e0_kev in enumerate(ein_kev):
            g = np.argmin(np.abs(ec_erg - e0_kev * KEV))
            E_in = ec_erg[g]
            e_label = f"{ec_erg[g] / KEV:.4g}"
            color = colors[idx]

            t0 = time.perf_counter()
            row_cpp = np.zeros(num_groups)
            for gp in range(num_groups):
                row_cpp[gp] = integrate_bin_cpp(
                    cpp_engine, E_in, eb_erg[gp], eb_erg[gp + 1], tau)
            cpp_time_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            row_py = np.zeros(num_groups)
            for gp in range(num_groups):
                row_py[gp] = integrate_bin_py(
                    E_in, eb_erg[gp], eb_erg[gp + 1], tau)
            py_time_total += time.perf_counter() - t0

            sigma_cpp = row_cpp / ewid_kev / MBARN
            ax.stairs(sigma_cpp, edges=eb_kev_arr, color=color,
                      linewidth=1.8,
                      label=f"$E_{{\\mathrm{{in}}}}$={e_label} keV")

            sigma_py = row_py / ewid_kev / MBARN
            ax.stairs(sigma_py, edges=eb_kev_arr, color=color,
                      linewidth=1.2, linestyle='--')

            if S_mc is not None:
                sigma_mc = S_mc[g, :] / ewid_kev / MBARN
                ax.stairs(sigma_mc, edges=eb_kev_arr, color=color,
                          linewidth=1.0, linestyle=':')

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

        handles, _ = ax.get_legend_handles_labels()
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
        plot_name = f"{name}.png"
        fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
        plt.close(fig)

        all_cpp_times.append(cpp_time_total)
        all_py_times.append(py_time_total)
        all_mc_times.append(mc_time)
        case_names.append(name)

        report.append(f"\n### {name} (T = {T_kev} keV)\n")
        report.append(f"![{name}](figs/{plot_name})\n")
        report.append(f"| Method | Time [s] |")
        report.append(f"|---|---|")
        report.append(f"| C++ + dblquad | {cpp_time_total:.2f} |")
        report.append(f"| Python + dblquad | {py_time_total:.2f} |")
        if S_mc is not None:
            report.append(f"| CMMC Monte Carlo | {mc_time:.2f} |")
        report.append(f"| Python/C++ slowdown | {py_time_total/cpp_time_total:.1f}x |")
        report.append("")

    # Summary timing bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(case_names))
    w = 0.25
    ax.bar(x - w, all_cpp_times, w, label='C++ + dblquad', color='#2ecc71',
           edgecolor='k', linewidth=0.5)
    ax.bar(x, all_py_times, w, label='Python + dblquad', color='steelblue',
           edgecolor='k', linewidth=0.5)
    ax.bar(x + w, all_mc_times, w, label='CMMC Monte Carlo', color='coral',
           edgecolor='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('Pomraning_', '') for n in case_names],
                       fontsize=9, rotation=15)
    ax.set_ylabel("Total time [s]")
    ax.set_title("Multigroup S-matrix Computation Time by Method")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    plot_name = "multigroup_timing_summary.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n### Timing Summary\n")
    report.append(f"![Timing Summary](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    report = []
    report.append("# Python vs C++ vs CMMC Comparison Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("This report compares the pure Python (`pycompton`) implementation of the")
    report.append("Compton scattering kernel against the C++ pybind11 implementation and the")
    report.append("CMMC Monte Carlo reference, covering pointwise accuracy, timing, and")
    report.append("multigroup Pomraning comparisons with plots.\n")
    report.append("---\n")

    print("Running Python vs C++ comparison report...")

    print("  [1/3] Pointwise kernel agreement...")
    t0 = time.time()
    section_pointwise(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [2/3] Pointwise timing...")
    t0 = time.time()
    section_timing(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [3/3] Pomraning multigroup comparison...")
    t0 = time.time()
    section_pomraning(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    outpath = os.path.join(GEN_DIR, "python_cpp_comparison.md")
    with open(outpath, "w") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nReport saved to: {outpath}")


if __name__ == "__main__":
    main()
