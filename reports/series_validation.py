"""
Series validation report: compares Section 4 series against quadrature and CMMC.

Generates a markdown report with embedded plots covering:
  1. Pointwise series vs quadrature agreement and timing
  2. Series vs quadrature spectra (sigma(E') curves)
  3. Pomraning multigroup 3-way comparison (series vs quadrature vs CMMC)
  4. Convergence diagnostics
  5. Aggregate timing summary

Usage:
    python3 reports/series_validation.py

Output:
    reports/generated/series_validation.md  (+ .png plots in figs/)
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
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'external', 'CMMC', 'cpp_modules'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from _compton_power_series import ComptonPowerSeries
from _compton_kernel_asymptotic_series import ComptonKernelAsymptoticSeries
from _compton_kernel_solver import ComptonKernelSolver

from _units import me_c2, kev, k_boltz, kev_kelvin, mbarn

XI_EPS = 1e-10

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


POINTWISE_CASES = [
    (1.0,   1.01,  0.0,   0.1),
    (1.0,   1.01,  0.0,   1.0),
    (1.0,   0.99,  0.0,   1.0),
    (1.0,   2.0,   0.0,   5.0),
    (1.0,   0.5,  -0.5,   5.0),
    (10.0,  10.5,  0.0,  20.0),
    (10.0,  9.5,   0.3,  20.0),
    (50.0,  55.0,  0.0,  20.0),
    (1.0,   1.01,  0.5, 100.0),
    (100.0, 101.0, 0.0, 100.0),
    (100.0, 80.0,  0.0, 100.0),
    (5.0,   5.0,   0.0,  10.0),
]

METHOD_COLORS = {
    'PowerSeries': 'steelblue',
    'Asymptotic': 'coral',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Pointwise agreement + timing
# ═══════════════════════════════════════════════════════════════════════════════

def section_pointwise(report):
    report.append("## 1. Pointwise Series vs Quadrature Agreement\n")
    report.append("Compares C++ series (Auto) against C++ quadrature (NL=256, PostIBP).\n")

    quad = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)
    series = ComptonKernelSolver()

    n_warmup = 5
    n_bench = 50

    report.append("| # | E [keV] | E' [keV] | xi | T [keV] | sigma_quad | sigma_series | rel_diff | method | terms | time_quad [us] | time_series [us] | speedup |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    rel_diffs = []
    methods = []
    t_quads = []
    t_series_list = []

    for i, (E_kev, Ep_kev, xi, T_kev) in enumerate(POINTWISE_CASES):
        E = E_kev * kev
        Ep = Ep_kev * kev
        T_K = T_kev * kev_kelvin

        for _ in range(n_warmup):
            quad.sigma_E(E, Ep, xi, T_K, 1.0)
            series.sigma_E(E, Ep, xi, T_K, 1.0)

        t0 = time.perf_counter()
        for _ in range(n_bench):
            qr = quad.sigma_E(E, Ep, xi, T_K, 1.0)
        t_q = (time.perf_counter() - t0) / n_bench * 1e6

        t0 = time.perf_counter()
        for _ in range(n_bench):
            sr = series.sigma_E(E, Ep, xi, T_K, 1.0)
        t_s = (time.perf_counter() - t0) / n_bench * 1e6

        scale = max(abs(qr.value), 1e-300)
        rd = abs(sr.value - qr.value) / scale
        speedup = t_q / t_s if t_s > 0 else float('inf')

        rel_diffs.append(max(rd, 1e-16))
        t_quads.append(t_q)
        t_series_list.append(t_s)

        report.append(
            f"| {i+1} | {E_kev} | {Ep_kev} | {xi} | {T_kev} | "
            f"{qr.value:.4e} | {sr.value:.4e} | {rd:.2e} | "
            f"{t_q:.1f} | {t_s:.1f} | {speedup:.1f}x |"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(POINTWISE_CASES))
    colors = [METHOD_COLORS.get(m, 'gray') for m in methods]
    ax1.bar(x, rel_diffs, color=colors, edgecolor='k', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i+1) for i in x], fontsize=8)
    ax1.set_xlabel("Test case #")
    ax1.set_ylabel("Relative difference vs quadrature")
    ax1.set_title("Series vs Quadrature Agreement")
    ax1.grid(True, alpha=0.3, axis='y')
    handles = [plt.Rectangle((0,0),1,1, fc=c, ec='k') for c in METHOD_COLORS.values()]
    ax1.legend(handles, list(METHOD_COLORS.keys()), fontsize=9)

    w = 0.35
    ax2.bar(x - w/2, t_quads, w, label='Quadrature NL=256', color='#2ecc71',
            edgecolor='k', linewidth=0.5)
    ax2.bar(x + w/2, t_series_list, w, label='Series (Auto)', color='steelblue',
            edgecolor='k', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in x], fontsize=8)
    ax2.set_xlabel("Test case #")
    ax2.set_ylabel("Time per eval [us]")
    ax2.set_title("Evaluation Time Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    plot_name = "series_pointwise.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Series vs Quadrature Pointwise](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Spectra
# ═══════════════════════════════════════════════════════════════════════════════

SPECTRA_CASES = [
    dict(E_kev=1.0,  T_kev=1.0,  Ep_range=(0.1, 5.0)),
    dict(E_kev=10.0, T_kev=20.0, Ep_range=(1.0, 30.0)),
    dict(E_kev=50.0, T_kev=100.0, Ep_range=(5.0, 150.0)),
]


def section_spectra(report):
    report.append("## 2. Series vs Quadrature Spectra\n")
    report.append("Plots sigma(E') vs E' for representative cases.\n")

    quad = ComptonKernelQuadrature(128, QuadratureForm.PostIBP)
    series = ComptonKernelSolver()

    for case in SPECTRA_CASES:
        E_kev = case["E_kev"]
        T_kev = case["T_kev"]
        Ep_lo, Ep_hi = case["Ep_range"]
        E = E_kev * kev
        T_K = T_kev * kev_kelvin
        xi = 0.0

        Ep_arr = np.linspace(Ep_lo * kev, Ep_hi * kev, 200)

        sigma_q = np.zeros(len(Ep_arr))
        sigma_s = np.zeros(len(Ep_arr))
        for j, Ep in enumerate(Ep_arr):
            try:
                sigma_q[j] = quad.sigma_E(E, Ep, xi, T_K, 1.0).value
                sigma_s[j] = series.sigma_E(E, Ep, xi, T_K, 1.0).value
            except Exception:
                pass

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)
        Ep_kev_arr = Ep_arr / kev
        ax1.plot(Ep_kev_arr, sigma_q, '-', color='steelblue', lw=1.8,
                 label='Quadrature NL=128')
        ax1.plot(Ep_kev_arr, sigma_s, '--', color='coral', lw=1.2,
                 label='Series (Auto)')
        ax1.set_ylabel(r'$\Sigma_E$ [cm$^2$/erg]')
        ax1.set_title(f'E = {E_kev} keV, T = {T_kev} keV, xi = {xi}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        mask = np.abs(sigma_q) > 1e-300
        reldiff = np.zeros_like(sigma_q)
        reldiff[mask] = np.abs(sigma_s[mask] - sigma_q[mask]) / np.abs(sigma_q[mask])
        ax2.semilogy(Ep_kev_arr, np.maximum(reldiff, 1e-16), '-', color='purple')
        ax2.set_xlabel("E' [keV]")
        ax2.set_ylabel("Relative diff")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_name = f"spectrum_E{E_kev}_T{T_kev}.png"
        fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
        plt.close(fig)

        report.append(f"\n### E = {E_kev} keV, T = {T_kev} keV\n")
        report.append(f"![Spectrum](figs/{plot_name})\n")

    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Pomraning multigroup 3-way comparison
# ═══════════════════════════════════════════════════════════════════════════════

POMRANING_CASES = [
    dict(name="Pomraning_1keV_low",  T_kev=1.0,  emax=75.0,
         ein_kev=[5.0, 10.0, 20.0, 40.0, 60.0],
         ylim=[1, 1e4], ref_dir="Pomeraning_1kev_low"),
    dict(name="Pomraning_20keV_low", T_kev=20.0, emax=140.0,
         ein_kev=[5.0, 10.0, 20.0, 40.0, 60.0],
         ylim=[1e-1, 1e3], ref_dir="Pomeraning_20kev_low"),
    dict(name="Pomraning_100keV", T_kev=100.0, emax=500.0,
         ein_kev=[30.0, 75.0, 150.0, 350.0],
         ylim=[1e-3, 1e2], ref_dir=None),
]


def make_energy_bins(emax_kev, n_bins=40):
    eb_kev = np.array(sorted(set(
        list(np.linspace(0.01, emax_kev, n_bins))
        + list(np.geomspace(0.01, emax_kev, n_bins // 2))
    )))
    return eb_kev * kev


def integrate_bin(engine, E_in, E_lo, E_hi, T_K):
    def integrand(Ep, xi):
        return engine.sigma_E(E_in, Ep, xi, T_K, 1.0).value
    val, _ = dblquad(integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                     lambda xi: E_lo, lambda xi: E_hi,
                     epsabs=1e-35, epsrel=1e-2)
    return 2.0 * np.pi * val


def integrate_bin_series(engine, E_in, E_lo, E_hi, T_K):
    def integrand(Ep, xi):
        r = engine.sigma_E(E_in, Ep, xi, T_K, 1.0)
        return r.value
    val, _ = dblquad(integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                     lambda xi: E_lo, lambda xi: E_hi,
                     epsabs=1e-35, epsrel=1e-2)
    return 2.0 * np.pi * val


def get_cmmc_matrix(T_kev, eb_erg, num_samples=200000):
    try:
        from _compton_matrix_mc import ComptonMatrixMC
    except ImportError:
        return None
    T_kelvin = T_kev * kev_kelvin
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
    report.append("## 3. Pomraning Multigroup 3-Way Comparison\n")
    report.append("For each Pomraning case, multigroup S-matrix computed using:")
    report.append("- **C++ quadrature** (solid): NL=64, PostIBP")
    report.append("- **C++ series** (dashed): Auto mode")
    report.append("- **CMMC Monte Carlo** (dotted): external MC reference")
    report.append("- **Pomraning (book)** (black circles): digitized reference\n")

    quad = ComptonKernelQuadrature(64, QuadratureForm.PostIBP)
    series = ComptonKernelSolver()

    all_times = []

    for case in POMRANING_CASES:
        name = case["name"]
        T_kev = case["T_kev"]
        emax = case["emax"]
        ein_kev = case["ein_kev"]
        ylim = case["ylim"]
        ref_dir = case.get("ref_dir")
        T_K = T_kev * kev_kelvin

        eb_erg = make_energy_bins(emax, n_bins=40)
        eb_kev_arr = eb_erg / kev
        ec_erg = 0.5 * (eb_erg[:-1] + eb_erg[1:])
        ewid_kev = np.diff(eb_erg) / kev
        num_groups = len(eb_erg) - 1

        print(f"    {name} ({num_groups} groups, {len(ein_kev)} E_in)...")

        ref_data = load_pomraning_reference(ref_dir)
        S_mc = get_cmmc_matrix(T_kev, eb_erg)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(ein_kev), 10)))

        t_quad_total = 0
        t_series_total = 0

        for idx, e0_kev in enumerate(ein_kev):
            g = np.argmin(np.abs(ec_erg - e0_kev * kev))
            E_in = ec_erg[g]
            e_label = f"{ec_erg[g] / kev:.4g}"
            color = colors[idx]

            t0 = time.perf_counter()
            row_quad = np.zeros(num_groups)
            for gp in range(num_groups):
                row_quad[gp] = integrate_bin(
                    quad, E_in, eb_erg[gp], eb_erg[gp + 1], T_K)
            t_quad_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            row_series = np.zeros(num_groups)
            for gp in range(num_groups):
                row_series[gp] = integrate_bin_series(
                    series, E_in, eb_erg[gp], eb_erg[gp + 1], T_K)
            t_series_total += time.perf_counter() - t0

            sigma_quad = row_quad / ewid_kev / mbarn
            ax.stairs(sigma_quad, edges=eb_kev_arr, color=color,
                      linewidth=1.8,
                      label=f"$E_{{\\mathrm{{in}}}}$={e_label} keV")

            sigma_series = row_series / ewid_kev / mbarn
            ax.stairs(sigma_series, edges=eb_kev_arr, color=color,
                      linewidth=1.2, linestyle='--')

            if S_mc is not None:
                sigma_mc = S_mc[g, :] / ewid_kev / mbarn
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
                   label='C++ series'),
            Line2D([0], [0], color='gray', linewidth=1.0, linestyle=':',
                   label='CMMC Monte Carlo'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=9)

        fig.tight_layout()
        plot_name = f"series_pomraning_{name}.png"
        fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
        plt.close(fig)

        speedup = t_quad_total / t_series_total if t_series_total > 0 else float('inf')
        all_times.append((name, t_quad_total, t_series_total))

        report.append(f"\n### {name} (T = {T_kev} keV)\n")
        report.append(f"![{name}](figs/{plot_name})\n")
        report.append("| Method | Time [s] |")
        report.append("|---|---|")
        report.append(f"| C++ quadrature + dblquad | {t_quad_total:.2f} |")
        report.append(f"| C++ series + dblquad | {t_series_total:.2f} |")
        report.append(f"| Series/quadrature speedup | {speedup:.1f}x |")
        report.append("")

    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def section_convergence(report):
    report.append("## 4. Convergence Diagnostics\n")

    series = ComptonKernelSolver()

    E_kev = 1.0
    Ep_kev = 1.5
    xi = 0.0
    E = E_kev * kev
    Ep = Ep_kev * kev

    taus = np.geomspace(0.0001, 0.5, 80)
    values_list = []
    n_success = 0

    for tau in taus:
        T_K = tau * me_c2 / k_boltz
        try:
            r = series.sigma_E(E, Ep, xi, T_K, 1.0)
            values_list.append(r.value)
            n_success += 1
        except RuntimeError:
            values_list.append(np.nan)

    values_arr = np.array(values_list)
    n_total = len(taus)

    fig, ax = plt.subplots(figsize=(10, 5))
    valid = np.isfinite(values_arr)
    ax.semilogy(taus[valid], np.abs(values_arr[valid]) + 1e-300, 'o-',
                color='steelblue', markersize=3)
    ax.set_xlabel(r"$\tau$ = kT / m_e c²")
    ax.set_ylabel("|sigma_E|")
    ax.set_title(f"Series sweep: E={E_kev} keV, E'={Ep_kev} keV, xi={xi}")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_name = "convergence_diagnostics.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"Sweep over tau for E={E_kev} keV, E'={Ep_kev} keV, xi={xi}:\n")
    report.append(f"- Successful evaluations: {n_success}/{n_total} ({100*n_success/n_total:.0f}%)")
    report.append(f"- Auto selects Asymptotic for small tau, PowerSeries for large tau")
    report.append(f"- Switching threshold: tau_alpha ~ 0.05\n")
    report.append(f"![Convergence](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Aggregate timing summary
# ═══════════════════════════════════════════════════════════════════════════════

def section_timing_summary(report):
    report.append("## 5. Aggregate Timing Summary\n")

    n_bench = 50
    n_warmup = 5

    methods_to_bench = [
        ("Quad NL=64", ComptonKernelQuadrature(64, QuadratureForm.PostIBP), True),
        ("Quad NL=128", ComptonKernelQuadrature(128, QuadratureForm.PostIBP), True),
        ("Quad NL=256", ComptonKernelQuadrature(256, QuadratureForm.PostIBP), True),
        ("Series Power", ComptonPowerSeries(), False),
        ("Series Asymp", ComptonKernelAsymptoticSeries(), False),
        ("Series Auto", ComptonKernelSolver(), False),
    ]

    results = {}
    for label, engine, is_quad in methods_to_bench:
        times = []
        for E_kev, Ep_kev, xi, T_kev in POINTWISE_CASES:
            E = E_kev * kev
            Ep = Ep_kev * kev
            T_K = T_kev * kev_kelvin
            try:
                for _ in range(n_warmup):
                    engine.sigma_E(E, Ep, xi, T_K, 1.0)
                t0 = time.perf_counter()
                for _ in range(n_bench):
                    engine.sigma_E(E, Ep, xi, T_K, 1.0)
                t_us = (time.perf_counter() - t0) / n_bench * 1e6
                times.append(t_us)
            except Exception:
                pass
        results[label] = times

    report.append("| Method | Mean [us] | Worst [us] | Speedup vs Quad NL=128 |")
    report.append("|---|---|---|---|")
    ref_mean = np.mean(results.get("Quad NL=128", [1.0]))
    for label in [m[0] for m in methods_to_bench]:
        if label in results and results[label]:
            ts = results[label]
            mean_t = np.mean(ts)
            worst_t = np.max(ts)
            speedup = ref_mean / mean_t if mean_t > 0 else 0
            report.append(f"| {label} | {mean_t:.1f} | {worst_t:.1f} | {speedup:.2f}x |")

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [m[0] for m in methods_to_bench]
    means = [np.mean(results.get(l, [0])) for l in labels]
    colors_list = ['#2ecc71', '#27ae60', '#1e8449', 'steelblue', 'coral', '#8e44ad']
    bars = ax.bar(labels, means, color=colors_list, edgecolor='k', linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{m:.1f}", ha='center', fontsize=9)
    ax.set_ylabel("Mean time per eval [us]")
    ax.set_title("Per-evaluation Timing: Quadrature vs Series")
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    fig.tight_layout()

    plot_name = "timing_summary.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Timing Summary](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Series Auto vs Quadrature Error Colorplot
# ═══════════════════════════════════════════════════════════════════════════════

HEATMAP_E_GRID = np.logspace(-1, 2.7, 50)
HEATMAP_T_GRID = np.logspace(-0.3, 2.7, 50)
HEATMAP_RATIOS = [0.5, 0.9, 1.01, 2.0, 5.0]
HEATMAP_XIS = [-0.5, 0.0, 0.5]

ASYMP_TAU_ALPHA_THRESHOLD = 0.02
GAMMA_DOUBLE_PRECISION_SAFE = 0.02

METHOD_ASYMPTOTIC = 0
METHOD_POWER = 1
METHOD_POWER_HP = 2


def auto_method_select(E_kev, ratio, xi, T_kev):
    E = E_kev * kev
    Ep = E * ratio
    gamma = E / me_c2
    gamma_p = Ep / me_c2
    tau = T_kev * kev_kelvin * k_boltz / me_c2

    a = 1.0 - xi
    dg = gamma_p - gamma
    omega2 = (1.0 + xi) / a
    gg_a = gamma * gamma_p * a
    factor1 = 1.0 + gg_a / 2.0
    factor2 = 1.0 + (dg * dg) / (2.0 * gg_a)
    Delta = np.sqrt(factor1 * factor2)
    lambda_plus = max(dg / 2.0 + Delta, 1.0)

    rho_plus = lambda_plus + gamma
    rho_minus = lambda_plus - gamma_p
    alpha_plus = 1.0 / np.sqrt(rho_plus**2 + omega2)
    alpha_minus = 1.0 / np.sqrt(rho_minus**2 + omega2)
    tau_alpha_max = tau * max(alpha_plus, alpha_minus)

    if tau_alpha_max < ASYMP_TAU_ALPHA_THRESHOLD:
        return METHOD_ASYMPTOTIC
    elif min(gamma, gamma_p) >= GAMMA_DOUBLE_PRECISION_SAFE:
        return METHOD_POWER
    else:
        return METHOD_POWER_HP


def section_auto_vs_quad_colorplot(report):
    report.append("## 6. Series Auto vs Quadrature Error Map\n")
    report.append("Relative discrepancy `|series_auto - Q256| / max(|auto|, |Q256|)` between")
    report.append("the series Auto `sigma_E` and the 256-point Gauss-Laguerre quadrature")
    report.append("`sigma_E` (PostIBP) over the (E, T) plane.\n")

    series = ComptonKernelSolver()
    quad = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)

    n_E = len(HEATMAP_E_GRID)
    n_T = len(HEATMAP_T_GRID)
    n_rows = len(HEATMAP_XIS)
    n_cols = len(HEATMAP_RATIOS)

    error_grids = {}
    method_grids = {}
    value_grids = {}
    all_rel_errs = []
    n_failed = 0

    total_panels = n_rows * n_cols
    done = 0

    for xi in HEATMAP_XIS:
        for ratio in HEATMAP_RATIOS:
            done += 1
            print(f'    auto vs quad panel {done}/{total_panels}: xi={xi}, ratio={ratio}')
            err_grid = np.full((n_T, n_E), np.nan)
            meth_grid = np.full((n_T, n_E), np.nan)
            val_grid = np.full((n_T, n_E), np.nan)
            for i, T_kev in enumerate(HEATMAP_T_GRID):
                T_K = T_kev * kev_kelvin
                for j, E_kev in enumerate(HEATMAP_E_GRID):
                    E = E_kev * kev
                    Ep = E * ratio
                    try:
                        r_s = series.sigma_E(E, Ep, xi, T_K, 1.0)
                        r_q = quad.sigma_E(E, Ep, xi, T_K, 1.0)
                        val_grid[i, j] = r_s.value
                        scale = max(abs(r_s.value), abs(r_q.value))
                        if scale > 1e-300:
                            rel = abs(r_s.value - r_q.value) / scale
                            err_grid[i, j] = rel
                            all_rel_errs.append(rel)
                        else:
                            err_grid[i, j] = 0.0
                        meth_grid[i, j] = auto_method_select(E_kev, ratio, xi, T_kev)
                    except (RuntimeError, Exception):
                        n_failed += 1
            error_grids[(xi, ratio)] = err_grid
            method_grids[(xi, ratio)] = meth_grid
            value_grids[(xi, ratio)] = val_grid

    # Error colorplot
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    norm = LogNorm(vmin=1e-16, vmax=1e0, clip=False)
    cmap = plt.cm.inferno_r.copy()
    cmap.set_bad('lightgray')
    cmap.set_under('skyblue')

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes[row][col]
            data = error_grids[(xi, ratio)]
            safe = np.where(np.isnan(data), np.nan, np.maximum(data, 1e-20))
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, safe,
                          norm=norm, cmap=cmap, shading='nearest')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == n_rows - 1:
                ax.set_xlabel('E [keV]')
            if col == 0:
                ax.set_ylabel('T [keV]')
            if row == 0:
                ax.set_title(f"E'/E = {ratio}")
            ax.text(0.97, 0.03, f'$\\xi={xi}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    pcm = axes[0][0].collections[0]
    all_axes = [ax for row_axes in axes for ax in row_axes]
    cbar = fig.colorbar(pcm, ax=all_axes, shrink=0.85, pad=0.03)
    cbar.set_label('|series_auto - Q256| / max(|auto|, |Q256|)')
    fig.suptitle('Series Auto vs Quadrature (Q256 PostIBP)', fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    err_fname = 'series_auto_vs_quad_TE.png'
    fig.savefig(os.path.join(FIGS_DIR, err_fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    report.append(f"![Series Auto vs Q256](figs/{err_fname})\n")
    report.append(f"Each panel shows the relative error on a {n_E}x{n_T} log-spaced grid "
                  f"in (E, T) space. Columns vary E'/E ({', '.join(str(r) for r in HEATMAP_RATIOS)}); "
                  f"rows vary xi ({', '.join(str(x) for x in HEATMAP_XIS)}). "
                  f"Sky-blue cells indicate negligible kernel values (both methods < 1e-300) "
                  f"or exact agreement; gray cells indicate convergence failure.\n")

    # Value colorplot (per-panel normalization)
    val_cmap = plt.cm.viridis.copy()
    val_cmap.set_bad('lightgray')

    fig_v, axes_v = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes_v[row][col]
            data = value_grids[(xi, ratio)]
            safe = np.where(np.isfinite(data) & (data > 0), data, np.nan)
            finite = safe[np.isfinite(safe)]
            if len(finite) > 0:
                pmin, pmax = np.min(finite), np.max(finite)
                pnorm = LogNorm(vmin=pmin, vmax=pmax)
            else:
                pnorm = LogNorm(vmin=1e-30, vmax=1e-15)
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, safe,
                          norm=pnorm, cmap=val_cmap, shading='nearest')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == n_rows - 1:
                ax.set_xlabel('E [keV]')
            if col == 0:
                ax.set_ylabel('T [keV]')
            if row == 0:
                ax.set_title(f"E'/E = {ratio}")
            ax.text(0.97, 0.03, f'$\\xi={xi}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            if len(finite) > 0:
                ax.text(0.03, 0.97,
                        f'{pmin:.0e}\n  to\n{pmax:.0e}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=6, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fig_v.suptitle(r'Series Auto: $\Sigma_E$ Value Map (per-panel scale)',
                   fontsize=13, y=1.01)
    fig_v.subplots_adjust(wspace=0.05, hspace=0.15)
    val_fname = 'series_auto_value_TE.png'
    fig_v.savefig(os.path.join(FIGS_DIR, val_fname), dpi=150, bbox_inches='tight')
    plt.close(fig_v)

    report.append("### Kernel value map\n")
    report.append(f"![Series Auto value map](figs/{val_fname})\n")
    report.append(f"`sigma_E` from series Auto on the same "
                  f"{n_E}x{n_T} (E, T) grid. Each panel is independently "
                  f"scaled to its own [min, max] range (annotated top-left). "
                  f"Gray cells indicate convergence failure.\n")

    # Statistics
    n_total = n_E * n_T * n_rows * n_cols
    errs_arr = np.array(all_rel_errs) if all_rel_errs else np.array([0.0])

    report.append("### Summary statistics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Grid points evaluated | {n_total - n_failed} / {n_total} |")
    report.append(f"| Failed (exception) | {n_failed} |")
    report.append(f"| Points with valid comparison | {len(all_rel_errs)} |")
    report.append("")

    if len(all_rel_errs) > 0:
        report.append("| Statistic | Value |")
        report.append("|-----------|-------|")
        report.append(f"| Minimum | {np.min(errs_arr):.2e} |")
        report.append(f"| Median | {np.median(errs_arr):.2e} |")
        report.append(f"| Mean | {np.mean(errs_arr):.2e} |")
        report.append(f"| 90th percentile | {np.percentile(errs_arr, 90):.2e} |")
        report.append(f"| 95th percentile | {np.percentile(errs_arr, 95):.2e} |")
        report.append(f"| 99th percentile | {np.percentile(errs_arr, 99):.2e} |")
        report.append(f"| Maximum | {np.max(errs_arr):.2e} |")
        report.append("")

        pct_1e8 = 100.0 * np.sum(errs_arr < 1e-8) / len(errs_arr)
        pct_1e6 = 100.0 * np.sum(errs_arr < 1e-6) / len(errs_arr)
        pct_1e4 = 100.0 * np.sum(errs_arr < 1e-4) / len(errs_arr)
        report.append(f"| Points with error < 1e-8 | {pct_1e8:.1f}% |")
        report.append(f"| Points with error < 1e-6 | {pct_1e6:.1f}% |")
        report.append(f"| Points with error < 1e-4 | {pct_1e4:.1f}% |")
        report.append("")

    # Method selection colorplot
    method_cmap = ListedColormap(['#2196F3', '#4CAF50', '#FF9800'])
    method_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], method_cmap.N)
    method_labels = ['Asymptotic', 'PowerSeries', 'PowerSeriesHP']

    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(4 * n_cols + 1.5, 3.5 * n_rows),
                               sharex=True, sharey=True, squeeze=False)

    for row, xi in enumerate(HEATMAP_XIS):
        for col, ratio in enumerate(HEATMAP_RATIOS):
            ax = axes2[row][col]
            data = method_grids[(xi, ratio)]
            ax.pcolormesh(HEATMAP_E_GRID, HEATMAP_T_GRID, data,
                          cmap=method_cmap, norm=method_norm, shading='nearest')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == n_rows - 1:
                ax.set_xlabel('E [keV]')
            if col == 0:
                ax.set_ylabel('T [keV]')
            if row == 0:
                ax.set_title(f"E'/E = {ratio}")
            ax.text(0.97, 0.03, f'$\\xi={xi}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    all_axes2 = [ax for row_axes in axes2 for ax in row_axes]
    cbar2 = fig2.colorbar(
        plt.cm.ScalarMappable(cmap=method_cmap, norm=method_norm),
        ax=all_axes2, shrink=0.85, pad=0.03, ticks=[0, 1, 2])
    cbar2.ax.set_yticklabels(method_labels)
    fig2.suptitle('Auto Method Selection Map (sigma_E)', fontsize=13, y=1.01)
    fig2.subplots_adjust(wspace=0.05, hspace=0.15)
    meth_fname = 'series_method_selection_TE.png'
    fig2.savefig(os.path.join(FIGS_DIR, meth_fname), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    report.append("### Auto method selection map\n")
    report.append(f"![Method selection map](figs/{meth_fname})\n")
    report.append("Blue = Asymptotic, Green = PowerSeries (double), Orange = PowerSeriesHP (DD). "
                  "The method is selected by the Auto dispatch logic based on "
                  "`tau * max(alpha+, alpha-) < 0.02` (Asymptotic) and "
                  "`min(gamma, gamma') >= 0.02` (PowerSeries vs HP).\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    report = []
    report.append("# Series Validation Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("This report validates the Section 4 series expansions (power series")
    report.append("and low-temperature asymptotic) against the existing Gauss-Laguerre")
    report.append("quadrature and CMMC Monte Carlo reference.\n")
    report.append("---\n")

    print("Running series validation report...")

    print("  [1/6] Pointwise agreement and timing...")
    t0 = time.time()
    section_pointwise(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [2/6] Spectra...")
    t0 = time.time()
    section_spectra(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [3/6] Pomraning multigroup comparison...")
    t0 = time.time()
    section_pomraning(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [4/6] Convergence diagnostics...")
    t0 = time.time()
    section_convergence(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [5/6] Timing summary...")
    t0 = time.time()
    section_timing_summary(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [6/6] Auto vs quadrature error colorplot...")
    t0 = time.time()
    section_auto_vs_quad_colorplot(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    outpath = os.path.join(GEN_DIR, "series_validation.md")
    with open(outpath, "w") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nReport saved to: {outpath}")


if __name__ == "__main__":
    main()
