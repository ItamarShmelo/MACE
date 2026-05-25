"""
Tolerance and convergence analysis for the Compton kernel.

Generates a comprehensive markdown report with embedded plots covering:
  1. Gauss-Laguerre quadrature order convergence (NL=64 vs 128 vs 256)
  2. Scipy outer integration tolerance sensitivity
  3. Post-IBP vs Pre-IBP agreement across temperature regimes
  4. Pointwise kernel error estimates from the C++ Richardson indicator
  5. Performance benchmarks
  6. Power series convergence vs number of terms
  7. Asymptotic series convergence and optimal truncation

Usage:
    python3 reports/convergence_analysis.py

Output:
    reports/generated/convergence_analysis.md  (+ .png plots)
"""
import sys
import os
import time
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm, scaled_K2
from _compton_kernel_series import ComptonKernelSeries, SeriesMethod

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9
XI_EPS = 1e-10

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
FIGS_DIR = os.path.join(GEN_DIR, 'figs')
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


def compute_bin_integral(engine, E_in, E_lo, E_hi, tau, epsrel_xi, epsrel_Ep):
    """Integrate kernel over one energy bin and all angles."""
    def xi_integrand(xi):
        def Ep_integrand(Ep):
            return engine.sigma_E(E_in, Ep, xi, tau, 1.0).value
        val, _ = quad(Ep_integrand, E_lo, E_hi, epsabs=1e-50, epsrel=epsrel_Ep)
        return val

    val, err = quad(xi_integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                    epsabs=1e-50, epsrel=epsrel_xi, limit=200)
    return 2.0 * np.pi * val, 2.0 * np.pi * err


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: GL order convergence
# ═══════════════════════════════════════════════════════════════════════════════

GL_TEST_CASES = [
    {"T_kev": 100.0, "E_in_kev": 100.0, "bins_kev": [(80, 120), (20, 60), (150, 250)]},
    {"T_kev": 20.0, "E_in_kev": 40.0, "bins_kev": [(35, 45), (10, 30), (50, 80)]},
    {"T_kev": 1.0, "E_in_kev": 10.0, "bins_kev": [(9, 11), (5, 8), (12, 15)]},
    {"T_kev": 0.1, "E_in_kev": 5.0, "bins_kev": [(4.5, 5.5), (3, 4), (6, 8)]},
]

NL_VALUES = [64, 128, 256]


def section_quadrature_order_convergence(report):
    report.append("## 1. Gauss-Laguerre Quadrature Order Convergence\n")
    report.append("Compares the bin-integrated kernel across NL values to quantify")
    report.append("the internal quadrature convergence. The outer scipy integration uses")
    report.append("tight tolerances (epsrel_xi=1e-6, epsrel_Ep=1e-8) so the difference")
    report.append("isolates the Gauss-Laguerre truncation error.\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for case_idx, case in enumerate(GL_TEST_CASES):
        T_kev = case["T_kev"]
        tau = T_kev * KEV / ME_C2
        E_in = case["E_in_kev"] * KEV
        ax = axes[case_idx]

        report.append(f"\n### T = {T_kev} keV (tau = {tau:.4e}), E_in = {case['E_in_kev']} keV\n")
        nl_header = " | ".join([f"S(NL={nl})" for nl in [64, 128, 256]])
        report.append(f"| Bin [keV] | {nl_header} | diff(64,128)/128 | diff(128,256)/256 |")
        report.append("|---|" + "---|" * (len([64, 128, 256]) + 2))

        for bin_idx, (bin_lo, bin_hi) in enumerate(case["bins_kev"]):
            E_lo = bin_lo * KEV
            E_hi = bin_hi * KEV

            all_results = {}
            for NL in NL_VALUES:
                eng = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PreIBP)
                val, _ = compute_bin_integral(eng, E_in, E_lo, E_hi, tau, 1e-6, 1e-8)
                all_results[NL] = val

            ref_val = all_results[256]
            if abs(ref_val) < 1e-50:
                report.append(f"| [{bin_lo}-{bin_hi}] | - | - | - | (underflow) | - |")
                continue

            errors = [abs(all_results[nl] - ref_val) / abs(ref_val)
                      for nl in NL_VALUES]
            ax.semilogy(NL_VALUES, [max(e, 1e-16) for e in errors],
                        'o-', label=f"[{bin_lo}-{bin_hi}] keV", markersize=5)

            d64 = abs(all_results[64] - all_results[128]) / abs(all_results[128]) if all_results[128] != 0 else 0
            d128 = abs(all_results[128] - all_results[256]) / abs(all_results[256])
            report.append(
                f"| [{bin_lo}-{bin_hi}] | {all_results[64]:.6e} | {all_results[128]:.6e} | "
                f"{all_results[256]:.6e} | {d64:.2e} | {d128:.2e} |"
            )

        ax.set_xlabel("NL (quadrature order)")
        ax.set_ylabel("Relative error vs NL=256")
        ax.set_title(f"T={T_kev} keV, E_in={case['E_in_kev']} keV")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-16)

    fig.suptitle("Gauss-Laguerre Quadrature Convergence", fontsize=14)
    fig.tight_layout()
    plot_name = "gl_convergence.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![GL Convergence](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Scipy tolerance sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

TOLERANCE_SETS = [
    {"label": "Loose",      "xi": 1e-2, "Ep": 1e-3},
    {"label": "Default",    "xi": 1e-3, "Ep": 1e-4},
    {"label": "Tight",      "xi": 1e-4, "Ep": 1e-6},
    {"label": "Very tight", "xi": 1e-6, "Ep": 1e-8},
    {"label": "Reference",  "xi": 1e-8, "Ep": 1e-10},
]

TOLERANCE_POINTS = [
    {"T_kev": 100.0, "E_in_kev": 100.0, "bin_kev": (60, 140)},
    {"T_kev": 100.0, "E_in_kev": 300.0, "bin_kev": (200, 400)},
    {"T_kev": 1.0, "E_in_kev": 10.0, "bin_kev": (8, 12)},
]


def section_scipy_tolerance_sensitivity(report):
    report.append("## 2. Scipy Integration Tolerance Sensitivity\n")
    report.append("Fixes NL=64 and varies the scipy `epsrel` to measure how the outer")
    report.append("integration tolerance affects the final result. Reference: tightest setting.\n")

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

    fig, axes = plt.subplots(1, len(TOLERANCE_POINTS), figsize=(5 * len(TOLERANCE_POINTS), 4))
    if len(TOLERANCE_POINTS) == 1:
        axes = [axes]

    for pt_idx, pt in enumerate(TOLERANCE_POINTS):
        T_kev = pt["T_kev"]
        tau = T_kev * KEV / ME_C2
        E_in = pt["E_in_kev"] * KEV
        E_lo = pt["bin_kev"][0] * KEV
        E_hi = pt["bin_kev"][1] * KEV
        ax = axes[pt_idx]

        report.append(f"\n### T={T_kev} keV, E_in={pt['E_in_kev']} keV, bin=[{pt['bin_kev'][0]}-{pt['bin_kev'][1]}] keV\n")
        report.append("| Tolerance | epsrel_xi | epsrel_Ep | S [cm^2] | Reported err | Rel diff vs ref | Time [ms] |")
        report.append("|---|---|---|---|---|---|---|")

        ref_tset = TOLERANCE_SETS[-1]
        ref_val, _ = compute_bin_integral(engine, E_in, E_lo, E_hi, tau,
                                         ref_tset["xi"], ref_tset["Ep"])

        epsrel_vals = []
        rel_diffs = []
        times_ms = []

        for tset in TOLERANCE_SETS:
            t0 = time.time()
            val, err = compute_bin_integral(engine, E_in, E_lo, E_hi, tau,
                                           tset["xi"], tset["Ep"])
            elapsed_ms = (time.time() - t0) * 1000

            rel_reported = err / abs(val) if val != 0 else 0
            rel_vs_ref = abs(val - ref_val) / abs(ref_val) if ref_val != 0 else 0

            epsrel_vals.append(tset["xi"])
            rel_diffs.append(max(rel_vs_ref, 1e-16))
            times_ms.append(elapsed_ms)

            report.append(
                f"| {tset['label']} | {tset['xi']:.0e} | {tset['Ep']:.0e} | "
                f"{val:.8e} | {rel_reported:.2e} | {rel_vs_ref:.2e} | {elapsed_ms:.0f} |"
            )

        ax.loglog(epsrel_vals[:-1], rel_diffs[:-1], 'o-b', label='Rel error')
        ax.loglog(epsrel_vals[:-1], epsrel_vals[:-1], '--', color='gray',
                  alpha=0.5, label='y = epsrel')
        ax2 = ax.twinx()
        ax2.semilogx(epsrel_vals, times_ms, 's-r', alpha=0.6, markersize=4)
        ax2.set_ylabel("Time [ms]", color='r', fontsize=9)

        ax.set_xlabel("epsrel_xi")
        ax.set_ylabel("Rel diff vs reference", color='b')
        ax.set_title(f"T={T_kev}, E_in={pt['E_in_kev']}", fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    fig.suptitle("Scipy Tolerance Sensitivity", fontsize=14)
    fig.tight_layout()
    plot_name = "tolerance_sensitivity.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Tolerance Sensitivity](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Post-IBP vs Pre-IBP
# ═══════════════════════════════════════════════════════════════════════════════

IBP_ENERGIES = [
    (100.0, 100.0, 0.0),
    (100.0, 50.0, 0.5),
    (20.0, 30.0, -0.3),
    (5.0, 5.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.8, 0.5),
    (0.1, 0.1, 0.0),
    (0.1, 0.08, -0.5),
    (0.01, 0.01, 0.0),
]

IBP_TEMPERATURES = [100.0, 10.0, 1.0, 0.1, 0.01]


def section_post_vs_pre_ibp(report):
    report.append("## 3. Post-IBP vs Pre-IBP Agreement\n")
    report.append("Both quadrature forms should give identical results (they represent")
    report.append("the same integral). Discrepancies indicate the post-IBP cancellation issue.\n")

    report.append("| T [keV] | E [keV] | E' [keV] | xi | tau | sigma_post | sigma_pre | Rel diff | Notes |")
    report.append("|---|---|---|---|---|---|---|---|---|")

    taus_all = []
    diffs_all = []
    labels_all = []

    for E_kev, Ep_kev, xi in IBP_ENERGIES:
        taus_case = []
        diffs_case = []

        for T_kev in IBP_TEMPERATURES:
            tau = T_kev * KEV / ME_C2
            E = E_kev * KEV
            Ep = Ep_kev * KEV

            eng_post = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)
            eng_pre = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PreIBP)

            try:
                r_post = eng_post.sigma_E(E, Ep, xi, tau, 1.0)
                r_pre = eng_pre.sigma_E(E, Ep, xi, tau, 1.0)
            except Exception as exc:
                report.append(
                    f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | {tau:.3e} | "
                    f"— | — | — | exception: {exc} |"
                )
                continue

            if abs(r_pre.value) < 1e-300:
                report.append(
                    f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | {tau:.3e} | "
                    f"{r_post.value:.4e} | {r_pre.value:.4e} | — | "
                    f"underflow (<1e-300): comparison not meaningful |"
                )
                continue

            rel_diff = abs(r_post.value - r_pre.value) / abs(r_pre.value)
            note = ""
            if rel_diff > 1e-3:
                note = "WARN: cancellation"
            elif rel_diff > 1e-6:
                note = "mild cancellation"

            taus_case.append(tau)
            diffs_case.append(max(rel_diff, 1e-16))

            report.append(
                f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | {tau:.3e} | "
                f"{r_post.value:.4e} | {r_pre.value:.4e} | {rel_diff:.2e} | {note} |"
            )

        if taus_case:
            taus_all.append(taus_case)
            diffs_all.append(diffs_case)
            labels_all.append(f"E={E_kev}, E'={Ep_kev}, xi={xi}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for taus_case, diffs_case, label in zip(taus_all, diffs_all, labels_all):
        ax.loglog(taus_case, diffs_case, 'o-', markersize=5, label=label)

    ax.axhline(1e-3, color='r', linestyle='--', alpha=0.5, label='WARN threshold')
    ax.axhline(1e-6, color='orange', linestyle='--', alpha=0.5, label='mild threshold')
    ax.set_xlabel(r"$\tau = kT / m_e c^2$")
    ax.set_ylabel("Relative difference |Post - Pre| / |Pre|")
    ax.set_title("Post-IBP vs Pre-IBP Agreement")
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout()
    plot_name = "post_vs_pre_ibp.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Post vs Pre IBP](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Richardson error estimates
# ═══════════════════════════════════════════════════════════════════════════════

RICHARDSON_POINTS = [
    (100.0, 100.0, 100.0, 0.0),
    (100.0, 100.0, 50.0, 0.5),
    (100.0, 300.0, 200.0, -0.3),
    (20.0, 20.0, 20.0, 0.0),
    (20.0, 20.0, 15.0, 0.7),
    (5.0, 10.0, 10.0, 0.0),
    (5.0, 10.0, 8.0, -0.5),
    (1.0, 5.0, 5.0, 0.0),
    (1.0, 5.0, 4.0, 0.5),
    (0.1, 1.0, 1.0, 0.0),
    (0.1, 1.0, 0.8, -0.8),
]


def section_pointwise_error_estimates(report):
    report.append("## 4. Built-in Richardson Error Estimates\n")
    report.append("The C++ kernel returns `estimated_rel_error = |IQ(N)-IQ(N/2)| / |value|`.")
    report.append("This section shows these estimates across the parameter space.\n")

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PreIBP)

    report.append("| T [keV] | E [keV] | E' [keV] | xi | sigma_E [cm^2/erg] | est. rel err | est. abs err |")
    report.append("|---|---|---|---|---|---|---|")

    taus = []
    rel_errs = []

    for T_kev, E_kev, Ep_kev, xi in RICHARDSON_POINTS:
        tau = T_kev * KEV / ME_C2
        E = E_kev * KEV
        Ep = Ep_kev * KEV

        try:
            r = engine.sigma_E(E, Ep, xi, tau, 1.0)
        except Exception as exc:
            report.append(
                f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | "
                f"— | — | exception: {exc} |"
            )
            continue

        if abs(r.value) < 1e-300:
            report.append(
                f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | "
                f"{r.value:.4e} | — | underflow (<1e-300) |"
            )
            continue

        taus.append(tau)
        rel_errs.append(r.estimated_rel_error)

        report.append(
            f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | "
            f"{r.value:.4e} | {r.estimated_rel_error:.2e} | {r.estimated_abs_error:.2e} |"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(len(taus)), rel_errs, 'o-', markersize=6)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:.1e}" for t in taus], rotation=45, fontsize=8)
    ax.set_xlabel(r"$\tau$ (test points)")
    ax.set_ylabel("Estimated relative error")
    ax.set_title("Richardson Error Estimates (NL=64, PreIBP)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_name = "richardson_errors.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Richardson Errors](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Timing
# ═══════════════════════════════════════════════════════════════════════════════

def section_timing(report):
    report.append("## 5. Performance Benchmarks\n")
    report.append("Timing for single-point evaluations and bin integrations.\n")

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)
    tau = 100.0 * KEV / ME_C2
    E = 100.0 * KEV
    Ep = 80.0 * KEV

    N_eval = 10000
    t0 = time.time()
    for _ in range(N_eval):
        engine.sigma_E(E, Ep, 0.0, tau, 1.0)
    t_single = (time.time() - t0) / N_eval * 1e6

    E_lo = 60.0 * KEV
    E_hi = 140.0 * KEV
    N_bins = 5
    t0 = time.time()
    for _ in range(N_bins):
        compute_bin_integral(engine, E, E_lo, E_hi, tau, 1e-6, 1e-8)
    t_bin = (time.time() - t0) / N_bins * 1000

    nl_times = {}
    for NL in [64, 128, 256]:
        eng = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PostIBP)
        t0 = time.time()
        for _ in range(N_eval):
            eng.sigma_E(E, Ep, 0.0, tau, 1.0)
        nl_times[NL] = (time.time() - t0) / N_eval * 1e6

    report.append("| Operation | NL | Time |")
    report.append("|---|---|---|")
    report.append(f"| Single sigma_E evaluation | 64 | {t_single:.1f} us |")
    report.append(f"| Bin integration (epsrel=1e-6) | 64 | {t_bin:.1f} ms |")
    for NL in [128, 256]:
        report.append(f"| Single sigma_E evaluation | {NL} | {nl_times[NL]:.1f} us |")

    fig, ax = plt.subplots(figsize=(7, 4))
    nls = sorted(nl_times.keys())
    times = [nl_times[nl] for nl in nls]
    ax.bar([str(nl) for nl in nls], times, color='steelblue', edgecolor='k')
    ax.set_xlabel("NL (quadrature order)")
    ax.set_ylabel("Time per evaluation [us]")
    ax.set_title("Single sigma_E Evaluation Cost")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    plot_name = "timing_vs_nl.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Timing vs NL](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Power series convergence vs number of terms
# ═══════════════════════════════════════════════════════════════════════════════

POWER_SERIES_CASES = [
    # (T_kev, E_kev, Ep_kev, xi) — warm/hot cases where power series converges well
    (20.0,  10.0,  10.5,  0.0),
    (50.0,  10.0,  10.5,  0.0),
    (100.0, 50.0,  55.0,  0.0),
    (100.0, 100.0, 80.0,  0.0),
    (20.0,  50.0,  55.0,  0.3),
    (10.0,  10.0,  9.5,   0.0),
]


def section_power_series_convergence(report):
    report.append("## 6. Power Series Convergence vs Number of Terms\n")
    report.append("Shows how the power series partial sum converges as `n_max` increases.")
    report.append("The reference value is from C++ quadrature (NL=256, PreIBP).")
    report.append("Each curve traces the relative error of the series truncated at n_max terms.\n")

    quad_eng = ComptonKernelQuadrature(NL=256, form=QuadratureForm.PreIBP)
    n_max_values = list(range(1, 81))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    report.append("| Case | T [keV] | E [keV] | E' [keV] | xi | Quad ref | Series (converged) | Terms | Rel err vs quad |")
    report.append("|---|---|---|---|---|---|---|---|---|")

    for ci, (T_kev, E_kev, Ep_kev, xi) in enumerate(POWER_SERIES_CASES):
        tau = T_kev * KEV / ME_C2
        E = E_kev * KEV
        Ep = Ep_kev * KEV

        ref = quad_eng.sigma_E(E, Ep, xi, tau, 1.0)
        ref_val = ref.value

        rel_errors = []
        values = []
        for nm in n_max_values:
            eng = ComptonKernelSeries(method=SeriesMethod.PowerSeries, eps_rel=1e-15, n_min=nm+1, n_max=nm)
            r = eng.sigma_E(E, Ep, xi, tau, 1.0)
            values.append(r.value)
            if abs(ref_val) > 1e-300:
                rel_errors.append(abs(r.value - ref_val) / abs(ref_val))
            else:
                rel_errors.append(0.0)

        conv_eng = ComptonKernelSeries(method=SeriesMethod.PowerSeries)
        conv_r = conv_eng.sigma_E(E, Ep, xi, tau, 1.0)
        if abs(ref_val) > 1e-300:
            conv_rel = abs(conv_r.value - ref_val) / abs(ref_val)
        else:
            conv_rel = 0.0

        report.append(
            f"| {ci+1} | {T_kev} | {E_kev} | {Ep_kev} | {xi} | {ref_val:.6e} | "
            f"{conv_r.value:.6e} | {conv_r.terms_used} | {conv_rel:.2e} |"
        )

        ax = axes[ci]
        plot_errs = [max(e, 1e-16) for e in rel_errors]
        ax.semilogy(n_max_values, plot_errs, '-', color='steelblue', linewidth=1.5)
        if conv_r.converged:
            ax.axvline(conv_r.terms_used, color='red', linestyle='--', alpha=0.7,
                       label=f'Converged at n={conv_r.terms_used}')
        ax.set_xlabel("n_max (terms)")
        ax.set_ylabel("Rel error vs quadrature")
        ax.set_title(f"T={T_kev}, E={E_kev}, E'={Ep_kev}, ξ={xi}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-16, top=1e2)
        ax.legend(fontsize=7)

    fig.suptitle("Power Series: Convergence vs Number of Terms", fontsize=14)
    fig.tight_layout()
    plot_name = "power_series_convergence.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Power Series Convergence](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Asymptotic series convergence and optimal truncation
# ═══════════════════════════════════════════════════════════════════════════════

ASYMPTOTIC_SERIES_CASES = [
    # (T_kev, E_kev, Ep_kev, xi) — cases where asymptotic series is the preferred method
    (0.1,   1.0,  1.01,  0.0),
    (1.0,   1.0,  1.01,  0.0),
    (0.5,   5.0,  5.5,   0.0),
    (0.1,   1.0,  0.99, -0.5),
    (2.0,  10.0, 10.5,   0.0),
    (5.0,  10.0, 10.5,   0.3),
]


def section_asymptotic_series_convergence(report):
    report.append("## 7. Asymptotic Series Convergence and Optimal Truncation\n")
    report.append("The asymptotic series is non-convergent: terms first decrease, reach a minimum,")
    report.append("then diverge. The optimal truncation is at the smallest term.")
    report.append("Top panels show the relative error of the partial sum vs quadrature reference.")
    report.append("Bottom panels show the term magnitude at each order, revealing the U-shape.\n")

    quad_eng = ComptonKernelQuadrature(NL=256, form=QuadratureForm.PreIBP)
    n_max_values = list(range(1, 61))

    fig, axes = plt.subplots(2, len(ASYMPTOTIC_SERIES_CASES),
                             figsize=(4 * len(ASYMPTOTIC_SERIES_CASES), 8))

    report.append("| Case | T [keV] | E [keV] | E' [keV] | xi | τ·α_max | Quad ref | Series (converged) | Optimal n | Rel err vs quad |")
    report.append("|---|---|---|---|---|---|---|---|---|---|")

    for ci, (T_kev, E_kev, Ep_kev, xi) in enumerate(ASYMPTOTIC_SERIES_CASES):
        tau = T_kev * KEV / ME_C2
        E = E_kev * KEV
        Ep = Ep_kev * KEV

        ref = quad_eng.sigma_E(E, Ep, xi, tau, 1.0)
        ref_val = ref.value

        rel_errors = []
        term_mags = []
        for nm in n_max_values:
            eng = ComptonKernelSeries(method=SeriesMethod.Asymptotic, eps_rel=1e-15, n_min=nm+1, n_max=nm)
            r = eng.sigma_E(E, Ep, xi, tau, 1.0)
            if abs(ref_val) > 1e-300:
                rel_errors.append(abs(r.value - ref_val) / abs(ref_val))
            else:
                rel_errors.append(0.0)
            term_mags.append(r.estimated_abs_error)

        conv_eng = ComptonKernelSeries(method=SeriesMethod.Asymptotic)
        conv_r = conv_eng.sigma_E(E, Ep, xi, tau, 1.0)
        if abs(ref_val) > 1e-300:
            conv_rel = abs(conv_r.value - ref_val) / abs(ref_val)
        else:
            conv_rel = 0.0

        gamma = E / ME_C2
        gamma_p = Ep / ME_C2
        a = 1.0 - xi
        dg = gamma_p - gamma
        q2 = dg**2 + 2*gamma*gamma_p*a
        delta = np.sqrt((1 + gamma*gamma_p*a/2) * (1 + dg**2/(2*gamma*gamma_p*a)))
        lp = dg/2 + delta
        rho_p = lp + gamma
        rho_m = lp - gamma_p
        omega2 = (1 + xi) / a
        alpha_p = 1.0 / np.sqrt(rho_p**2 + omega2)
        alpha_m = 1.0 / np.sqrt(rho_m**2 + omega2)
        tau_alpha_max = tau * max(alpha_p, alpha_m)

        report.append(
            f"| {ci+1} | {T_kev} | {E_kev} | {Ep_kev} | {xi} | {tau_alpha_max:.4f} | "
            f"{ref_val:.6e} | {conv_r.value:.6e} | {conv_r.terms_used} | {conv_rel:.2e} |"
        )

        ax_err = axes[0, ci]
        plot_errs = [max(e, 1e-16) for e in rel_errors]
        ax_err.semilogy(n_max_values, plot_errs, '-', color='coral', linewidth=1.5)
        if conv_r.converged:
            ax_err.axvline(conv_r.terms_used, color='green', linestyle='--', alpha=0.7,
                           label=f'Truncated at n={conv_r.terms_used}')
        best_n = n_max_values[np.argmin(rel_errors)]
        ax_err.axvline(best_n, color='blue', linestyle=':', alpha=0.5,
                       label=f'Best n={best_n}')
        ax_err.set_ylabel("Rel error vs quadrature")
        ax_err.set_title(f"T={T_kev}, E={E_kev}, E'={Ep_kev}\nξ={xi}, τα={tau_alpha_max:.3f}",
                         fontsize=8)
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(bottom=1e-16, top=1e2)
        ax_err.legend(fontsize=6)

        ax_term = axes[1, ci]
        plot_terms = [max(t, 1e-300) for t in term_mags]
        ax_term.semilogy(n_max_values, plot_terms, '-', color='darkorange', linewidth=1.5)
        if conv_r.converged:
            ax_term.axvline(conv_r.terms_used, color='green', linestyle='--', alpha=0.7)
        ax_term.set_xlabel("n_max (terms)")
        ax_term.set_ylabel("| σ₀ × smallest term |")
        ax_term.set_title("Term magnitude", fontsize=9)
        ax_term.grid(True, alpha=0.3)

    fig.suptitle("Asymptotic Series: Convergence, Divergence, and Optimal Truncation", fontsize=13)
    fig.tight_layout()
    plot_name = "asymptotic_series_convergence.png"
    fig.savefig(os.path.join(FIGS_DIR, plot_name), dpi=150)
    plt.close(fig)

    report.append(f"\n![Asymptotic Series Convergence](figs/{plot_name})\n")
    report.append("")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    report = []
    report.append("# Convergence and Tolerance Analysis Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("This report quantifies the numerical accuracy of the Compton kernel")
    report.append("across parameter regimes, covering the Gauss-Laguerre quadrature")
    report.append("convergence, external scipy integration, and series term convergence.\n")
    report.append("---\n")

    print("Running convergence analysis...")

    print("  [1/7] Quadrature order convergence...")
    t0 = time.time()
    section_quadrature_order_convergence(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [2/7] Scipy tolerance sensitivity...")
    t0 = time.time()
    section_scipy_tolerance_sensitivity(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [3/7] Post-IBP vs Pre-IBP agreement...")
    t0 = time.time()
    section_post_vs_pre_ibp(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [4/7] Pointwise error estimates...")
    t0 = time.time()
    section_pointwise_error_estimates(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [5/7] Performance benchmarks...")
    t0 = time.time()
    section_timing(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [6/7] Power series convergence...")
    t0 = time.time()
    section_power_series_convergence(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [7/7] Asymptotic series convergence...")
    t0 = time.time()
    section_asymptotic_series_convergence(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    outpath = os.path.join(GEN_DIR, "convergence_analysis.md")
    with open(outpath, "w") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nReport saved to: {outpath}")


if __name__ == "__main__":
    main()
