"""
Tolerance and convergence analysis for the Compton kernel quadrature.

Generates a comprehensive markdown report covering:
  1. Gauss-Laguerre quadrature order convergence (NL=64 vs 128 vs 256)
  2. Scipy outer integration tolerance sensitivity
  3. Post-IBP vs Pre-IBP agreement across temperature regimes
  4. Pointwise kernel error estimates from the C++ Richardson indicator

Usage:
    python3 tests/convergence_analysis.py

Output:
    reports/convergence_analysis.md
"""
import sys
import os
import time
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm, scaled_K2

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9
XI_EPS = 1e-10

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


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


def section_quadrature_order_convergence(report):
    """Test NL=64 vs 128 vs 256 across multiple regimes."""
    report.append("## 1. Gauss-Laguerre Quadrature Order Convergence\n")
    report.append("Compares the bin-integrated kernel at NL=64, 128, 256 to quantify")
    report.append("the internal quadrature convergence. The outer scipy integration uses")
    report.append("tight tolerances (epsrel_xi=1e-6, epsrel_Ep=1e-8) so the difference")
    report.append("isolates the Gauss-Laguerre truncation error.\n")

    test_cases = [
        {"T_kev": 100.0, "E_in_kev": 100.0, "bins_kev": [(80, 120), (20, 60), (150, 250)]},
        {"T_kev": 20.0, "E_in_kev": 40.0, "bins_kev": [(35, 45), (10, 30), (50, 80)]},
        {"T_kev": 1.0, "E_in_kev": 10.0, "bins_kev": [(9, 11), (5, 8), (12, 15)]},
        {"T_kev": 0.1, "E_in_kev": 5.0, "bins_kev": [(4.5, 5.5), (3, 4), (6, 8)]},
    ]

    NL_values = [64, 128, 256]

    for case in test_cases:
        T_kev = case["T_kev"]
        tau = T_kev * KEV / ME_C2
        E_in = case["E_in_kev"] * KEV

        report.append(f"\n### T = {T_kev} keV (tau = {tau:.4e}), E_in = {case['E_in_kev']} keV\n")
        report.append("| Bin [keV] | S(NL=64) | S(NL=128) | S(NL=256) | diff(64,128)/128 | diff(128,256)/256 |")
        report.append("|---|---|---|---|---|---|")

        for bin_lo, bin_hi in case["bins_kev"]:
            E_lo = bin_lo * KEV
            E_hi = bin_hi * KEV
            results = {}
            for NL in NL_values:
                eng = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PreIBP)
                val, _ = compute_bin_integral(eng, E_in, E_lo, E_hi, tau, 1e-6, 1e-8)
                results[NL] = val

            if abs(results[256]) < 1e-50:
                report.append(f"| [{bin_lo}-{bin_hi}] | - | - | - | (underflow) | - |")
                continue

            d64 = abs(results[64] - results[128]) / abs(results[128]) if results[128] != 0 else 0
            d128 = abs(results[128] - results[256]) / abs(results[256])

            report.append(
                f"| [{bin_lo}-{bin_hi}] | {results[64]:.6e} | {results[128]:.6e} | "
                f"{results[256]:.6e} | {d64:.2e} | {d128:.2e} |"
            )

    report.append("")


def section_scipy_tolerance_sensitivity(report):
    """Vary scipy tolerances and measure the effect on the result."""
    report.append("## 2. Scipy Integration Tolerance Sensitivity\n")
    report.append("Fixes NL=64 and varies the scipy `epsrel` to measure how the outer")
    report.append("integration tolerance affects the final result. Reference: tightest setting.\n")

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

    tolerance_sets = [
        {"label": "Loose", "xi": 1e-2, "Ep": 1e-3},
        {"label": "Default", "xi": 1e-3, "Ep": 1e-4},
        {"label": "Tight", "xi": 1e-4, "Ep": 1e-6},
        {"label": "Very tight", "xi": 1e-6, "Ep": 1e-8},
        {"label": "Reference", "xi": 1e-8, "Ep": 1e-10},
    ]

    test_points = [
        {"T_kev": 100.0, "E_in_kev": 100.0, "bin_kev": (60, 140)},
        {"T_kev": 100.0, "E_in_kev": 300.0, "bin_kev": (200, 400)},
        {"T_kev": 1.0, "E_in_kev": 10.0, "bin_kev": (8, 12)},
    ]

    for pt in test_points:
        T_kev = pt["T_kev"]
        tau = T_kev * KEV / ME_C2
        E_in = pt["E_in_kev"] * KEV
        E_lo = pt["bin_kev"][0] * KEV
        E_hi = pt["bin_kev"][1] * KEV

        report.append(f"\n### T={T_kev} keV, E_in={pt['E_in_kev']} keV, bin=[{pt['bin_kev'][0]}-{pt['bin_kev'][1]}] keV\n")
        report.append("| Tolerance | epsrel_xi | epsrel_Ep | S [cm^2] | Reported err | Rel diff vs ref | Time [ms] |")
        report.append("|---|---|---|---|---|---|---|")

        # First pass: compute reference value
        ref_tset = tolerance_sets[-1]
        ref_val, _ = compute_bin_integral(engine, E_in, E_lo, E_hi, tau,
                                         ref_tset["xi"], ref_tset["Ep"])

        for tset in tolerance_sets:
            t0 = time.time()
            val, err = compute_bin_integral(engine, E_in, E_lo, E_hi, tau,
                                           tset["xi"], tset["Ep"])
            elapsed_ms = (time.time() - t0) * 1000

            rel_reported = err / abs(val) if val != 0 else 0
            rel_vs_ref = abs(val - ref_val) / abs(ref_val) if ref_val != 0 else 0

            report.append(
                f"| {tset['label']} | {tset['xi']:.0e} | {tset['Ep']:.0e} | "
                f"{val:.8e} | {rel_reported:.2e} | {rel_vs_ref:.2e} | {elapsed_ms:.0f} |"
            )

    report.append("")


def section_post_vs_pre_ibp(report):
    """Compare PostIBP and PreIBP forms across temperature regimes."""
    report.append("## 3. Post-IBP vs Pre-IBP Agreement\n")
    report.append("Both quadrature forms should give identical results (they represent")
    report.append("the same integral). Discrepancies indicate the post-IBP cancellation issue.\n")

    test_points = [
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

    report.append("| T [keV] | E [keV] | E' [keV] | xi | tau | sigma_post | sigma_pre | Rel diff | Notes |")
    report.append("|---|---|---|---|---|---|---|---|---|")

    for E_kev, Ep_kev, xi in test_points:
        for T_kev in [100.0, 10.0, 1.0, 0.1]:
            tau = T_kev * KEV / ME_C2
            E = E_kev * KEV
            Ep = Ep_kev * KEV

            eng_post = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)
            eng_pre = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PreIBP)

            try:
                r_post = eng_post.sigma_E(E, Ep, xi, tau, 1.0)
                r_pre = eng_pre.sigma_E(E, Ep, xi, tau, 1.0)
            except Exception:
                continue

            if abs(r_pre.value) < 1e-300:
                continue

            rel_diff = abs(r_post.value - r_pre.value) / abs(r_pre.value)
            note = ""
            if rel_diff > 1e-3:
                note = "WARN: cancellation"
            elif rel_diff > 1e-6:
                note = "mild cancellation"

            if rel_diff < 1e-12 and T_kev > 1.0:
                continue  # Skip uninteresting perfect-agreement lines

            report.append(
                f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | {tau:.3e} | "
                f"{r_post.value:.4e} | {r_pre.value:.4e} | {rel_diff:.2e} | {note} |"
            )

    report.append("")


def section_pointwise_error_estimates(report):
    """Show the built-in Richardson error estimates for representative points."""
    report.append("## 4. Built-in Richardson Error Estimates\n")
    report.append("The C++ kernel returns `estimated_rel_error = |IQ(N)−IQ(N/2)| / |value|`.")
    report.append("This section shows these estimates across the parameter space.\n")

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PreIBP)

    report.append("| T [keV] | E [keV] | E' [keV] | xi | sigma_E [cm^2/erg] | est. rel err | est. abs err |")
    report.append("|---|---|---|---|---|---|---|")

    scan_points = [
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

    for T_kev, E_kev, Ep_kev, xi in scan_points:
        tau = T_kev * KEV / ME_C2
        E = E_kev * KEV
        Ep = Ep_kev * KEV

        try:
            r = engine.sigma_E(E, Ep, xi, tau, 1.0)
        except Exception:
            continue

        if abs(r.value) < 1e-300:
            continue

        report.append(
            f"| {T_kev} | {E_kev} | {Ep_kev} | {xi} | "
            f"{r.value:.4e} | {r.estimated_rel_error:.2e} | {r.estimated_abs_error:.2e} |"
        )

    report.append("")


def section_timing(report):
    """Benchmark the evaluation speed."""
    report.append("## 5. Performance Benchmarks\n")
    report.append("Timing for single-point evaluations and bin integrations.\n")

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)
    tau = 100.0 * KEV / ME_C2
    E = 100.0 * KEV
    Ep = 80.0 * KEV

    # Single-point benchmark
    N_eval = 10000
    t0 = time.time()
    for _ in range(N_eval):
        engine.sigma_E(E, Ep, 0.0, tau, 1.0)
    t_single = (time.time() - t0) / N_eval * 1e6  # microseconds

    # Bin integration benchmark
    E_lo = 60.0 * KEV
    E_hi = 140.0 * KEV
    N_bins = 5
    t0 = time.time()
    for _ in range(N_bins):
        compute_bin_integral(engine, E, E_lo, E_hi, tau, 1e-6, 1e-8)
    t_bin = (time.time() - t0) / N_bins * 1000  # milliseconds

    report.append("| Operation | NL | Time |")
    report.append("|---|---|---|")
    report.append(f"| Single sigma_E evaluation | 64 | {t_single:.1f} us |")
    report.append(f"| Bin integration (epsrel=1e-6) | 64 | {t_bin:.1f} ms |")

    for NL in [128, 256]:
        eng = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PostIBP)
        t0 = time.time()
        for _ in range(N_eval):
            eng.sigma_E(E, Ep, 0.0, tau, 1.0)
        t_nl = (time.time() - t0) / N_eval * 1e6
        report.append(f"| Single sigma_E evaluation | {NL} | {t_nl:.1f} us |")

    report.append("")


def main():
    report = []
    report.append("# Convergence and Tolerance Analysis Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("This report quantifies the numerical accuracy of the Compton kernel")
    report.append("quadrature across parameter regimes, covering both the internal")
    report.append("Gauss-Laguerre convergence and the external scipy integration.\n")
    report.append("---\n")

    print("Running convergence analysis...")

    print("  [1/5] Quadrature order convergence...")
    t0 = time.time()
    section_quadrature_order_convergence(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [2/5] Scipy tolerance sensitivity...")
    t0 = time.time()
    section_scipy_tolerance_sensitivity(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [3/5] Post-IBP vs Pre-IBP agreement...")
    t0 = time.time()
    section_post_vs_pre_ibp(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [4/5] Pointwise error estimates...")
    t0 = time.time()
    section_pointwise_error_estimates(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    print("  [5/5] Performance benchmarks...")
    t0 = time.time()
    section_timing(report)
    print(f"        Done ({time.time()-t0:.1f}s)")

    # Write report
    outpath = os.path.join(REPORTS_DIR, "convergence_analysis.md")
    with open(outpath, "w") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nReport saved to: {outpath}")


if __name__ == "__main__":
    main()
