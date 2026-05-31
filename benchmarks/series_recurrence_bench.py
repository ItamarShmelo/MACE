"""
Benchmark script for series recurrence optimizations.

Measures performance before/after optimization, storing values, timings,
and diagnostic parameters to JSON. Uses shared instrumentation via
compute_params to avoid formula duplication.
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
from _compton_kernel_series import ComptonKernelSeries, SeriesMethod, ehat_cf
from pycompton.compton_kernel_series import sigma_E_series
from pycompton.compton_kernel_quadrature import compute_params, me_c2, kev

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = 1
VECTOR_SIZES = [1_000, 10_000, 100_000]
N_TIMING_REPEATS = 5

TOLERANCE_SETTINGS = {
    "rel_tol": 1e-14,
    "abs_floor": 1e-30,
    "scale_aware": True,
}

# Benchmark points: (E_keV, E_prime_keV, xi, T_keV)
BENCHMARK_POINTS = [
    # Power series regime
    {"label": "power_near_elastic", "E_keV": 1, "Ep_keV": 1.01, "xi": 0.0, "T_keV": 100},
    {"label": "power_moderate", "E_keV": 10, "Ep_keV": 10.5, "xi": 0.0, "T_keV": 20},
    {"label": "power_high_T", "E_keV": 100, "Ep_keV": 101, "xi": 0.0, "T_keV": 100},
    # Asymptotic regime
    {"label": "asymp_cold_elastic", "E_keV": 1, "Ep_keV": 1.01, "xi": 0.0, "T_keV": 0.1},
    {"label": "asymp_inelastic", "E_keV": 1, "Ep_keV": 2.0, "xi": 0.0, "T_keV": 1.0},
    {"label": "asymp_mid_T", "E_keV": 10, "Ep_keV": 10.5, "xi": 0.0, "T_keV": 5.0},
    {"label": "asymp_high_E", "E_keV": 100, "Ep_keV": 101, "xi": 0.0, "T_keV": 5.0},
    # Forward scattering (xi near 1)
    {"label": "forward_1", "E_keV": 1, "Ep_keV": 1.5, "xi": 0.95, "T_keV": 100},
    {"label": "forward_2", "E_keV": 10, "Ep_keV": 12, "xi": 0.9, "T_keV": 50},
    # Backscatter (xi near -1)
    {"label": "backscatter_1", "E_keV": 1, "Ep_keV": 0.5, "xi": -0.95, "T_keV": 100},
    {"label": "backscatter_2", "E_keV": 10, "Ep_keV": 8, "xi": -0.9, "T_keV": 50},
    # Near regime switch
    {"label": "regime_switch_1", "E_keV": 1, "Ep_keV": 1.5, "xi": 0.0, "T_keV": 40},
    {"label": "regime_switch_2", "E_keV": 5, "Ep_keV": 5.5, "xi": 0.3, "T_keV": 40},
    # Large x (inelastic, barely power)
    {"label": "large_x_1", "E_keV": 100, "Ep_keV": 200, "xi": -0.5, "T_keV": 40},
    {"label": "large_x_2", "E_keV": 100, "Ep_keV": 500, "xi": -0.9, "T_keV": 30},
]


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_diagnostics(E_keV, Ep_keV, xi, T_keV):
    """Compute diagnostic parameters via shared compute_params."""
    import math
    E = E_keV * kev
    Ep = Ep_keV * kev
    gamma = E / me_c2
    gamma_p = Ep / me_c2
    tau = T_keV * kev / me_c2

    p = compute_params(gamma, gamma_p, xi, tau)
    omega = math.sqrt(p.omega2)
    b = omega / (2.0 * tau)

    tau_alpha_plus = tau * p.alpha_plus
    tau_alpha_minus = tau * p.alpha_minus
    method = "asymptotic" if max(tau_alpha_plus, tau_alpha_minus) < 0.05 else "power"

    diag = {
        "tau": tau,
        "method": method,
        "tau_alpha_plus": tau_alpha_plus,
        "tau_alpha_minus": tau_alpha_minus,
        "zeta_plus": p.rho_plus * p.alpha_plus,
        "zeta_minus": p.rho_minus * p.alpha_minus,
    }

    if method == "power":
        theta_plus = math.asinh(p.rho_plus / omega)
        theta_minus = math.asinh(p.rho_minus / omega)
        diag["x_plus"] = b * math.exp(theta_plus)
        diag["y_plus"] = b * math.exp(-theta_plus)
        diag["x_minus"] = b * math.exp(theta_minus)
        diag["y_minus"] = b * math.exp(-theta_minus)

    return diag


def benchmark_single_point(pt, kernel):
    """Benchmark a single point: C++ single-call and Python."""
    E = pt["E_keV"] * kev
    Ep = pt["Ep_keV"] * kev
    xi = pt["xi"]
    tau = pt["T_keV"] * kev / me_c2

    # C++ single call (measures pybind + kernel)
    times_cpp_single = []
    result = None
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        result = kernel.sigma_E(E, Ep, xi, tau, 1.0)
        t1 = time.perf_counter()
        times_cpp_single.append(t1 - t0)

    # Pure Python call
    times_python = []
    py_result = None
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        py_result = sigma_E_series(E, Ep, xi, tau, Ne=1.0)
        t1 = time.perf_counter()
        times_python.append(t1 - t0)

    return {
        "cpp_value": result.value,
        "cpp_terms": result.terms_used,
        "cpp_single_call_us": np.median(times_cpp_single) * 1e6,
        "python_value": py_result.value,
        "python_terms": py_result.terms_used,
        "python_time_us": np.median(times_python) * 1e6,
    }


def benchmark_vectorized(pt, kernel, N):
    """Benchmark vectorized C++ throughput with N identical E' values."""
    E = pt["E_keV"] * kev
    Ep = pt["Ep_keV"] * kev
    xi = pt["xi"]
    tau = pt["T_keV"] * kev / me_c2

    Ep_arr = np.full(N, Ep)

    times = []
    values = None
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        values, errors, terms = kernel.sigma_E_vec(E, Ep_arr, xi, tau, 1.0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_time = np.median(times)
    per_eval_us = median_time / N * 1e6

    return {
        "N": N,
        "total_time_ms": median_time * 1e3,
        "per_eval_us": per_eval_us,
        "value_first": float(values[0]),
    }


def benchmark_heterogeneous(pt, kernel, N=10_000):
    """Benchmark with heterogeneous E' values spanning a range."""
    E = pt["E_keV"] * kev
    Ep_center = pt["Ep_keV"] * kev
    xi = pt["xi"]
    tau = pt["T_keV"] * kev / me_c2

    Ep_lo = Ep_center * 0.9
    Ep_hi = Ep_center * 1.1
    Ep_arr = np.linspace(Ep_lo, Ep_hi, N)

    times = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        values, errors, terms = kernel.sigma_E_vec(E, Ep_arr, xi, tau, 1.0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_time = np.median(times)
    per_eval_us = median_time / N * 1e6

    return {
        "N": N,
        "total_time_ms": median_time * 1e3,
        "per_eval_us": per_eval_us,
        "vector_type": "heterogeneous",
    }


def run_benchmark(output_file, implementation_mode="baseline", amplification_budget=None):
    """Run the full benchmark suite and save results."""
    kernel = ComptonKernelSeries()

    results = {
        "schema_version": SCHEMA_VERSION,
        "implementation_mode": implementation_mode,
        "vector_sizes": VECTOR_SIZES,
        "tolerance_settings": TOLERANCE_SETTINGS,
        "amplification_budget": amplification_budget,
        "git_commit_hash": get_git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "points": [],
    }

    for pt in BENCHMARK_POINTS:
        print(f"  Benchmarking: {pt['label']}...")
        diag = compute_diagnostics(pt["E_keV"], pt["Ep_keV"], pt["xi"], pt["T_keV"])
        single = benchmark_single_point(pt, kernel)

        # Vectorized benchmarks at multiple sizes
        vec_results = {}
        for N in VECTOR_SIZES:
            vec_results[str(N)] = benchmark_vectorized(pt, kernel, N)

        # Heterogeneous vector
        hetero = benchmark_heterogeneous(pt, kernel, N=10_000)

        point_result = {
            "label": pt["label"],
            "inputs": {
                "E_keV": pt["E_keV"],
                "Ep_keV": pt["Ep_keV"],
                "xi": pt["xi"],
                "T_keV": pt["T_keV"],
            },
            "diagnostics": diag,
            "single_call": single,
            "vectorized_identical": vec_results,
            "vectorized_heterogeneous": hetero,
        }
        results["points"].append(point_result)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark saved to: {output_file}")
    print_summary(results)
    return results


def print_summary(results):
    """Print a summary table of benchmark results."""
    print(f"\n{'='*80}")
    print(f"Benchmark Summary ({results['implementation_mode']})")
    print(f"{'='*80}")
    print(f"{'Point':<25} {'Method':<10} {'C++ vec(10k) us':<18} {'Python us':<12} {'Terms'}")
    print(f"{'-'*80}")

    for pt in results["points"]:
        label = pt["label"]
        method = pt["diagnostics"]["method"]
        vec_us = pt["vectorized_identical"]["10000"]["per_eval_us"]
        py_us = pt["single_call"]["python_time_us"]
        terms = pt["single_call"]["cpp_terms"]
        print(f"{label:<25} {method:<10} {vec_us:<18.3f} {py_us:<12.1f} {terms}")


def estimate_series_conditioning(pt):
    """
    Estimate the conditioning number for a series evaluation point.

    For the power series, P_plus and P_minus can nearly cancel, amplifying
    any difference in individual terms. The conditioning number is:
      cond = max(|P_plus|, |P_minus|) / |Psi + P_plus - P_minus|

    Returns the effective relative tolerance for this point.
    """
    import math as _math
    inp = pt["inputs"]
    E = inp["E_keV"] * kev
    Ep = inp["Ep_keV"] * kev
    xi = inp["xi"]
    tau = inp["T_keV"] * kev / me_c2

    gamma = E / me_c2
    gamma_p = Ep / me_c2
    p = compute_params(gamma, gamma_p, xi, tau)

    method = pt["diagnostics"]["method"]
    if method != "power":
        return TOLERANCE_SETTINGS["rel_tol"]

    omega = _math.sqrt(p.omega2)
    b = omega / (2.0 * tau)
    theta_plus = _math.asinh(p.rho_plus / omega)
    theta_minus = _math.asinh(p.rho_minus / omega)
    x_plus = b * _math.exp(theta_plus)
    y_plus = b * _math.exp(-theta_plus)
    x_minus = b * _math.exp(theta_minus)
    y_minus = b * _math.exp(-theta_minus)

    if y_plus > _POISSON_Y_MAX or y_minus > _POISSON_Y_MAX:
        return TOLERANCE_SETTINGS["rel_tol"]

    w_plus = _math.exp(-y_plus)
    w_minus = _math.exp(-y_minus)
    P_plus = 0.0
    P_minus = 0.0

    from pycompton.compton_kernel_series import ehat_expn as _ehat
    for n in range(200):
        coeff_plus = p.A_plus + 2.0 * n / p.a
        coeff_minus = p.A_minus + 2.0 * n / p.a
        t_plus = w_plus * coeff_plus * _ehat(n + 1, x_plus)
        t_minus = w_minus * coeff_minus * _ehat(n + 1, x_minus)
        P_plus += t_plus
        P_minus += t_minus
        term_mag = abs(t_plus) + abs(t_minus)
        S_n = abs(P_plus) + abs(P_minus)
        if n >= 4 and term_mag / (S_n + 1e-300) < 1e-12:
            break
        if n < 199:
            w_plus *= y_plus / (n + 1)
            w_minus *= y_minus / (n + 1)

    result = abs(p.Psi + P_plus - P_minus)
    if result < 1e-300:
        return 1.0

    sum_abs = abs(P_plus) + abs(P_minus) + abs(p.Psi)
    conditioning = sum_abs / result
    eps_machine = 2.2e-16
    effective_tol = max(TOLERANCE_SETTINGS["rel_tol"], conditioning * eps_machine * 10)
    return effective_tol


_POISSON_Y_MAX = 500.0


def compare_results(before_file, after_file):
    """Compare before/after results for correctness and speedup."""
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)

    rel_tol = TOLERANCE_SETTINGS["rel_tol"]
    abs_floor = TOLERANCE_SETTINGS["abs_floor"]

    print(f"\n{'='*80}")
    print("Correctness and Performance Comparison")
    print(f"{'='*80}")
    print(f"{'Point':<25} {'|diff|':<12} {'rel_err':<12} {'eff_tol':<12} {'Speedup(vec)':<14} {'Pass'}")
    print(f"{'-'*90}")

    all_pass = True
    for bp, ap in zip(before["points"], after["points"]):
        v_before = bp["single_call"]["cpp_value"]
        v_after = ap["single_call"]["cpp_value"]

        abs_err = abs(v_before - v_after)
        effective_tol = estimate_series_conditioning(bp)

        if abs_err < max(abs_floor, effective_tol * abs(v_before)):
            passed = True
            rel_err = abs_err / max(abs(v_before), abs_floor) if abs(v_before) > abs_floor else 0.0
        else:
            rel_err = abs_err / max(abs(v_before), abs_floor)
            passed = rel_err < effective_tol

        vec_before = bp["vectorized_identical"]["10000"]["per_eval_us"]
        vec_after = ap["vectorized_identical"]["10000"]["per_eval_us"]
        speedup = vec_before / vec_after if vec_after > 0 else float('inf')

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{bp['label']:<25} {abs_err:<12.2e} {rel_err:<12.2e} {effective_tol:<12.2e} {speedup:<14.2f} {status}")

    print(f"\n{'Overall: ' + ('ALL PASS' if all_pass else 'FAILED')}")

    # Python comparison
    print(f"\n{'Point':<25} {'Py before us':<14} {'Py after us':<14} {'Py speedup'}")
    print(f"{'-'*60}")
    for bp, ap in zip(before["points"], after["points"]):
        py_b = bp["single_call"]["python_time_us"]
        py_a = ap["single_call"]["python_time_us"]
        speedup = py_b / py_a if py_a > 0 else float('inf')
        print(f"{bp['label']:<25} {py_b:<14.1f} {py_a:<14.1f} {speedup:<.2f}")

    return all_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Series recurrence benchmark")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--mode", default="baseline",
                        choices=["baseline", "recurrence"],
                        help="Implementation mode label")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two benchmark files")
    parser.add_argument("--budget", type=float, default=None,
                        help="Amplification budget value (for metadata)")
    args = parser.parse_args()

    if args.compare:
        success = compare_results(args.compare[0], args.compare[1])
        sys.exit(0 if success else 1)

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__),
            f"{'before' if args.mode == 'baseline' else 'after'}.json"
        )

    print(f"Running series recurrence benchmark ({args.mode})...")
    run_benchmark(args.output, implementation_mode=args.mode,
                  amplification_budget=args.budget)
