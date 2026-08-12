"""
Generate benchmarks/VERIFICATION_REPORT.md from JSON result files.
"""

import json
import os
import platform
import subprocess
import sys


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_environment_info():
    """Collect environment information."""
    info = {}
    info["os"] = f"{platform.system()} {platform.release()}"

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, IndexError):
        info["cpu"] = platform.processor() or "unknown"

    info["cores"] = os.cpu_count()

    try:
        result = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, timeout=5
        )
        first_line = result.stdout.split("\n")[0]
        info["compiler"] = first_line
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["compiler"] = "unknown"

    info["python"] = platform.python_version()

    return info


def format_sci(value):
    """Format a number in scientific notation."""
    if value == 0:
        return "`0`"
    return f"`{value:.2e}`"


def generate_report(results_dir):
    """Generate the verification report markdown."""
    point_kernel = load_json(os.path.join(results_dir, "results_point_kernel.json"))
    multigroup_1t = load_json(os.path.join(results_dir, "results_multigroup_1t.json"))
    multigroup_15t = load_json(os.path.join(results_dir, "results_multigroup_15t.json"))
    stress = load_json(os.path.join(results_dir, "results_stress.json"))

    env = get_environment_info()

    lines = []
    lines.append("# Verification report: Compton kernel approximate benchmark")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- CPU: {env['cpu']}")
    lines.append(f"- visible cores: {env['cores']}")
    lines.append(f"- operating system: {env['os']}")
    lines.append(f"- compiler: {env['compiler']}")
    lines.append(f"- Python: {env['python']}")
    lines.append(f"- interface: pybind11 (Python bindings)")
    lines.append("")

    # Point-kernel accuracy
    lines.append("## Point-kernel accuracy")
    lines.append("")
    acc = point_kernel["accuracy"]
    lines.append(
        "Comparison of `ComptonKernelApproximateSolver` against `ComptonKernelSolver`"
        f" on the structured {acc['points']}-point grid:"
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| points | {acc['points']} |")
    lines.append(f"| solver failures | {acc['solver_failures']} |")
    lines.append(f"| median relative error | {format_sci(acc['relative_median'])} |")
    lines.append(f"| 95th percentile | {format_sci(acc['relative_p95'])} |")
    lines.append(f"| 99th percentile | {format_sci(acc['relative_p99'])} |")
    lines.append(f"| maximum | {format_sci(acc['relative_max'])} |")
    lines.append("")
    lines.append("Raw `ComptonKernelApproximate` diagnostic statistics:")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| accepted points | {acc['approximate_accepted']} |")
    lines.append(f"| failures | {acc['approximate_failures']} |")
    lines.append(f"| rejections (self-error >= 3e-4) | {acc['approximate_rejections']} |")
    if "approximate_relative_max" in acc:
        lines.append(
            f"| max relative error (accepted) | {format_sci(acc['approximate_relative_max'])} |"
        )
    lines.append("")

    # Point-kernel timing
    lines.append("## Point-kernel performance")
    lines.append("")
    timing = point_kernel["timing"]
    lines.append("Timing via `sigma_E_vec` (vectorized C++ loop, 9 samples of 120 repeats):")
    lines.append("")
    lines.append("| Evaluator | Median time | Relative speed |")
    lines.append("|---|---:|---:|")
    orig_ns = timing["original_solver"]["median_ns"]
    approx_ns = timing["approximate_solver"]["median_ns"]
    raw_ns = timing["raw_approximate_accepted_subset"]["median_ns"]
    lines.append(f"| `ComptonKernelSolver` | {orig_ns:.0f} ns | 1.00x |")
    lines.append(
        f"| `ComptonKernelApproximateSolver` | {approx_ns:.0f} ns | {orig_ns/approx_ns:.2f}x |"
    )
    lines.append(
        f"| Raw `ComptonKernelApproximate` (accepted subset) | {raw_ns:.0f} ns | {orig_ns/raw_ns:.2f}x |"
    )
    lines.append("")

    # Multigroup single-thread
    lines.append("## Multigroup-multiangle matrices")
    lines.append("")
    lines.append(
        "Configuration: 4 groups [0.1, 1, 10, 100, 1000] keV, 4 angle bins, "
        "uniform weight, cutoff disabled."
    )
    lines.append("")
    lines.append("### Single thread")
    lines.append("")
    lines.append(
        "| kT | Existing solver | Approximate solver | Speedup | "
        "Matrix L1 error | Maximum significant-cell error | Maximum row-sum error |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for m in multigroup_1t["measurements"]:
        lines.append(
            f"| {m['T_kev']:.0f} keV | {m['original_ms']:.2f} ms | "
            f"{m['approximate_ms']:.2f} ms | {m['speedup']:.2f}x | "
            f"{format_sci(m['accuracy']['l1_relative'])} | "
            f"{format_sci(m['accuracy']['max_significant_relative'])} | "
            f"{format_sci(m['accuracy']['max_row_sum_relative'])} |"
        )
    lines.append("")

    # Multigroup 15-thread
    lines.append("### Fifteen OpenMP threads")
    lines.append("")
    lines.append(
        "| kT | Existing solver | Approximate solver | Speedup |"
    )
    lines.append("|---:|---:|---:|---:|")
    for m in multigroup_15t["measurements"]:
        lines.append(
            f"| {m['T_kev']:.0f} keV | {m['original_ms']:.2f} ms | "
            f"{m['approximate_ms']:.2f} ms | {m['speedup']:.2f}x |"
        )
    lines.append("")

    # Stress test
    lines.append("## Five-percent stress calibration")
    lines.append("")
    lines.append(
        "Four matrix families swept across 14 temperatures from 1 to 250 keV. "
        "Pass criterion: maximum significant-cell error < 5%."
    )
    lines.append("")

    overall_pass = stress["overall_pass"]
    worst_error = 0.0
    worst_scenario = ""
    worst_T = 0.0

    for scenario in stress["scenarios"]:
        lines.append(f"### {scenario['name']}")
        lines.append("")
        lines.append(
            "| kT (keV) | L1 error | Max significant-cell error | Row-sum error | Pass |"
        )
        lines.append("|---:|---:|---:|---:|:---:|")
        for t in scenario["temperatures"]:
            pass_str = "yes" if t["pass"] else "**NO**"
            lines.append(
                f"| {t['T_kev']} | {format_sci(t['l1_relative'])} | "
                f"{format_sci(t['max_significant_relative'])} | "
                f"{format_sci(t['max_row_sum_relative'])} | {pass_str} |"
            )
            if t["max_significant_relative"] > worst_error:
                worst_error = t["max_significant_relative"]
                worst_scenario = scenario["name"]
                worst_T = t["T_kev"]
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    if overall_pass:
        lines.append("**PASSED**: All significant-cell errors remain below 5%.")
    else:
        lines.append("**FAILED**: One or more significant-cell errors exceeded 5%.")
    lines.append("")
    lines.append(f"- Worst significant-cell error: {format_sci(worst_error)} "
                 f"({worst_scenario} at {worst_T} keV)")
    lines.append(f"- Point-kernel solver speedup: {point_kernel['speedup_solver']:.2f}x")
    lines.append(
        f"- Multigroup speedup (single-thread, T=100 keV): "
        f"{multigroup_1t['measurements'][2]['speedup']:.2f}x"
    )
    lines.append(
        f"- Multigroup speedup (15 threads, T=100 keV): "
        f"{multigroup_15t['measurements'][2]['speedup']:.2f}x"
    )
    lines.append("")

    report_path = os.path.join(results_dir, "VERIFICATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__))
    generate_report(results_dir)
