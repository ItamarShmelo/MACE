"""Compare deterministic and Monte Carlo multigroup Compton matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix._units import kev, kev_kelvin, sigma_thomson


OUTPUT_DIR = Path(__file__).with_name("output")


def main() -> None:
    boundaries_kev = np.array([1.0, 5.0, 20.0, 100.0])
    boundaries = (boundaries_kev * kev).tolist()
    temperature = 10.0 * kev_kelvin

    weight = cm.PlanckWeightFunction(
        cap_x=25.0,
        group_boundaries=boundaries,
    )
    deterministic = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries,
        weight_function=weight,
        config=cm.MGIntegrationConfig(cutoff_ratio=None),
    )
    monte_carlo = cm.ComptonMonteCarloKernel(
        energy_group_boundaries=boundaries,
        weight_function=weight,
        config=cm.MCIntegrationConfig(
            num_samples=500_000,
            seed=42,
            discard_out_of_grid=True,
        ),
    )

    sigma_deterministic = deterministic.compute_sigma_matrix(
        cds.ComptonKernelSolver(),
        T=temperature,
    )
    sigma_monte_carlo = monte_carlo.compute_sigma_matrix(T=temperature)

    total_relative_difference = np.sum(
        np.abs(sigma_monte_carlo - sigma_deterministic)
    ) / np.sum(np.abs(sigma_deterministic))

    matrices = [sigma_deterministic, sigma_monte_carlo]
    titles = ["Deterministic", "Monte Carlo"]
    log_values = [
        np.log10(np.maximum(matrix / sigma_thomson, 1e-300))
        for matrix in matrices
    ]
    color_min = min(float(np.min(values)) for values in log_values)
    color_max = max(float(np.max(values)) for values in log_values)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for axis, values, title in zip(axes, log_values, titles, strict=True):
        image = axis.imshow(
            values,
            origin="lower",
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_title(title)
        axis.set_xlabel("Outgoing group")
    axes[0].set_ylabel("Incoming group")
    fig.colorbar(image, ax=axes, label=r"$\log_{10}(\sigma/\sigma_T)$")
    fig.suptitle(
        f"L1 relative difference = {total_relative_difference:.3%}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "monte_carlo_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"L1 relative difference: {total_relative_difference:.3%}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
