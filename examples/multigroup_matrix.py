"""Compute deterministic multigroup, multiangle Compton matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix._units import kev, kev_kelvin, sigma_thomson


OUTPUT_DIR = Path(__file__).with_name("output")


def main() -> None:
    boundaries_kev = np.array([0.5, 2.0, 10.0, 50.0, 200.0])
    boundaries = (boundaries_kev * kev).tolist()
    temperature = 10.0 * kev_kelvin

    kernel = cds.ComptonKernelSolver()
    weight = cm.PlanckWeightFunction(
        cap_x=25.0,
        group_boundaries=boundaries,
    )
    multigroup = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries,
        weight_function=weight,
    )

    angle_resolved = multigroup.compute_sigma_matrix(
        kernel,
        num_angle_bins=4,
        T=temperature,
    )
    sigma = angle_resolved.sum(axis=2)
    dsigma_dT = multigroup.compute_dsigma_dT_matrix(kernel, T=temperature)

    scaled_sigma = np.maximum(sigma / sigma_thomson, 1e-300)
    scaled_derivative = temperature * dsigma_dT / sigma_thomson
    derivative_limit = max(float(np.max(np.abs(scaled_derivative))), 1e-15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    image_sigma = axes[0].imshow(np.log10(scaled_sigma), origin="lower")
    axes[0].set_title(r"$\log_{10}(\sigma_{g\to g'}/\sigma_T)$")
    axes[0].set_xlabel("Outgoing group")
    axes[0].set_ylabel("Incoming group")
    fig.colorbar(image_sigma, ax=axes[0])

    image_derivative = axes[1].imshow(
        scaled_derivative,
        origin="lower",
        cmap="coolwarm",
        vmin=-derivative_limit,
        vmax=derivative_limit,
    )
    axes[1].set_title(r"$T(d\sigma/dT)/\sigma_T$")
    axes[1].set_xlabel("Outgoing group")
    axes[1].set_ylabel("Incoming group")
    fig.colorbar(image_derivative, ax=axes[1])
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = OUTPUT_DIR / "multigroup_matrix.png"
    data_path = OUTPUT_DIR / "multigroup_matrix.npz"
    fig.savefig(plot_path, dpi=180)
    np.savez(
        data_path,
        boundaries_kev=boundaries_kev,
        sigma=sigma,
        angle_resolved=angle_resolved,
        dsigma_dT=dsigma_dT,
    )

    print(f"Angle-resolved matrix shape: {angle_resolved.shape}")
    print(f"Saved {plot_path}")
    print(f"Saved {data_path}")


if __name__ == "__main__":
    main()
