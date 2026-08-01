"""Plot a pointwise thermal Compton kernel and its temperature derivative."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compton_matrix._compton_differential_cross_section as cds
from compton_matrix._units import kev, kev_kelvin


OUTPUT_DIR = Path(__file__).with_name("output")


def main() -> None:
    incident_energy_kev = 10.0
    temperature_kev = 10.0
    xi = 0.0

    scattered_energy_kev = np.geomspace(0.5, 100.0, 300)
    scattered_energy = scattered_energy_kev * kev

    kernel = cds.ComptonKernelSolver()
    values, relative_errors = kernel.sigma_E_vec(
        incident_energy_kev * kev,
        scattered_energy,
        xi,
        temperature_kev * kev_kelvin,
    )
    derivatives, _ = kernel.dsigma_E_dT_vec(
        incident_energy_kev * kev,
        scattered_energy,
        xi,
        temperature_kev * kev_kelvin,
    )

    value_scale = np.max(np.abs(values))
    derivative_scale = np.max(np.abs(derivatives))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(scattered_energy_kev, values / value_scale)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized kernel")
    axes[0].grid(alpha=0.25)

    axes[1].plot(scattered_energy_kev, derivatives / derivative_scale)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Scattered photon energy [keV]")
    axes[1].set_ylabel("Normalized temperature derivative")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"E = {incident_energy_kev:g} keV, "
        f"kBT = {temperature_kev:g} keV, xi = {xi:g}"
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "point_kernel.png"
    fig.savefig(output_path, dpi=180)

    peak_index = int(np.argmax(values))
    print(f"Peak scattered energy: {scattered_energy_kev[peak_index]:.4g} keV")
    print(f"Largest estimated relative error: {np.max(relative_errors):.3e}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
