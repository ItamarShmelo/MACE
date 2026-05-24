"""
Plot comparing the direct quadrature kernel against CMMC Monte Carlo,
in the same style as external/CMMC/examples/compton_pomraning.py.

Usage:
    python3 tests/plot_comparison.py

Outputs saved to tests/output/
"""
import sys
import os
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'CMMC', 'cpp_modules'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from _compton_matrix_mc import ComptonMatrixMC

from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams.update({'font.size': 14})

# Constants
ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9
KEV_KELVIN = KEV / 1.380649e-16
BARN = 1e-24
MBARN = 1e-3 * BARN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

XI_EPS = 1e-10


def compute_quadrature_spectrum(engine, E_in_erg, ec_erg, eb_erg, tau,
                                report_errors=False):
    """
    Compute the scattering spectrum sigma(E') for a given E_in,
    integrating over xi for each energy bin.
    Returns S[g'] = 2*pi * integral_{-1}^{1} integral_{E_lo}^{E_hi} Sigma_E dE' dxi

    If report_errors=True, also returns array of absolute integration errors.
    """
    num_groups = len(ec_erg)
    S = np.zeros(num_groups)
    S_err = np.zeros(num_groups)

    for gp in range(num_groups):
        E_lo = eb_erg[gp]
        E_hi = eb_erg[gp + 1]

        def xi_integrand(xi):
            def Ep_integrand(Ep):
                return engine.sigma_E(E_in_erg, Ep, xi, tau, 1.0).value
            val, _ = quad(Ep_integrand, E_lo, E_hi, epsabs=1e-50, epsrel=1e-8)
            return val

        val, err = quad(xi_integrand, -1.0 + XI_EPS, 1.0 - XI_EPS,
                        epsabs=1e-50, epsrel=1e-6, limit=200)
        S[gp] = 2.0 * np.pi * val
        S_err[gp] = 2.0 * np.pi * err

    if report_errors:
        return S, S_err
    return S


def main():
    cases = [
        dict(T=1., emax=75., ein=[5., 10., 20., 40., 60.],
             ylim=[1, 1e4], name="quadrature_vs_mc_1kev_low"),
        dict(T=20., emax=140., ein=[5., 10., 20., 40., 60.],
             ylim=[1e-1, 1e3], name="quadrature_vs_mc_20kev_low"),
        dict(T=100., emax=500., ein=[30., 50., 100., 200., 300.],
             ylim=[1e-2, 1e2], name="quadrature_vs_mc_100kev"),
    ]

    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

    for case in cases:
        print(f"\n{'='*60}")
        print(f"Case: T={case['T']} keV, E_in={case['ein']} keV")
        print(f"{'='*60}")

        # Build energy grid (same style as CMMC example)
        eb_kev = np.array(sorted(list(set(
            list(np.linspace(0.01, case["emax"] / 10, 50)) +
            list(np.linspace(0.01, case["emax"], 200)) +
            list(np.geomspace(0.01, case["emax"], 50))
        ))))

        eb_erg = eb_kev * KEV
        ec_erg = 0.5 * (eb_erg[:-1] + eb_erg[1:])
        ec_kev = ec_erg / KEV
        ewid_kev = np.diff(eb_kev)

        T_kelvin = case["T"] * KEV_KELVIN
        tau = case["T"] * KEV / ME_C2

        # CMMC Monte Carlo
        print("  Computing CMMC Monte Carlo...")
        mc = ComptonMatrixMC(
            energy_groups_centers=ec_erg.tolist(),
            energy_groups_boundaries=eb_erg.tolist(),
            num_of_samples=int(2e5),
            force_detailed_balance=True,
            seed=0,
        )
        S_mc = np.array(mc.calculate_S_matrix(temperature=T_kelvin))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))

        mc_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(case["ein"])))
        quad_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(case["ein"])))

        for idx, e0_kev in enumerate(case["ein"]):
            e0_erg = e0_kev * KEV
            g = np.argmin(np.abs(ec_erg - e0_erg))
            actual_ein_kev = ec_kev[g]

            # MC result
            mc_spectrum = S_mc[g, :] / ewid_kev / MBARN
            ax.stairs(edges=eb_kev, values=mc_spectrum,
                      color=mc_colors[idx], linewidth=1.2, linestyle='-',
                      label=f"MC  $E_{{in}}$={actual_ein_kev:.1f} keV")

            # Quadrature result
            print(f"  Computing quadrature for E_in={actual_ein_kev:.1f} keV...")
            S_quad, S_err = compute_quadrature_spectrum(
                engine, ec_erg[g], ec_erg, eb_erg, tau, report_errors=True)
            quad_spectrum = S_quad / ewid_kev / MBARN
            ax.plot(ec_kev, quad_spectrum,
                    color=quad_colors[idx], linewidth=2.0, linestyle='--',
                    label=f"Quad $E_{{in}}$={actual_ein_kev:.1f} keV")

            # Print integration error report
            print(f"    Integration errors for E_in={actual_ein_kev:.1f} keV:")
            print(f"    {'Bin center [keV]':>18s}  {'S [cm^2]':>12s}  "
                  f"{'Abs err [cm^2]':>14s}  {'Rel err':>10s}")
            print(f"    {'-'*60}")
            max_S = np.max(np.abs(S_quad))
            for gp in range(len(ec_kev)):
                if np.abs(S_quad[gp]) < 1e-6 * max_S:
                    continue
                rel = S_err[gp] / np.abs(S_quad[gp]) if S_quad[gp] != 0 else 0.0
                print(f"    {ec_kev[gp]:18.2f}  {S_quad[gp]:12.4e}  "
                      f"{S_err[gp]:14.4e}  {rel:10.2e}")
            print()

        ax.set_yscale("log")
        ax.set_ylim(case["ylim"])
        ax.set_xlim([0., eb_kev[-1]])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Compton Scattering Kernel — T = {case['T']} keV\n"
                     f"(solid = MC, dashed = quadrature)")
        ax.set_xlabel("Final photon energy $E'$ [keV]")
        ax.set_ylabel(r"$\sigma(E')$ [mbarn/keV]")
        ax.legend(loc="best", fontsize=9, ncol=2)

        outpath = os.path.join(OUTPUT_DIR, f"{case['name']}.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=150)
        print(f"  Saved: {outpath}")

        outpath_pdf = os.path.join(OUTPUT_DIR, f"{case['name']}.pdf")
        fig.savefig(outpath_pdf)
        print(f"  Saved: {outpath_pdf}")

        plt.close(fig)

    print("\nDone! All plots saved to tests/output/")


if __name__ == "__main__":
    main()
