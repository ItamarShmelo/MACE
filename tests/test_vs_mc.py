"""
Slow tests comparing the quadrature implementation against CMMC Monte Carlo.
Run with: pytest tests/test_vs_mc.py --run-slow
"""
import sys
import os
import numpy as np
import pytest
from scipy.integrate import dblquad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'CMMC', 'cpp_modules'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9
KEV_KELVIN = KEV / 1.380649e-16
SIGMA_T = 6.652458732160e-25

XI_EPS = 1e-10


def angular_factor(mode: str) -> float:
    if mode == "average":
        return 0.5
    if mode == "moment0":
        return 2.0 * np.pi
    raise ValueError(f"Unknown angular normalization mode: {mode}")


def integrate_bin(engine, E_in, E_lo, E_hi, tau, angular_norm_mode="moment0"):
    C_omega = angular_factor(angular_norm_mode)

    def integrand(Ep, xi):
        return engine.sigma_E(E_in, Ep, xi, tau, 1.0).value

    val, err = dblquad(
        integrand,
        -1.0 + XI_EPS, 1.0 - XI_EPS,
        lambda xi: E_lo, lambda xi: E_hi,
        epsabs=1e-35, epsrel=1e-2
    )
    return C_omega * val, C_omega * err


def get_cmmc_matrix(T_kev, eb_erg, num_samples=500000):
    """Generate reference S-matrix from CMMC."""
    try:
        from _compton_matrix_mc import ComptonMatrixMC
    except ImportError:
        pytest.skip("CMMC _compton_matrix_mc not importable")

    T_kelvin = T_kev * KEV_KELVIN
    ec = 0.5 * (eb_erg[:-1] + eb_erg[1:])

    compton_engine = ComptonMatrixMC(
        energy_groups_centers=ec.tolist(),
        energy_groups_boundaries=eb_erg.tolist(),
        num_of_samples=num_samples,
        force_detailed_balance=False,
        seed=42,
    )
    S_mat = np.array(compton_engine.calculate_S_matrix(temperature=T_kelvin))
    return S_mat


def get_quad_matrix(T_kev, eb_erg, angular_norm_mode="moment0"):
    """Generate S-matrix from quadrature."""
    engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

    tau = T_kev * KEV / ME_C2
    num_groups = len(eb_erg) - 1
    ec = 0.5 * (eb_erg[:-1] + eb_erg[1:])

    S_quad = np.zeros((num_groups, num_groups))

    for g in range(num_groups):
        E_in = ec[g]
        for gp in range(num_groups):
            val, err = integrate_bin(
                engine, E_in,
                eb_erg[gp], eb_erg[gp + 1],
                tau, angular_norm_mode
            )
            S_quad[g, gp] = val

    return S_quad


def _assert_matrix_agreement(S_mc, S_quad, num_groups, rtol=0.5,
                              skip_small_frac=1e-3, label=""):
    """Compare two S-matrices element-wise with tolerance."""
    for g in range(num_groups):
        for gp in range(num_groups):
            mc_val = S_mc[g, gp]
            quad_val = S_quad[g, gp]

            if abs(mc_val) < 1e-30 and abs(quad_val) < 1e-30:
                continue

            diag_scale = max(S_mc[g, g], S_quad[g, g])
            if max(abs(mc_val), abs(quad_val)) < skip_small_frac * diag_scale:
                continue

            scale = max(abs(mc_val), abs(quad_val))
            rel_diff = abs(mc_val - quad_val) / scale

            assert rel_diff < rtol, (
                f"{label}MC comparison failed at g={g}, gp={gp}: "
                f"mc={mc_val:.4e}, quad={quad_val:.4e}, "
                f"rel_diff={rel_diff:.4f}"
            )


@pytest.mark.slow
class TestVsMonteCarlo:
    """Compare quadrature S-matrix against CMMC Monte Carlo results."""

    def test_100kev(self):
        """
        Compare at T=100 keV (tau~0.2) where off-diagonal scattering is
        significant and both methods are well-resolved.
        """
        T_kev = 100.0
        eb_kev = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=1000000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        # All elements should agree within 50% (MC noise + integration error)
        num_groups = len(eb_kev) - 1
        for g in range(num_groups):
            for gp in range(num_groups):
                mc_val = S_mc[g, gp]
                quad_val = S_quad[g, gp]

                if abs(mc_val) < 1e-30 and abs(quad_val) < 1e-30:
                    continue

                scale = max(abs(mc_val), abs(quad_val))
                rel_diff = abs(mc_val - quad_val) / scale

                assert rel_diff < 0.5, (
                    f"MC comparison failed at g={g}, gp={gp}: "
                    f"mc={mc_val:.4e}, quad={quad_val:.4e}, "
                    f"rel_diff={rel_diff:.4f}"
                )

    def test_20kev(self):
        """
        Compare at T=20 keV (tau~0.04). Off-diagonal still significant
        for neighboring groups.
        """
        T_kev = 20.0
        eb_kev = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=1000000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        num_groups = len(eb_kev) - 1
        for g in range(num_groups):
            for gp in range(num_groups):
                mc_val = S_mc[g, gp]
                quad_val = S_quad[g, gp]

                if abs(mc_val) < 1e-30 and abs(quad_val) < 1e-30:
                    continue

                # Skip elements that are small relative to diagonal
                # (MC noise dominates for these elements)
                diag_scale = max(S_mc[g, g], S_quad[g, g])
                if max(abs(mc_val), abs(quad_val)) < 1e-3 * diag_scale:
                    continue

                scale = max(abs(mc_val), abs(quad_val))
                rel_diff = abs(mc_val - quad_val) / scale

                assert rel_diff < 0.5, (
                    f"MC comparison failed at g={g}, gp={gp}: "
                    f"mc={mc_val:.4e}, quad={quad_val:.4e}, "
                    f"rel_diff={rel_diff:.4f}"
                )

    def test_diagonal_1kev(self):
        """
        At T=1 keV, verify that diagonal elements (elastic-like scattering)
        agree well, even though off-diagonal is exponentially suppressed.
        """
        T_kev = 1.0
        eb_kev = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=500000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        num_groups = len(eb_kev) - 1
        for g in range(num_groups):
            mc_val = S_mc[g, g]
            quad_val = S_quad[g, g]
            rel_diff = abs(mc_val - quad_val) / max(abs(mc_val), abs(quad_val))

            assert rel_diff < 0.05, (
                f"Diagonal comparison failed at g={g}: "
                f"mc={mc_val:.4e}, quad={quad_val:.4e}, "
                f"rel_diff={rel_diff:.4f}"
            )

    def test_pomraning_1kev_low(self):
        """Pomraning T=1 keV, low-energy incoming photons."""
        T_kev = 1.0
        eb_kev = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 75.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=1000000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        num_groups = len(eb_kev) - 1
        _assert_matrix_agreement(S_mc, S_quad, num_groups, rtol=0.7,
                                  skip_small_frac=0.1,
                                  label="Pomraning 1keV low: ")

    def test_pomraning_1kev_high(self):
        """Pomraning T=1 keV, high-energy incoming photons.

        At T=1 keV the scattering kernel is very narrow, so off-diagonal
        bins in this coarse grid are dominated by MC noise.  We use a
        generous skip threshold to test only the dynamically significant
        matrix elements.
        """
        T_kev = 1.0
        eb_kev = np.array([10.0, 40.0, 80.0, 120.0, 200.0, 300.0, 340.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=1000000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        num_groups = len(eb_kev) - 1
        _assert_matrix_agreement(S_mc, S_quad, num_groups, rtol=0.7,
                                  skip_small_frac=0.25,
                                  label="Pomraning 1keV high: ")

    def test_pomraning_20kev_low(self):
        """Pomraning T=20 keV, low-energy incoming photons."""
        T_kev = 20.0
        eb_kev = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 140.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=1000000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        num_groups = len(eb_kev) - 1
        _assert_matrix_agreement(S_mc, S_quad, num_groups, rtol=0.7,
                                  skip_small_frac=0.1,
                                  label="Pomraning 20keV low: ")

    def test_pomraning_20kev_high(self):
        """Pomraning T=20 keV, high-energy incoming photons."""
        T_kev = 20.0
        eb_kev = np.array([10.0, 40.0, 80.0, 120.0, 200.0, 300.0, 440.0])
        eb_erg = eb_kev * KEV

        S_mc = get_cmmc_matrix(T_kev, eb_erg, num_samples=1000000)
        S_quad = get_quad_matrix(T_kev, eb_erg, "moment0")

        num_groups = len(eb_kev) - 1
        _assert_matrix_agreement(S_mc, S_quad, num_groups, rtol=0.7,
                                  skip_small_frac=0.1,
                                  label="Pomraning 20keV high: ")
