"""
Tests for ComptonKernelSeries(Auto): accuracy, edge cases, domain validation,
and physical consistency.
"""

import sys
import os
import math
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

from _compton_kernel_series import ComptonKernelSeries, SeriesMethod
from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm

ME_C2 = 9.109383713928e-28 * (2.99792458e10)**2
KEV = 1.602176634e-9
K_BOLTZ = 1.380649e-16
KEV_KELVIN = KEV / K_BOLTZ


@pytest.fixture
def solver():
    return ComptonKernelSeries(SeriesMethod.Auto)


@pytest.fixture
def quad256():
    return ComptonKernelQuadrature(256, QuadratureForm.PostIBP)


# ═══════════════════════════════════════════════════════════════════════════════
# Domain validation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainValidation:
    def test_E_zero(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(0.0, 1.0 * KEV, 0.0, 1e6, 1.0)

    def test_E_negative(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(-1.0 * KEV, 1.0 * KEV, 0.0, 1e6, 1.0)

    def test_Eprime_zero(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(1.0 * KEV, 0.0, 0.0, 1e6, 1.0)

    def test_T_zero(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(1.0 * KEV, 1.5 * KEV, 0.0, 0.0, 1.0)

    def test_T_negative(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(1.0 * KEV, 1.5 * KEV, 0.0, -1e6, 1.0)

    def test_xi_plus_one(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(1.0 * KEV, 1.5 * KEV, 1.0, 1e6, 1.0)

    def test_xi_minus_one(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(1.0 * KEV, 1.5 * KEV, -1.0, 1e6, 1.0)

    def test_xi_beyond_plus_one(self, solver):
        with pytest.raises(Exception):
            solver.sigma_E(1.0 * KEV, 1.5 * KEV, 1.1, 1e6, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Accuracy tests: solver results vs quadrature reference
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccuracy:
    ACCURACY_CASES = [
        (1.0, 1.5, 0.0, 0.01),
        (1.0, 1.5, 0.0, 1.0),
        (10.0, 15.0, 0.0, 5.0),
        (10.0, 10.5, 0.0, 20.0),
        (100.0, 150.0, 0.0, 50.0),
        (100.0, 101.0, 0.0, 100.0),
        (50.0, 55.0, 0.3, 20.0),
        (1.0, 2.0, -0.5, 5.0),
    ]

    def test_solver_vs_quadrature(self, solver, quad256):
        """Solver should agree with Q256 to within solver's reported error or 1e-6."""
        for E_keV, Ep_keV, xi, T_keV in self.ACCURACY_CASES:
            E = E_keV * KEV; Ep = Ep_keV * KEV; T = T_keV * KEV_KELVIN
            sr = solver.sigma_E(E, Ep, xi, T, 1.0)
            qr = quad256.sigma_E(E, Ep, xi, T, 1.0)

            if abs(qr.value) < 1e-300:
                continue

            rel_diff = abs(sr.value - qr.value) / abs(qr.value)
            tolerance = max(sr.estimated_rel_error, qr.estimated_rel_error, 1e-6)
            assert rel_diff < tolerance, (
                f"E={E_keV}, Ep={Ep_keV}, xi={xi}, T={T_keV}: "
                f"rel_diff={rel_diff:.2e}, tol={tolerance:.2e}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Edge-case tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_xi_near_plus_one(self, solver):
        """xi very close to +1 should not crash."""
        E = 1.0 * KEV; Ep = 1.5 * KEV; T = 5.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.999, T, 1.0)
        assert math.isfinite(r.value)

    def test_xi_near_minus_one(self, solver):
        """xi very close to -1 should not crash."""
        E = 1.0 * KEV; Ep = 1.5 * KEV; T = 5.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, -0.999, T, 1.0)
        assert math.isfinite(r.value)

    def test_near_elastic(self, solver):
        """E'/E very close to 1 should produce valid result."""
        E = 10.0 * KEV; Ep = 10.001 * KEV; T = 5.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.0, T, 1.0)
        assert math.isfinite(r.value)
        assert r.value >= 0.0

    def test_strongly_separated_energies(self, solver):
        """E'/E = 10 should work."""
        E = 1.0 * KEV; Ep = 10.0 * KEV; T = 50.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.0, T, 1.0)
        assert math.isfinite(r.value)
        assert r.value >= 0.0

    def test_very_small_T(self, solver):
        """Very cold electrons (deep asymptotic regime)."""
        E = 10.0 * KEV; Ep = 10.5 * KEV; T = 0.001 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.0, T, 1.0)
        assert math.isfinite(r.value)

    def test_high_T(self, solver):
        """Hot electrons (power series or quadrature regime)."""
        E = 100.0 * KEV; Ep = 120.0 * KEV; T = 500.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.0, T, 1.0)
        assert math.isfinite(r.value)
        assert r.value >= 0.0

    def test_sigma0_underflow(self, solver):
        """Large energy transfer at low T: sigma0 underflows, kernel is negligible."""
        E = 1.0 * KEV; Ep = 100.0 * KEV; T = 0.01 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.0, T, 1.0)
        assert r.value == 0.0 or abs(r.value) < 1e-100


# ═══════════════════════════════════════════════════════════════════════════════
# Out-of-calibration-domain tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutOfDomain:
    def test_very_high_T_raises(self, solver):
        """T well beyond series convergence domain raises RuntimeError."""
        E = 1.0 * KEV; Ep = 1.5 * KEV; T = 2000.0 * KEV_KELVIN
        with pytest.raises(RuntimeError):
            solver.sigma_E(E, Ep, 0.0, T, 1.0)

    def test_extreme_energy_ratio(self, solver):
        """E'/E = 100 beyond calibration grid."""
        E = 1.0 * KEV; Ep = 100.0 * KEV; T = 200.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.0, T, 1.0)
        assert math.isfinite(r.value)

    def test_xi_0999_works(self, solver):
        """xi=0.999 beyond calibration grid (calibrated to 0.9)."""
        E = 10.0 * KEV; Ep = 15.0 * KEV; T = 10.0 * KEV_KELVIN
        r = solver.sigma_E(E, Ep, 0.999, T, 1.0)
        assert math.isfinite(r.value)


# ═══════════════════════════════════════════════════════════════════════════════
# Physical consistency tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhysicalConsistency:
    def test_non_negativity(self, solver):
        """Kernel should be non-negative everywhere."""
        cases = [
            (1.0, 1.5, 0.0, 1.0),
            (1.0, 1.01, 0.0, 10.0),
            (10.0, 10.5, 0.5, 20.0),
            (100.0, 101.0, 0.0, 100.0),
            (1.0, 2.0, -0.5, 50.0),
            (50.0, 55.0, 0.3, 5.0),
        ]
        for E_keV, Ep_keV, xi, T_keV in cases:
            E = E_keV * KEV; Ep = Ep_keV * KEV; T = T_keV * KEV_KELVIN
            r = solver.sigma_E(E, Ep, xi, T, 1.0)
            assert r.value >= 0.0, (
                f"Negative value at E={E_keV}, Ep={Ep_keV}, xi={xi}, T={T_keV}: "
                f"value={r.value:.2e}"
            )

    def test_quadrature_convergence(self, solver):
        """Cross-check: Q256 vs Q128 should show convergence for well-behaved cases."""
        q128 = ComptonKernelQuadrature(128, QuadratureForm.PostIBP)
        q256 = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)

        E = 10.0 * KEV; Ep = 15.0 * KEV; xi = 0.0; T = 20.0 * KEV_KELVIN
        r128 = q128.sigma_E(E, Ep, xi, T, 1.0)
        r256 = q256.sigma_E(E, Ep, xi, T, 1.0)

        if abs(r256.value) > 1e-300:
            disc = abs(r256.value - r128.value) / abs(r256.value)
            assert disc < 1e-6, f"Q128 vs Q256 discrepancy: {disc:.2e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized API test
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorized:
    def test_sigma_E_vec(self, solver):
        """Vectorized API should produce same results as scalar."""
        E = 10.0 * KEV
        Ep_arr = np.array([10.5, 11.0, 12.0, 15.0, 20.0]) * KEV
        xi = 0.0; T = 10.0 * KEV_KELVIN

        values, errors = solver.sigma_E_vec(E, Ep_arr, xi, T, 1.0)

        for i, Ep in enumerate(Ep_arr):
            r = solver.sigma_E(E, float(Ep), xi, T, 1.0)
            assert abs(values[i] - r.value) < 1e-300 + 1e-15 * abs(r.value)
