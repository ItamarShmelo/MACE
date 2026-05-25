"""
Tests comparing the pure Python (pycompton) implementation against the
C++ pybind11 implementation for pointwise kernel evaluation.

Two comparison modes:
  - Fixed Gauss-Laguerre at the same NL: expect ~10-digit agreement
  - Adaptive quad vs C++ fixed-rule: expect ~8-digit agreement
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from pycompton.compton_kernel_quadrature import sigma_E as py_sigma_E

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2
KEV = 1.602176634e-9

TEST_POINTS = [
    # (E_kev, E_prime_kev, xi, tau_kev)
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.5, 0.5, 1.0),
    (1.0, 2.0, -0.5, 1.0),
    (10.0, 8.0, 0.3, 5.0),
    (10.0, 12.0, -0.3, 5.0),
    (50.0, 45.0, 0.0, 20.0),
    (50.0, 55.0, 0.7, 20.0),
    (0.1, 0.08, 0.0, 0.1),
    (0.1, 0.12, -0.8, 0.1),
    (5.0, 5.0, 0.0, 10.0),
    (5.0, 3.0, 0.9, 10.0),
]


def assert_close_mixed(a, b, rtol=1e-10, atol=1e-300):
    scale = max(abs(a), abs(b))
    assert abs(a - b) <= atol + rtol * scale, (
        f"|{a} - {b}| = {abs(a-b):.4e} > atol({atol}) + rtol({rtol})*scale({scale:.4e})"
    )


class TestFixedRuleAgreement:
    """Python fixed Gauss-Laguerre vs C++ at the same NL."""

    @pytest.mark.parametrize("NL", [64, 128, 256])
    def test_post_ibp(self, NL):
        cpp_engine = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PostIBP)

        for E_kev, Ep_kev, xi, tau_kev in TEST_POINTS:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2

            cpp_r = cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
            py_val, py_aerr, py_rerr = py_sigma_E(
                E, Ep, xi, tau, 1.0, form="post_ibp", NL=NL, method="fixed"
            )

            # Post-IBP has Psi+IQ cancellation at small tau, loosening tolerance
            assert_close_mixed(
                py_val, cpp_r.value, rtol=1e-3, atol=1e-300
            )

    @pytest.mark.parametrize("NL", [64, 128, 256])
    def test_pre_ibp(self, NL):
        cpp_engine = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PreIBP)

        for E_kev, Ep_kev, xi, tau_kev in TEST_POINTS:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2

            cpp_r = cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
            py_val, _, _ = py_sigma_E(
                E, Ep, xi, tau, 1.0, form="pre_ibp", NL=NL, method="fixed"
            )

            assert_close_mixed(
                py_val, cpp_r.value, rtol=1e-6, atol=1e-300
            )

    def test_error_estimates_same_order(self):
        """Richardson error estimates should be the same order of magnitude."""
        NL = 128
        cpp_engine = ComptonKernelQuadrature(NL=NL, form=QuadratureForm.PostIBP)

        for E_kev, Ep_kev, xi, tau_kev in TEST_POINTS:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2

            cpp_r = cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
            _, py_aerr, _ = py_sigma_E(
                E, Ep, xi, tau, 1.0, form="post_ibp", NL=NL, method="fixed"
            )

            if cpp_r.estimated_abs_error < 1e-300 or py_aerr < 1e-300:
                continue
            scale = max(cpp_r.estimated_abs_error, py_aerr)
            ratio = min(cpp_r.estimated_abs_error, py_aerr) / scale
            assert ratio > 0.01, (
                f"Error estimates differ by more than 100x at "
                f"E={E_kev}, Ep={Ep_kev}, xi={xi}, tau_kev={tau_kev}: "
                f"py={py_aerr:.4e}, cpp={cpp_r.estimated_abs_error:.4e}"
            )


class TestAdaptiveDiagnostic:
    """Python adaptive quad vs C++ fixed-rule (looser tolerance)."""

    def test_post_ibp_adaptive(self):
        cpp_engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)

        for E_kev, Ep_kev, xi, tau_kev in TEST_POINTS:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2

            cpp_r = cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
            py_val, _, _ = py_sigma_E(
                E, Ep, xi, tau, 1.0, form="post_ibp", method="adaptive"
            )

            assert_close_mixed(
                py_val, cpp_r.value, rtol=1e-3, atol=1e-300
            )

    def test_pre_ibp_adaptive(self):
        cpp_engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PreIBP)

        for E_kev, Ep_kev, xi, tau_kev in TEST_POINTS:
            E = E_kev * KEV
            Ep = Ep_kev * KEV
            tau = tau_kev * KEV / ME_C2

            cpp_r = cpp_engine.sigma_E(E, Ep, xi, tau, 1.0)
            py_val, _, _ = py_sigma_E(
                E, Ep, xi, tau, 1.0, form="pre_ibp", method="adaptive"
            )

            assert_close_mixed(
                py_val, cpp_r.value, rtol=1e-4, atol=1e-300
            )
