"""
Tests for the temperature derivative ∂Σ_E/∂τ of the Compton kernel.

Validates the analytic derivative implementation against:
  - Richardson-extrapolated finite differences
  - Pre-IBP vs Post-IBP consistency
  - Small-τ numerical stability
  - Bessel helper accuracy (scaled_K1, kappa_ratio)
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
from _compton_common import SigmaResult
from _compton_kernel_quadrature import (
    ComptonKernelQuadrature, QuadratureForm,
    scaled_K1, scaled_K2, kappa_ratio,
)

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2  # erg
KEV = 1.602176634e-9  # erg


def _to_erg(E_kev):
    return E_kev * KEV


NICE_POINTS = [
    # (E_kev, E_prime_kev, xi, tau_kev)
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.5, 0.5, 1.0),
    (1.0, 2.0, -0.5, 1.0),
    (10.0, 8.0, 0.3, 5.0),
    (50.0, 45.0, 0.0, 20.0),
    (5.0, 5.0, 0.0, 10.0),
    (5.0, 3.0, 0.9, 10.0),
]


class TestFiniteDifference:
    """Compare analytic dsigma_E_dtau against Richardson-extrapolated FD."""

    def _fd_richardson(self, engine, E, Ep, xi, tau, Ne, h):
        """Centered FD at h and h/2, Richardson-extrapolated to O(h^4)."""
        def fd(step):
            vp = engine.sigma_E(E, Ep, xi, tau + step, Ne).value
            vm = engine.sigma_E(E, Ep, xi, tau - step, Ne).value
            return (vp - vm) / (2.0 * step)

        fd_h = fd(h)
        fd_h2 = fd(h / 2.0)
        return (4.0 * fd_h2 - fd_h) / 3.0

    def _run_fd_test(self, form, tol):
        engine = ComptonKernelQuadrature(256, form)
        passed = 0

        for E_kev, Ep_kev, xi, tau_kev in NICE_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            tau = tau_kev * KEV / ME_C2

            sig = engine.sigma_E(E, Ep, xi, tau, 1.0)
            if sig.estimated_rel_error > 1e-6:
                continue

            h = 1e-4 * tau
            fd_rich = self._fd_richardson(engine, E, Ep, xi, tau, 1.0, h)
            analytic = engine.dsigma_E_dtau(E, Ep, xi, tau, 1.0)

            if abs(fd_rich) < 1e-300:
                continue

            rel_err = abs(analytic.value - fd_rich) / abs(fd_rich)
            assert rel_err < tol, (
                f"FD mismatch at E={E_kev}, Ep={Ep_kev}, xi={xi}, tau={tau_kev}: "
                f"analytic={analytic.value}, fd_rich={fd_rich}, rel_err={rel_err}"
            )
            passed += 1

        assert passed >= 4, (
            f"Insufficient converged points: {passed}/7 (need >=4)"
        )

    def test_pre_ibp(self):
        self._run_fd_test(QuadratureForm.PreIBP, tol=1e-3)

    def test_post_ibp(self):
        self._run_fd_test(QuadratureForm.PostIBP, tol=1e-2)

    def test_error_estimate_quality(self):
        """Richardson error estimate should be roughly consistent with actual FD error."""
        engine = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)
        checked = 0

        for E_kev, Ep_kev, xi, tau_kev in NICE_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            tau = tau_kev * KEV / ME_C2

            sig = engine.sigma_E(E, Ep, xi, tau, 1.0)
            if sig.estimated_rel_error > 1e-6:
                continue

            h = 1e-4 * tau
            fd_rich = self._fd_richardson(engine, E, Ep, xi, tau, 1.0, h)
            analytic = engine.dsigma_E_dtau(E, Ep, xi, tau, 1.0)

            if abs(analytic.value) < 1e-300:
                continue

            actual_rel = abs(analytic.value - fd_rich) / abs(analytic.value)
            if actual_rel < 1e-12:
                continue

            assert analytic.estimated_rel_error < 100.0 * actual_rel, (
                f"Richardson estimate wildly off at E={E_kev}, Ep={Ep_kev}: "
                f"reported={analytic.estimated_rel_error}, actual~={actual_rel}"
            )
            checked += 1

        assert checked >= 2, f"Only {checked} points checked for error quality"


class TestPrePostIBPConsistency:
    """Verify pre-IBP and post-IBP derivatives agree."""

    def test_agreement(self):
        engine_pre = ComptonKernelQuadrature(256, QuadratureForm.PreIBP)
        engine_post = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)

        for E_kev, Ep_kev, xi, tau_kev in NICE_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            tau = tau_kev * KEV / ME_C2

            r_pre = engine_pre.dsigma_E_dtau(E, Ep, xi, tau, 1.0)
            r_post = engine_post.dsigma_E_dtau(E, Ep, xi, tau, 1.0)

            scale = max(abs(r_pre.value), abs(r_post.value))
            if scale < 1e-300:
                continue

            rel_diff = abs(r_pre.value - r_post.value) / scale
            assert rel_diff < 1e-4, (
                f"Pre/post IBP mismatch at E={E_kev}, Ep={Ep_kev}, xi={xi}, "
                f"tau={tau_kev}: pre={r_pre.value}, post={r_post.value}, "
                f"rel_diff={rel_diff}"
            )


class TestSmallTauStability:
    """Verify dsigma_E_dtau returns finite values at small temperatures."""

    @pytest.mark.parametrize("tau_kev", [0.01, 0.1, 1.0])
    def test_finite(self, tau_kev):
        engine = ComptonKernelQuadrature(128, QuadratureForm.PreIBP)
        E = _to_erg(1.0)
        Ep = _to_erg(1.0)
        tau = tau_kev * KEV / ME_C2

        r = engine.dsigma_E_dtau(E, Ep, 0.0, tau, 1.0)
        assert np.isfinite(r.value), f"Non-finite at tau_kev={tau_kev}: {r.value}"
        assert np.isfinite(r.estimated_abs_error)
        assert np.isfinite(r.estimated_rel_error)


class TestKappaBessel:
    """Unit tests for scaled_K1 and kappa_ratio."""

    def test_scaled_K1_vs_scipy(self):
        from scipy.special import kve
        x_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 49.0]
        for x in x_values:
            ref = kve(1, x)
            val = scaled_K1(x)
            rel = abs(val - ref) / abs(ref)
            assert rel < 1e-10, f"scaled_K1 mismatch at x={x}: rel={rel}"

    def test_scaled_K1_asymptotic(self):
        from scipy.special import kve
        x_values = [50.0, 100.0, 200.0, 500.0, 1000.0]
        for x in x_values:
            ref = kve(1, x)
            val = scaled_K1(x)
            rel = abs(val - ref) / abs(ref)
            assert rel < 1e-8, f"scaled_K1 asymptotic mismatch at x={x}: rel={rel}"

    def test_scaled_K1_invalid(self):
        with pytest.raises(Exception):
            scaled_K1(0.0)
        with pytest.raises(Exception):
            scaled_K1(-1.0)
        with pytest.raises(Exception):
            scaled_K1(float('inf'))

    def test_kappa_cold_limit(self):
        """kappa -> 1 as tau -> 0 (x = 1/tau -> inf)."""
        for tau in [1e-4, 1e-3, 1e-2]:
            k = kappa_ratio(tau)
            assert abs(k - 1.0) < 0.1, f"kappa={k} not near 1 at tau={tau}"

    def test_kappa_hot_limit(self):
        """kappa -> 1/(2*tau) as tau -> inf."""
        for tau in [50.0, 100.0]:
            k = kappa_ratio(tau)
            expected = 1.0 / (2.0 * tau)
            rel = abs(k - expected) / expected
            assert rel < 0.1, f"kappa={k} not near 1/(2*tau)={expected} at tau={tau}"

    def test_kappa_finiteness(self):
        taus = np.logspace(-4, 2, 50)
        for tau in taus:
            k = kappa_ratio(tau)
            assert np.isfinite(k), f"kappa non-finite at tau={tau}"
            assert k > 0, f"kappa non-positive at tau={tau}"
