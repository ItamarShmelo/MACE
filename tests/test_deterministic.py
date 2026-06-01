"""
Fast deterministic tests for the Compton kernel quadrature implementation.
These must pass before any Monte Carlo comparison is attempted.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp_modules'))
from _compton_common import SigmaResult
from _compton_kernel_quadrature import (
    ComptonKernelQuadrature, QuadratureForm, scaled_K2,
    gauss_laguerre_rule,
)

ME_C2 = 9.109383713928e-28 * (2.99792458000e10)**2  # erg
K_BOLTZ = 1.380649e-16  # erg/K


def assert_close_mixed(a, b, rtol=1e-8, atol=1e-300):
    scale = max(abs(a), abs(b))
    assert abs(a - b) <= atol + rtol * scale, (
        f"|{a} - {b}| = {abs(a-b)} > atol({atol}) + rtol({rtol})*scale({scale})"
    )


def assert_close_moderate(a, b, rtol=1e-10, atol=1e-300):
    scale = max(abs(a), abs(b))
    assert abs(a - b) <= atol + rtol * scale, (
        f"|{a} - {b}| = {abs(a-b)} > atol({atol}) + rtol({rtol})*scale({scale})"
    )


class TestScaledK2:
    """Compare scaled_K2(x) against scipy.special.kve(2, x)."""

    def test_moderate_x(self):
        from scipy.special import kve
        x_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 49.0]
        for x in x_values:
            ref = kve(2, x)
            val = scaled_K2(x)
            assert_close_moderate(val, ref), f"Failed at x={x}"

    def test_large_x_asymptotic(self):
        from scipy.special import kve
        x_values = [50.0, 100.0, 200.0, 500.0]
        for x in x_values:
            ref = kve(2, x)
            val = scaled_K2(x)
            assert_close_mixed(val, ref), f"Failed at x={x}"

    def test_invalid_input(self):
        with pytest.raises(Exception):
            scaled_K2(0.0)
        with pytest.raises(Exception):
            scaled_K2(-1.0)
        with pytest.raises(Exception):
            scaled_K2(float('inf'))


# Representative test points spanning different regimes
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

KEV = 1.602176634e-9  # erg
KEV_KELVIN = KEV / K_BOLTZ  # 1 keV in Kelvin


def _to_erg(E_kev):
    return E_kev * KEV


def _to_kelvin(T_kev):
    return T_kev * KEV_KELVIN


class TestFiniteOutput:
    """Verify sigma_E returns finite values over representative points."""

    def test_finite_output(self):
        engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)
        for E_kev, Ep_kev, xi, T_kev in TEST_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            T = _to_kelvin(T_kev)
            r = engine.sigma_E(E, Ep, xi, T, 1.0)
            assert np.isfinite(r.value), (
                f"Non-finite at E={E_kev}, Ep={Ep_kev}, xi={xi}, T_kev={T_kev}"
            )
            assert np.isfinite(r.estimated_abs_error)
            assert np.isfinite(r.estimated_rel_error)


class TestPositivity:
    """Verify sigma_E is non-negative (tolerant)."""

    def test_positivity(self):
        engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)
        for E_kev, Ep_kev, xi, T_kev in TEST_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            T = _to_kelvin(T_kev)
            r = engine.sigma_E(E, Ep, xi, T, 1.0)
            r_rev = engine.sigma_E(Ep, E, xi, T, 1.0)
            local_scale = max(abs(r.value), abs(r_rev.value), 1e-300)
            assert r.value >= -1e-12 * local_scale, (
                f"Negative at E={E_kev}, Ep={Ep_kev}, xi={xi}: {r.value}"
            )


class TestDetailedBalance:
    """
    Test energy-form detailed balance:
    E^2 * exp(-E/(tau*me_c2)) * Sigma_E(E->E', xi, tau)
    = E'^2 * exp(-E'/(tau*me_c2)) * Sigma_E(E'->E, xi, tau)
    """

    def test_detailed_balance(self):
        engine = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)
        for E_kev, Ep_kev, xi, T_kev in TEST_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            T = _to_kelvin(T_kev)
            tau = T_kev * KEV / ME_C2

            r_fwd = engine.sigma_E(E, Ep, xi, T, 1.0)
            r_rev = engine.sigma_E(Ep, E, xi, T, 1.0)

            gamma = E / ME_C2
            gamma_p = Ep / ME_C2

            log_lhs = 2.0 * np.log(E) - gamma / tau + np.log(abs(r_fwd.value) + 1e-300)
            log_rhs = 2.0 * np.log(Ep) - gamma_p / tau + np.log(abs(r_rev.value) + 1e-300)

            if abs(r_fwd.value) < 1e-300 and abs(r_rev.value) < 1e-300:
                continue

            assert abs(log_lhs - log_rhs) < 1e-6, (
                f"Detailed balance failed at E={E_kev}, Ep={Ep_kev}, xi={xi}, "
                f"T_kev={T_kev}: log_lhs={log_lhs}, log_rhs={log_rhs}, "
                f"diff={abs(log_lhs - log_rhs)}"
            )


class TestNLConvergence:
    """Verify convergence with increasing quadrature order."""

    def test_convergence_pre_ibp(self):
        """Pre-IBP converges uniformly (no Psi/IQ cancellation)."""
        engine_128 = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PreIBP)
        engine_256 = ComptonKernelQuadrature(NL=256, form=QuadratureForm.PreIBP)

        for E_kev, Ep_kev, xi, T_kev in TEST_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            T = _to_kelvin(T_kev)

            r_128 = engine_128.sigma_E(E, Ep, xi, T, 1.0)
            r_256 = engine_256.sigma_E(E, Ep, xi, T, 1.0)

            if abs(r_256.value) < 1e-300:
                continue

            rel_diff = abs(r_256.value - r_128.value) / (abs(r_256.value) + 1e-300)
            assert rel_diff < 5e-6, (
                f"NL convergence failed at E={E_kev}, Ep={Ep_kev}, xi={xi}, "
                f"T_kev={T_kev}: rel_diff={rel_diff}"
            )

    def test_convergence_post_ibp_moderate(self):
        """Post-IBP converges for moderate tau (not small-tau regime)."""
        engine_128 = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)
        engine_256 = ComptonKernelQuadrature(NL=256, form=QuadratureForm.PostIBP)

        for E_kev, Ep_kev, xi, T_kev in TEST_POINTS:
            if T_kev < 0.5:
                continue
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            T = _to_kelvin(T_kev)

            r_128 = engine_128.sigma_E(E, Ep, xi, T, 1.0)
            r_256 = engine_256.sigma_E(E, Ep, xi, T, 1.0)

            if abs(r_256.value) < 1e-300:
                continue

            rel_diff = abs(r_256.value - r_128.value) / (abs(r_256.value) + 1e-300)
            assert rel_diff < 5e-6, (
                f"NL convergence failed at E={E_kev}, Ep={Ep_kev}, xi={xi}, "
                f"T_kev={T_kev}: rel_diff={rel_diff}"
            )


class TestPostVsPreIBP:
    """Verify that PostIBP and PreIBP forms agree."""

    def test_agreement(self):
        engine_post = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PostIBP)
        engine_pre = ComptonKernelQuadrature(NL=128, form=QuadratureForm.PreIBP)

        for E_kev, Ep_kev, xi, T_kev in TEST_POINTS:
            E = _to_erg(E_kev)
            Ep = _to_erg(Ep_kev)
            T = _to_kelvin(T_kev)

            r_post = engine_post.sigma_E(E, Ep, xi, T, 1.0)
            r_pre = engine_pre.sigma_E(E, Ep, xi, T, 1.0)

            if T_kev < 0.5:
                assert_close_mixed(r_post.value, r_pre.value, rtol=1e-3, atol=1e-300)
            else:
                assert_close_mixed(r_post.value, r_pre.value, rtol=1e-6, atol=1e-300)


class TestAngularNormalization:
    """
    Verify that the total cross section (integrated over all angles and
    outgoing energies) gives a physically reasonable value.
    In the low-energy limit, it should approach sigma_Thomson.
    """

    def test_low_energy_total(self):
        from scipy.integrate import dblquad

        engine = ComptonKernelQuadrature(NL=64, form=QuadratureForm.PostIBP)

        SIGMA_T = 6.652458732160e-25  # cm^2
        E_kev = 0.01  # very low energy (10 eV)
        T_kev = 10.0  # hot plasma so scattering is quasi-elastic
        E = _to_erg(E_kev)
        T = _to_kelvin(T_kev)

        E_lo = _to_erg(E_kev * 0.5)
        E_hi = _to_erg(E_kev * 2.0)

        XI_EPS = 1e-8

        def integrand(Ep, xi):
            return engine.sigma_E(E, Ep, xi, T, 1.0).value

        val, err = dblquad(
            integrand,
            -1.0 + XI_EPS, 1.0 - XI_EPS,
            lambda xi: E_lo, lambda xi: E_hi,
            epsabs=1e-30, epsrel=1e-3
        )

        total_sigma = 0.5 * val  # angular average factor
        # For low energy and hot electrons, should be order-of-magnitude sigma_T
        assert total_sigma > 0, f"Total cross section is non-positive: {total_sigma}"
        ratio = total_sigma / SIGMA_T
        assert 0.01 < ratio < 100.0, (
            f"Total cross section {total_sigma} is not within "
            f"order-of-magnitude of sigma_T={SIGMA_T}, ratio={ratio}"
        )


class TestGaussLaguerreVsScipy:
    """Compare C++ Gauss-Laguerre nodes/weights against scipy."""

    def test_nodes_and_weights(self):
        from scipy.special import roots_laguerre

        for NL in [32, 64, 128, 256]:
            cpp_nodes, cpp_weights = gauss_laguerre_rule(NL)
            scipy_nodes, scipy_weights = roots_laguerre(NL)

            np.testing.assert_allclose(
                cpp_nodes, scipy_nodes, rtol=1e-11,
                err_msg=f"Nodes differ at NL={NL}"
            )
            np.testing.assert_allclose(
                cpp_weights, scipy_weights, rtol=1e-11,
                err_msg=f"Weights differ at NL={NL}"
            )
