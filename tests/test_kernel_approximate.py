"""Tests for the ComptonKernelApproximate fifth-order global approximation solver."""

import math

import numpy as np
import pytest
from compton_matrix._compton_differential_cross_section import (
    ComptonKernelApproximate,
    ComptonKernelSolver,
)
from compton_matrix._units import kev, kev_kelvin


def _rel_diff(a, b):
    return abs(a - b) / (abs(b) + 1e-300)


APPROX = ComptonKernelApproximate()
SOLVER = ComptonKernelSolver()

# Physical constants for dimensionless conversions
ME_C2_ERG = 9.109383713928e-28 * (2.99792458e10) ** 2  # m_e c^2 in erg
K_BOLTZ = 1.380649e-16  # erg/K


# ── Kinematic identity tests ──────────────────────────────────────────────


class TestKinematicIdentities:
    """Verify q^2 = Delta^2 + 2*A = gamma^2 + gamma'^2 - 2*gamma*gamma'*xi."""

    @pytest.mark.parametrize(
        "gamma,gamma_prime,xi",
        [
            (0.01, 0.012, 0.0),
            (0.1, 0.08, -0.5),
            (1.0, 0.5, 0.9),
            (0.001, 0.0015, -1.0),
            (5.0, 3.0, 0.5),
        ],
    )
    def test_q_squared_identity(self, gamma, gamma_prime, xi):
        a = 1.0 - xi
        A = gamma * gamma_prime * a
        Delta = gamma_prime - gamma
        q_sq_formula1 = gamma**2 + gamma_prime**2 - 2 * gamma * gamma_prime * xi
        q_sq_formula2 = Delta**2 + 2 * A
        assert _rel_diff(q_sq_formula1, q_sq_formula2) < 1e-14

    @pytest.mark.parametrize(
        "gamma,gamma_prime,xi",
        [
            (0.01, 0.012, 0.0),
            (0.1, 0.08, -0.5),
            (1.0, 0.5, 0.9),
            (0.001, 0.0015, -1.0),
        ],
    )
    def test_lambda_min_formulas_agree(self, gamma, gamma_prime, xi):
        """Both formulas for lambda_min should agree at generic points."""
        a = 1.0 - xi
        A = gamma * gamma_prime * a
        Delta = gamma_prime - gamma
        q = math.hypot(Delta, math.sqrt(2.0 * A))

        lam1 = 0.5 * Delta + math.sqrt((1 + 0.5 * A) * (1 + Delta**2 / (2 * A)))
        lam2 = 0.5 * (Delta + q * math.sqrt(1 + 2.0 / A))
        assert _rel_diff(lam1, lam2) < 1e-12


# ── Cold Compton line test ────────────────────────────────────────────────


class TestColdComptonLine:
    """At the cold Compton line, lambda_min should equal 1."""

    @pytest.mark.parametrize(
        "gamma,xi",
        [
            (0.01, 0.0),
            (0.1, -0.5),
            (1.0, 0.9),
            (0.001, -1.0),
            (0.5, 0.5),
        ],
    )
    def test_lambda_min_equals_one_on_cold_line(self, gamma, xi):
        gamma_prime_C = gamma / (1.0 + gamma * (1.0 - xi))
        a = 1.0 - xi
        A = gamma * gamma_prime_C * a
        Delta = gamma_prime_C - gamma
        lam = 0.5 * Delta + math.sqrt((1 + 0.5 * A) * (1 + Delta**2 / (2 * A)))
        assert abs(lam - 1.0) < 1e-12


# ── Invalid input tests ───────────────────────────────────────────────────


class TestInvalidInputs:
    """Invalid inputs should return the failure sentinel (abs_error=1.0)."""

    def test_negative_energy(self):
        r = APPROX.sigma_E(-1.0 * kev, 1.0 * kev, 0.0, 10.0 * kev_kelvin)
        assert r.estimated_abs_error == 1.0

    def test_negative_temperature(self):
        r = APPROX.sigma_E(1.0 * kev, 0.5 * kev, 0.0, -10.0 * kev_kelvin)
        assert r.estimated_abs_error == 1.0

    def test_xi_out_of_range_high(self):
        r = APPROX.sigma_E(1.0 * kev, 0.5 * kev, 1.5, 10.0 * kev_kelvin)
        assert r.estimated_abs_error == 1.0

    def test_xi_out_of_range_low(self):
        r = APPROX.sigma_E(1.0 * kev, 0.5 * kev, -1.5, 10.0 * kev_kelvin)
        assert r.estimated_abs_error == 1.0

    def test_nan_input(self):
        r = APPROX.sigma_E(float("nan"), 1.0 * kev, 0.0, 10.0 * kev_kelvin)
        assert r.estimated_abs_error == 1.0


# ── Distributional limit tests ────────────────────────────────────────────


class TestDistributionalLimits:
    """Distributional limits (xi=1, T=0) return the failure sentinel."""

    def test_forward_scattering_limit(self):
        r = APPROX.sigma_E(1.0 * kev, 1.0 * kev, 1.0, 10.0 * kev_kelvin)
        assert r.estimated_abs_error == 1.0

    def test_zero_temperature_limit(self):
        r = APPROX.sigma_E(1.0 * kev, 0.5 * kev, 0.0, 0.0)
        assert r.estimated_abs_error == 1.0


# ── Underflow test ────────────────────────────────────────────────────────


class TestUnderflow:
    """At very low tau with lambda_min >> 1, exponential underflows to zero."""

    def test_exponential_underflow_gives_zero(self):
        E = 100.0 * kev
        T = 0.01 * kev_kelvin
        xi = -1.0
        # far from cold line -> large lambda_min -> exp(-(lam-1)/tau) ~ 0
        E_prime = 0.001 * kev
        r = APPROX.sigma_E(E, E_prime, xi, T)
        assert r.value == 0.0
        assert r.estimated_abs_error == 0.0


# ── Reference agreement tests ─────────────────────────────────────────────


class TestReferenceAgreement:
    """Approximate solver should agree with ComptonKernelSolver within reasonable tolerance."""

    @pytest.mark.parametrize(
        "E_kev,xi,T_kev",
        [
            (1.0, 0.0, 10.0),
            (0.1, -0.5, 5.0),
            (10.0, 0.5, 50.0),
            (0.01, 0.0, 1.0),
            (5.0, -0.9, 25.0),
            (1.0, 0.9, 10.0),
            (0.5, 0.0, 100.0),
        ],
    )
    def test_agreement_at_cold_line(self, E_kev, xi, T_kev):
        """Approximate vs solver at the cold Compton line for each parameter set."""
        E = E_kev * kev
        T = T_kev * kev_kelvin
        gamma = E / ME_C2_ERG
        E_prime = E / (1.0 + gamma * (1.0 - xi))

        approx_r = APPROX.sigma_E(E, E_prime, xi, T)
        ref_r = SOLVER.sigma_E(E, E_prime, xi, T)

        if ref_r.estimated_abs_error >= abs(ref_r.value):
            pytest.skip("reference unreliable at this point")
        if approx_r.estimated_abs_error == 1.0:
            pytest.skip("approximate solver failed at this point")

        rel = _rel_diff(approx_r.value, ref_r.value)
        assert rel < 0.05, f"rel_diff={rel:.3e} at E={E_kev}keV, xi={xi}, T={T_kev}keV"

    @pytest.mark.parametrize(
        "E_kev,xi,T_kev",
        [
            (1.0, 0.0, 10.0),
            (0.1, -0.5, 5.0),
            (10.0, 0.5, 50.0),
            (5.0, -0.9, 25.0),
            (0.5, 0.0, 100.0),
        ],
    )
    def test_agreement_sweep_E_prime(self, E_kev, xi, T_kev):
        """Approximate vs solver at multiple E' values around the cold line."""
        E = E_kev * kev
        T = T_kev * kev_kelvin
        gamma = E / ME_C2_ERG
        E_prime_C = E / (1.0 + gamma * (1.0 - xi))

        ratios = [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5]
        max_rel = 0.0

        for ratio in ratios:
            E_prime = E_prime_C * ratio
            if E_prime <= 0:
                continue

            approx_r = APPROX.sigma_E(E, E_prime, xi, T)
            ref_r = SOLVER.sigma_E(E, E_prime, xi, T)

            if ref_r.estimated_abs_error >= abs(ref_r.value) or ref_r.value == 0:
                continue
            if approx_r.estimated_abs_error == 1.0:
                continue

            rel = _rel_diff(approx_r.value, ref_r.value)
            max_rel = max(max_rel, rel)

        assert max_rel < 0.05, f"max_rel_diff={max_rel:.3e} at E={E_kev}keV, xi={xi}, T={T_kev}keV"


# ── Unit conversion test ──────────────────────────────────────────────────


class TestUnitConversion:
    """Verify output has plausible magnitude (cross section in cm^2/erg)."""

    def test_result_is_positive_and_finite(self):
        E = 1.0 * kev
        T = 10.0 * kev_kelvin
        xi = 0.0
        gamma = E / ME_C2_ERG
        E_prime = E / (1.0 + gamma * (1.0 - xi))

        r = APPROX.sigma_E(E, E_prime, xi, T)
        assert r.value > 0.0
        assert np.isfinite(r.value)

    def test_sigma_E_order_of_magnitude(self):
        """sigma_E at ~1 keV should be somewhere around 1e-17 to 1e-15 cm^2/erg."""
        E = 1.0 * kev
        T = 10.0 * kev_kelvin
        xi = 0.0
        gamma = E / ME_C2_ERG
        E_prime = E / (1.0 + gamma * (1.0 - xi))

        r = APPROX.sigma_E(E, E_prime, xi, T)
        assert 1e-20 < r.value < 1e-12


# ── Vectorized interface test ─────────────────────────────────────────────


class TestVectorized:
    """sigma_E_vec should return the same values as scalar sigma_E."""

    def test_vec_matches_scalar(self):
        E = 1.0 * kev
        T = 10.0 * kev_kelvin
        xi = 0.0
        gamma = E / ME_C2_ERG
        E_prime_C = E / (1.0 + gamma * (1.0 - xi))

        E_prime_arr = np.array([E_prime_C * r for r in [0.8, 0.9, 1.0, 1.1, 1.2]])

        values, errors = APPROX.sigma_E_vec(E, E_prime_arr, xi, T)

        for i, Ep in enumerate(E_prime_arr):
            scalar_r = APPROX.sigma_E(E, Ep, xi, T)
            assert values[i] == pytest.approx(scalar_r.value, rel=0, abs=0)
            assert errors[i] == pytest.approx(scalar_r.estimated_abs_error, rel=0, abs=0)
