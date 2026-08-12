"""Accuracy, binding, and dispatch tests for the fast approximate kernel."""

import numpy as np
import pytest

import compton_matrix as mace


def relative_difference(value, reference):
    return abs(value - reference) / (abs(reference) + 1e-300)


APPROXIMATE_POINTS = [
    (0.1, 0.102, 0.5, 0.1),
    (10.0, 9.5, -0.5, 5.0),
    (50.0, 55.0, 0.25, 25.0),
    (100.0, 105.0, 0.9, 70.0),
]


@pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", APPROXIMATE_POINTS)
def test_approximate_kernel_below_one_percent(E_kev, Ep_kev, xi, T_kev):
    approximate = mace.ComptonKernelApproximate()
    reference = mace.ComptonKernelQuadrature(256, mace.QuadratureForm.PostIBP)
    E = E_kev * mace.kev
    E_prime = Ep_kev * mace.kev
    T = T_kev * mace.kev_kelvin

    expected = reference.sigma_E(E, E_prime, xi, T)
    if expected.estimated_rel_error > 1e-5:
        pytest.skip("quadrature reference is not converged")
    result = approximate.sigma_E(E, E_prime, xi, T)
    assert result.value > 0.0
    assert relative_difference(result.value, expected.value) < 0.01


def test_approximate_temperature_derivative_matches_finite_difference():
    kernel = mace.ComptonKernelApproximate()
    E = 10.0 * mace.kev
    E_prime = 10.2 * mace.kev
    xi = 0.25
    T = 5.0 * mace.kev_kelvin
    step = 1e-5 * T

    analytic = kernel.dsigma_E_dT(E, E_prime, xi, T).value
    upper = kernel.sigma_E(E, E_prime, xi, T + step).value
    lower = kernel.sigma_E(E, E_prime, xi, T - step).value
    finite_difference = (upper - lower) / (2.0 * step)
    assert relative_difference(analytic, finite_difference) < 1e-7


def test_approximate_vector_binding_matches_scalar():
    kernel = mace.ComptonKernelApproximate()
    E = 10.0 * mace.kev
    energies = np.array([9.5, 10.0, 10.5]) * mace.kev
    values, errors = kernel.sigma_E_vec(
        E,
        energies,
        0.25,
        5.0 * mace.kev_kelvin,
    )
    expected = np.array(
        [
            kernel.sigma_E(E, E_prime, 0.25, 5.0 * mace.kev_kelvin).value
            for E_prime in energies
        ]
    )
    np.testing.assert_array_equal(values, expected)
    assert np.all(errors >= 0.0)


SOLVER_POINTS = [
    # Fast asymptotic case
    (10.0, 9.9, 0.0, 1.0),
    # Fast approximate case
    (50.0, 52.0, 0.5, 50.0),
    # Photon-energy-domain rejection and power-series case
    (1500.0, 1764.5110845, -0.5, 100.0),
]


@pytest.mark.parametrize("E_kev,Ep_kev,xi,T_kev", SOLVER_POINTS)
def test_three_case_solver_below_one_percent(E_kev, Ep_kev, xi, T_kev):
    solver = mace.ComptonKernelApproximateSolver()
    reference = mace.ComptonKernelSolver()
    E = E_kev * mace.kev
    E_prime = Ep_kev * mace.kev
    T = T_kev * mace.kev_kelvin
    result = solver.sigma_E(E, E_prime, xi, T)
    expected = reference.sigma_E(E, E_prime, xi, T)
    assert result.value >= 0.0
    assert relative_difference(result.value, expected.value) < 0.01


def test_approximate_solver_is_accepted_by_multigroup_binding():
    boundaries = np.array([1.0, 2.0, 4.0]) * mace.kev
    config = mace.MGIntegrationConfig(
        cutoff_ratio=None,
        xi_order=4,
        xi_tail_order=4,
        ep_edge_order=4,
        ep_interior_order=4,
        e_panel_order=4,
    )
    multigroup = mace.ComptonMultigroupKernel(
        boundaries,
        mace.UniformWeightFunction(),
        config,
    )
    matrix = multigroup.compute_sigma_matrix(
        mace.ComptonKernelApproximateSolver(),
        num_angle_bins=2,
        T=5.0 * mace.kev_kelvin,
    )
    assert matrix.shape == (2, 2, 2)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
