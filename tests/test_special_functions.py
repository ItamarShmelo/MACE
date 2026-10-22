"""
Tests for the scaled modified Bessel functions used by the kernel prefactor.

scaled_K1 and scaled_K2 switch from Boost to the Hankel asymptotic expansion
at x = 50.  Both branches must reach double precision, since the temperature
derivative of the kernel depends on kappa = K1/K2 through (lambda_+ - kappa)/tau^2,
which amplifies an error in kappa by 1/tau^2.
"""

import numpy as np
import pytest
from compton_matrix._compton_differential_cross_section import (
    kappa_ratio,
    scaled_K1,
    scaled_K2,
)
from scipy.special import kve

X_GRID = np.geomspace(0.1, 1e4, 400)


def _max_rel(values, reference):
    return float(np.max(np.abs(np.asarray(values) / reference - 1.0)))


@pytest.mark.parametrize(
    "func,order", [(scaled_K1, 1), (scaled_K2, 2)], ids=["K1", "K2"]
)
def test_scaled_bessel_vs_scipy(func, order):
    values = [func(x) for x in X_GRID]
    assert _max_rel(values, kve(order, X_GRID)) < 1e-14


def test_hankel_branch_near_switch():
    """The asymptotic branch starts at x = 50, where a short expansion is worst."""
    x = np.linspace(50.0, 60.0, 101)
    assert _max_rel([scaled_K1(v) for v in x], kve(1, x)) < 1e-14
    assert _max_rel([scaled_K2(v) for v in x], kve(2, x)) < 1e-14


def test_kappa_ratio_vs_scipy():
    tau = 1.0 / X_GRID
    values = [kappa_ratio(t) for t in tau]
    assert _max_rel(values, kve(1, X_GRID) / kve(2, X_GRID)) < 1e-14
