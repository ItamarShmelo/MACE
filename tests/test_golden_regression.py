"""
Small golden regression checks for sigma_E.

These values are pinned from the current validated implementation to detect
unintentional numerical drift across refactors.
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cpp_modules"))

from _compton_kernel_quadrature import ComptonKernelQuadrature, QuadratureForm
from _compton_kernel_series import ComptonKernelSeries, SeriesMethod

ME_C2 = 9.109383713928e-28 * (2.99792458e10) ** 2
KEV = 1.602176634e-9
K_BOLTZ = 1.380649e-16
KEV_KELVIN = KEV / K_BOLTZ
REL_TOL = 5e-12


def _erg(E_kev: float) -> float:
    return E_kev * KEV


def _T_kelvin(T_kev: float) -> float:
    return T_kev * KEV_KELVIN


def _assert_rel_close(actual: float, expected: float, rel_tol: float = REL_TOL) -> None:
    assert math.isfinite(actual)
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=0.0), (
        f"Golden mismatch: actual={actual:.17e}, expected={expected:.17e}, "
        f"reldiff={abs(actual - expected) / (abs(expected) + 1e-300):.3e}"
    )


def test_quadrature_sigma_e_golden() -> None:
    quad = ComptonKernelQuadrature(256, QuadratureForm.PostIBP)
    r = quad.sigma_E(_erg(10.0), _erg(12.0), -0.3, _T_kelvin(5.0), 1.0)
    _assert_rel_close(r.value, 3.08235492600825279e-18)


def test_series_power_sigma_e_golden() -> None:
    series = ComptonKernelSeries(SeriesMethod.PowerSeriesHighPrecision)
    r = series.sigma_E(_erg(10.0), _erg(10.5), 0.0, _T_kelvin(100.0), 1.0)
    _assert_rel_close(r.value, 1.40557117111646695e-18)


def test_series_asymptotic_sigma_e_golden() -> None:
    series = ComptonKernelSeries(SeriesMethod.Asymptotic)
    r = series.sigma_E(_erg(1.0), _erg(1.01), 0.0, _T_kelvin(0.1), 1.0)
    _assert_rel_close(r.value, 4.18888613261258852e-16)


def test_series_auto_sigma_e_golden() -> None:
    series = ComptonKernelSeries(SeriesMethod.Auto)
    r = series.sigma_E(_erg(100.0), _erg(150.0), 0.0, _T_kelvin(100.0), 1.0)
    _assert_rel_close(r.value, 9.88033440640529926e-20)
