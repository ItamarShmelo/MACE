"""
Number-weighted photon transfer multiplier tests.

Validates NumberWeightedEnergyTransferMultiplier and NumberWeightMultiplier:
  1. Pointwise values (ratio to the ConstantMultiplier xi integral)
  2. Constructor validation
  3. Identity with the energy-weighted in/out matrices for sigma and dsigma/dT
  4. Non-negative off-diagonal transfer coefficients
  5. Group detailed balance of the raw pairs with a Wien weight
"""

import compton_matrix._compton_kernel_multipliers as km
import compton_matrix._compton_multigroup as cm
import numpy as np
import pytest
from compton_matrix._compton_differential_cross_section import ComptonKernelSolver
from compton_matrix._units import k_boltz, kev, kev_kelvin

KERNEL = ComptonKernelSolver()

BOUNDARIES_KEV = np.geomspace(0.5, 20.0, 7)
BOUNDARIES_ERG = BOUNDARIES_KEV * kev
CENTERS_ERG = np.sqrt(BOUNDARIES_ERG[:-1] * BOUNDARIES_ERG[1:])


def _number_transfer(boundaries=BOUNDARIES_ERG, centers=CENTERS_ERG):
    return km.NumberWeightedEnergyTransferMultiplier(
        energy_group_boundaries=list(boundaries),
        energy_group_centers=list(centers),
    )


def _number_weight(boundaries=BOUNDARIES_ERG, centers=CENTERS_ERG):
    return km.NumberWeightMultiplier(
        energy_group_boundaries=list(boundaries),
        energy_group_centers=list(centers),
    )


def _planck_kernel(boundaries=BOUNDARIES_ERG):
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=list(boundaries),
        weight_function=cm.CappedPlanckWeightFunction(cap_x=25.0),
        config=cm.MGIntegrationConfig(cutoff_ratio=None),
    )


def _group(E):
    return int(np.searchsorted(BOUNDARIES_ERG, E, side="right")) - 1


def _off_diagonal(G):
    return ~np.eye(G, dtype=bool)


# ---------------------------------------------------------------------------
# 1. Pointwise values
# ---------------------------------------------------------------------------


class TestPointwise:
    """The xi integral scales with the multiplier value at (E, E')."""

    T = 5.0 * kev_kelvin
    mg = _planck_kernel()

    def _ratio(self, multiplier, E, Ep):
        with_multiplier = np.asarray(self.mg.compute_xi_integral_sigma(KERNEL, E, Ep, 1, self.T, multiplier))
        plain = np.asarray(self.mg.compute_xi_integral_sigma(KERNEL, E, Ep, 1, self.T, cm.ConstantMultiplier()))
        return float(with_multiplier.sum() / plain.sum())

    @pytest.mark.parametrize(("E_kev", "Ep_kev"), [(1.3, 3.1), (4.0, 1.2), (7.7, 15.0)])
    def test_transfer_off_diagonal(self, E_kev, Ep_kev):
        E, Ep = E_kev * kev, Ep_kev * kev
        g, gp = _group(E), _group(Ep)
        assert g != gp
        expected = (CENTERS_ERG[g] / E) * (Ep - E) / (CENTERS_ERG[gp] - CENTERS_ERG[g])
        assert self._ratio(_number_transfer(), E, Ep) == pytest.approx(expected, rel=1e-12)

    def test_transfer_diagonal(self):
        E, Ep = 2.0 * kev, 2.1 * kev
        g = _group(E)
        assert g == _group(Ep)
        assert self._ratio(_number_transfer(), E, Ep) == pytest.approx(CENTERS_ERG[g] / E, rel=1e-12)

    @pytest.mark.parametrize(("E_kev", "Ep_kev"), [(1.3, 3.1), (2.0, 2.1), (7.7, 5.0)])
    def test_number_weight(self, E_kev, Ep_kev):
        E, Ep = E_kev * kev, Ep_kev * kev
        expected = CENTERS_ERG[_group(E)] / E
        assert self._ratio(_number_weight(), E, Ep) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [km.NumberWeightedEnergyTransferMultiplier, km.NumberWeightMultiplier])
class TestValidation:
    def test_size_mismatch(self, cls):
        with pytest.raises(ValueError, match="centers.size"):
            cls(energy_group_boundaries=[1.0, 2.0, 3.0], energy_group_centers=[1.5])

    def test_not_increasing(self, cls):
        with pytest.raises(ValueError, match="strictly increasing"):
            cls(energy_group_boundaries=[1.0, 3.0, 2.0], energy_group_centers=[1.5, 2.5])


# ---------------------------------------------------------------------------
# 3. Identity with the energy-weighted in/out matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T_kev", [1.0e-3, 20.0, 1.0e3])
class TestInOutIdentity:
    """sigma_new(g->g') (E_g' - E_g)/E_g equals sigma[E'/E] - sigma[1] off the diagonal."""

    def _check(self, new, sigma_in, sigma_out):
        G = len(CENTERS_ERG)
        factor = (CENTERS_ERG[np.newaxis, :] - CENTERS_ERG[:, np.newaxis]) / CENTERS_ERG[:, np.newaxis]
        lhs = new * factor
        rhs = sigma_in - sigma_out
        mask = _off_diagonal(G)
        scale = np.max(np.abs(sigma_in), axis=1, keepdims=True) * np.ones((1, G))
        tested = mask & (scale > 0.0)
        np.testing.assert_allclose(lhs[tested], rhs[tested], rtol=1e-9, atol=1e-12 * scale[tested].max())

    def test_sigma(self, T_kev):
        T = T_kev * kev_kelvin
        mg = _planck_kernel()
        new = np.asarray(mg.compute_sigma_matrix(KERNEL, T=T, multiplier=_number_transfer()))
        sigma_in = np.asarray(mg.compute_sigma_matrix(KERNEL, T=T, multiplier=km.EpOverEMultiplier()))
        sigma_out = np.asarray(mg.compute_sigma_matrix(KERNEL, T=T, multiplier=cm.ConstantMultiplier()))
        self._check(new, sigma_in, sigma_out)

    def test_dsigma_dT(self, T_kev):
        T = T_kev * kev_kelvin
        mg = _planck_kernel()
        new = np.asarray(mg.compute_dsigma_dT_matrix(KERNEL, T=T, multiplier=_number_transfer()))
        dsigma_in = np.asarray(mg.compute_dsigma_dT_matrix(KERNEL, T=T, multiplier=km.EpOverEMultiplier()))
        dsigma_out = np.asarray(mg.compute_dsigma_dT_matrix(KERNEL, T=T, multiplier=cm.ConstantMultiplier()))
        self._check(new, dsigma_in, dsigma_out)


# ---------------------------------------------------------------------------
# 4. Non-negativity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T_kev", [1.0e-3, 5.0, 1.0e3])
def test_off_diagonal_nonnegative(T_kev):
    mg = _planck_kernel()
    sigma = np.asarray(mg.compute_sigma_matrix(KERNEL, T=T_kev * kev_kelvin, multiplier=_number_transfer()))
    assert np.all(sigma[_off_diagonal(len(CENTERS_ERG))] >= 0.0)
    events = np.asarray(mg.compute_sigma_matrix(KERNEL, T=T_kev * kev_kelvin, multiplier=_number_weight()))
    assert np.all(events >= 0.0)


# ---------------------------------------------------------------------------
# 5. Group detailed balance with a Wien weight
# ---------------------------------------------------------------------------


class TestWienDetailedBalance:
    """With w = x^3 e^{-x}, the number-weighted pairs balance up to quadrature error.

    The kernel satisfies E^2 e^{-E/kT} Sigma(E->E') = E'^2 e^{-E'/kT} Sigma(E'->E),
    and the split factor is symmetric under (E, g) <-> (E', g'), so
    sigma(g->g') D_g / E_g = sigma(g'->g) D_g' / E_g' for the number weight.
    """

    T = 1.0 * kev_kelvin
    WEIGHT = cm.CappedWienWeightFunction(cap_x=500.0)

    def _pair_imbalance(self, multiplier):
        mg = cm.ComptonMultigroupKernel(
            energy_group_boundaries=list(BOUNDARIES_ERG),
            weight_function=self.WEIGHT,
            config=cm.MGIntegrationConfig(cutoff_ratio=None),
        )
        sigma = np.asarray(mg.compute_sigma_matrix(KERNEL, T=self.T, multiplier=multiplier))
        D = np.array(
            [
                self.WEIGHT.compute_denominator(lo, hi, self.T)
                for lo, hi in zip(BOUNDARIES_ERG[:-1], BOUNDARIES_ERG[1:], strict=True)
            ]
        )
        flux = sigma * (D / CENTERS_ERG)[:, np.newaxis]
        G = len(CENTERS_ERG)
        upper = np.triu(np.ones((G, G), dtype=bool), k=1)
        scale = flux.max()
        tested = upper & (flux > 1e-12 * scale) & (flux.T > 1e-12 * scale)
        return np.abs(flux[tested] / flux.T[tested] - 1.0)

    def test_number_weighted_transfer_balances(self):
        assert self._pair_imbalance(_number_transfer()).max() < 1e-6

    def test_number_weight_balances(self):
        assert self._pair_imbalance(_number_weight()).max() < 1e-6

    def test_energy_weighted_transfer_does_not_balance(self):
        energy_transfer = km.EnergyTransferMultiplier(
            energy_group_boundaries=list(BOUNDARIES_ERG),
            energy_group_centers=list(CENTERS_ERG),
        )
        assert self._pair_imbalance(energy_transfer).max() > 1e-3


def test_kT_consistency():
    """Sanity: the Wien test groups lie below x = 25 at T = 1 keV."""
    assert BOUNDARIES_ERG[-1] / (k_boltz * TestWienDetailedBalance.T) < 25.0
