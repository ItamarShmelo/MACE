"""
Compton scattering cross-section and multigroup integration.

Provides adaptive evaluation of the Klein-Nishina thermal Compton kernel
and multigroup/multiangle scattering matrix construction via deterministic
quadrature or Monte Carlo sampling.
"""

from compton_matrix._compton_common import ComptonResult
from compton_matrix._compton_differential_cross_section import (
    ComptonKernelAsymptoticSeries,
    ComptonKernelQuadrature,
    ComptonKernelSolver,
    ComptonPowerSeries,
    QuadratureForm,
)
from compton_matrix._compton_multigroup import (
    ComptonMonteCarloKernel,
    ComptonMultigroupKernel,
    MCIntegrationConfig,
    MGIntegrationConfig,
    PlanckWeightFunction,
    UniformWeightFunction,
    WienWeightFunction,
)
from compton_matrix._units import (
    k_boltz,
    kev,
    kev_kelvin,
    mbarn,
    me_c2,
    sigma_thomson,
)

__all__ = [
    "ComptonKernelAsymptoticSeries",
    "ComptonKernelQuadrature",
    "ComptonKernelSolver",
    "ComptonMonteCarloKernel",
    "ComptonMultigroupKernel",
    "ComptonPowerSeries",
    "ComptonResult",
    "MCIntegrationConfig",
    "MGIntegrationConfig",
    "PlanckWeightFunction",
    "QuadratureForm",
    "UniformWeightFunction",
    "WienWeightFunction",
    "k_boltz",
    "kev",
    "kev_kelvin",
    "mbarn",
    "me_c2",
    "sigma_thomson",
]
