"""
Weighted multigroup-multiangle Compton scattering matrix
"""
from __future__ import annotations
import collections.abc
import compton_matrix._compton_differential_cross_section
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CappedPlanckWeightFunction', 'CappedWienWeightFunction', 'ComptonMonteCarloKernel', 'ComptonMultigroupKernel', 'ConstantMultiplier', 'KernelMultiplier', 'MCIntegrationConfig', 'MGIntegrationConfig', 'PlanckWeightFunction', 'RidgeBounds', 'UniformWeightFunction', 'WeightFunction', 'WienWeightFunction', 'adaptive_legendre_integrate', 'adaptive_log_legendre_integrate', 'adaptive_rlog_legendre_integrate', 'cold_recoil_hi', 'cold_recoil_lo', 'compute_ridge_bounds', 'gauss_legendre_rule', 'ridge_thermal_width']
class ComptonMonteCarloKernel:
    def __init__(self, energy_group_boundaries: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], weight_function: WeightFunction, config: MCIntegrationConfig = ...) -> None:
        ...
    @typing.overload
    def compute_kernel_derivative_contribution(self, num_angle_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_kernel_derivative_contribution(self, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_dsigma_dT_matrix(self, num_angle_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_dsigma_dT_matrix(self, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_sigma_matrix(self, num_angle_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_sigma_matrix(self, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def group_boundaries(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def group_centers(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def num_groups(self) -> int:
        ...
class ComptonMultigroupKernel:
    def __init__(self, energy_group_boundaries: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], weight_function: ..., config: MGIntegrationConfig = ...) -> None:
        ...
    def compute_Ep_xi_integral_dsigma_dT(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, E: typing.SupportsFloat | typing.SupportsIndex, Ep_lo: typing.SupportsFloat | typing.SupportsIndex, Ep_hi: typing.SupportsFloat | typing.SupportsIndex, num_xi_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        """
        Integrate dsigma_E/dT over E' range and xi bins for fixed E
        """
    def compute_Ep_xi_integral_sigma(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, E: typing.SupportsFloat | typing.SupportsIndex, Ep_lo: typing.SupportsFloat | typing.SupportsIndex, Ep_hi: typing.SupportsFloat | typing.SupportsIndex, num_xi_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        """
        Integrate sigma_E over E' range and xi bins for fixed E
        """
    @typing.overload
    def compute_kernel_derivative_contribution(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, num_angle_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_kernel_derivative_contribution(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_dsigma_dT_matrix(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, num_angle_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_dsigma_dT_matrix(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_sigma_matrix(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, num_angle_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @typing.overload
    def compute_sigma_matrix(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def compute_xi_integral_dsigma_dT(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, E: typing.SupportsFloat | typing.SupportsIndex, Ep: typing.SupportsFloat | typing.SupportsIndex, num_xi_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        """
        Integrate dsigma_E/dT over xi bins for fixed (E, E')
        """
    def compute_xi_integral_sigma(self, kernel: compton_matrix._compton_differential_cross_section.ComptonKernelSolver, E: typing.SupportsFloat | typing.SupportsIndex, Ep: typing.SupportsFloat | typing.SupportsIndex, num_xi_bins: typing.SupportsInt | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex, multiplier: KernelMultiplier = ...) -> numpy.typing.NDArray[numpy.float64]:
        """
        Integrate sigma_E over xi bins for fixed (E, E')
        """
    @property
    def group_boundaries(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def num_groups(self) -> int:
        ...
class ConstantMultiplier(KernelMultiplier):
    def __init__(self) -> None:
        ...
class KernelMultiplier:
    pass
class MCIntegrationConfig:
    discard_out_of_grid: bool
    def __init__(self, num_samples: typing.SupportsInt | typing.SupportsIndex = 1000000, seed: typing.SupportsInt | typing.SupportsIndex = -1, discard_out_of_grid: bool = True) -> None:
        ...
    @property
    def num_samples(self) -> int:
        ...
    @num_samples.setter
    def num_samples(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MGIntegrationConfig:
    def __init__(self, cutoff_ratio: typing.SupportsFloat | typing.SupportsIndex | None = 1e-08, xi_order: typing.SupportsInt | typing.SupportsIndex | None = None, xi_peak_k: typing.SupportsFloat | typing.SupportsIndex = 5.0, xi_tail_order: typing.SupportsInt | typing.SupportsIndex | None = None, ep_k_cut: typing.SupportsFloat | typing.SupportsIndex = 5.0, ep_k_in: typing.SupportsFloat | typing.SupportsIndex = 2.0, ep_edge_order: typing.SupportsInt | typing.SupportsIndex | None = None, ep_interior_order: typing.SupportsInt | typing.SupportsIndex | None = None, e_panel_order: typing.SupportsInt | typing.SupportsIndex | None = None, log_e_panel_ratio: typing.SupportsFloat | typing.SupportsIndex = 2.0, e_boundary_k: typing.SupportsFloat | typing.SupportsIndex = 5.0) -> None:
        ...
    def effective_e_panel_order(self) -> int:
        ...
    def effective_ep_edge_order(self) -> int:
        ...
    def effective_ep_interior_order(self) -> int:
        ...
    def effective_xi_order(self) -> int:
        ...
    def effective_xi_tail_order(self) -> int:
        ...
    @property
    def cutoff_ratio(self) -> float | None:
        ...
    @cutoff_ratio.setter
    def cutoff_ratio(self, arg0: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    @property
    def e_boundary_k(self) -> float:
        ...
    @e_boundary_k.setter
    def e_boundary_k(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def e_panel_order(self) -> int | None:
        ...
    @e_panel_order.setter
    def e_panel_order(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None:
        ...
    @property
    def ep_edge_order(self) -> int | None:
        ...
    @ep_edge_order.setter
    def ep_edge_order(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None:
        ...
    @property
    def ep_interior_order(self) -> int | None:
        ...
    @ep_interior_order.setter
    def ep_interior_order(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None:
        ...
    @property
    def ep_k_cut(self) -> float:
        ...
    @ep_k_cut.setter
    def ep_k_cut(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ep_k_in(self) -> float:
        ...
    @ep_k_in.setter
    def ep_k_in(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def log_e_panel_ratio(self) -> float:
        ...
    @log_e_panel_ratio.setter
    def log_e_panel_ratio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def xi_order(self) -> int | None:
        ...
    @xi_order.setter
    def xi_order(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None:
        ...
    @property
    def xi_peak_k(self) -> float:
        ...
    @xi_peak_k.setter
    def xi_peak_k(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def xi_tail_order(self) -> int | None:
        ...
    @xi_tail_order.setter
    def xi_tail_order(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None:
        ...
class CappedPlanckWeightFunction(WeightFunction):
    def __init__(self, *, cap_x: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def cap_x(self) -> float:
        ...
    def compute_denominator(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_denominator_dT(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_log_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def peak_energy(self, T: typing.SupportsFloat | typing.SupportsIndex) -> float | None:
        ...
    def weight(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class RidgeBounds:
    @property
    def cold_hi(self) -> float:
        ...
    @property
    def cold_lo(self) -> float:
        ...
    @property
    def sigma_hi(self) -> float:
        ...
    @property
    def sigma_lo(self) -> float:
        ...
class UniformWeightFunction(WeightFunction):
    def __init__(self) -> None:
        ...
    def compute_denominator(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_denominator_dT(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_log_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def peak_energy(self, T: typing.SupportsFloat | typing.SupportsIndex) -> float | None:
        ...
    def weight(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class PlanckWeightFunction(WeightFunction):
    def __init__(self, *, cap_x: typing.SupportsFloat | typing.SupportsIndex, group_boundaries: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        """
        Note: group_boundaries must match the energy_group_boundaries
        passed to the kernel constructor.
        """
    def cap_x(self) -> float:
        ...
    def compute_denominator(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_denominator_dT(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_log_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def peak_energy(self, T: typing.SupportsFloat | typing.SupportsIndex) -> float | None:
        ...
    def weight(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class WeightFunction:
    pass
class WienWeightFunction(WeightFunction):
    def __init__(self, *, group_boundaries: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        """
        Note: group_boundaries must match the energy_group_boundaries
        passed to the kernel constructor.
        """
    def compute_denominator(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_denominator_dT(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_log_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def peak_energy(self, T: typing.SupportsFloat | typing.SupportsIndex) -> float | None:
        ...
    def weight(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class CappedWienWeightFunction(WeightFunction):
    def __init__(self, *, cap_x: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def cap_x(self) -> float:
        ...
    def compute_denominator(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_denominator_dT(self, E_left: typing.SupportsFloat | typing.SupportsIndex, E_right: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_log_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def d_weight_dT(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def peak_energy(self, T: typing.SupportsFloat | typing.SupportsIndex) -> float | None:
        ...
    def weight(self, E: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
def adaptive_legendre_integrate(integrand: collections.abc.Callable, base_order: typing.SupportsInt | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_depth: typing.SupportsInt | typing.SupportsIndex = 15) -> float:
    """
    Adaptive Gauss-Legendre integration of f over [a, b]
    """
def adaptive_log_legendre_integrate(integrand: collections.abc.Callable, base_order: typing.SupportsInt | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_depth: typing.SupportsInt | typing.SupportsIndex = 15) -> float:
    """
    Adaptive log-space GL integration of f over [a, b] (clusters nodes near a)
    """
def adaptive_rlog_legendre_integrate(integrand: collections.abc.Callable, base_order: typing.SupportsInt | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_depth: typing.SupportsInt | typing.SupportsIndex = 15) -> float:
    """
    Adaptive reflected-log GL integration of f over [a, b] (clusters nodes near b)
    """
def cold_recoil_hi(E: typing.SupportsFloat | typing.SupportsIndex, xi_hi: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Upper edge of the cold Compton recoil band [erg] (T=0 limit)
    """
def cold_recoil_lo(E: typing.SupportsFloat | typing.SupportsIndex, xi_lo: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Lower edge of the cold Compton recoil band [erg] (T=0 limit)
    """
def compute_ridge_bounds(E: typing.SupportsFloat | typing.SupportsIndex, xi_lo: typing.SupportsFloat | typing.SupportsIndex, xi_hi: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> RidgeBounds:
    """
    Ridge bounds with cold endpoints and thermal widths [erg]
    """
def gauss_legendre_rule(N: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Compute N-point Gauss-Legendre nodes and weights on [-1, 1]
    """
def ridge_thermal_width(E: typing.SupportsFloat | typing.SupportsIndex, xi: typing.SupportsFloat | typing.SupportsIndex, T: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Local thermal width of the Compton ridge in E' [erg]
    """
