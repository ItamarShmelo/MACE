"""
Concrete kernel multipliers for multigroup Compton integrals
"""
from __future__ import annotations
import collections.abc
import compton_matrix._compton_multigroup
import typing
__all__: list[str] = ['EMinusEpMultiplier', 'EnergyTransferMultiplier', 'EpMultiplier', 'EpOverEMultiplier', 'InducedEmissionRatioMultiplier']
class EMinusEpMultiplier(compton_matrix._compton_multigroup.KernelMultiplier):
    def __init__(self) -> None:
        ...
class EnergyTransferMultiplier(compton_matrix._compton_multigroup.KernelMultiplier):
    def __init__(self, energy_group_boundaries: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], energy_group_centers: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
class EpMultiplier(compton_matrix._compton_multigroup.KernelMultiplier):
    def __init__(self) -> None:
        ...
class EpOverEMultiplier(compton_matrix._compton_multigroup.KernelMultiplier):
    def __init__(self) -> None:
        ...
class InducedEmissionRatioMultiplier(compton_matrix._compton_multigroup.KernelMultiplier):
    def __init__(self) -> None:
        ...
