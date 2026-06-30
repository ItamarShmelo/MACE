"""
Concrete kernel multipliers for multigroup Compton integrals
"""
from __future__ import annotations
import collections.abc
import compton_matrix._compton_multigroup
import typing
__all__: list[str] = ['EnergyTransferMultiplier']
class EnergyTransferMultiplier(compton_matrix._compton_multigroup.KernelMultiplier):
    def __init__(self, energy_group_boundaries: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], energy_group_centers: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
