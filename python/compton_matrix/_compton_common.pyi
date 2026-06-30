"""
Shared types for Compton scattering kernel modules
"""
from __future__ import annotations
__all__: list[str] = ['ComptonResult']
class ComptonResult:
    @property
    def estimated_abs_error(self) -> float:
        ...
    @property
    def estimated_rel_error(self) -> float:
        ...
    @property
    def value(self) -> float:
        ...
