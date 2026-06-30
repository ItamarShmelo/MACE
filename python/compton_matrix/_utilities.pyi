"""
Quadrature utilities: Gauss-Legendre and Gauss-Laguerre rules
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['gauss_laguerre_rule', 'gauss_legendre_rule']
def gauss_laguerre_rule(N: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Compute N-point Gauss-Laguerre nodes and weights
    """
def gauss_legendre_rule(N: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Compute N-point Gauss-Legendre nodes and weights on [-1, 1]
    """
