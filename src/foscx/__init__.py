from ._numba import set_numba_enabled
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .foscx import FOSCX

__all__ = ["FOSCX", "set_numba_enabled"]

def __getattr__(name):
    if name == "FOSCX":
        from .foscx import FOSCX
        return FOSCX
    raise AttributeError(name)