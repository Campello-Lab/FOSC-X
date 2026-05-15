from ._numba import set_numba_enabled


def __getattr__(name):

    if name == "FOSCX":
        from .foscx import FOSCX
        return FOSCX

    raise AttributeError(name)


__all__ = [
    "FOSCX",
    "set_numba_enabled",
]