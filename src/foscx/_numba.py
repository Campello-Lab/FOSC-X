"""Lightweight compatibility layer for optional numba support."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_NUMBA_ENV = "FOSCX_USE_NUMBA"


def set_numba_enabled(enabled: bool):
    """
    Enable/disable numba before constructing FOSCX objects.

    Must be called before first use of FOSCX.
    """
    os.environ[_NUMBA_ENV] = (
        "1" if enabled else "0"
    )


def numba_enabled() -> bool:
    return os.environ.get(
        _NUMBA_ENV,
        "1"
    ).lower() not in {"0", "false", "no"}


def _ensure_numba_cache():

    if not numba_enabled():
        return

    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    try:
        cache_root = os.environ.get(
            "XDG_CACHE_HOME",
            Path.home() / ".cache"
        )

        cache_dir = Path(cache_root) / "foscx" / "numba"

        cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

    except OSError:

        try:
            cache_dir = (
                Path(tempfile.gettempdir())
                / "foscx_numba"
            )

            cache_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

        except OSError:
            pass


def njit(*args, **kwargs):
    """
    Runtime optional replacement for numba.njit.
    """

    if numba_enabled():

        _ensure_numba_cache()

        try:
            from numba import njit as _numba_njit

            return _numba_njit(
                *args,
                **kwargs
            )

        except ImportError:
            pass

    # fallback identity decorator

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False