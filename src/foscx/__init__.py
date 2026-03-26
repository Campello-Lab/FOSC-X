import os
from pathlib import Path

def _ensure_numba_cache():
    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    cache_dir = Path.home() / ".cache" / "foscx" / "numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


from .foscx import FOSCX

__all__ = ["FOSCX"]
