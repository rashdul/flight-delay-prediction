from __future__ import annotations

import sys
from pathlib import Path


def activate() -> None:
    """
    Make local packages importable without hardcoding absolute paths.

    Usage (e.g. in notebooks):
        import bootstrap_paths
        bootstrap_paths.activate()
        from aeroapi_market.Flights import Flights
    """

    supplychain_root = Path(__file__).resolve().parent

    local_src_dirs = [
        supplychain_root / "aeromarket-api" / "src",
        supplychain_root / "aeroapi-python" / "src",
        supplychain_root / "openmeteo_api" / "src",
    ]

    for src_dir in local_src_dirs:
        if src_dir.is_dir():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
