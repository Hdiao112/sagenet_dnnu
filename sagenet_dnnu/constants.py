# -*- coding: utf-8 -*-
"""
Physical constants used by the SageNet Delta-Neff integrator.

This module acts as a thin compatibility layer.

Priority:
1. Try to load constants from the local `global_param.py` placed in the
   same `sagenet_dnnu/` package directory.
2. If that fails, fall back to a lightweight standalone definition.

Only the constants needed by the dnnu conversion are exposed:
    Neff0, Omega_ph2, Omega_nh2, ln10.
"""

from __future__ import annotations

import importlib
import math
from typing import Any


def _get_float(module: Any, name: str) -> float:
    if not hasattr(module, name):
        raise AttributeError(f"`global_param.py` does not define `{name}`.")
    return float(getattr(module, name))


def _load_from_local_global_param():
    """
    Load constants from `sagenet_dnnu/global_param.py`.

    This import may fail if `global_param.py` depends on files/modules that
    are not present, e.g. `th.dat` or `utils_new.functions`.
    """
    package = __package__ or "sagenet_dnnu"
    gp = importlib.import_module(f"{package}.global_param")

    neff0 = _get_float(gp, "Neff0")
    omega_nh2 = _get_float(gp, "Omega_nh2")
    ln10_value = _get_float(gp, "ln10")

    # Useful for diagnostics, but not strictly required by the integrator.
    omega_ph2 = float(getattr(gp, "Omega_ph2", float("nan")))

    tcmb = getattr(gp, "TCMB", None)
    if tcmb is not None:
        tcmb = float(tcmb)

    return {
        "Neff0": neff0,
        "Omega_ph2": omega_ph2,
        "Omega_nh2": omega_nh2,
        "ln10": ln10_value,
        "TCMB": tcmb,
        "CONSTANT_SOURCE": f"{package}.global_param",
    }


def _fallback_constants():
    """
    Lightweight fallback aligned with the current MCMC convention.

    This avoids making `sagenet_dnnu` unusable when the full upstream
    `global_param.py` dependencies are unavailable.
    """
    neff0 = 3.044
    omega_ph2 = 2.4729e-5
    omega_nh2 = omega_ph2 * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * neff0

    return {
        "Neff0": neff0,
        "Omega_ph2": omega_ph2,
        "Omega_nh2": omega_nh2,
        "ln10": math.log(10.0),
        "TCMB": None,
        "CONSTANT_SOURCE": "standalone_fallback",
    }


try:
    _vals = _load_from_local_global_param()
except Exception as exc:
    _vals = _fallback_constants()
    _vals["CONSTANT_SOURCE"] += f"; local_global_param_failed={exc!r}"


Neff0: float = _vals["Neff0"]
Omega_ph2: float = _vals["Omega_ph2"]
Omega_nh2: float = _vals["Omega_nh2"]
ln10: float = _vals["ln10"]
TCMB = _vals["TCMB"]
CONSTANT_SOURCE: str = _vals["CONSTANT_SOURCE"]


__all__ = [
    "Neff0",
    "Omega_ph2",
    "Omega_nh2",
    "ln10",
    "TCMB",
    "CONSTANT_SOURCE",
]
