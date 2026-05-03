# -*- coding: utf-8 -*-
"""
Helpers for the dnnu integrator.

Contents
--------
* ``maybe_log10f``       : auto-detect whether the SageNet ``f`` array is
                           already log10(f) or linear f in Hz.
* ``clean_sort_unique``  : drop non-finite x, clamp non-finite y, sort and
                           unique-ify x while keeping y aligned.
* ``simpson_atol_from_dnnu_tol`` : convert a user-facing absolute tolerance
                           on dnnu into the raw integral atol that the
                           adaptive Simpson driver expects.
* ``g2_to_dnnu``         : final unit conversion g2 -> Delta N_eff.

All functions are pure and have no global state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import Neff0, Omega_nh2, ln10


def maybe_log10f(f) -> Tuple[np.ndarray, str]:
    """Auto-detect whether ``f`` is already log10(f) or linear f in Hz.

    The SageNet predictor in some versions returns the frequency grid
    directly in Hz, in others as log10(f). Down-stream integration is
    always done in log10-space, so we normalise the input here.

    Returns
    -------
    log10f : np.ndarray
        Frequency grid expressed in log10(Hz).
    mode : str
        Tag describing how the conversion was decided. Useful for
        diagnostic logs / regression tests.
    """
    f = np.asarray(f, dtype=float)
    if f.size == 0:
        return f, "empty"

    fmin = np.nanmin(f)
    fmax = np.nanmax(f)

    # Non-finite extrema can't be linear Hz; assume already log10.
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        return f, "assume_log10f_nonfinite_seen"
    if fmin <= 0:
        return f, "assume_log10f_nonpositive_seen"

    # Heuristic: log10(f) for SGWB lives in roughly [-20, +12], and is
    # comfortably within (-200, 200). Linear f in Hz would normally
    # exceed 10 by far at the upper end.
    if (-200.0 < fmin < 200.0) and (-200.0 < fmax < 200.0):
        if (fmin < 0) or (fmax < 10):
            return f, "assume_log10f_by_range"

    return np.log10(f), "converted_linear_f_to_log10"


def clean_sort_unique(
    x_in,
    y_in,
    *,
    clamp_y_nonfinite_to: float = -300.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean ``x``, keep ``y`` aligned by index.

    * Rows with non-finite ``x`` are dropped.
    * Non-finite ``y`` values are clamped to ``clamp_y_nonfinite_to``
      (a tiny log10Omega), not dropped, so we don't accidentally
      throw away the high-frequency tail of the spectrum.
    * The result is sorted in ``x`` and unique-ified (first occurrence
      kept on duplicates).

    Raises
    ------
    ValueError
        If fewer than two valid samples survive, or if x is not strictly
        increasing after deduplication (defensive check, should never trip).
    """
    x = np.asarray(x_in, dtype=float)
    y = np.asarray(y_in, dtype=float)

    mx = np.isfinite(x)
    x = x[mx]
    y = y[mx]

    y = np.where(np.isfinite(y), y, float(clamp_y_nonfinite_to))

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x, idx = np.unique(x, return_index=True)
    y = y[idx]

    if x.size < 2:
        raise ValueError("Not enough valid x points after cleaning (need >=2).")

    dx = np.diff(x)
    if not np.all(dx > 0):
        raise ValueError(
            f"x not strictly increasing after unique/sort. min(dx)={dx.min()}"
        )

    return x, y


def simpson_atol_from_dnnu_tol(H0: float, dnnu_tol_abs: float) -> float:
    """Convert an absolute tolerance on dnnu into a raw-integral atol.

    The chain of unit conversions is::

        dnnu = Neff0 * g2 / Omega_nu
            => g2_atol = dnnu_tol_abs * Omega_nu / Neff0
        g2 = ln10 * I_raw
            => I_raw_atol = g2_atol / ln10

    where ``I_raw = ∫ Omega(x) dx`` with ``x = log10(f)``.

    This is the heart of the dynamic-tolerance idea: the integration
    precision is *physics-driven* (we ask for dnnu to e.g. 1e-6) rather
    than driven by an arbitrary number of grid points.
    """
    h = float(H0) / 100.0
    Omega_nu_val = Omega_nh2 / (h * h)
    g2_atol = float(dnnu_tol_abs) * (Omega_nu_val / Neff0)
    return float(g2_atol / ln10)


def g2_to_dnnu(g2: float, H0: float) -> float:
    """Convert the integral ``g2`` to Delta N_eff (``dnnu``)."""
    h = float(H0) / 100.0
    Omega_nu_val = Omega_nh2 / (h * h)
    return float(Neff0 * float(g2) / Omega_nu_val)


__all__ = [
    "maybe_log10f",
    "clean_sort_unique",
    "simpson_atol_from_dnnu_tol",
    "g2_to_dnnu",
]
