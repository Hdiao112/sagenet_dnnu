# -*- coding: utf-8 -*-
"""
Adaptive Simpson with PCHIP-interpolated integrand in log10-Omega domain.

Why PCHIP in log domain?
------------------------
The SGWB spectrum from SageNet covers many orders of magnitude in
``Omega_GW`` and is non-uniformly sampled (the predictor inserts extra
points around peaks). Plain Simpson on this raw grid can produce
*negative* integral values when adjacent intervals span a sharp peak,
because the parabolic fit overshoots and goes below zero between samples.

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is shape-preserving:
it never overshoots and preserves monotonicity within each interval. Doing
the interpolation in ``log10(Omega)`` and exponentiating before integrating
guarantees the integrand is strictly positive, and re-establishes a
geometrically smooth curve that Simpson can integrate without artefacts.

The adaptive driver below splits intervals only where the local Simpson
error estimate exceeds the tolerance, so dense sampling is automatic in
peak regions and sparse elsewhere.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator


class InterpLogOmegaPCHIP:
    """PCHIP interpolator for ``Omega(x)`` operating in log10-Omega.

    Parameters
    ----------
    x : array
        Strictly increasing 1D array of log10(f) values.
    ylog : array
        log10(Omega_GW) values matching ``x``.

    Notes
    -----
    Queries outside ``[x[0], x[-1]]`` are clamped to the boundary value
    rather than extrapolated. Extrapolation of a PCHIP fit at the edges
    of an SGWB spectrum is meaningless and would silently bias the integral.
    """

    def __init__(self, x: np.ndarray, ylog: np.ndarray):
        self.x = np.asarray(x, dtype=float)
        self.ylog = np.asarray(ylog, dtype=float)

        self._x_min = float(self.x[0])
        self._x_max = float(self.x[-1])

        self._pchip = PchipInterpolator(self.x, self.ylog, extrapolate=False)
        self._y_left = float(self.ylog[0])
        self._y_right = float(self.ylog[-1])

    def __call__(self, xq: float) -> float:
        if xq <= self._x_min:
            yq = self._y_left
        elif xq >= self._x_max:
            yq = self._y_right
        else:
            yq = float(self._pchip(xq))
        return float(10.0 ** yq)


def _simpson_interval(fa: float, fm: float, fb: float, a: float, b: float) -> float:
    """Single-interval Simpson estimate ``(b-a)/6 * (f(a) + 4 f(m) + f(b))``."""
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0


def adaptive_simpson_interpolated(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    rtol: float,
    atol: float,
    max_depth: int,
    max_evals: int,
) -> Tuple[float, Dict[str, Any], np.ndarray]:
    """Adaptive Simpson with local error control by interval splitting.

    Acceptance rule for an interval ``[a_i, b_i]``::

        S  = simpson(a_i, b_i)
        S2 = simpson(a_i, m_i) + simpson(m_i, b_i)
        err ~= |S2 - S| / 15
        accept if  err <= atol + rtol * |S2|

    Returns
    -------
    integral : float
        The integral estimate.
    diagnostics : dict
        Convergence flag, error estimate, eval count, accepted/split
        interval counts, and the depth actually reached.
    eval_points : np.ndarray
        Sorted, deduplicated array of the abscissas at which the
        integrand was evaluated. Useful for the trapz fallback.
    """
    m = 0.5 * (a + b)
    fa = f(a)
    fm = f(m)
    fb = f(b)

    eval_points = [a, m, b]
    eval_count = 3

    S = _simpson_interval(fa, fm, fb, a, b)
    stack = [(a, b, fa, fm, fb, S, 0)]  # (a, b, fa, fm, fb, S, depth)

    total = 0.0
    err_est_total = 0.0
    accepted = 0
    split = 0
    max_depth_used = 0
    converged = True

    while stack:
        a_i, b_i, fa_i, fm_i, fb_i, S_i, depth = stack.pop()
        max_depth_used = max(max_depth_used, depth)

        m_i = 0.5 * (a_i + b_i)
        lm = 0.5 * (a_i + m_i)
        rm = 0.5 * (m_i + b_i)

        flm = f(lm)
        frm = f(rm)
        eval_points.extend([lm, rm])
        eval_count += 2

        if eval_count > max_evals:
            converged = False
            break

        S_left = _simpson_interval(fa_i, flm, fm_i, a_i, m_i)
        S_right = _simpson_interval(fm_i, frm, fb_i, m_i, b_i)
        S2 = S_left + S_right

        err = abs(S2 - S_i) / 15.0
        tol = float(atol) + float(rtol) * abs(S2)

        if (err <= tol) or (depth >= max_depth):
            total += S2
            err_est_total += err
            accepted += 1
            if depth >= max_depth and err > tol:
                converged = False
        else:
            split += 1
            if depth + 1 > max_depth:
                # Defensive: shouldn't trigger because of the check above,
                # but keep it for safety.
                converged = False
                total += S2
                err_est_total += err
                accepted += 1
                continue

            # LIFO: push right then left so left is processed first.
            stack.append((m_i, b_i, fm_i, frm, fb_i, S_right, depth + 1))
            stack.append((a_i, m_i, fa_i, flm, fm_i, S_left, depth + 1))

    diag = {
        "simpson_converged": bool(converged),
        "simpson_rtol": float(rtol),
        "simpson_atol": float(atol),
        "simpson_max_depth": int(max_depth),
        "simpson_max_depth_used": int(max_depth_used),
        "simpson_max_evals": int(max_evals),
        "simpson_eval_count": int(eval_count),
        "simpson_accepted_intervals": int(accepted),
        "simpson_split_events": int(split),
        "simpson_err_est": float(err_est_total),
    }

    eval_points_arr = np.unique(np.asarray(eval_points, dtype=float))
    return float(total), diag, eval_points_arr


__all__ = [
    "InterpLogOmegaPCHIP",
    "adaptive_simpson_interpolated",
]
