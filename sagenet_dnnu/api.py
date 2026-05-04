# -*- coding: utf-8 -*-
"""
Top-level API for the SageNet dnnu integrator.

The simplest entry point is :func:`compute_dnnu`, which takes the SageNet
prediction dictionary (or just ``f`` and ``log10OmegaGW`` arrays) plus
``H0`` and returns ``Delta N_eff`` together with a diagnostics dict.

For users who want the full SageNet prediction step folded in, use
:func:`compute_dnnu_from_predictor`. That function only requires that you
already have a ``GWPredictor`` instance from the ``sagenetgw`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import integrate

from .constants import ln10
from .integrator import InterpLogOmegaPCHIP, adaptive_simpson_interpolated
from .utils import (
    clean_sort_unique,
    g2_to_dnnu,
    maybe_log10f,
    simpson_atol_from_dnnu_tol,
)


# -----------------------------------------------------------------------------
# Defaults — copied from the upstream Cobaya theory; reproducible and
# expressed once here so they can be inspected / overridden.
# -----------------------------------------------------------------------------
DEFAULT_DNNU_TOL_ABS: float = 1e-6
DEFAULT_SIMPSON_RTOL: float = 1e-5
DEFAULT_SIMPSON_MAX_DEPTH: int = 35
DEFAULT_SIMPSON_MAX_EVALS: int = 500_000
DEFAULT_EDGE_TRIM: float = 0.0
DEFAULT_CLAMP_LOG10OMEGA_NONFINITE_TO: float = -300.0
DEFAULT_FALLBACK_TRAPZ_ON_FAIL: bool = True
# Physical guards on the final dnnu. ``None`` keeps the legacy behaviour
# (no guard); the conventional cosmological cut is 5.0 (legacy gate from
# the upstream Cobaya theory).
DEFAULT_REJECT_ABOVE_DNNU: Optional[float] = None
DEFAULT_REJECT_BELOW_DNNU: Optional[float] = None


@dataclass
class IntegratorConfig:
    """Knobs for the dnnu integrator. All fields have sensible defaults.

    Attributes
    ----------
    dnnu_tol_abs : float
        Absolute tolerance on dnnu. The Simpson driver is given an
        atol consistent with this requirement; see
        :func:`sagenet_dnnu.utils.simpson_atol_from_dnnu_tol`.
    simpson_rtol : float
        Relative tolerance for adaptive Simpson.
    simpson_max_depth : int
        Maximum recursion depth for adaptive Simpson.
    simpson_max_evals : int
        Hard cap on integrand evaluations (safety net).
    edge_trim : float
        If > 0, trim the first and last ``edge_trim`` units of the
        log10(f) range before integrating. Useful for debugging boundary
        behaviour. Default 0 (no trim).
    clamp_log10omega_nonfinite_to : float
        Non-finite log10Omega values are clamped to this number so that
        they don't poison the integrand. Default -300, i.e. effectively zero.
    fallback_to_trapz_on_fail : bool
        If Simpson fails to converge (or returns a negative integral),
        fall back to trapezoid integration on the points that Simpson
        already evaluated.
    reject_above_dnnu : float or None
        If not None and the final ``dnnu`` exceeds this value, the result
        is flagged as rejected: ``IntegrationResult.dnnu`` becomes ``nan``
        and ``diagnostics['rejected'] = True`` with
        ``diagnostics['rejected_reason'] = 'dnnu>reject_above_dnnu'``.
        ``g2`` and the raw integration diagnostics are still returned for
        inspection. Default ``None`` (no upper guard). The conventional
        cosmological cut is 5.0 (matches the legacy gate of the upstream
        Cobaya theory).
    reject_below_dnnu : float or None
        Symmetric lower guard. If not None and dnnu falls below this
        value, the result is flagged as rejected. Useful to reject
        spurious negative dnnu when ``fallback_to_trapz_on_fail=False``.
        Default ``None``.
    """

    dnnu_tol_abs: float = DEFAULT_DNNU_TOL_ABS
    simpson_rtol: float = DEFAULT_SIMPSON_RTOL
    simpson_max_depth: int = DEFAULT_SIMPSON_MAX_DEPTH
    simpson_max_evals: int = DEFAULT_SIMPSON_MAX_EVALS
    edge_trim: float = DEFAULT_EDGE_TRIM
    clamp_log10omega_nonfinite_to: float = DEFAULT_CLAMP_LOG10OMEGA_NONFINITE_TO
    fallback_to_trapz_on_fail: bool = DEFAULT_FALLBACK_TRAPZ_ON_FAIL
    reject_above_dnnu: Optional[float] = DEFAULT_REJECT_ABOVE_DNNU
    reject_below_dnnu: Optional[float] = DEFAULT_REJECT_BELOW_DNNU


@dataclass
class IntegrationResult:
    """Return type for :func:`compute_dnnu` / :func:`compute_g2`.

    ``diagnostics`` is a free-form dict that includes:

    * the f-mode detected (``log10`` vs linear Hz)
    * the Simpson atol derived from ``dnnu_tol_abs``
    * the trapz baseline for cross-checking
    * the relative difference between Simpson and trapz
    * which method actually produced the final number (Simpson or trapz fallback)
    * full Simpson convergence diagnostics
    * if a guard was tripped: ``rejected``, ``rejected_reason``,
      ``dnnu_raw`` (the dnnu before the guard masked it to NaN)
    """

    dnnu: float
    g2: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Core: g2 integration
# -----------------------------------------------------------------------------
def compute_g2(
    f_like,
    log10OmegaGW_like,
    *,
    H0: float,
    config: Optional[IntegratorConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute the integral ``g2 = ln(10) * ∫ Omega(x) dx`` with ``x = log10(f)``.

    Parameters
    ----------
    f_like : array_like
        Frequency grid. Either linear Hz or log10(Hz); auto-detected.
    log10OmegaGW_like : array_like
        log10 of the GW energy density spectrum on the same grid.
    H0 : float
        Hubble constant in km/s/Mpc, used only to convert the user-facing
        dnnu tolerance into a raw-integral atol.
    config : IntegratorConfig, optional
        Override defaults.

    Returns
    -------
    g2 : float
    diag : dict

    Notes
    -----
    The dnnu reject guards (``reject_above_dnnu`` / ``reject_below_dnnu``)
    are applied in :func:`compute_dnnu`, NOT here. ``compute_g2`` always
    returns the raw integral.
    """
    cfg = config or IntegratorConfig()

    x0, f_mode = maybe_log10f(f_like)
    x, ylog = clean_sort_unique(
        x0, log10OmegaGW_like,
        clamp_y_nonfinite_to=cfg.clamp_log10omega_nonfinite_to,
    )

    if cfg.edge_trim > 0.0:
        xmin, xmax = float(x.min()), float(x.max())
        keep = (x >= xmin + cfg.edge_trim) & (x <= xmax - cfg.edge_trim)
        if keep.sum() >= 2:
            x = x[keep]
            ylog = ylog[keep]

    omega_native = np.power(10.0, ylog)
    if not np.all(np.isfinite(omega_native)):
        raise ValueError("omega has NaN/Inf after 10**log10OmegaGW.")
    if omega_native.min() < 0:
        raise ValueError("omega negative after 10**log10OmegaGW (should not happen).")

    # Baseline trapz on the native grid (raw variable).
    I_raw_trapz = integrate.trapezoid(y=omega_native, x=x)
    g2_trapz = float(I_raw_trapz * ln10)

    # Translate the dnnu tolerance into a raw-integral atol.
    simpson_atol = simpson_atol_from_dnnu_tol(
        H0=float(H0), dnnu_tol_abs=cfg.dnnu_tol_abs,
    )

    # Adaptive Simpson on PCHIP-interpolated integrand.
    f_interp = InterpLogOmegaPCHIP(x, ylog)
    a, b = float(x[0]), float(x[-1])

    I_raw_simp, simp_diag, eval_points = adaptive_simpson_interpolated(
        f_interp, a, b,
        rtol=cfg.simpson_rtol,
        atol=simpson_atol,
        max_depth=cfg.simpson_max_depth,
        max_evals=cfg.simpson_max_evals,
    )
    g2_simp = float(I_raw_simp * ln10)

    g2_final = g2_simp
    method_final = "simpson_local"
    fallback_used = False

    bad = (
        (not simp_diag["simpson_converged"])
        or (not np.isfinite(g2_simp))
        or (g2_simp < 0.0)
    )
    if bad:
        if cfg.fallback_to_trapz_on_fail:
            # Re-evaluate the interpolated integrand at every point Simpson
            # actually visited and take the trapezoid. This is strictly >=0
            # because the integrand is >=0.
            omega_eval = np.array([f_interp(float(xq)) for xq in eval_points], dtype=float)
            I_raw_tr_ref = integrate.trapezoid(y=omega_eval, x=eval_points)
            g2_final = float(I_raw_tr_ref * ln10)
            method_final = "trapz_refined_fallback"
            fallback_used = True
        else:
            method_final = "simpson_local_no_fallback"

    rel_diff = (
        (g2_final - g2_trapz) / g2_trapz if g2_trapz != 0 else float("nan")
    )

    diag = {
        "f_mode": f_mode,
        "dnnu_tol_abs": float(cfg.dnnu_tol_abs),
        "simpson_atol_raw_from_dnnu": float(simpson_atol),
        "g2_trapz": float(g2_trapz),
        "g2_simpson_local_final": float(g2_final),
        "g2_rel_diff": float(rel_diff),
        "simpson_method_final": method_final,
        "simpson_fallback_used": bool(fallback_used),
        "n_input_points": int(np.asarray(f_like).size),
        "n_clean_points": int(x.size),
    }
    diag.update(simp_diag)
    return float(g2_final), diag


# -----------------------------------------------------------------------------
# Public wrappers
# -----------------------------------------------------------------------------
def _apply_dnnu_guards(
    dnnu_raw: float,
    cfg: IntegratorConfig,
    diag: Dict[str, Any],
) -> float:
    """Apply optional upper / lower guards on dnnu.

    Mutates ``diag`` in place to record the guard outcome and returns the
    guarded dnnu value (``nan`` if any guard tripped, otherwise unchanged).
    """
    rejected = False
    reason = None

    if not np.isfinite(dnnu_raw):
        rejected = True
        reason = "dnnu_nonfinite"
    else:
        if cfg.reject_above_dnnu is not None and dnnu_raw > float(cfg.reject_above_dnnu):
            rejected = True
            reason = f"dnnu>{float(cfg.reject_above_dnnu):g}"
        elif cfg.reject_below_dnnu is not None and dnnu_raw < float(cfg.reject_below_dnnu):
            rejected = True
            reason = f"dnnu<{float(cfg.reject_below_dnnu):g}"

    diag["rejected"] = bool(rejected)
    diag["rejected_reason"] = reason if rejected else ""
    diag["reject_above_dnnu"] = (
        float(cfg.reject_above_dnnu) if cfg.reject_above_dnnu is not None else None
    )
    diag["reject_below_dnnu"] = (
        float(cfg.reject_below_dnnu) if cfg.reject_below_dnnu is not None else None
    )
    diag["dnnu_raw"] = float(dnnu_raw)

    return float("nan") if rejected else float(dnnu_raw)


def compute_dnnu(
    prediction_or_f,
    log10OmegaGW=None,
    *,
    H0: float,
    config: Optional[IntegratorConfig] = None,
) -> IntegrationResult:
    """Compute ``Delta N_eff`` (a.k.a. ``dnnu``) from a SageNet spectrum.

    Two calling conventions are supported.

    1. **Prediction dict** (recommended)::

           pred = predictor.predict({...})
           result = compute_dnnu(pred, H0=67.32)
           print(result.dnnu)

       The dict is expected to expose the keys ``"f"`` and
       ``"log10OmegaGW"``, exactly as returned by
       ``sagenetgw.classes.GWPredictor.predict``.

    2. **Raw arrays**::

           result = compute_dnnu(f, log10OmegaGW, H0=67.32)

    Optional reject guards
    ----------------------
    Pass an :class:`IntegratorConfig` with ``reject_above_dnnu`` (and/or
    ``reject_below_dnnu``) to turn on a physical cut::

        cfg = IntegratorConfig(reject_above_dnnu=5.0)
        r = compute_dnnu(pred, H0=67.32, config=cfg)
        if np.isnan(r.dnnu):
            print("rejected:", r.diagnostics["rejected_reason"])
            print("raw value was:", r.diagnostics["dnnu_raw"])

    When a guard trips, ``IntegrationResult.dnnu`` is set to ``nan`` and
    the diagnostics dict carries the keys ``rejected`` (bool),
    ``rejected_reason`` (str), and ``dnnu_raw`` (the un-masked value).
    ``g2`` and all integration diagnostics are returned unchanged.

    Parameters
    ----------
    prediction_or_f : Mapping or array_like
        Either a SageNet prediction dict, or the ``f`` array.
    log10OmegaGW : array_like, optional
        Required when the first argument is an array.
    H0 : float
        Hubble constant in km/s/Mpc.
    config : IntegratorConfig, optional
        Tuning knobs. See :class:`IntegratorConfig`.

    Returns
    -------
    IntegrationResult
        ``.dnnu``, ``.g2``, and ``.diagnostics``. ``.dnnu`` is ``nan`` if
        a configured reject guard tripped; check
        ``.diagnostics['rejected']``.
    """
    if isinstance(prediction_or_f, Mapping):
        if log10OmegaGW is not None:
            raise TypeError(
                "When the first argument is a prediction dict, "
                "log10OmegaGW must not be passed."
            )
        try:
            f_like = prediction_or_f["f"]
            log_like = prediction_or_f["log10OmegaGW"]
        except KeyError as exc:
            raise KeyError(
                "Prediction dict must contain keys 'f' and 'log10OmegaGW'."
            ) from exc
    else:
        if log10OmegaGW is None:
            raise TypeError(
                "compute_dnnu requires log10OmegaGW when the first "
                "argument is an array."
            )
        f_like = prediction_or_f
        log_like = log10OmegaGW

    cfg = config or IntegratorConfig()

    g2, diag = compute_g2(f_like, log_like, H0=H0, config=cfg)
    dnnu_raw = g2_to_dnnu(g2, H0)
    dnnu = _apply_dnnu_guards(dnnu_raw, cfg, diag)

    return IntegrationResult(dnnu=float(dnnu), g2=float(g2), diagnostics=diag)


def compute_dnnu_from_predictor(
    predictor,
    params: Mapping[str, float],
    *,
    config: Optional[IntegratorConfig] = None,
) -> IntegrationResult:
    """Convenience: run the predictor and integrate in one call.

    ``params`` must contain the key ``"H0"`` along with whatever else
    ``predictor.predict`` requires. Reject guards configured on
    ``IntegratorConfig`` apply here as well.

    Example
    -------
    >>> from sagenetgw.classes import GWPredictor
    >>> from sagenet_dnnu import compute_dnnu_from_predictor, IntegratorConfig
    >>> predictor = GWPredictor(model_type="Transformer", device="cpu")
    >>> cfg = IntegratorConfig(reject_above_dnnu=5.0)   # legacy gate
    >>> result = compute_dnnu_from_predictor(predictor, {
    ...     "r": 3.9585109e-05, "n_t": 1.0116972,
    ...     "kappa10": 110.42477, "T_re": 0.17453859, "DN_re": 39.366618,
    ...     "Omega_bh2": 0.0223828, "Omega_ch2": 0.1201075,
    ...     "H0": 67.32117, "A_s": 2.100549e-9,
    ... }, config=cfg)
    >>> print(result.dnnu, result.diagnostics["rejected"])
    """
    if "H0" not in params:
        raise KeyError("`params` must contain 'H0'.")
    pred = predictor.predict(dict(params))
    return compute_dnnu(pred, H0=float(params["H0"]), config=config)


__all__ = [
    "DEFAULT_DNNU_TOL_ABS",
    "DEFAULT_SIMPSON_RTOL",
    "DEFAULT_SIMPSON_MAX_DEPTH",
    "DEFAULT_SIMPSON_MAX_EVALS",
    "DEFAULT_EDGE_TRIM",
    "DEFAULT_CLAMP_LOG10OMEGA_NONFINITE_TO",
    "DEFAULT_FALLBACK_TRAPZ_ON_FAIL",
    "DEFAULT_REJECT_ABOVE_DNNU",
    "DEFAULT_REJECT_BELOW_DNNU",
    "IntegratorConfig",
    "IntegrationResult",
    "compute_g2",
    "compute_dnnu",
    "compute_dnnu_from_predictor",
]
