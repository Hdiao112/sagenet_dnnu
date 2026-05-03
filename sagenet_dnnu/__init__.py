# -*- coding: utf-8 -*-
"""
sagenet_dnnu
============

A drop-in companion package for `SageNet <https://github.com/...>`_ that
turns a predicted SGWB spectrum into ``Delta N_eff`` (``dnnu``) using
PCHIP-interpolated adaptive Simpson integration with physics-driven
dynamic tolerance.

Quick start::

    from sagenetgw.classes import GWPredictor
    from sagenet_dnnu import compute_dnnu_from_predictor

    predictor = GWPredictor(model_type="Transformer", device="cpu")
    result = compute_dnnu_from_predictor(predictor, {
        "r": 3.9585109e-05, "n_t": 1.0116972,
        "kappa10": 110.42477, "T_re": 0.17453859, "DN_re": 39.366618,
        "Omega_bh2": 0.0223828, "Omega_ch2": 0.1201075,
        "H0": 67.32117, "A_s": 2.100549e-9,
    })
    print(result.dnnu, result.diagnostics["simpson_method_final"])

Or, integrate a spectrum you already have::

    from sagenet_dnnu import compute_dnnu
    result = compute_dnnu(prediction_dict, H0=67.32)

"""

from .api import (
    DEFAULT_CLAMP_LOG10OMEGA_NONFINITE_TO,
    DEFAULT_DNNU_TOL_ABS,
    DEFAULT_EDGE_TRIM,
    DEFAULT_FALLBACK_TRAPZ_ON_FAIL,
    DEFAULT_SIMPSON_MAX_DEPTH,
    DEFAULT_SIMPSON_MAX_EVALS,
    DEFAULT_SIMPSON_RTOL,
    IntegrationResult,
    IntegratorConfig,
    compute_dnnu,
    compute_dnnu_from_predictor,
    compute_g2,
)
from .constants import Neff0, Omega_nh2, ln10
from .integrator import InterpLogOmegaPCHIP, adaptive_simpson_interpolated
from .utils import (
    clean_sort_unique,
    g2_to_dnnu,
    maybe_log10f,
    simpson_atol_from_dnnu_tol,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Top-level API
    "compute_dnnu",
    "compute_dnnu_from_predictor",
    "compute_g2",
    "IntegratorConfig",
    "IntegrationResult",
    # Defaults
    "DEFAULT_DNNU_TOL_ABS",
    "DEFAULT_SIMPSON_RTOL",
    "DEFAULT_SIMPSON_MAX_DEPTH",
    "DEFAULT_SIMPSON_MAX_EVALS",
    "DEFAULT_EDGE_TRIM",
    "DEFAULT_CLAMP_LOG10OMEGA_NONFINITE_TO",
    "DEFAULT_FALLBACK_TRAPZ_ON_FAIL",
    # Lower-level helpers
    "InterpLogOmegaPCHIP",
    "adaptive_simpson_interpolated",
    "maybe_log10f",
    "clean_sort_unique",
    "simpson_atol_from_dnnu_tol",
    "g2_to_dnnu",
    # Constants
    "Neff0",
    "Omega_nh2",
    "ln10",
]
