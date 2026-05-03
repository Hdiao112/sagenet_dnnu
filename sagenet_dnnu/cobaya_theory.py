# -*- coding: utf-8 -*-
"""
Optional Cobaya `Theory` adapter.

Keeps the gate logic from the upstream pipeline (legacy dnnu>5 cut,
hard/soft guard modes for parth/alterbbn backends) so existing Cobaya
configs keep working unchanged. Importing this module requires
``cobaya`` and ``sagenetgw`` to be installed.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from cobaya.theory import Theory  # type: ignore
except ImportError as exc:  # pragma: no cover - import-time guard only
    raise ImportError(
        "sagenet_dnnu.cobaya_theory requires `cobaya`. "
        "Install it with `pip install cobaya`."
    ) from exc

try:
    from sagenetgw.classes import GWPredictor  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sagenet_dnnu.cobaya_theory requires `sagenetgw`. "
        "Install SageNet first."
    ) from exc

from .api import IntegratorConfig, compute_g2
from .utils import g2_to_dnnu


# -----------------------------------------------------------------------------
# Guard defaults (mirror upstream sagenet_final.py)
# -----------------------------------------------------------------------------
DEFAULT_DEVICE = "cuda"
DEFAULT_BACKEND = "parth"
DEFAULT_GUARD_MODE = "hard"
DEFAULT_DNNU_MAX_LEGACY = 5.0
DEFAULT_DNNU_MAX_STRICT = 1.0
DEFAULT_ETA10_MAX_HARD = 9.0


def _eta10_from_omegabh2(omegabh2: float) -> float:
    """Mapping used by the upstream pipeline:
    ``eta10 = 273.3036 * Omega_b h^2 * (1 + 7.16958e-3 * 0.25)``.
    """
    return 273.3036 * float(omegabh2) * (1.0 + 7.16958e-3 * 0.25)


class SageNetDnnuTheory(Theory):
    """Cobaya `Theory` providing ``dn_nu`` and ``kappa`` as derived params.

    Returning ``False`` from ``calculate`` flags the point as physics-rejected
    (posterior = -inf), so downstream BBN evaluations are short-circuited.

    Configuration via environment variables (all optional)::

        SAGENET_DEVICE                  cuda | cpu      (default: cuda)
        SAGENET_BACKEND                 parth | alterbbn(default: parth)
        SAGENET_GUARD_MODE              hard | soft     (default: hard)
        SAGENET_DNNU_TOL_ABS            float           (default: 1e-6)
        SAGENET_SIMPSON_RTOL            float           (default: 1e-5)
        SAGENET_SIMPSON_MAX_DEPTH       int             (default: 35)
        SAGENET_SIMPSON_MAX_EVALS       int             (default: 500000)
        SAGENET_EDGE_TRIM               float           (default: 0.0)
        SAGENET_CLAMP_LOG10OMEGA_NONFINITE_TO  float    (default: -300.0)
        SAGENET_FALLBACK_TRAPZ_ON_FAIL  0 | 1           (default: 1)
        SAGENET_DNNU_MAX_LEGACY         float           (default: 5.0)
        SAGENET_DNNU_MAX_STRICT         float           (default: 1.0)
        SAGENET_ETA10_MAX_HARD          float           (default: 9.0)
    """

    params = {
        "dn_nu": {"derived": True},
        "kappa": {"derived": True},
    }

    def initialize(self):
        print("\n>>> [SageNetDnnuTheory] Initializing...")

        self.device = os.environ.get("SAGENET_DEVICE", DEFAULT_DEVICE).strip()
        print(f">>> [SageNetDnnuTheory] device={self.device}")

        self.sagenet = GWPredictor(model_type="Transformer", device=self.device)
        print(">>> [SageNetDnnuTheory] Model loaded.")

        self.config = IntegratorConfig(
            dnnu_tol_abs=float(os.environ.get("SAGENET_DNNU_TOL_ABS", 1e-6)),
            simpson_rtol=float(os.environ.get("SAGENET_SIMPSON_RTOL", 1e-5)),
            simpson_max_depth=int(os.environ.get("SAGENET_SIMPSON_MAX_DEPTH", 35)),
            simpson_max_evals=int(os.environ.get("SAGENET_SIMPSON_MAX_EVALS", 500_000)),
            edge_trim=float(os.environ.get("SAGENET_EDGE_TRIM", 0.0)),
            clamp_log10omega_nonfinite_to=float(
                os.environ.get("SAGENET_CLAMP_LOG10OMEGA_NONFINITE_TO", -300.0)
            ),
            fallback_to_trapz_on_fail=bool(
                int(os.environ.get("SAGENET_FALLBACK_TRAPZ_ON_FAIL", 1))
            ),
        )

        self.backend = os.environ.get("SAGENET_BACKEND", DEFAULT_BACKEND).strip().lower()
        self.guard_mode = os.environ.get("SAGENET_GUARD_MODE", DEFAULT_GUARD_MODE).strip().lower()
        self.dnnu_max_legacy = float(os.environ.get("SAGENET_DNNU_MAX_LEGACY", DEFAULT_DNNU_MAX_LEGACY))
        self.dnnu_max_strict = float(os.environ.get("SAGENET_DNNU_MAX_STRICT", DEFAULT_DNNU_MAX_STRICT))
        self.eta10_max_hard = float(os.environ.get("SAGENET_ETA10_MAX_HARD", DEFAULT_ETA10_MAX_HARD))

        if self.backend not in ("parth", "alterbbn"):
            raise ValueError(f"Invalid backend='{self.backend}'.")
        if self.guard_mode not in ("hard", "soft"):
            raise ValueError(f"Invalid guard_mode='{self.guard_mode}'.")

        print(
            ">>> [SageNetDnnuTheory] Active guard config:\n"
            f"    backend={self.backend}\n"
            f"    guard_mode={self.guard_mode}\n"
            f"    dnnu_max_legacy={self.dnnu_max_legacy}\n"
            f"    dnnu_max_strict={self.dnnu_max_strict}\n"
            f"    eta10_max_hard={self.eta10_max_hard}\n"
        )

    def get_requirements(self):
        return {
            "log10r": None,
            "n_t": None,
            "log10kappa10": None,
            "log10Tre": None,
            "DN_re": None,
            "omegabh2": None,
            "omegach2": None,
            "H0": None,
            "ln10_10As": None,
        }

    def get_can_provide(self):
        return ["dn_nu", "kappa"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        try:
            input_params = {
                "r":         10 ** float(params_values_dict["log10r"]),
                "n_t":       float(params_values_dict["n_t"]),
                "kappa10":   10 ** float(params_values_dict["log10kappa10"]),
                "T_re":      10 ** float(params_values_dict["log10Tre"]),
                "DN_re":     float(params_values_dict["DN_re"]),
                "Omega_bh2": float(params_values_dict["omegabh2"]),
                "Omega_ch2": float(params_values_dict["omegach2"]),
                "H0":        float(params_values_dict["H0"]),
                "A_s":       float(np.exp(float(params_values_dict["ln10_10As"])) * 1e-10),
            }

            pred = self.sagenet.predict(input_params)
            f_pred = np.asarray(pred["f"], dtype=float)
            logOm_pred = np.asarray(pred["log10OmegaGW"], dtype=float)

            g2_final, _diag = compute_g2(
                f_pred, logOm_pred,
                H0=input_params["H0"],
                config=self.config,
            )
            dnnu = g2_to_dnnu(g2_final, input_params["H0"])

            # Legacy gate: dnnu > legacy => reject.
            if (not np.isfinite(dnnu)) or (float(dnnu) > self.dnnu_max_legacy):
                return False

            # Mode-specific gate.
            if self.guard_mode == "hard":
                if float(dnnu) > self.dnnu_max_strict:
                    return False
                if self.backend == "parth":
                    eta10 = _eta10_from_omegabh2(input_params["Omega_bh2"])
                    if (not np.isfinite(eta10)) or (float(eta10) > self.eta10_max_hard):
                        return False
            else:  # soft
                if self.backend == "parth":
                    if float(dnnu) > self.dnnu_max_strict:
                        return False
                # soft + alterbbn: only legacy gate applies.

            if want_derived:
                state.setdefault("derived", {})
                state["derived"]["kappa"] = float(input_params["kappa10"])
                state["derived"]["dn_nu"] = float(dnnu)

            return True

        except Exception as e:
            print(f">>> [ERROR] SageNetDnnuTheory.calculate failed: {e!r}")
            raise


__all__ = ["SageNetDnnuTheory"]
