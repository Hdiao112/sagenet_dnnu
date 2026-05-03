#!/usr/bin/env python3
"""Minimal usage example: predict + integrate + plot.

Mirrors the snippet you sent, with one extra line that turns the
predicted spectrum into Delta N_eff via the dnnu integrator.
"""

import numpy as np
import matplotlib.pyplot as plt

from sagenetgw.classes import GWPredictor
from sagenet_dnnu import compute_dnnu


def main():
    predictor = GWPredictor(model_type="Transformer", device="cpu")

    params = {
        "r":         3.9585109e-05,
        "n_t":       1.0116972,
        "kappa10":   1.42477,
        "T_re":      0.17453859,
        "DN_re":     39.366618,
        "Omega_bh2": 0.0223828,
        "Omega_ch2": 0.1201075,
        "H0":        67.32117,
        "A_s":       2.100549e-9,
    }

    prediction = predictor.predict(params)

    # Integrate the SGWB spectrum to get Delta N_eff.
    result = compute_dnnu(prediction, H0=params["H0"])

    print(f"dnnu (Delta N_eff)      = {result.dnnu:.6e}")
    print(f"g2                      = {result.g2:.6e}")
    print(f"method                  = {result.diagnostics['simpson_method_final']}")
    print(f"simpson_converged       = {result.diagnostics['simpson_converged']}")
    print(f"f_mode (auto-detect)    = {result.diagnostics['f_mode']}")
    print(f"trapz baseline g2       = {result.diagnostics['g2_trapz']:.6e}")
    print(f"rel diff Simpson-trapz  = {result.diagnostics['g2_rel_diff']:.3e}")

    pred_coords = np.column_stack((prediction["f"], prediction["log10OmegaGW"]))
    plt.plot(pred_coords[:, 0], pred_coords[:, 1], "--", color="royalblue", marker=".")
    plt.xlabel("log10(f / Hz)")
    plt.ylabel(r"$\log_{10} \Omega_{\rm GW}$")
    plt.title(f"SageNet SGWB — dnnu = {result.dnnu:.3e}")
    plt.tight_layout()
    plt.savefig("sgwb_with_dnnu.png", dpi=120)
    print("\nSaved figure: sgwb_with_dnnu.png")


if __name__ == "__main__":
    main()
