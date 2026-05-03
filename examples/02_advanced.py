#!/usr/bin/env python3
"""Advanced usage examples.

1. Run the predictor + integrator in one call.
2. Tune the integrator with a custom IntegratorConfig.
3. Sweep a single parameter and watch dnnu respond.
"""

import numpy as np

from sagenetgw.classes import GWPredictor
from sagenet_dnnu import (
    IntegratorConfig,
    compute_dnnu,
    compute_dnnu_from_predictor,
)


BASE_PARAMS = {
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


def example_one_shot():
    print("=" * 60)
    print("Example 1: one-shot predict + integrate")
    print("=" * 60)

    predictor = GWPredictor(model_type="Transformer", device="cpu")
    result = compute_dnnu_from_predictor(predictor, BASE_PARAMS)
    print(f"  dnnu = {result.dnnu:.6e}")
    print(f"  g2   = {result.g2:.6e}\n")
    return predictor


def example_custom_config(predictor):
    print("=" * 60)
    print("Example 2: custom IntegratorConfig")
    print("=" * 60)

    # Tighter integration (slower, more accurate).
    cfg_tight = IntegratorConfig(
        dnnu_tol_abs=1e-9,
        simpson_rtol=1e-7,
        simpson_max_depth=40,
    )

    # Looser integration (faster, less accurate).
    cfg_loose = IntegratorConfig(
        dnnu_tol_abs=1e-4,
        simpson_rtol=1e-3,
    )

    pred = predictor.predict(BASE_PARAMS)

    r_tight = compute_dnnu(pred, H0=BASE_PARAMS["H0"], config=cfg_tight)
    r_loose = compute_dnnu(pred, H0=BASE_PARAMS["H0"], config=cfg_loose)

    print(f"  tight:  dnnu = {r_tight.dnnu:.10e}  evals = {r_tight.diagnostics['simpson_eval_count']}")
    print(f"  loose:  dnnu = {r_loose.dnnu:.10e}  evals = {r_loose.diagnostics['simpson_eval_count']}\n")


def example_parameter_sweep(predictor):
    print("=" * 60)
    print("Example 3: r-sweep, watch dnnu vs r")
    print("=" * 60)

    rs = np.logspace(-6, -3, 7)
    print(f"  {'r':>12s}    {'dnnu':>14s}    method")
    for r in rs:
        params = dict(BASE_PARAMS)
        params["r"] = float(r)
        result = compute_dnnu_from_predictor(predictor, params)
        print(
            f"  {r:>12.3e}    {result.dnnu:>14.6e}    "
            f"{result.diagnostics['simpson_method_final']}"
        )


def main():
    predictor = example_one_shot()
    example_custom_config(predictor)
    example_parameter_sweep(predictor)


if __name__ == "__main__":
    main()
