# sagenet-dnnu

> **PCHIP-interpolated adaptive Simpson integrator for SageNet SGWB spectra.**
> Turns a `GWPredictor` prediction into ΔNₑff (`dnnu`) — robustly, in one line.

[![tests](https://github.com/your-org/sagenet-dnnu/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/sagenet-dnnu/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What this is

[SageNet](https://github.com/) (`sagenetgw.classes.GWPredictor`) predicts a
stochastic gravitational-wave background (SGWB) spectrum
`(f, log10ΩGW(f))`. To use that spectrum in a BBN / Cobaya pipeline you
typically need to integrate it into a single number — the extra effective
relativistic species `ΔNₑff`, here called **`dnnu`**.

Naively, that integral is

$$
\Delta N_\mathrm{eff} \;=\; \frac{N_\mathrm{eff}^{(0)}}{\Omega_\nu h^2 / h^2}\;
\ln 10 \int \Omega_\mathrm{GW}(\log_{10} f)\, \mathrm{d}(\log_{10} f).
$$

In practice running plain `scipy.integrate.simpson` or `numpy.trapz` on the
raw SageNet output is fragile: SageNet returns a non-uniform grid with
extra samples around peaks, the spectrum spans many decades in `ΩGW`, and
Simpson's parabolic fit can **go negative** between samples on a sharp peak
(see *Why this exists* below).

This package implements the integration recipe used in production by the
upstream Cobaya theory in `sagenet_final.py`, and ships it as a clean,
standalone, testable Python module:

- **PCHIP** (shape-preserving cubic Hermite) interpolation in `log10Ω`-domain — never overshoots, never produces negative values.
- **Adaptive Simpson** with local-error control — refines automatically wherever the spectrum is curved.
- **Physics-driven dynamic tolerance** — you specify the absolute tolerance you want on `dnnu`; the integrator computes the corresponding raw-integral atol (which depends on `H₀`).
- **Trapezoid fallback** — if Simpson fails to converge, the result is recomputed by trapezoid on every point Simpson actually visited. Strictly non-negative.
- **`f`-mode auto-detection** — feed it `f` in Hz or already in `log10(f)`; it figures out which.

Everything is implemented in pure NumPy + SciPy. No GPU, no `cobaya`, no `sagenetgw` required for the integrator itself.

---

## Install

```bash
pip install sagenet-dnnu
```

Or, from a clone of this repo:

```bash
git clone https://github.com/your-org/sagenet-dnnu.git
cd sagenet-dnnu
pip install -e .
```

Optional extras:

| Extra      | Adds                              | Use when                                       |
| ---------- | --------------------------------- | ---------------------------------------------- |
| `[cobaya]` | `cobaya`                          | you want to use `SageNetDnnuTheory` in a Cobaya run. |
| `[test]`   | `pytest`                          | you want to run the test suite.                |
| `[examples]` | `matplotlib`                    | you want to run the plotting example.          |
| `[all]`    | all of the above                  | one-liner install for development.             |

```bash
pip install "sagenet-dnnu[all]"
```

> SageNet itself (`sagenetgw`) is **not** a dependency of this package. You only need it if you want to actually call `predictor.predict(...)`. The integrator works on any `(f, log10ΩGW)` arrays.

---

## Quick start

The snippet below is exactly the SageNet example you already have, with one extra line at the end:

```python
from sagenetgw.classes import GWPredictor
import numpy as np
from matplotlib import pyplot as plt

from sagenet_dnnu import compute_dnnu          # <-- new

predictor = GWPredictor(model_type="Transformer", device="cpu")
prediction = predictor.predict({
    "r":         3.9585109e-05,
    "n_t":       1.0116972,
    "kappa10":   110.42477,
    "T_re":      0.17453859,
    "DN_re":     39.366618,
    "Omega_bh2": 0.0223828,
    "Omega_ch2": 0.1201075,
    "H0":        67.32117,
    "A_s":       2.100549e-9,
})

# ---- new: integrate the spectrum into ΔN_eff ----
result = compute_dnnu(prediction, H0=67.32117)
print("dnnu =", result.dnnu)
print("method:", result.diagnostics["simpson_method_final"])
# -------------------------------------------------

pred_coords = np.column_stack((prediction["f"], prediction["log10OmegaGW"]))
plt.plot(pred_coords[:, 0], pred_coords[:, 1], "--",
         color="royalblue", marker=".")
```

That's it. `result` is an `IntegrationResult` with three fields:

```python
result.dnnu          # ΔN_eff (the number you usually want)
result.g2            # the raw integral g2 = ln(10) * ∫ Ω d(log10 f)
result.diagnostics   # dict with f-mode, atol used, trapz baseline,
                     # convergence flag, eval count, ...
```

### One-shot: predict + integrate

```python
from sagenet_dnnu import compute_dnnu_from_predictor

result = compute_dnnu_from_predictor(predictor, {
    "r": ..., "n_t": ..., "kappa10": ...,
    "H0": 67.32117,    # required key
    ...
})
```

### Raw arrays, without a SageNet prediction dict

```python
result = compute_dnnu(f_array, log10OmegaGW_array, H0=67.32)
```

`f_array` may be either linear Hz (e.g. `1e-12 .. 1e8`) or already log10
(e.g. `-12 .. 8`). The integrator detects this from the value range; the
detected mode is reported in `result.diagnostics["f_mode"]`.

---

## Why this exists — algorithm in 1 minute

The upstream Cobaya theory shipped a hand-rolled integrator inside
`sagenet_final.py`. This repo extracts that integrator, documents it,
tests it, and packages it. The choices it makes are not arbitrary; they
fix concrete numerical failure modes.

### Failure mode 1: Simpson gives negative integrals

Simpson fits a parabola through three samples. On a non-uniform grid that
straddles a sharp SGWB peak, the parabola can dip below zero between
samples, so the *integral* of a strictly positive spectrum comes out
negative. Empirically this happens on a substantial fraction of SageNet
outputs (`native_neg_count: 11721 / 13184` rows in one validation set).

**Fix:** interpolate `log10ΩGW` with PCHIP (shape-preserving — provably
no overshoot), exponentiate, and integrate that. PCHIP guarantees the
integrand is `> 0` everywhere, so Simpson on the interpolated curve is
also positive. The same validation set drops to `Inter_neg_count: 0`.

### Failure mode 2: fixed tolerances are wrong by orders of magnitude

A point in parameter space with `H₀ = 60` and another with `H₀ = 80`
need different raw-integral tolerances to hit the same precision on
`dnnu`. Hard-coding a Simpson `atol` either over-spends compute on easy
points or under-resolves hard ones.

**Fix:** the user specifies `dnnu_tol_abs` (absolute tolerance on the
final `ΔNₑff`). The integrator translates that into the equivalent
raw-integral atol *for the current* `H₀`, every call:

```
g2_atol     = dnnu_tol_abs * (Ω_ν h² / h²) / N_eff⁰
I_raw_atol  = g2_atol / ln 10
```

### Failure mode 3: Simpson refuses to converge

Pathological spectra (`max_evals` exceeded, or `S < 0` survives despite
PCHIP) trigger a fallback: trapezoid integration over **the points
Simpson already evaluated**. This is monotonic in the integrand and
strictly non-negative, so it always returns a sane answer.

### Validation against numerical ODE backend

From the upstream error analysis (`Sagenet_dnnu vs Numerical dnnu`):

```
Comparison vs PartheNoPE numerical backend, positive-dnnu sample (10k):
                Simpson (PCHIP)   Trapz (raw)     Winner
  H2/H          1.14e-04          1.15e-04        Simpson
  Y_p           4.81e-05          4.85e-05        Simpson
  He3/H         4.13e-05          4.17e-05        Simpson
  Li7/H         8.88e-05          8.98e-05        Simpson

Negative-dnnu sample (i.e. raw Simpson would have gone negative):
  H2/H          1.48e-03          1.52e-03        Simpson (PCHIP)
  Y_p           4.82e-04          4.95e-04        Simpson (PCHIP)
  He3/H         7.06e-04          7.29e-04        Simpson (PCHIP)
  Li7/H         1.30e-03          1.33e-03        Simpson (PCHIP)
```

Errors are below the BBN observational uncertainties for D/H, Yp, ³He/H,
⁷Li/H in essentially all sampled configurations.

---

## Tuning: `IntegratorConfig`

Defaults match the upstream pipeline byte-for-byte. Override only if you
know what you're doing.

```python
from sagenet_dnnu import IntegratorConfig, compute_dnnu

cfg = IntegratorConfig(
    dnnu_tol_abs=1e-7,         # tighter -> more compute, more precision
    simpson_rtol=1e-6,
    simpson_max_depth=40,
    simpson_max_evals=1_000_000,
    edge_trim=0.0,             # trim x-range by this many decades on each end
    clamp_log10omega_nonfinite_to=-300.0,
    fallback_to_trapz_on_fail=True,
)

result = compute_dnnu(prediction, H0=67.32, config=cfg)
```

| Field | Default | Meaning |
| --- | --- | --- |
| `dnnu_tol_abs` | `1e-6` | Absolute tolerance on the final `dnnu`. The Simpson `atol` is derived from this. |
| `simpson_rtol` | `1e-5` | Relative tolerance for adaptive Simpson. |
| `simpson_max_depth` | `35` | Max recursion depth. |
| `simpson_max_evals` | `500 000` | Hard cap on integrand evaluations. |
| `edge_trim` | `0.0` | Drop `edge_trim` units of `log10(f)` from each end before integrating. Diagnostic only. |
| `clamp_log10omega_nonfinite_to` | `-300.0` | Replace non-finite `log10ΩGW` with this (effectively zero contribution). |
| `fallback_to_trapz_on_fail` | `True` | If Simpson fails or returns `< 0`, fall back to trapezoid on already-evaluated points. |

---

## Diagnostics

Every call returns a `diagnostics` dict. The fields are stable and
machine-readable — handy for Cobaya runs that want to log per-point
behaviour.

```python
{
    "f_mode": "converted_linear_f_to_log10",    # or "assume_log10f_by_range" etc.
    "dnnu_tol_abs": 1e-6,
    "simpson_atol_raw_from_dnnu": 7.43e-09,
    "g2_trapz": 1.234e-09,                      # baseline trapz on native grid
    "g2_simpson_local_final": 1.235e-09,        # Simpson (or trapz fallback) result
    "g2_rel_diff": 8.1e-04,                     # (Simpson - trapz) / trapz
    "simpson_method_final": "simpson_local",    # or "trapz_refined_fallback"
    "simpson_fallback_used": False,
    "simpson_converged": True,
    "simpson_eval_count": 137,
    "simpson_accepted_intervals": 67,
    "simpson_split_events": 70,
    "simpson_max_depth_used": 12,
    "simpson_err_est": 4.2e-12,
    # ...
}
```

---

## Cobaya integration

If you previously used `sagenet_final.py::SageNetTheoryFinal`, swap the
import:

```yaml
# my_cobaya.yaml
theory:
  sagenet_dnnu.cobaya_theory.SageNetDnnuTheory:
    # no extra fields needed; configure via env vars below
```

Environment variables (all optional, all default to the upstream defaults):

```
SAGENET_DEVICE                  cuda | cpu       (default: cuda)
SAGENET_BACKEND                 parth | alterbbn (default: parth)
SAGENET_GUARD_MODE              hard | soft      (default: hard)
SAGENET_DNNU_TOL_ABS            float            (default: 1e-6)
SAGENET_SIMPSON_RTOL            float            (default: 1e-5)
SAGENET_SIMPSON_MAX_DEPTH       int              (default: 35)
SAGENET_SIMPSON_MAX_EVALS       int              (default: 500000)
SAGENET_DNNU_MAX_LEGACY         float            (default: 5.0)   # always: dnnu>this -> reject
SAGENET_DNNU_MAX_STRICT         float            (default: 1.0)   # used by hard/soft modes
SAGENET_ETA10_MAX_HARD          float            (default: 9.0)   # only hard+parth
```

Guard logic (kept identical to upstream `sagenet_final.py`):

| Mode | Backend | Reject when |
| ---- | ------- | ----------- |
| any | any | `dnnu > 5` (legacy gate, always on) |
| `hard` | `parth` | `dnnu > 1` **or** `eta10 > 9` |
| `hard` | `alterbbn` | `dnnu > 1` |
| `soft` | `parth` | `dnnu > 1` |
| `soft` | `alterbbn` | (only legacy gate) |

---

## API reference

```python
# Top-level
compute_dnnu(prediction_or_f, log10OmegaGW=None, *, H0, config=None) -> IntegrationResult
compute_dnnu_from_predictor(predictor, params, *, config=None) -> IntegrationResult
compute_g2(f_like, log10OmegaGW_like, *, H0, config=None) -> (g2, diagnostics)

# Config / result
IntegratorConfig(dnnu_tol_abs=..., simpson_rtol=..., ...)
IntegrationResult(dnnu, g2, diagnostics)

# Lower-level building blocks (advanced)
InterpLogOmegaPCHIP(x, ylog)               # callable: omega(x_query)
adaptive_simpson_interpolated(f, a, b, ...) # bring your own integrand
maybe_log10f(f) -> (log10f, mode_str)
clean_sort_unique(x, y, *, clamp_y_nonfinite_to=-300.0)
simpson_atol_from_dnnu_tol(H0, dnnu_tol_abs)
g2_to_dnnu(g2, H0)

# Constants
Neff0, Omega_nh2, ln10
```

---

## Examples

| File | What it does |
| ---- | ------------ |
| [`examples/01_minimal.py`](examples/01_minimal.py) | Predict + integrate + plot. |
| [`examples/02_advanced.py`](examples/02_advanced.py) | Custom `IntegratorConfig`, parameter sweep. |

```bash
python examples/01_minimal.py
```

---

## Testing

```bash
pip install -e ".[test]"
pytest -q
```

The test suite covers:

- f-mode detection (Hz vs log10),
- input cleaning (NaN/Inf handling, dedup, sort),
- the dnnu ⇄ atol round-trip,
- PCHIP edge clamping and positivity,
- adaptive Simpson on a constant and on a polynomial (analytical truth),
- end-to-end on a log-Gaussian SGWB toy spectrum (analytical truth, low- and high-precision configs),
- robustness on irregular spike-grids,
- API error cases (missing keys, conflicting argument forms).

---

## Project layout

```
sagenet-dnnu/
├── sagenet_dnnu/
│   ├── __init__.py
│   ├── api.py             # compute_dnnu, compute_g2, compute_dnnu_from_predictor
│   ├── integrator.py      # PCHIP interp + adaptive Simpson
│   ├── utils.py           # f-mode detect, cleaning, atol conversion
│   ├── constants.py       # Neff0, Omega_nh2, ln10
│   ├── cobaya_theory.py   # optional Cobaya `Theory` adapter
│   └── tests/
│       └── test_integrator.py
├── examples/
│   ├── 01_minimal.py
│   └── 02_advanced.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Differences vs `sagenet_final.py`

If you're coming from the upstream file, here's what changed:

1. **No `cobaya` / `sagenetgw` import at top level.** Both are optional now. The integrator runs on bare numpy/scipy.
2. **No `utils_new.global_param` dependency.** The constants `Neff0`, `Omega_nh2`, `ln10` live in `sagenet_dnnu.constants` and are derivable.
3. **No environment-variable side effects in the core API.** Defaults are real Python defaults via `IntegratorConfig`. Env vars only apply to the optional Cobaya adapter (where they preserve byte-for-byte upstream behaviour).
4. **Public types.** `IntegrationResult` and `IntegratorConfig` replace the bare tuple `(g2, diag)`. The lower-level `compute_g2` still returns the tuple.
5. **Tests + CI.** New.
6. The Cobaya theory class was renamed `SageNetTheoryFinal` → `SageNetDnnuTheory` to flag the package boundary; behaviour is otherwise identical.

The numerical algorithm — PCHIP-in-log10Ω, adaptive Simpson with local error control, dnnu-driven atol, trapz fallback — is byte-for-byte the same.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this package in published work, please cite the SageNet paper
along with this repository.
