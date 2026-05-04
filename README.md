# sagenet-dnnu

> **Fast and robust integration of SageNet SGWB spectra into**
> \(\Delta N_\mathrm{eff}\) **for parameter scans, MCMC exploration, and
> Cobaya-style pipelines.**

[![tests](https://github.com/<your-user-or-org>/sagenet-dnnu/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-user-or-org>/sagenet-dnnu/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What this is

[SageNet](https://github.com/) (`sagenetgw.classes.GWPredictor`) predicts a
stochastic gravitational-wave background (SGWB) spectrum,

$$
\Delta N_\mathrm{eff} \;=\; \frac{N_\mathrm{eff}^{(0)}}{\Omega_\nu h^2 / h^2}\,
\ln 10 \int \Omega_\mathrm{GW}(\log_{10} f)\, \mathrm{d}(\log_{10} f).
$$

To use that spectrum in a BBN or Cobaya pipeline, one usually needs to
compress it into a single effective radiation variable,

$$
\Delta N_\mathrm{eff},
$$

called **`dnnu`** in this package.

`sagenet-dnnu` provides a standalone, testable integration module that turns a
SageNet SGWB spectrum into \(\Delta N_\mathrm{eff}\). It is designed for
large-scale applications where repeatedly running a direct numerical SGWB/ODE
backend would be too expensive, such as:

- fast parameter-space scans,
- MCMC exploration,
- gate screening,
- trend studies,
- identification of physically interesting regions for numerical follow-up.

The package implements the SGWB-to-\(\Delta N_\mathrm{eff}\) integration recipe
used in the upstream Cobaya theory pipeline, but exposes it as a clean Python
API independent of `cobaya` and independent of `sagenetgw` at import time.

---

## Why this package exists

A naive direct application of `scipy.integrate.simpson` or `numpy.trapz` to the
raw SageNet output can be numerically fragile. The main difficulties are:

1. SageNet returns a non-uniform frequency grid.
2. The grid often contains extra samples around spectral features.
3. \(\Omega_\mathrm{GW}\) spans many orders of magnitude.
4. Polynomial Simpson interpolation on a sharp peak can become unstable on a
   non-uniform grid.
5. A fixed absolute integration tolerance does not correspond to the same
   tolerance in \(\Delta N_\mathrm{eff}\) for different cosmological parameters.

`sagenet-dnnu` addresses these issues using:

- **PCHIP interpolation in the \(\log_{10}\Omega_\mathrm{GW}\) domain**  
  Shape-preserving interpolation avoids artificial overshoot and keeps the
  reconstructed SGWB integrand positive.

- **Adaptive Simpson integration with local error control**  
  The integrator refines the curve where the SGWB spectrum is curved or sharply
  peaked.

- **A physics-driven tolerance**  
  The user specifies the desired absolute tolerance on \(\Delta N_\mathrm{eff}\).
  The code converts this into the corresponding raw-integral tolerance for the
  current cosmological parameters.

- **A non-negative trapezoid fallback**  
  If adaptive Simpson fails to converge or returns an invalid result, the code
  falls back to trapezoid integration on the refined grid visited by Simpson.

- **Automatic frequency-mode detection**  
  The input frequency array can be either linear frequency \(f\) or already
  \(\log_{10}f\). The detected mode is recorded in the diagnostics.

The integrator itself uses only NumPy and SciPy. It does not require `cobaya`
or `sagenetgw` unless you want to use the optional Cobaya adapter or call a
SageNet predictor directly.

---

## Install

### From GitHub

```bash
pip install git+https://github.com/Hdiao112/sagenet-dnnu.git
```

To install from a specific branch:

```bash
pip install git+https://github.com/Hdiao112/sagenet-dnnu.git@main
```

### Local development install(Recommand)

```bash
git clone https://github.com/Hdiao112/sagenet-dnnu.git
cd sagenet-dnnu
pip install -e .
```

### PyPI install

After the package is published to PyPI, it can be installed with:

```bash
pip install sagenet-dnnu
```

### Optional extras

| Extra        | Adds         | Use when |
| ------------ | ------------ | -------- |
| `[cobaya]`   | `cobaya`     | you want to use `SageNetDnnuTheory` in a Cobaya run |
| `[test]`     | `pytest`     | you want to run the test suite |
| `[examples]` | `matplotlib` | you want to run plotting examples |
| `[all]`      | all of above | one-line install for development |

```bash
pip install -e ".[all]"
```

> SageNet itself (`sagenetgw`) is **not** a core dependency of this package.
> You only need it if you want to call `GWPredictor.predict(...)`. The
> integrator works on any compatible `(f, log10OmegaGW)` arrays.

---

## Quick start

The example below follows the standard SageNet usage pattern and adds one
integration step.

```python
from sagenetgw.classes import GWPredictor
import numpy as np
from matplotlib import pyplot as plt

from sagenet_dnnu import compute_dnnu, IntegratorConfig

predictor = GWPredictor(model_type="Transformer", device="cpu")

prediction = predictor.predict({
    "r":         3.9585109e-05,
    "n_t":       1.0116972,
    "kappa10":   1.42477,
    "T_re":      0.17453859,
    "DN_re":     39.366618,
    "Omega_bh2": 0.0223828,
    "Omega_ch2": 0.1201075,
    "H0":        67.32117,
    "A_s":       2.100549e-9,
})

# Integrate the SGWB spectrum into Delta N_eff.
# `reject_above_dnnu=5.0` enables the standard cosmological cut: points with
# dnnu > 5 are physically excluded by BBN/CMB and almost always sit in
# SageNet's extrapolation region, so we mask them to NaN by default. The
# pre-mask value is preserved in result.diagnostics["dnnu_raw"].
config = IntegratorConfig(reject_above_dnnu=5.0)
result = compute_dnnu(prediction, H0=67.32117, config=config)

if result.diagnostics["rejected"]:
    print(f"point rejected: {result.diagnostics['rejected_reason']} "
          f"(raw dnnu was {result.diagnostics['dnnu_raw']:.3f})")
else:
    print(f"dnnu   = {result.dnnu:.6e}")
    print(f"g2     = {result.g2:.6e}")
    print(f"method = {result.diagnostics['simpson_method_final']}")

pred_coords = np.column_stack((prediction["f"], prediction["log10OmegaGW"]))
plt.plot(pred_coords[:, 0], pred_coords[:, 1], "--",
         color="royalblue", marker=".")
plt.xlabel("log10(f)")
plt.ylabel(r"$\log_{10}\,\Omega_{\rm GW}$")
plt.show()
```

The returned object is an `IntegrationResult`:

```python
result.dnnu          # Delta N_eff
result.g2            # raw SGWB integral after conversion
result.diagnostics   # machine-readable diagnostics
```

---

## One-shot: predict and integrate

```python
from sagenet_dnnu import compute_dnnu_from_predictor

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

result = compute_dnnu_from_predictor(predictor, params)
print(result.dnnu)
```

if try to see how the threshold works, here's a group of params that should get dnnu > 5 and should return dnnu=nan：

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from matplotlib import pyplot as plt

from sagenetgw.classes import GWPredictor

from sagenet_dnnu import compute_dnnu, IntegratorConfig


# =============================================================================
# 1. Test row (your row_index = 5632)
# =============================================================================
row = {
    "row_index": 5632,
    "omegabh2":      0.073688103,
    "omegach2":      0.08447487,
    "H0":            33.906496,
    "ln10_10As":     3.8834262,
    "log10r":       -33.99478,
    "n_t":           4.2286621,
    "log10kappa10": -0.49245602,
    "log10Tre":      5.3597445,
    "DN_re":         1.6802285,

    # Stored CSV values, for comparison only.
    "dnnu_simpson_local":   56.40103691845641,
    "dnnu_trapz":           56.54147358465502,
    "dnnu_simpson_uniform": 57.45712169635855,
    "dn_nu_inputcol":       53.37966,
}


# =============================================================================
# 2. Convert log10/cobaya-style row into SageNet predictor input
# =============================================================================
sagenet_params = {
    "r":         10 ** float(row["log10r"]),
    "n_t":       float(row["n_t"]),
    "kappa10":   10 ** float(row["log10kappa10"]),
    "T_re":      10 ** float(row["log10Tre"]),
    "DN_re":     float(row["DN_re"]),
    "Omega_bh2": float(row["omegabh2"]),
    "Omega_ch2": float(row["omegach2"]),
    "H0":        float(row["H0"]),
    "A_s":       np.exp(float(row["ln10_10As"])) * 1e-10,
}

print(">>> Testing row_index =", row["row_index"])
print("\n=== SageNet physical input params ===")
for k, v in sagenet_params.items():
    print(f"{k:10s} = {v:.16e}")


# =============================================================================
# 3. Run SageNet predictor
# =============================================================================
predictor = GWPredictor(model_type="Transformer", device="cpu")
prediction = predictor.predict(sagenet_params)

f = np.asarray(prediction["f"])
log10OmegaGW = np.asarray(prediction["log10OmegaGW"])

print("\n=== Raw SageNet prediction diagnostics ===")
print("prediction keys:", list(prediction.keys()))
print("f shape:", f.shape, "  log10OmegaGW shape:", log10OmegaGW.shape)
print("f min/max:           ", np.nanmin(f), np.nanmax(f))
print("log10OmegaGW min/max:", np.nanmin(log10OmegaGW), np.nanmax(log10OmegaGW))
print("nan in f:", bool(np.isnan(f).any()),
      "  nan in log10OmegaGW:", bool(np.isnan(log10OmegaGW).any()))


# =============================================================================
# 4. Configure the dnnu guard (NEW API)
# -----------------------------------------------------------------------------
# - reject_above_dnnu = 5.0  -> if dnnu > 5, mask result to NaN
# - reject_below_dnnu = None -> no lower guard (default)
# Diagnostics added when a guard trips:
#       rejected (bool)
#       rejected_reason (str, e.g. "dnnu>5")
#       dnnu_raw (float, the un-masked value)
#       reject_above_dnnu / reject_below_dnnu (echoed back)
# =============================================================================
config = IntegratorConfig(
    reject_above_dnnu=5.0,
    # reject_below_dnnu=1e-30,   # uncomment to also drop floor-noise points
)


# =============================================================================
# 5. Integrate
# =============================================================================
result = compute_dnnu(
    prediction,
    H0=float(row["H0"]),
    config=config,
)

print("\n=== sagenet_dnnu output ===")
print("dnnu                 =", result.dnnu)
print("g2                   =", result.g2)

dnnu_raw = result.diagnostics["dnnu_raw"]
print("dnnu_raw (pre-gate)  =", dnnu_raw)
print("dnnu_is_nan          =", bool(np.isnan(result.dnnu)))
print("rejected             =", result.diagnostics["rejected"])
print("rejected_reason      =", result.diagnostics["rejected_reason"])
print("reject_above_dnnu    =", result.diagnostics["reject_above_dnnu"])
print("reject_below_dnnu    =", result.diagnostics["reject_below_dnnu"])
```

---

## Raw-array usage

You can also use the integrator without SageNet:

```python
from sagenet_dnnu import compute_dnnu

result = compute_dnnu(
    f_array,
    log10OmegaGW_array,
    H0=67.32,
)
```

`f_array` may be either linear frequency, for example

```text
1e-18, 1e-17, ..., 1e-3
```

or already \(\log_{10}f\), for example

```text
-18, -17, ..., -3
```

The detected mode is stored in:

```python
result.diagnostics["f_mode"]
```

---

## Scope and limitations

`sagenet-dnnu` is intended as a **fast SGWB-to-\(\Delta N_\mathrm{eff}\)
reconstruction tool** for SageNet-based workflows.

It is suitable for common large-scale research tasks such as:

- broad parameter-space scans,
- MCMC exploration,
- fast likelihood evaluation,
- gate screening,
- trend studies,
- locating interesting regions for more expensive numerical follow-up.

It should not be interpreted as proving that a SageNet-based pipeline is
numerically identical to a direct SGWB/ODE solver at every point. The difference
between

\[
\Delta N_\mathrm{eff}^{\rm SageNet+sagenet\text{-}dnnu}
\]

and

\[
\Delta N_\mathrm{eff}^{\rm numerical\ SGWB/ODE}
\]

is a **total emulator-to-solver discrepancy**. It can include:

- SageNet spectral-emulation error,
- frequency-grid and interpolation differences,
- convention differences between pipelines,
- residual numerical-solver uncertainty,
- and the much smaller internal quadrature uncertainty of `sagenet-dnnu`.

In the validation samples considered here, this total discrepancy is usually
small compared with the current observational error budget and is not expected
to produce conclusion-level changes in broad scans or trend analyses. However,
important scientific points should still be checked with the direct numerical
backend.

We recommend direct numerical re-evaluation for:

- final best-fit or benchmark points,
- posterior-tail points,
- samples close to hard selection boundaries,
- regions near the edge of the SageNet training domain,
- future CMB-S4-level or other high-precision forecast studies.

---

## Validation strategy

We validate the package at two different levels.

### 1. Internal quadrature stability

First, we test the stability of the integrator itself by comparing:

- default integration settings,
- tightened-tolerance settings,
- and stored Simpson-integrated values from the upstream pipeline.

In physically relevant regions, this internal integration difference is
typically at the \(\mathcal{O}(10^{-4})\) level or smaller in
\(\Delta N_\mathrm{eff}\). This is well below the current observational
uncertainty scale for \(N_\mathrm{eff}\), so the quadrature error of the
integrator is negligible for current broad-scan applications.

### 2. Comparison with a direct numerical SGWB/ODE backend

Second, we compare the full SageNet-based reconstruction against a direct
numerical SGWB/ODE backend. This comparison is not a pure integration-error
test. It measures the total SageNet-to-numerical discrepancy.

In the tested sample, this discrepancy is typically at the

\[
\mathcal{O}(10^{-3})-\mathcal{O}(10^{-2})
\]

level in \(\Delta N_\mathrm{eff}\). For current Planck/BAO/BBN-level
applications, this is usually subdominant to the observational error budget and
is suitable for broad scans, MCMC exploration, gate screening, and trend
studies.

This does **not** mean that the SageNet-based reconstruction should be treated
as a high-precision drop-in replacement for the direct numerical SGWB/ODE solver
in all cases. Rather, it provides a fast and validated surrogate for large-scale
exploration, with numerical re-evaluation recommended for key points.

### 3. Hard-boundary check

The numerical pipeline uses

\[
\Delta N_\mathrm{eff} = 5
\]

as a hard boundary in the tested setup. In the 50k-sample SageNet validation
set used for this hard-boundary check, all SageNet-integrated points satisfied

\[
\Delta N_\mathrm{eff} - 5 \le 0 .
\]

Thus, within this tested sample, the SageNet-based reconstruction did not
spuriously cross the numerical hard boundary \(\Delta N_\mathrm{eff}=5\).

### 4. Downstream BBN-abundance checks

The upstream validation also checks the impact of the SGWB-to-\(\Delta
N_\mathrm{eff}\) reconstruction on downstream BBN abundance predictions. In
the tested samples, the abundance-level differences induced by the
SageNet-based reconstruction are generally smaller than the current
observational error budget. This supports using `sagenet-dnnu` for broad scans
and MCMC exploration, while retaining direct numerical SGWB/ODE re-evaluation
for final benchmark, boundary, and high-precision forecast points.

---

## Algorithm in one minute

The SGWB contribution is obtained by integrating the predicted spectrum over
\(\log_{10}f\). In practice, the implementation proceeds as follows:

1. Clean the input arrays: remove non-finite values, sort by frequency, and
   merge duplicated frequency points.
2. Detect whether the input frequency is linear \(f\) or already
   \(\log_{10}f\).
3. Interpolate \(\log_{10}\Omega_\mathrm{GW}\) using PCHIP.
4. Exponentiate the interpolant to obtain a positive SGWB integrand.
5. Integrate with adaptive Simpson using a tolerance derived from the requested
   absolute tolerance on \(\Delta N_\mathrm{eff}\).
6. If Simpson fails or returns an invalid value, fall back to trapezoid
   integration on the refined points already evaluated.
7. Convert the raw integral to \(\Delta N_\mathrm{eff}\).

The diagnostics record the detected frequency mode, integration method,
tolerances, convergence status, number of evaluations, and fallback usage.

---

## Tuning: `IntegratorConfig`

Defaults are chosen to match the upstream production pipeline. Most users should
not need to change them.

```python
from sagenet_dnnu import IntegratorConfig, compute_dnnu

cfg = IntegratorConfig(
    dnnu_tol_abs=1e-7,          # tighter -> more compute, smaller quadrature error
    simpson_rtol=1e-6,
    simpson_max_depth=40,
    simpson_max_evals=1_000_000,
    edge_trim=0.0,
    clamp_log10omega_nonfinite_to=-300.0,
    fallback_to_trapz_on_fail=True,
)

result = compute_dnnu(prediction, H0=67.32, config=cfg)
```

| Field | Default | Meaning |
| --- | ---: | --- |
| `dnnu_tol_abs` | `1e-6` | Absolute tolerance target on final `dnnu` |
| `simpson_rtol` | `1e-5` | Relative tolerance for adaptive Simpson |
| `simpson_max_depth` | `35` | Maximum recursion depth |
| `simpson_max_evals` | `500000` | Hard cap on integrand evaluations |
| `edge_trim` | `0.0` | Optional trim in `log10(f)` units at both edges |
| `clamp_log10omega_nonfinite_to` | `-300.0` | Replacement for non-finite `log10OmegaGW` values |
| `fallback_to_trapz_on_fail` | `True` | Use refined-grid trapezoid fallback if Simpson fails |

---

## Diagnostics

Every call returns a diagnostics dictionary. These fields are designed to be
machine-readable and useful in Cobaya runs.

Example:

```python
{
    "f_mode": "assume_log10f_nonpositive_seen",
    "n_input_points": 256,
    "n_clean_points": 256,
    "dnnu_tol_abs": 1e-6,
    "simpson_atol_raw_from_dnnu": 5.38e-12,
    "g2_trapz": 2.92e-05,
    "g2_simpson_local_final": 2.91e-05,
    "g2_rel_diff": -3.71e-03,
    "simpson_method_final": "simpson_local",
    "simpson_fallback_used": False,
    "simpson_converged": True,
    "simpson_eval_count": 945,
    "simpson_accepted_intervals": 236,
    "simpson_split_events": 235,
    "simpson_max_depth_used": 16,
    "simpson_err_est": 3.0e-10,
}
```

The most useful fields are:

| Field | Meaning |
| --- | --- |
| `f_mode` | How the input frequency was interpreted |
| `n_clean_points` | Number of points used after cleaning |
| `g2_trapz` | Native-grid trapezoid baseline |
| `g2_simpson_local_final` | Final Simpson or fallback result |
| `g2_rel_diff` | Relative difference between final and native trapz estimate |
| `simpson_method_final` | Final integration path |
| `simpson_fallback_used` | Whether fallback was used |
| `simpson_converged` | Whether adaptive Simpson converged |
| `simpson_eval_count` | Number of function evaluations |
| `simpson_err_est` | Internal Simpson error estimate |

---

## Cobaya integration

If you previously used the upstream `SageNetTheoryFinal`-style Cobaya theory,
you can use the optional adapter:

```yaml
theory:
  sagenet_dnnu.cobaya_theory.SageNetDnnuTheory:
    # configure via environment variables if needed
```

Environment variables:

```text
SAGENET_DEVICE                  cuda | cpu       default: cuda
SAGENET_BACKEND                 parth | alterbbn default: parth
SAGENET_GUARD_MODE              hard | soft      default: hard
SAGENET_DNNU_TOL_ABS            float            default: 1e-6
SAGENET_SIMPSON_RTOL            float            default: 1e-5
SAGENET_SIMPSON_MAX_DEPTH       int              default: 35
SAGENET_SIMPSON_MAX_EVALS       int              default: 500000
SAGENET_DNNU_MAX_LEGACY         float            default: 5.0
SAGENET_DNNU_MAX_STRICT         float            default: 1.0
SAGENET_ETA10_MAX_HARD          float            default: 9.0
```

### Guard interpretation

The legacy ceiling

\[
\Delta N_\mathrm{eff} > 5
\]

is treated here as the hard numerical boundary used by the tested pipeline.

Some upstream configurations also use a stricter

\[
\Delta N_\mathrm{eff} > 1
\]

guard. This stricter guard is project-specific and should be understood as a
conservative domain-control choice for selected backends, training ranges, or
pipeline configurations. It should **not** be read as a universal physical
boundary.

| Guard | Interpretation |
| --- | --- |
| `dnnu > 5` | Legacy numerical hard boundary used in this pipeline |
| `dnnu > 1` | Optional project-specific conservative guard |
| `eta10 > 9` | Optional backend-specific guard for selected PArthENoPE-style configurations |

---

## API reference

Top-level API:

```python
compute_dnnu(prediction_or_f, log10OmegaGW=None, *, H0, config=None)
compute_dnnu_from_predictor(predictor, params, *, config=None)
compute_g2(f_like, log10OmegaGW_like, *, H0, config=None)
```

Returned object:

```python
IntegrationResult(
    dnnu,        # Delta N_eff
    g2,          # converted raw integral
    diagnostics  # dict
)
```

Configuration:

```python
IntegratorConfig(
    dnnu_tol_abs=...,
    simpson_rtol=...,
    simpson_max_depth=...,
    simpson_max_evals=...,
    edge_trim=...,
    clamp_log10omega_nonfinite_to=...,
    fallback_to_trapz_on_fail=...,
)
```

Lower-level utilities:

```python
InterpLogOmegaPCHIP(x, ylog)
adaptive_simpson_interpolated(f, a, b, ...)
maybe_log10f(f)
clean_sort_unique(x, y, *, clamp_y_nonfinite_to=-300.0)
simpson_atol_from_dnnu_tol(H0, dnnu_tol_abs)
g2_to_dnnu(g2, H0)
```

Constants:

```python
Neff0
Omega_nh2
ln10
```

---

## Examples

| File | Description |
| --- | --- |
| [`examples/01_minimal.py`](examples/01_minimal.py) | Predict, integrate, and plot |
| [`examples/02_advanced.py`](examples/02_advanced.py) | Custom `IntegratorConfig` and parameter sweep |

Run:

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

- frequency-mode detection,
- input cleaning,
- duplicate handling,
- NaN/Inf handling,
- \(\Delta N_\mathrm{eff}\)-to-tolerance conversion,
- PCHIP positivity,
- adaptive Simpson on analytic functions,
- end-to-end toy SGWB spectra,
- irregular spike-grid robustness,
- API error cases.

---

## Project layout

```text
sagenet-dnnu/
├── sagenet_dnnu/
│   ├── __init__.py
│   ├── api.py
│   ├── integrator.py
│   ├── utils.py
│   ├── constants.py
│   ├── cobaya_theory.py
│   └── tests/
│       └── test_integrator.py
├── examples/
│   ├── 01_minimal.py
│   └── 02_advanced.py
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
└── README.md
```

---

## Differences from the upstream Cobaya implementation

Compared with the upstream monolithic theory file, this package makes the
integration logic easier to reuse and test:

1. **No mandatory `cobaya` or `sagenetgw` import at top level**  
   Both are optional.

2. **No runtime dependence on project-local `utils_new` modules**  
   Constants and conversion utilities live inside `sagenet_dnnu`.

3. **No environment-variable side effects in the core API**  
   Core defaults are explicit Python defaults through `IntegratorConfig`.

4. **Typed public result object**  
   `IntegrationResult` replaces informal tuples.

5. **Standalone tests and CI support**  
   The numerical integration logic can be tested independently of a full
   cosmology pipeline.

6. **Optional Cobaya adapter**  
   The adapter preserves the intended upstream behaviour while keeping the core
   integrator independent.

---

## Recommended workflow

For expensive SGWB+BBN studies, the recommended workflow is:

1. Use SageNet + `sagenet-dnnu` for broad scans and MCMC exploration.
2. Identify best-fit points, posterior-tail samples, boundary samples, and
   interesting physical regions.
3. Recompute representative key points with the direct numerical SGWB/ODE
   backend.
4. Use the numerical re-evaluation as a validation reference for final
   scientific claims.

This workflow preserves the speed advantage of the SageNet-based pipeline while
avoiding the over-interpretation of emulator results as exact numerical-solver
outputs.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this package in published work, please cite the SageNet paper and
this repository. If the package is used as part of a SGWB+BBN inference pipeline,
please also describe whether final benchmark or boundary points were rechecked
with the direct numerical backend.
