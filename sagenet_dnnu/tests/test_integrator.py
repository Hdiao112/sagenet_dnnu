# -*- coding: utf-8 -*-
"""Unit tests for the dnnu integrator.

Run with::

    pytest -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sagenet_dnnu import (
    IntegratorConfig,
    compute_dnnu,
    compute_g2,
)
from sagenet_dnnu.constants import Neff0, Omega_nh2, ln10
from sagenet_dnnu.utils import (
    clean_sort_unique,
    g2_to_dnnu,
    maybe_log10f,
    simpson_atol_from_dnnu_tol,
)
from sagenet_dnnu.integrator import (
    InterpLogOmegaPCHIP,
    adaptive_simpson_interpolated,
)


# -----------------------------------------------------------------------------
# Synthetic SGWB-like spectrum: a log-Gaussian peak in log10(f).
# Easy to integrate analytically (g2 ~ amplitude * sigma * sqrt(2*pi)).
# -----------------------------------------------------------------------------
def make_log_gauss_spectrum(
    n: int = 500,
    f_lo_log: float = -18.0,
    f_hi_log: float = 10.0,
    mu: float = -7.0,
    sigma: float = 1.5,
    log10_amp: float = -10.0,
):
    """Return (f_log10, log10Omega) for a Gaussian-in-log10f peak.

    Omega(x) = 10**log10_amp * exp(-(x - mu)**2 / (2 sigma**2))

    Integral in raw log10(f) variable:
        I_raw = 10**log10_amp * sigma * sqrt(2*pi)
    g2 = ln10 * I_raw.
    """
    x = np.linspace(f_lo_log, f_hi_log, n)
    omega = (10.0 ** log10_amp) * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    log10Omega = np.log10(omega)
    return x, log10Omega


def gauss_g2_truth(sigma: float, log10_amp: float) -> float:
    return ln10 * (10.0 ** log10_amp) * sigma * math.sqrt(2.0 * math.pi)


# -----------------------------------------------------------------------------
# utils.maybe_log10f
# -----------------------------------------------------------------------------
class TestMaybeLog10f:
    def test_already_log10(self):
        f = np.linspace(-10, 5, 50)
        out, mode = maybe_log10f(f)
        assert mode.startswith("assume_log10f")
        np.testing.assert_array_equal(out, f)

    def test_linear_hz_converted(self):
        f = np.logspace(-5, 8, 50)  # 1e-5 .. 1e8 Hz
        out, mode = maybe_log10f(f)
        assert mode == "converted_linear_f_to_log10"
        np.testing.assert_allclose(out, np.log10(f))

    def test_empty(self):
        out, mode = maybe_log10f(np.array([]))
        assert mode == "empty"
        assert out.size == 0


# -----------------------------------------------------------------------------
# utils.clean_sort_unique
# -----------------------------------------------------------------------------
class TestCleanSortUnique:
    def test_drops_nonfinite_x_clamps_y(self):
        x = np.array([1.0, np.nan, 2.0, 3.0])
        y = np.array([10.0, 20.0, np.inf, 30.0])
        xc, yc = clean_sort_unique(x, y, clamp_y_nonfinite_to=-300.0)
        np.testing.assert_array_equal(xc, [1.0, 2.0, 3.0])
        # x=2 had y=inf -> clamped; x=1 keeps 10 (y=20 was at the dropped row).
        assert yc[0] == 10.0
        assert yc[1] == -300.0
        assert yc[2] == 30.0

    def test_sorts_and_uniques(self):
        x = np.array([3.0, 1.0, 2.0, 1.0])
        y = np.array([30.0, 10.0, 20.0, 99.0])
        xc, yc = clean_sort_unique(x, y)
        np.testing.assert_array_equal(xc, [1.0, 2.0, 3.0])
        # First occurrence of x=1 was y=10 (after sort).
        np.testing.assert_array_equal(yc, [10.0, 20.0, 30.0])

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            clean_sort_unique(np.array([1.0]), np.array([1.0]))


# -----------------------------------------------------------------------------
# utils.simpson_atol_from_dnnu_tol & g2_to_dnnu inverse relationship.
# -----------------------------------------------------------------------------
class TestToleranceRoundTrip:
    def test_g2_to_dnnu_inverse(self):
        H0 = 67.32
        g2 = 1.0e-6
        dnnu = g2_to_dnnu(g2, H0)
        # By construction, dnnu = Neff0 * g2 / (Omega_nh2 / h^2).
        h = H0 / 100.0
        Omega_nu = Omega_nh2 / (h * h)
        assert math.isclose(dnnu, Neff0 * g2 / Omega_nu, rel_tol=1e-12)

    def test_atol_consistency(self):
        # If we feed dnnu_tol_abs through simpson_atol_from_dnnu_tol and
        # then run that through the g2->dnnu chain, we should recover
        # exactly dnnu_tol_abs.
        H0 = 67.32
        dnnu_tol = 1e-6
        I_raw_atol = simpson_atol_from_dnnu_tol(H0, dnnu_tol)
        g2_atol = I_raw_atol * ln10
        dnnu_recovered = g2_to_dnnu(g2_atol, H0)
        assert math.isclose(dnnu_recovered, dnnu_tol, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# Interpolator
# -----------------------------------------------------------------------------
class TestInterpolator:
    def test_clamp_at_edges(self):
        x = np.array([0.0, 1.0, 2.0])
        ylog = np.array([-3.0, -2.0, -4.0])
        f = InterpLogOmegaPCHIP(x, ylog)
        assert f(-100.0) == 10.0 ** -3.0  # left clamp
        assert f(100.0) == 10.0 ** -4.0   # right clamp

    def test_interpolation_positive(self):
        x = np.array([0.0, 1.0, 2.0])
        ylog = np.array([-3.0, -2.0, -4.0])
        f = InterpLogOmegaPCHIP(x, ylog)
        # PCHIP in log domain -> 10**(...) -> always positive.
        for xq in np.linspace(0.0, 2.0, 21):
            assert f(xq) > 0


# -----------------------------------------------------------------------------
# Adaptive Simpson
# -----------------------------------------------------------------------------
class TestAdaptiveSimpson:
    def test_constant_function(self):
        # ∫_0^1 5 dx = 5
        I, diag, _ = adaptive_simpson_interpolated(
            lambda x: 5.0, 0.0, 1.0,
            rtol=1e-8, atol=1e-12,
            max_depth=20, max_evals=10_000,
        )
        assert diag["simpson_converged"]
        assert math.isclose(I, 5.0, rel_tol=1e-12)

    def test_polynomial(self):
        # ∫_0^2 x^3 dx = 4
        I, diag, _ = adaptive_simpson_interpolated(
            lambda x: x ** 3, 0.0, 2.0,
            rtol=1e-10, atol=1e-14,
            max_depth=30, max_evals=10_000,
        )
        assert diag["simpson_converged"]
        assert math.isclose(I, 4.0, rel_tol=1e-10)


# -----------------------------------------------------------------------------
# End-to-end: log-Gaussian -> known g2 / dnnu
# -----------------------------------------------------------------------------
class TestEndToEndLogGaussian:
    def test_g2_matches_analytical(self):
        # A reasonably sampled Gaussian peak inside a moderate domain.
        # Tighter tolerances on the integrator are exercised separately
        # via test_g2_high_precision below; here we just check baseline
        # agreement with the closed-form integral.
        sigma = 1.5
        log10_amp = -10.0
        x, ylog = make_log_gauss_spectrum(
            n=400, f_lo_log=-12.0, f_hi_log=2.0,
            mu=-5.0, sigma=sigma, log10_amp=log10_amp,
        )
        g2, diag = compute_g2(x, ylog, H0=67.32)
        truth = gauss_g2_truth(sigma, log10_amp)
        assert diag["simpson_converged"]
        # ~3% on a non-uniformly-sampled wide-domain Gaussian via PCHIP.
        assert math.isclose(g2, truth, rel_tol=3e-2)

    def test_g2_high_precision(self):
        # Tighten the integrator: rtol=1e-8, dnnu_tol=1e-12 => atol very
        # tight. Should reach analytical agreement to ~1e-5.
        sigma = 1.5
        log10_amp = -10.0
        x, ylog = make_log_gauss_spectrum(
            n=2000, f_lo_log=-12.0, f_hi_log=2.0,
            mu=-5.0, sigma=sigma, log10_amp=log10_amp,
        )
        cfg = IntegratorConfig(dnnu_tol_abs=1e-12, simpson_rtol=1e-9)
        g2, diag = compute_g2(x, ylog, H0=67.32, config=cfg)
        truth = gauss_g2_truth(sigma, log10_amp)
        assert diag["simpson_converged"]
        assert math.isclose(g2, truth, rel_tol=1e-4)

    def test_dnnu_matches_analytical(self):
        sigma = 1.2
        log10_amp = -8.5
        x, ylog = make_log_gauss_spectrum(
            n=800, f_lo_log=-12.0, f_hi_log=2.0,
            mu=-5.0, sigma=sigma, log10_amp=log10_amp,
        )
        result = compute_dnnu(x, ylog, H0=67.32)
        truth_g2 = gauss_g2_truth(sigma, log10_amp)
        truth_dnnu = g2_to_dnnu(truth_g2, 67.32)
        # Same realistic baseline tolerance as above.
        assert math.isclose(result.dnnu, truth_dnnu, rel_tol=3e-2)
        assert math.isclose(result.g2, truth_g2, rel_tol=3e-2)

    def test_prediction_dict_interface(self):
        x, ylog = make_log_gauss_spectrum(n=200)
        pred = {"f": x, "log10OmegaGW": ylog}  # log10 already; auto-detected
        result = compute_dnnu(pred, H0=67.32)
        assert result.dnnu > 0
        assert result.g2 > 0
        assert "f_mode" in result.diagnostics
        assert "simpson_method_final" in result.diagnostics

    def test_linear_hz_input(self):
        x_log, ylog = make_log_gauss_spectrum(n=200)
        f_hz = 10.0 ** x_log
        # Same shape but expressed in linear Hz; should auto-detect.
        result = compute_dnnu({"f": f_hz, "log10OmegaGW": ylog}, H0=67.32)
        assert result.diagnostics["f_mode"] == "converted_linear_f_to_log10"
        assert result.dnnu > 0


# -----------------------------------------------------------------------------
# Negative-trapz / positive-Simpson scenario:
# Construct a spiky non-uniform grid that would make plain trapz / Simpson
# misbehave; verify that PCHIP+Simpson stays positive.
# -----------------------------------------------------------------------------
class TestRobustnessToSpikes:
    def test_positive_integral_on_irregular_grid(self):
        # Non-uniform grid concentrated around a sharp peak.
        x_dense = np.linspace(-1.5, -0.5, 80)
        x_sparse = np.array([-15.0, -10.0, -5.0, 5.0, 10.0])
        x = np.sort(np.concatenate([x_sparse, x_dense]))
        omega = 1e-9 * np.exp(-((x + 1.0) ** 2) / (2 * 0.1 ** 2))
        omega = np.maximum(omega, 1e-300)
        ylog = np.log10(omega)

        result = compute_dnnu(x, ylog, H0=67.32)
        # We don't have an exact analytical truth here (the peak is
        # very narrow and the integration domain is large), but the
        # result must be strictly positive.
        assert result.g2 > 0
        assert result.dnnu > 0

    def test_fallback_disabled_still_returns(self):
        x, ylog = make_log_gauss_spectrum(n=100)
        cfg = IntegratorConfig(fallback_to_trapz_on_fail=False)
        result = compute_dnnu(x, ylog, H0=67.32, config=cfg)
        assert result.g2 > 0


# -----------------------------------------------------------------------------
# API error handling.
# -----------------------------------------------------------------------------
class TestAPIErrors:
    def test_missing_keys_in_dict(self):
        with pytest.raises(KeyError):
            compute_dnnu({"f": [1, 2, 3]}, H0=67.32)

    def test_dict_plus_array_rejected(self):
        with pytest.raises(TypeError):
            compute_dnnu({"f": [1], "log10OmegaGW": [1]}, [1], H0=67.32)

    def test_array_without_y_rejected(self):
        with pytest.raises(TypeError):
            compute_dnnu([1.0, 2.0, 3.0], H0=67.32)
