# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-03

### Added
- Initial public release.
- `compute_dnnu`, `compute_dnnu_from_predictor`, `compute_g2` top-level API.
- `IntegratorConfig` / `IntegrationResult` data classes.
- `InterpLogOmegaPCHIP` and `adaptive_simpson_interpolated` lower-level building blocks.
- Helpers: `maybe_log10f`, `clean_sort_unique`, `simpson_atol_from_dnnu_tol`, `g2_to_dnnu`.
- Constants module (`Neff0`, `Omega_nh2`, `ln10`).
- Optional Cobaya adapter `sagenet_dnnu.cobaya_theory.SageNetDnnuTheory` mirroring upstream `SageNetTheoryFinal` behaviour.
- 22 unit tests covering f-mode detection, input cleaning, tolerance round-trip, PCHIP edge clamping, adaptive Simpson on analytical integrands, end-to-end log-Gaussian validation, and irregular-grid robustness.
- GitHub Actions CI matrix on Python 3.9–3.12.
- Two runnable examples in `examples/`.
