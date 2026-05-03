# -*- coding: utf-8 -*-
"""
Physical constants used by the dnnu integrator.

These values are taken from the upstream SageNet pipeline and are kept
separate from the algorithmic code so that downstream users can override
them (e.g. for sensitivity studies) without monkey-patching the integrator.

References
----------
Omega_nh2  : neutrino energy density today (Omega_nu * h^2), assuming
             three massless species with the standard CMB temperature.
Neff0      : effective number of neutrino species in the Standard Model.
ln10       : natural log of 10, used to convert log10-domain integrals
             back to natural-log domain.
"""

import numpy as np

# Standard-Model effective number of relativistic neutrino species today.
Neff0: float = 3.046

# Neutrino energy density parameter today, Omega_nu * h^2.
# Derived from Omega_gamma * h^2 ~ 2.4729e-5 and the (7/8)*(4/11)^(4/3)*Neff
# scaling. The value below matches the one used in the upstream pipeline.
Omega_nh2: float = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * Neff0 * 2.4729e-5

# ln(10), used to convert ∫ Omega d(log10 f) -> ∫ Omega d(ln f).
ln10: float = float(np.log(10.0))

__all__ = ["Neff0", "Omega_nh2", "ln10"]
