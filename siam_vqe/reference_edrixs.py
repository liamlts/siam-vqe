"""EDRIXS-faithful impurity/bath energies for the L1 NiO single-orbital SIAM.

This module ports the closed-form `CT_imp_bath` formula directly from EDRIXS's
`edrixs/utils.py` (Wang et al., Comp. Phys. Commun. 243, 151 (2019)) so the
package does not depend on the edrixs Fortran/Python distribution at runtime.

The formula solves a 3-equation linear system for the impurity (E_d) and bath
(E_L) energies, given the charge-transfer convention that Delta is the
d^n -> d^(n+1) L_bar transition energy in the atomic limit with Coulomb costs
subtracted (Zaanen-Sawatzky-Allen 1985; Haverkort PRB 85, 165113 (2012)):

    E_d = (10*Delta - n*(19 + n)*U_dd/2) / (10 + n)
    E_L = n*((1 + n)*U_dd/2 - Delta) / (10 + n)

`compute_l1_levels()` adds the eg-branch crystal-field shifts (+0.6 * 10Dq) on
top of (E_d, E_L) to match example_3's per-orbital convention for the dz²,
dx²-y² orbital pair (lines 156-202 of example_03_AIM_XAS.py).
"""

from __future__ import annotations


def compute_l1_levels(
    U_dd: float = 7.3,
    Delta: float = 4.7,
    nd: int = 8,
    ten_dq: float = 0.56,
    ten_dq_bath: float = 1.44,
) -> tuple[float, float]:
    """Return (eps_d, eps_p) for the L1 NiO SIAM in eV.

    Parameters
    ----------
    U_dd : on-site Coulomb (Haverkort NiO value, default 7.3 eV).
    Delta : charge-transfer energy (default 4.7 eV).
    nd : nominal impurity occupancy reference (default 8 = Ni d^8).
    ten_dq : impurity cubic crystal field (default 0.56 eV).
    ten_dq_bath : bath cubic crystal field (default 1.44 eV).

    Returns
    -------
    (eps_d, eps_p) : tuple of floats in eV, with +0.6*10Dq eg shifts applied.
    """
    e_d = (10 * Delta - nd * (19 + nd) * U_dd / 2) / (10 + nd)
    e_l = nd * ((1 + nd) * U_dd / 2 - Delta) / (10 + nd)
    eps_d = e_d + 0.6 * ten_dq
    eps_p = e_l + 0.6 * ten_dq_bath
    return float(eps_d), float(eps_p)
