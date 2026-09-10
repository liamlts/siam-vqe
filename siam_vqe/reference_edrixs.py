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

from dataclasses import dataclass


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


# L2 Kanamori convention: locked in Phase 3 Task 1 (see
# docs/superpowers/notes/2026-05-25-kanamori-convention.md).
# STK Racah pins the ¹A₁g multiplet gap (4J = 16B + 4C = 4.022 eV) exactly.
_L2_J_CONVENTION = "stk"
_L2_F2 = 9.787  # eV
_L2_F4 = 6.078  # eV


@dataclass(frozen=True)
class L2Levels:
    """Kanamori + bath parameters for the L2 NiO e_g² SIAM.

    All values in eV.
    """

    U: float
    U_prime: float
    J_H: float
    V: float
    eps_d: float
    eps_p: float


def compute_l2_levels(
    F2: float = _L2_F2,
    F4: float = _L2_F4,
    U_dd: float = 7.3,
    V_eg: float = 2.06,
    Delta: float = 4.7,
    nd: int = 8,
    ten_dq: float = 0.56,
    ten_dq_bath: float = 1.44,
) -> L2Levels:
    """Compute Kanamori (U, U', J_H), hybridization V, and bath levels (ε_d, ε_p)
    for the L2 NiO e_g² SIAM.

    Returns an L2Levels dataclass. The J convention is set by _L2_J_CONVENTION.
    """
    if _L2_J_CONVENTION == "pavarini":
        U_eg = U_dd + 4 * F2 / 49 + 36 * F4 / 441
        J_H = 3 * F2 / 49 + 20 * F4 / 441
    elif _L2_J_CONVENTION == "stk":
        # Sugano-Tanabe-Kamimura Racah-derived: pins ¹A₁g multiplet gap exactly.
        B = (9 * F2 - 5 * F4) / 441
        C = 5 * F4 / 63
        A = U_dd - 49 * F4 / 441
        U_eg = A + 4 * B + 3 * C
        J_H = 4 * B + C
    else:
        raise ValueError(
            f"Unknown Kanamori J convention: {_L2_J_CONVENTION!r}. "
            "Must be 'pavarini' or 'stk'; see Phase 3 task 1."
        )

    U_prime = U_eg - 2 * J_H

    # Reuse compute_l1_levels' CT_imp_bath core for ε_d, ε_p.
    eps_d, eps_p = compute_l1_levels(
        U_dd=U_dd, Delta=Delta, nd=nd, ten_dq=ten_dq, ten_dq_bath=ten_dq_bath
    )

    return L2Levels(
        U=U_eg,
        U_prime=U_prime,
        J_H=J_H,
        V=V_eg,
        eps_d=eps_d,
        eps_p=eps_p,
    )
