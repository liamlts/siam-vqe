"""Fermionic Hamiltonian factories for siam_vqe.

Mode ordering convention used throughout:
    mode 0 = site 0, spin up
    mode 1 = site 1, spin up
    mode 2 = site 0, spin down
    mode 3 = site 1, spin down

This "all-up-then-all-down" ordering matches Qiskit Nature's default for
spinful Hamiltonians and keeps Sz tapering straightforward.
"""

from __future__ import annotations

from qiskit_nature.second_q.operators import FermionicOp


def hubbard_dimer(U: float, t: float, eps: float = 0.0) -> FermionicOp:
    """2-site, 1-orbital, spinful Hubbard dimer.

    H = eps Σ_{iσ} n_{iσ}
        - t Σ_σ (c†_{0σ} c_{1σ} + h.c.)
        + U Σ_i n_{i↑} n_{i↓}

    Parameters
    ----------
    U : on-site Coulomb repulsion
    t : nearest-neighbor hopping (positive number; sign convention is H_hop = -t ...)
    eps : on-site energy applied to every mode

    Returns
    -------
    FermionicOp on 4 spin-orbitals.
    """
    labels: dict[str, float] = {}

    # On-site eps for every mode (only if nonzero, to keep the op small).
    if eps != 0.0:
        for mode in range(4):
            labels[f"+_{mode} -_{mode}"] = eps

    # Hopping: -t between (0,1) up and (2,3) down.
    for src, dst in [(0, 1), (2, 3)]:
        labels[f"+_{src} -_{dst}"] = -t
        labels[f"+_{dst} -_{src}"] = -t

    # On-site U: site 0 = modes (0, 2); site 1 = modes (1, 3).
    for up_mode, dn_mode in [(0, 2), (1, 3)]:
        # n_up n_dn = (c†_up c_up)(c†_dn c_dn)
        labels[f"+_{up_mode} -_{up_mode} +_{dn_mode} -_{dn_mode}"] = U

    return FermionicOp(labels, num_spin_orbitals=4)


def observables_dimer() -> dict[str, FermionicOp]:
    """Return observable FermionicOps for the Hubbard dimer.

    Mode ordering matches `hubbard_dimer`: (0,1)=up at sites (0,1); (2,3)=down.

    Returns dict with keys:
        n_total      — total particle number
        n_site0      — particle number on site 0 (up + down)
        n_site1      — particle number on site 1
        S2           — total spin-squared S^2 = Sx^2 + Sy^2 + Sz^2
        double_occ   — Σ_i n_{i↑} n_{i↓} (double occupancy summed over sites)
    """
    # Per-mode number operators.
    n0u = FermionicOp({"+_0 -_0": 1.0}, num_spin_orbitals=4)
    n1u = FermionicOp({"+_1 -_1": 1.0}, num_spin_orbitals=4)
    n0d = FermionicOp({"+_2 -_2": 1.0}, num_spin_orbitals=4)
    n1d = FermionicOp({"+_3 -_3": 1.0}, num_spin_orbitals=4)

    n_site0 = (n0u + n0d).simplify()
    n_site1 = (n1u + n1d).simplify()
    n_total = (n_site0 + n_site1).simplify()

    # Double occupancy: Σ_i n_{i↑} n_{i↓}.
    double_occ = FermionicOp(
        {
            "+_0 -_0 +_2 -_2": 1.0,  # site 0
            "+_1 -_1 +_3 -_3": 1.0,  # site 1
        },
        num_spin_orbitals=4,
    )

    # Spin operators. For each site i:
    #   S+_i = c†_{i↑} c_{i↓}
    #   S-_i = c†_{i↓} c_{i↑}
    #   Sz_i = 0.5 (n_{i↑} - n_{i↓})
    # Total: S± = Σ_i S±_i, Sz = Σ_i Sz_i.
    # S^2 = Sz^2 + 0.5 (S+ S- + S- S+).
    s_plus = FermionicOp(
        {"+_0 -_2": 1.0, "+_1 -_3": 1.0},
        num_spin_orbitals=4,
    )
    s_minus = FermionicOp(
        {"+_2 -_0": 1.0, "+_3 -_1": 1.0},
        num_spin_orbitals=4,
    )
    sz = FermionicOp(
        {
            "+_0 -_0": 0.5,
            "+_1 -_1": 0.5,
            "+_2 -_2": -0.5,
            "+_3 -_3": -0.5,
        },
        num_spin_orbitals=4,
    )
    s2 = (sz @ sz + 0.5 * (s_plus @ s_minus + s_minus @ s_plus)).simplify()

    return {
        "n_total": n_total,
        "n_site0": n_site0,
        "n_site1": n_site1,
        "S2": s2,
        "double_occ": double_occ,
    }


def nio_l1_anderson(U: float, V: float, eps_d: float, eps_p: float) -> FermionicOp:
    """L1 NiO single-orbital Anderson impurity model on 4 spin-orbitals.

    Mode ordering (inherits Phase 1's all-up-then-all-down):
        0 = d_up    (impurity, up)
        1 = p_up    (bath, up)
        2 = d_dn    (impurity, down)
        3 = p_dn    (bath, down)

    H = eps_d (n_0 + n_2)
        + eps_p (n_1 + n_3)
        + V Σ_σ (d†_σ p_σ + h.c.)      (sign +V per EDRIXS convention)
        + U n_{d↑} n_{d↓}              (Coulomb only on impurity)

    Parameters
    ----------
    U : on-site Coulomb repulsion on the impurity orbital.
    V : impurity-bath hybridization. Sign convention is +V (matches
        EDRIXS hyb[bath, orb] = +Veg in example_3).
    eps_d : impurity on-site energy.
    eps_p : bath on-site energy.

    Returns
    -------
    FermionicOp on 4 spin-orbitals.
    """
    labels: dict[str, float] = {
        # Impurity on-site energy (modes 0, 2 for d_up, d_dn).
        "+_0 -_0": eps_d,
        "+_2 -_2": eps_d,
        # Bath on-site energy (modes 1, 3 for p_up, p_dn).
        "+_1 -_1": eps_p,
        "+_3 -_3": eps_p,
        # Hopping d↔p, per spin. Sign +V per EDRIXS convention.
        "+_0 -_1": V,
        "+_1 -_0": V,
        "+_2 -_3": V,
        "+_3 -_2": V,
        # Coulomb only on impurity (modes 0, 2).
        "+_0 -_0 +_2 -_2": U,
    }
    return FermionicOp(labels, num_spin_orbitals=4)


def observables_l1() -> dict[str, FermionicOp]:
    """Observables for the L1 NiO SIAM (4 spin-orbitals).

    Mode ordering matches `nio_l1_anderson`:
        0 = d_up, 1 = p_up, 2 = d_dn, 3 = p_dn.

    Returns dict with keys:
        n_d_total    — impurity particle number (n_{d↑} + n_{d↓})
        n_p_total    — bath particle number (n_{p↑} + n_{p↓})
        S2_d         — impurity-only total spin squared
        double_occ_d — impurity double occupancy n_{d↑} n_{d↓}
    """
    n_d_up = FermionicOp({"+_0 -_0": 1.0}, num_spin_orbitals=4)
    n_d_dn = FermionicOp({"+_2 -_2": 1.0}, num_spin_orbitals=4)
    n_p_up = FermionicOp({"+_1 -_1": 1.0}, num_spin_orbitals=4)
    n_p_dn = FermionicOp({"+_3 -_3": 1.0}, num_spin_orbitals=4)

    n_d_total = (n_d_up + n_d_dn).simplify()
    n_p_total = (n_p_up + n_p_dn).simplify()

    # Impurity-only S² = Sz_d² + 0.5 (S+_d S-_d + S-_d S+_d).
    s_plus_d = FermionicOp({"+_0 -_2": 1.0}, num_spin_orbitals=4)
    s_minus_d = FermionicOp({"+_2 -_0": 1.0}, num_spin_orbitals=4)
    sz_d = FermionicOp(
        {"+_0 -_0": 0.5, "+_2 -_2": -0.5},
        num_spin_orbitals=4,
    )
    s2_d = (sz_d @ sz_d + 0.5 * (s_plus_d @ s_minus_d + s_minus_d @ s_plus_d)).simplify()

    double_occ_d = FermionicOp({"+_0 -_0 +_2 -_2": 1.0}, num_spin_orbitals=4)

    return {
        "n_d_total": n_d_total,
        "n_p_total": n_p_total,
        "S2_d": s2_d,
        "double_occ_d": double_occ_d,
    }
