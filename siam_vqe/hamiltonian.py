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


# L1 mode ordering: see nio_l1_anderson docstring. Modes 0 and 2 are the
# impurity d-orbital (up, dn); 1 and 3 are the ligand p-orbital. Downstream
# consumers (e.g. core-hole helpers) should import this constant rather than
# hardcoding indices, so a single source of truth governs the L1 layout.
_L1_IMPURITY_MODES: tuple[int, int] = (0, 2)


def nio_l1_anderson(U: float, V: float, eps_d: float, eps_p: float) -> FermionicOp:
    """L1 NiO single-orbital Anderson impurity model on 4 spin-orbitals.

    Mode ordering (inherits Phase 1's all-up-then-all-down):
        0 = d_up    (impurity, up)
        1 = p_up    (bath, up)
        2 = d_dn    (impurity, down)
        3 = p_dn    (bath, down)

    The impurity mode indices are also exported as `_L1_IMPURITY_MODES`
    for downstream consumers that need to act on the impurity orbital
    (e.g. the L1 core-hole potential).

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


def nio_l2_kanamori(
    U: float,
    U_prime: float,
    J_H: float,
    V: float,
    eps_d: float,
    eps_p: float,
) -> FermionicOp:
    """L2 NiO e_g² Anderson impurity model on 8 spin-orbitals.

    Mode ordering (per Phase 3 spec §3.1, extends Phase 2's all-up-then-all-down):
        0 = eg_a ↑ (impurity dz²-r², up)
        1 = eg_b ↑ (impurity dx²-y², up)
        2 = p_eg(a) ↑ (ligand a, up)
        3 = p_eg(b) ↑ (ligand b, up)
        4 = eg_a ↓
        5 = eg_b ↓
        6 = p_eg(a) ↓
        7 = p_eg(b) ↓

    H = ε_d Σ_{α∈{a,b}, σ} n_{d_α σ}
        + ε_p Σ_{α∈{a,b}, σ} n_{p_α σ}
        + V   Σ_{α∈{a,b}, σ} (d†_{α σ} p_{α σ} + h.c.)
        + U   Σ_α n_{d_α↑} n_{d_α↓}
        + U'  Σ_{α≠β} n_{d_α↑} n_{d_β↓}
        + (U' - J_H) Σ_{α<β, σ} n_{d_α σ} n_{d_β σ}
        − J_H Σ_{α≠β} d†_{α↑} d_{α↓} d†_{β↓} d_{β↑}    (spin-flip)
        + J_H Σ_{α≠β} d†_{α↑} d†_{α↓} d_{β↓} d_{β↑}    (pair-hop)

    Parameters
    ----------
    U : intra-orbital impurity Coulomb (same orbital, opposite spin).
    U_prime : inter-orbital impurity Coulomb (different orbitals, opposite spin).
    J_H : Hund's exchange.
    V : impurity-bath hybridization (diagonal per symmetry channel, +V sign).
    eps_d : impurity on-site energy.
    eps_p : bath on-site energy.

    Returns
    -------
    FermionicOp on 8 spin-orbitals.
    """
    labels: dict[str, float] = {}

    # On-site impurity (modes 0, 1, 4, 5)
    for mode in (0, 1, 4, 5):
        labels[f"+_{mode} -_{mode}"] = eps_d
    # On-site bath (modes 2, 3, 6, 7)
    for mode in (2, 3, 6, 7):
        labels[f"+_{mode} -_{mode}"] = eps_p

    # Diagonal hybridization per symmetry channel (a: 0↔2, 4↔6; b: 1↔3, 5↔7)
    for d_mode, p_mode in [(0, 2), (1, 3), (4, 6), (5, 7)]:
        labels[f"+_{d_mode} -_{p_mode}"] = V
        labels[f"+_{p_mode} -_{d_mode}"] = V

    # Intra-orbital U: (eg_a↑ eg_a↓) and (eg_b↑ eg_b↓)
    for d_up, d_dn in [(0, 4), (1, 5)]:
        labels[f"+_{d_up} -_{d_up} +_{d_dn} -_{d_dn}"] = U

    # Inter-orbital U', opposite spin: (eg_a↑ eg_b↓) and (eg_b↑ eg_a↓)
    for d_up, d_dn in [(0, 5), (1, 4)]:
        labels[f"+_{d_up} -_{d_up} +_{d_dn} -_{d_dn}"] = U_prime

    # Inter-orbital (U' - J), same spin: (eg_a↑ eg_b↑) and (eg_a↓ eg_b↓)
    for d_a, d_b in [(0, 1), (4, 5)]:
        labels[f"+_{d_a} -_{d_a} +_{d_b} -_{d_b}"] = U_prime - J_H

    # Spin-flip: -J Σ_{α≠β} d†_{α↑} d_{α↓} d†_{β↓} d_{β↑}
    # α=a, β=b: -J  d†_{0} d_{4} d†_{5} d_{1}
    # α=b, β=a: -J  d†_{1} d_{5} d†_{4} d_{0}
    labels["+_0 -_4 +_5 -_1"] = -J_H
    labels["+_1 -_5 +_4 -_0"] = -J_H

    # Pair-hop: +J Σ_{α≠β} d†_{α↑} d†_{α↓} d_{β↓} d_{β↑}
    # α=a, β=b: +J  d†_{0} d†_{4} d_{5} d_{1}
    # α=b, β=a: +J  d†_{1} d†_{5} d_{4} d_{0}
    labels["+_0 +_4 -_5 -_1"] = J_H
    labels["+_1 +_5 -_4 -_0"] = J_H

    return FermionicOp(labels, num_spin_orbitals=8)


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


def observables_l2() -> dict[str, FermionicOp]:
    """Observables for the L2 NiO e_g² SIAM (8 spin-orbitals).

    Mode ordering matches `nio_l2_kanamori`:
        0=eg_a↑, 1=eg_b↑, 2=p_a↑, 3=p_b↑, 4=eg_a↓, 5=eg_b↓, 6=p_a↓, 7=p_b↓.

    Returns dict with keys:
        n_d_total          — Σ_{α∈{a,b}, σ} n_{d_α σ}
        n_p_total          — Σ_{α∈{a,b}, σ} n_{p_α σ}
        S2_d               — impurity total spin squared (sum across e_g)
        double_occ_d       — Σ_α n_{d_α↑} n_{d_α↓}
        n_d_a_minus_n_d_b  — n_{eg_a, total} − n_{eg_b, total} (orbital polarization)
    """
    n_d_modes = [0, 1, 4, 5]
    n_p_modes = [2, 3, 6, 7]

    n_d_total = FermionicOp(
        {f"+_{m} -_{m}": 1.0 for m in n_d_modes}, num_spin_orbitals=8
    )
    n_p_total = FermionicOp(
        {f"+_{m} -_{m}": 1.0 for m in n_p_modes}, num_spin_orbitals=8
    )

    s_plus_d = FermionicOp(
        {"+_0 -_4": 1.0, "+_1 -_5": 1.0}, num_spin_orbitals=8
    )
    s_minus_d = FermionicOp(
        {"+_4 -_0": 1.0, "+_5 -_1": 1.0}, num_spin_orbitals=8
    )
    sz_d = FermionicOp(
        {
            "+_0 -_0": 0.5, "+_1 -_1": 0.5,
            "+_4 -_4": -0.5, "+_5 -_5": -0.5,
        },
        num_spin_orbitals=8,
    )
    s2_d = (sz_d @ sz_d + 0.5 * (s_plus_d @ s_minus_d + s_minus_d @ s_plus_d)).simplify()

    double_occ_d = FermionicOp(
        {
            "+_0 -_0 +_4 -_4": 1.0,
            "+_1 -_1 +_5 -_5": 1.0,
        },
        num_spin_orbitals=8,
    )

    n_d_a_minus_n_d_b = FermionicOp(
        {
            "+_0 -_0": 1.0, "+_4 -_4": 1.0,
            "+_1 -_1": -1.0, "+_5 -_5": -1.0,
        },
        num_spin_orbitals=8,
    )

    return {
        "n_d_total": n_d_total.simplify(),
        "n_p_total": n_p_total.simplify(),
        "S2_d": s2_d,
        "double_occ_d": double_occ_d,
        "n_d_a_minus_n_d_b": n_d_a_minus_n_d_b.simplify(),
    }
