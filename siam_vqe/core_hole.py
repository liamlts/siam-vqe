"""Effective core-hole potential for L-edge XAS on the L3 NiO SIAM.

The core hole is treated in the effective-core-hole approximation: the 2p
core electron is removed from the Hilbert space, and its attractive
interaction with the d-electrons is encoded as a monopole potential
V_core = -U_dc Σ_{α,σ} n_{d_α σ} on the valence d-orbitals.

V_core is strictly particle-conserving on the valence subsystem. The XAS
final-state sector ((10, 9) or (9, 10)) encodes the +1 particle shift
relative to the (9, 9) initial state.

References:
- Haverkort PRB 85, 165113 (2012): U_dc as F^0_dc Slater-Condon integral.
- EDRIXS example_3: U_dc = 8.5 eV.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.tapering_l3 import tapered_l3_pauli


@dataclass(frozen=True)
class CoreHoleParams:
    """Effective core-hole potential parameters.

    Attributes
    ----------
    U_dc : float
        Monopole core-valence attraction (F^0_dc) in eV. EDRIXS example_3
        default = 8.5.
    include_multipoles : bool
        If True, add G^1_dc and G^3_dc Slater-Condon multipoles (v2 feature;
        not implemented in v1).
    """
    U_dc: float = 8.5
    include_multipoles: bool = False


def v_core_operator(params: L3Params, ch: CoreHoleParams) -> FermionicOp:
    """Build V_core = -U_dc Σ_{α∈d, σ} n_{d_α σ} on the L3 spin-orbital lattice.

    Mode ordering (Phase 4 convention):
        0..4   d↑   (eg, eg, t2g, t2g, t2g)
        5..9   bath↑
        10..14 d↓
        15..19 bath↓

    V_core acts only on d-orbital modes; the bath sees no core-hole
    potential.
    """
    if ch.include_multipoles:
        raise NotImplementedError(
            "G^1_dc / G^3_dc multipoles are deferred to Phase 5 v2."
        )

    half = params.num_spin_orbitals // 2
    labels: dict[str, float] = {}
    for alpha in range(params.num_d_orbitals):
        for spin_offset in (0, half):
            mode = alpha + spin_offset
            labels[f"+_{mode} -_{mode}"] = -ch.U_dc

    return FermionicOp(labels, num_spin_orbitals=params.num_spin_orbitals)


def nio_l3_core_hole_hamiltonian(
    params: L3Params,
    ch: CoreHoleParams,
) -> FermionicOp:
    """H' = H + V_core for the L3 NiO SIAM.

    H is the Phase 4 valence Hamiltonian (no core hole); V_core is the
    effective monopole core-hole potential on the d-electrons.

    The (10, 9) and (9, 10) sectors of H' carry the XAS final-state
    manifold for the two spin channels.
    """
    H = nio_l3_hamiltonian(params)
    V = v_core_operator(params, ch)
    return (H + V).simplify()


def tapered_l3_h_prime_pauli(
    params: L3Params,
    ch: CoreHoleParams,
    *,
    num_particles: tuple[int, int],
) -> SparsePauliOp:
    """Build H' = H + V_core and project to its tapered Pauli form in the
    requested (n_up, n_dn) sector.

    For XAS final states, use (10, 9) or (9, 10). The (9, 9) sector is
    valid (gives Phase 4's tapered H with a U_dc shift on its d-occupation
    expectation value) but not physically meaningful for XAS.
    """
    H_prime = nio_l3_core_hole_hamiltonian(params, ch)
    return tapered_l3_pauli(H_prime, num_particles=num_particles)
