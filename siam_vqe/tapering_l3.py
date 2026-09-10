"""L3 tapering orchestrator.

Wraps `mappings.to_qubit_op(scheme='parity_tapered')` to apply
parity + (N=18, S_z=0) Z₂ tapering to the L3 Hamiltonian.
18 qubits (down from 20-spin-orbital bare JW).
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian_l3 import L3Params
from siam_vqe.mappings import to_qubit_op
from siam_vqe.reference_ed import _build_sector_basis


def tapered_l3_pauli(
    fermionic_op: FermionicOp,
    num_particles: tuple[int, int] = (9, 9),
) -> SparsePauliOp:
    """JW + parity + (N, S_z) Z₂ tapering for the L3 Hamiltonian.

    Parameters
    ----------
    fermionic_op : FermionicOp
        Output of `nio_l3_hamiltonian` (20 spin-orbitals).
    num_particles : tuple (n_up, n_down)
        Sector projection. Default (9, 9) for d⁸ + 10 bath electrons, S_z=0.

    Returns
    -------
    SparsePauliOp on 18 qubits.
    """
    return to_qubit_op(fermionic_op, scheme="parity_tapered",
                       num_particles=num_particles)


def num_tapered_qubits(params: L3Params) -> int:
    """Return the qubit count after parity + (N, S_z) tapering.

    Bare JW: 2 × n_spatial. Parity + (N, S_z) tapers off 2 qubits.
    """
    return params.num_spin_orbitals - 2


def _tapering_basis_map(
    num_spin_orbitals: int,
    num_particles: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the index-and-sign mapping between sector basis and tapered Hilbert space.

    Returns:
        sector_basis : np.ndarray of occupation-number ints, shape (sector_dim,)
        tap : structured array shape (sector_dim, 2): col 0 = tapered_idx (uint32),
            col 1 = sign (int8)
    """
    sector_basis = _build_sector_basis(
        num_d_spin_orbitals=num_spin_orbitals,
        n_up=num_particles[0],
        n_down=num_particles[1],
    )

    n = num_spin_orbitals
    half = n // 2
    tap_idx = np.zeros(len(sector_basis), dtype=np.uint32)
    signs = np.ones(len(sector_basis), dtype=np.int8)

    for i, occ in enumerate(sector_basis):
        # Parity-basis transform: bit i = XOR over bits 0..i of |occ⟩.
        par_bits = 0
        running_par = 0
        for bit_i in range(n):
            running_par ^= (occ >> bit_i) & 1
            par_bits |= running_par << bit_i

        # Remove bit (n-1): total-parity qubit (S_z-parity)
        bits_after_total = (par_bits & ((1 << (n - 1)) - 1))
        # Remove bit (half - 1): up-spin block parity (N-parity for up)
        upper = bits_after_total >> half
        lower = bits_after_total & ((1 << (half - 1)) - 1)
        new_bits = (upper << (half - 1)) | lower

        tap_idx[i] = new_bits
        signs[i] = 1  # see note below

    return np.array(sector_basis, dtype=np.int64), np.stack([tap_idx, signs], axis=1)


def project_full_to_tapered(
    psi_full: np.ndarray,
    num_particles: tuple[int, int] = (9, 9),
) -> np.ndarray:
    """Project a bare-JW statevector (2^N) onto the tapered space (2^(N-2))."""
    n = int(np.log2(len(psi_full)))
    if 2**n != len(psi_full):
        raise ValueError(f"psi_full length {len(psi_full)} is not a power of 2")
    sector_basis, tap = _tapering_basis_map(n, num_particles)
    n_tap = n - 2
    psi_tapered = np.zeros(2**n_tap, dtype=complex)
    for i, occ in enumerate(sector_basis):
        psi_tapered[tap[i, 0]] += tap[i, 1] * psi_full[occ]
    return psi_tapered


def lift_tapered_to_full_sector(
    psi_tapered: np.ndarray,
    num_particles: tuple[int, int] = (9, 9),
    num_spin_orbitals: int = 20,
) -> np.ndarray:
    """Inverse of project_full_to_tapered, restricted to (n_up, n_down) sector.

    Returns a vector of length = sector dimension (100 for L3 defaults).
    """
    sector_basis, tap = _tapering_basis_map(num_spin_orbitals, num_particles)
    psi_sector = np.zeros(len(sector_basis), dtype=complex)
    for i in range(len(sector_basis)):
        psi_sector[i] = tap[i, 1] * psi_tapered[tap[i, 0]]
    return psi_sector
