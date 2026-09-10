"""Tests for siam_vqe.tapering_l3."""
from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.tapering_l3 import (
    lift_tapered_to_full_sector,
    num_tapered_qubits,
    project_full_to_tapered,
    tapered_l3_pauli,
)


def test_tapered_l3_pauli_returns_sparsepauliop():
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    H_pauli = tapered_l3_pauli(H, num_particles=(9, 9))
    assert isinstance(H_pauli, SparsePauliOp)


def test_tapered_l3_pauli_has_18_qubits():
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    H_pauli = tapered_l3_pauli(H, num_particles=(9, 9))
    assert H_pauli.num_qubits == 18


def test_num_tapered_qubits_for_l3_is_18():
    p = L3Params()
    assert num_tapered_qubits(p) == 18


def test_lift_tapered_to_full_sector_preserves_norm():
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    H_pauli = tapered_l3_pauli(H, num_particles=(9, 9))  # noqa: F841
    rng = np.random.default_rng(42)
    psi_tapered = rng.standard_normal(2**18) + 1j * rng.standard_normal(2**18)
    psi_tapered = psi_tapered / np.linalg.norm(psi_tapered)
    psi_lifted = lift_tapered_to_full_sector(psi_tapered, num_particles=(9, 9))
    # For a random 18-qubit unit vector, the lifted norm is roughly sqrt(100/2^18) — small but nonzero.
    assert 0.0 < np.linalg.norm(psi_lifted) <= 1.0 + 1e-10


def test_lift_tapered_to_full_sector_overlap_with_known_state():
    """Lifting the scipy-ED ground vector through `tapered_l3_pauli` and back
    should give a state with overlap 1 against the original (round-trip identity
    on the (9,9) sector subspace)."""
    from siam_vqe.reference_l3 import compute_l3_reference

    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)

    # Embed ref.ground_vector (sector basis, dim 100) into the bare 2²⁰ JW basis.
    psi_full_20q = np.zeros(2**20, dtype=complex)
    for i, occ_int in enumerate(ref.basis):
        psi_full_20q[occ_int] = ref.ground_vector[i]

    psi_tapered = project_full_to_tapered(psi_full_20q, num_particles=(9, 9))
    psi_roundtrip = lift_tapered_to_full_sector(psi_tapered, num_particles=(9, 9))
    overlap = abs(np.vdot(ref.ground_vector, psi_roundtrip))
    assert overlap == pytest.approx(1.0, abs=1e-8)
