"""Tests for siam_vqe.mappings — fermion-to-qubit mapping schemes."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from siam_vqe.hamiltonian import hubbard_dimer
from siam_vqe.mappings import to_qubit_op


@pytest.mark.parametrize("scheme", ["jw", "parity", "parity_tapered"])
def test_to_qubit_op_returns_sparse_pauli(scheme: str, dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    num_particles = (1, 1)  # (n_up, n_down) at half-filling
    pop = to_qubit_op(fop, scheme=scheme, num_particles=num_particles)
    assert isinstance(pop, SparsePauliOp)


def test_jw_dimer_qubit_count(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="jw", num_particles=(1, 1))
    assert pop.num_qubits == 4


def test_parity_dimer_qubit_count(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity", num_particles=(1, 1))
    assert pop.num_qubits == 4


def test_parity_tapered_dimer_reduces_qubits(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    # Parity mapping + two-qubit reduction via N + Sz removes 2 qubits.
    assert pop.num_qubits == 2


def test_all_schemes_share_groundstate_energy(dimer_params: dict[str, float]) -> None:
    """JW / parity / parity_tapered should agree on the N=(1,1) sector GS energy.

    JW and parity return Hamiltonians on the full Fock space; we project them
    into the same N=2 sector that parity_tapered already encodes, then compare
    the lowest eigenvalues.
    """
    num_particles = (1, 1)
    fop = hubbard_dimer(**dimer_params)
    from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
    from qiskit_nature.second_q.operators import FermionicOp

    jw_mapper = JordanWignerMapper()
    parity_mapper = ParityMapper()

    # Build particle-number operator and N=2 projection mask in JW basis.
    n_op = FermionicOp(
        {"+_0 -_0": 1.0, "+_1 -_1": 1.0, "+_2 -_2": 1.0, "+_3 -_3": 1.0},
        num_spin_orbitals=4,
    )
    n_diag_jw = np.diag(jw_mapper.map(n_op).to_matrix()).real
    n2_mask_jw = np.abs(n_diag_jw - 2.0) < 1e-8

    # Build N=2 projection mask in parity basis.
    n_mat_p = parity_mapper.map(n_op).to_matrix()
    n_diag_p = np.diag(n_mat_p).real
    n2_mask_p = np.abs(n_diag_p - 2.0) < 1e-8

    energies = []
    for scheme in ["jw", "parity", "parity_tapered"]:
        pop = to_qubit_op(fop, scheme=scheme, num_particles=num_particles)
        h = pop.to_matrix()
        if scheme == "parity_tapered":
            # Already in the (1,1) sector; full spectrum is in-sector.
            eigvals = np.linalg.eigvalsh(h)
        elif scheme == "jw":
            h_proj = h[np.ix_(n2_mask_jw, n2_mask_jw)]
            eigvals = np.linalg.eigvalsh(h_proj)
        else:
            # parity basis: project via parity-basis N=2 mask
            h_proj = h[np.ix_(n2_mask_p, n2_mask_p)]
            eigvals = np.linalg.eigvalsh(h_proj)
        energies.append(float(eigvals[0]))
    # All three N=2-sector GS energies should agree.
    assert energies[0] == pytest.approx(energies[1], abs=1e-10)
    assert energies[0] == pytest.approx(energies[2], abs=1e-10)
