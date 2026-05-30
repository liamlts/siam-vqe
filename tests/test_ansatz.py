"""Tests for siam_vqe.ansatz — parametrized ansatz circuits."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from siam_vqe.ansatz import efficient_su2_ansatz, uccsd_ansatz
from siam_vqe.hamiltonian import hubbard_dimer
from siam_vqe.mappings import to_qubit_op


def test_efficient_su2_returns_circuit_and_x0() -> None:
    circuit, x0 = efficient_su2_ansatz(num_qubits=4, reps=2, seed=20260524)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4
    assert circuit.num_parameters > 0
    assert x0.shape == (circuit.num_parameters,)


def test_efficient_su2_x0_is_small(rng: np.random.Generator) -> None:
    _, x0 = efficient_su2_ansatz(num_qubits=4, reps=2, seed=42)
    # Initial point should be Gaussian noise close to zero (~ N(0, 0.1)).
    assert float(np.std(x0)) < 0.5
    assert float(np.abs(x0).max()) < 1.0


def test_uccsd_ansatz_returns_circuit_and_x0(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = uccsd_ansatz(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        mapper_scheme="parity_tapered",
    )
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == pop.num_qubits
    assert circuit.num_parameters > 0
    assert x0.shape == (circuit.num_parameters,)


def test_uccsd_x0_is_zero(dimer_params: dict[str, float]) -> None:
    """UCCSD with x0 = 0 reduces to Hartree-Fock (the reference state)."""
    _, x0 = uccsd_ansatz(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        mapper_scheme="parity_tapered",
    )
    assert np.allclose(x0, 0.0)
