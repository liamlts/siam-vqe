"""Parametrized ansatz factories.

Each factory returns (circuit, initial_point). The initial point is chosen to be
a sensible starting place for the optimizer:
    - EfficientSU2:  small Gaussian noise around zero (avoids barren plateaus)
    - UCCSD:         zeros (reduces to the Hartree-Fock reference state)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

MapperScheme = Literal["jw", "parity", "parity_tapered"]


def efficient_su2_ansatz(
    num_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    seed: int | None = None,
) -> tuple[QuantumCircuit, np.ndarray]:
    """Hardware-efficient ansatz. Returns (circuit, initial_point)."""
    ansatz = EfficientSU2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        insert_barriers=False,
    )
    circuit = ansatz.decompose()
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=0.0, scale=0.1, size=circuit.num_parameters)
    return circuit, x0


def uccsd_ansatz(
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    mapper_scheme: MapperScheme = "parity_tapered",
    reps: int = 1,
) -> tuple[QuantumCircuit, np.ndarray]:
    """UCCSD ansatz initialized at Hartree-Fock (x0 = 0).

    Parameters
    ----------
    num_spatial_orbitals : number of spatial (not spin) orbitals.
    num_particles : (n_up, n_down) electron counts.
    mapper_scheme : one of "jw", "parity", "parity_tapered".
    reps : number of UCC repetitions (Trotter steps). reps>=2 avoids local
        minima in the Trotterized landscape for small systems.
    """
    mapper = _build_mapper(mapper_scheme, num_particles)
    initial_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=reps,
    )
    circuit = ansatz.decompose()
    x0 = np.zeros(circuit.num_parameters)
    return circuit, x0


def _build_mapper(
    scheme: MapperScheme,
    num_particles: tuple[int, int],
) -> JordanWignerMapper | ParityMapper:
    if scheme == "jw":
        return JordanWignerMapper()
    if scheme == "parity":
        return ParityMapper()
    if scheme == "parity_tapered":
        return ParityMapper(num_particles=num_particles)
    raise ValueError(f"unknown mapper scheme: {scheme!r}")
