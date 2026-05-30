"""Fermion-to-qubit mapping wrappers for siam_vqe.

Three schemes are exposed:
    "jw"              — bare Jordan-Wigner
    "parity"          — parity mapping
    "parity_tapered"  — parity + Z2 symmetry reduction (particle number + Sz),
                        removing 2 qubits for spinful Hamiltonians.

The `num_particles` argument (n_up, n_down) is REQUIRED for "parity_tapered"
because the tapering must project onto the correct symmetry sector.
"""

from __future__ import annotations

from typing import Literal

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators import FermionicOp

Scheme = Literal["jw", "parity", "parity_tapered"]


def to_qubit_op(
    fop: FermionicOp,
    scheme: Scheme,
    num_particles: tuple[int, int] | None = None,
) -> SparsePauliOp:
    """Map a FermionicOp to a SparsePauliOp under the requested scheme.

    Parameters
    ----------
    fop : FermionicOp on N spin-orbitals.
    scheme : one of "jw", "parity", "parity_tapered".
    num_particles : (n_up, n_down). Required when scheme="parity_tapered".

    Returns
    -------
    SparsePauliOp; qubit count = N for "jw"/"parity", N-2 for "parity_tapered".
    """
    if scheme == "jw":
        return _ensure_sparse_pauli(JordanWignerMapper().map(fop))

    if scheme == "parity":
        return _ensure_sparse_pauli(ParityMapper().map(fop))

    if scheme == "parity_tapered":
        if num_particles is None:
            raise ValueError("parity_tapered scheme requires num_particles=(n_up, n_dn)")
        mapper = ParityMapper(num_particles=num_particles)
        return _ensure_sparse_pauli(mapper.map(fop))

    raise ValueError(f"unknown mapping scheme: {scheme!r}")


def _ensure_sparse_pauli(op: object) -> SparsePauliOp:
    """Qiskit Nature mappers return SparsePauliOp; this is a typed identity guard."""
    if not isinstance(op, SparsePauliOp):
        raise TypeError(f"expected SparsePauliOp, got {type(op).__name__}")
    return op
