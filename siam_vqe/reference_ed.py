"""Classical exact diagonalization via scipy.sparse for small qubit Hamiltonians.

This module is the validation oracle for VQE runs. It uses the SAME SparsePauliOp
that VQE consumes, so any disagreement is in the VQE side — not in differing
Hamiltonian constructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qiskit.quantum_info import SparsePauliOp


@dataclass(frozen=True)
class EDResult:
    """Result of an exact diagonalization.

    Attributes
    ----------
    energies : (k,) ndarray of the k lowest eigenvalues, sorted ascending.
    vectors  : (D, k) ndarray of the corresponding eigenvectors as columns.
    hamiltonian : the SparsePauliOp that was diagonalized (for consistency).
    """

    energies: np.ndarray
    vectors: np.ndarray
    hamiltonian: SparsePauliOp


def exact_diag(hamiltonian: SparsePauliOp, k: int = 4) -> EDResult:
    """Return the k lowest eigenvalues + eigenvectors of `hamiltonian`.

    For ≤8 qubits we use dense eigh (fast, exact). For >8 qubits we use
    scipy.sparse.linalg.eigsh on the CSR form. Always returns dense
    eigenvectors (`vectors` is np.ndarray, not sparse) for downstream
    overlap/expectation calculations.
    """
    n_qubits = hamiltonian.num_qubits
    dim = 1 << n_qubits

    if n_qubits <= 8:
        h_dense = hamiltonian.to_matrix()
        eigvals, eigvecs = np.linalg.eigh(h_dense)
        return EDResult(
            energies=eigvals[:k].copy(),
            vectors=eigvecs[:, :k].copy(),
            hamiltonian=hamiltonian,
        )

    # Sparse path for larger problems.
    h_sparse = sp.csr_matrix(hamiltonian.to_matrix(sparse=True))
    # eigsh needs k < dim - 1; clamp k defensively.
    k_eff = min(k, dim - 2)
    eigvals, eigvecs = spla.eigsh(h_sparse, k=k_eff, which="SA")
    idx = np.argsort(eigvals)
    return EDResult(
        energies=eigvals[idx],
        vectors=eigvecs[:, idx],
        hamiltonian=hamiltonian,
    )


def _build_sector_basis(num_d_spin_orbitals: int, n_up: int, n_down: int) -> list[int]:
    """Enumerate occupation-number basis states in (n_up, n_down) sector.

    Mode ordering: spin-up modes are the first half, spin-down modes the second.
    Each state is an integer whose bits index occupied modes.
    """
    half = num_d_spin_orbitals // 2
    up_states = [
        sum(1 << i for i in combo)
        for combo in combinations(range(half), n_up)
    ]
    dn_states = [
        sum(1 << (i + half) for i in combo)
        for combo in combinations(range(half), n_down)
    ]
    return sorted(u | d for u in up_states for d in dn_states)


def _fermionic_op_to_sparse_matrix(op: Any, basis: list[int]) -> sp.csr_matrix:
    """Build a sparse matrix of `op` in the given basis (occupation-number Fock states).

    Each FermionicOp label like '+_2 -_0 +_5 -_3' is applied bit-by-bit, with
    Jordan-Wigner signs from the count of occupied modes with lower index.
    """
    import scipy.sparse as sps
    n = len(basis)
    state_to_idx = {s: i for i, s in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for label, coeff in op.items():
        for col, state_in in enumerate(basis):
            sign = 1
            state = state_in
            ok = True
            for token in reversed(label.split()):
                kind, idx_str = token.split("_")
                idx = int(idx_str)
                bit = 1 << idx
                if kind == "+":
                    if state & bit:
                        ok = False
                        break
                    lower_mask = bit - 1
                    sign *= (-1) ** bin(state & lower_mask).count("1")
                    state |= bit
                elif kind == "-":
                    if not (state & bit):
                        ok = False
                        break
                    lower_mask = bit - 1
                    sign *= (-1) ** bin(state & lower_mask).count("1")
                    state &= ~bit
                else:
                    raise ValueError(f"bad token {token}")
            if not ok:
                continue
            if state not in state_to_idx:
                continue
            rows.append(state_to_idx[state])
            cols.append(col)
            vals.append(sign * coeff)
    return sps.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)
