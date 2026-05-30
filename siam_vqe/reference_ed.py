"""Classical exact diagonalization via scipy.sparse for small qubit Hamiltonians.

This module is the validation oracle for VQE runs. It uses the SAME SparsePauliOp
that VQE consumes, so any disagreement is in the VQE side — not in differing
Hamiltonian constructions.
"""

from __future__ import annotations

from dataclasses import dataclass

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
