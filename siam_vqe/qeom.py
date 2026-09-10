"""qEOM on top of Phase 4's tapered statevector ground state.

Bauer 2020 / McClean 2017 / Ollitrault 2020 formulation. The qEOM
matrices M and S are built from commutators on the ψ'_GS reference;
solving M X = E S X (generalized eigenvalue problem) yields excited
states with energies E_n and amplitudes X.

Cross-sector matrix elements arise because the dipole D = Σ c†_d maps
the Phase 4 (9, 9) ground state to (10, 9) ⊕ (9, 10). We lift both
sides to the full untapered JW basis (2²⁰ dim at L3, 16 MB complex128
— easily fits), apply the operator, and take the inner product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


def cross_sector_matrix_element(
    *,
    psi_left_full: np.ndarray,
    op: FermionicOp,
    psi_right_full: np.ndarray,
    num_spin_orbitals: int,
) -> complex:
    """Return ⟨psi_left | op | psi_right⟩ in the full 2^N JW basis.

    Both statevectors must live in the full untapered JW Hilbert space
    (use ``lift_tapered_to_full_sector`` to bridge from tapered form).
    The operator is mapped to its qubit representation via Jordan-Wigner
    and applied as a sparse matrix.
    """
    if len(psi_left_full) != 2**num_spin_orbitals:
        raise ValueError(
            f"psi_left length {len(psi_left_full)} ≠ "
            f"2^{num_spin_orbitals} = {2**num_spin_orbitals}"
        )
    if len(psi_right_full) != 2**num_spin_orbitals:
        raise ValueError(
            f"psi_right length {len(psi_right_full)} ≠ "
            f"2^{num_spin_orbitals} = {2**num_spin_orbitals}"
        )

    op_qubit = JordanWignerMapper().map(op)
    op_mat = op_qubit.to_matrix(sparse=True)
    out = op_mat @ psi_right_full
    return complex(np.vdot(psi_left_full, out))


def build_eom_matrices(
    H_pauli: SparsePauliOp,
    pool: list[SparsePauliOp],
    psi_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build M and S matrices for the Bauer qEOM problem.

    M_{mn} = ⟨ψ_ref | [T_m†, [H, T_n]] | ψ_ref⟩  (double commutator)
    S_{mn} = ⟨ψ_ref | [T_m†, T_n] | ψ_ref⟩       (overlap / metric)

    Both are Hermitized numerically: M ← ½(M + M†), S ← ½(S + S†).
    """
    H = H_pauli.to_matrix(sparse=True)
    Ts = [T.to_matrix(sparse=True) for T in pool]
    Ts_dag = [T.conjugate().T for T in Ts]
    K = len(pool)

    H_psi = H @ psi_ref
    T_psi = [T @ psi_ref for T in Ts]
    Tdag_psi = [Td @ psi_ref for Td in Ts_dag]
    HT_psi = [H @ tp for tp in T_psi]  # H T_n |ψ⟩

    M = np.zeros((K, K), dtype=complex)
    S = np.zeros((K, K), dtype=complex)
    for m in range(K):
        Tdag_m = Ts_dag[m]
        for n in range(K):
            # M_{mn} = ⟨ψ| T_m† [H, T_n] |ψ⟩ + ⟨ψ| [T_m†, H] T_n |ψ⟩,
            # symmetrized below. We compute the standard form first:
            #   ⟨ψ| T_m† H T_n |ψ⟩ − ⟨ψ| T_m† T_n H |ψ⟩
            # then add the h.c. and ÷ 2.
            term_a = np.vdot(Tdag_m.conjugate().T @ psi_ref, HT_psi[n])  # ⟨ψ|T_m† H T_n|ψ⟩
            term_b = np.vdot(Tdag_m.conjugate().T @ psi_ref, Ts[n] @ H_psi)  # ⟨ψ|T_m† T_n H|ψ⟩
            M[m, n] = term_a - term_b

            # S_{mn} = ⟨ψ|T_m† T_n|ψ⟩ − ⟨ψ|T_n T_m†|ψ⟩
            term_c = np.vdot(Tdag_m.conjugate().T @ psi_ref, Ts[n] @ psi_ref)
            term_d = np.vdot(psi_ref, Ts[n] @ Tdag_psi[m])  # ⟨ψ| T_n T_m†|ψ⟩
            S[m, n] = term_c - term_d

    # Hermitize numerically
    M = 0.5 * (M + M.conj().T)
    S = 0.5 * (S + S.conj().T)
    return M, S


@dataclass(frozen=True)
class QEOMResult:
    """Result of solving the qEOM generalized eigenvalue problem.

    Attributes
    ----------
    energies_eV : shape (K',) — sorted excitation energies (eV) relative
        to ⟨ψ_ref|H|ψ_ref⟩.
    amplitudes : shape (K', K'') — excited-state amplitudes (rows = states,
        cols = pool operators in the surviving subspace).
    pool_size : original pool size K.
    s_dropped : number of S-eigenvalues dropped by the pseudoinverse cleanup.
    """

    energies_eV: np.ndarray
    amplitudes: np.ndarray
    pool_size: int
    s_dropped: int


def solve_qeom(
    M: np.ndarray,
    S: np.ndarray,
    s_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenvalue problem M X = E S X via
    S-pseudoinverse projection.

    1. Diagonalize S = V Λ V†.
    2. Keep only modes with Λ_i > s_tol × Λ_max (others are null directions).
    3. In the surviving subspace, solve the standard eigenproblem
       Λ^{-1/2} V† M V Λ^{-1/2} y = E y.
    4. Map y back to the original pool basis: X = V Λ^{-1/2} y.

    Returns (energies, amplitudes). Energies are sorted ascending.
    """
    # Symmetrize for safety
    M_sym = 0.5 * (M + M.conj().T)
    S_sym = 0.5 * (S + S.conj().T)

    lam_S, V_S = scipy.linalg.eigh(S_sym)
    lam_max = max(lam_S.max(), 1.0)
    keep = lam_S > s_tol * lam_max

    if not np.any(keep):
        # All-null S: no qEOM solutions
        return np.zeros(0), np.zeros((0, M.shape[0]))

    V_keep = V_S[:, keep]
    lam_keep = lam_S[keep]
    inv_sqrt_lam = 1.0 / np.sqrt(lam_keep)

    # Project M into the well-conditioned subspace
    M_proj = V_keep.conj().T @ M_sym @ V_keep
    M_proj = (inv_sqrt_lam[:, None] * M_proj) * inv_sqrt_lam[None, :]
    M_proj = 0.5 * (M_proj + M_proj.conj().T)  # restore symmetry

    energies, y = scipy.linalg.eigh(M_proj)
    # Map y back to the original pool basis
    X = V_keep @ (inv_sqrt_lam[:, None] * y)
    return energies, X.T  # rows = states


def solve_qeom_struct(
    M: np.ndarray,
    S: np.ndarray,
    s_tol: float = 1e-8,
) -> QEOMResult:
    """Convenience wrapper returning a `QEOMResult` instead of (e, X)."""
    energies, X = solve_qeom(M, S, s_tol=s_tol)
    return QEOMResult(
        energies_eV=energies,
        amplitudes=X,
        pool_size=M.shape[0],
        s_dropped=M.shape[0] - len(energies),
    )


def compute_spectral_weights(
    *,
    psi_gs_full: np.ndarray,
    psi_prime_gs_full: np.ndarray,
    pool_fermi: list[FermionicOp | SparsePauliOp],
    amplitudes: np.ndarray,
    dipole: FermionicOp,
    num_spin_orbitals: int,
) -> np.ndarray:
    """Compute |⟨F_n|D|ψ_GS⟩|² for each qEOM excited state.

    Per Phase 5 spec §4.5:
        ⟨F_n | D | ψ_GS⟩ = Σ_m X_n[m]* · ⟨ψ'_GS | T_m† D | ψ_GS⟩

    The matrix element ⟨ψ'_GS | T_m† D | ψ_GS⟩ is cross-sector: D maps
    the initial GS to a different (Nu, Nd) sector, then T_m† acts in the
    final sector (where ψ'_GS lives). Both states must be lifted to the
    full 2^N JW Hilbert space; the operator products are applied as
    sparse matrices.

    Parameters
    ----------
    psi_gs_full : initial ground state, full JW basis (2^N).
    psi_prime_gs_full : final-sector reference state used as the qEOM
        expansion point, full JW basis (2^N).
    pool_fermi : pool operators T_m. Each entry may be either a
        FermionicOp (mapped to qubits via Jordan-Wigner) or already a
        SparsePauliOp (used directly). Must be the same pool passed to
        `build_eom_matrices`.
    amplitudes : shape (K', K) array of qEOM amplitudes (rows = excited
        states, cols = pool operators).
    dipole : dipole operator D (FermionicOp).
    num_spin_orbitals : N — number of spin-orbitals (qubits).

    Returns
    -------
    weights : shape (K',) array of nonnegative spectral weights
        |⟨F_n|D|ψ_GS⟩|².
    """
    if len(psi_gs_full) != 2**num_spin_orbitals:
        raise ValueError(
            f"psi_gs length {len(psi_gs_full)} ≠ "
            f"2^{num_spin_orbitals} = {2**num_spin_orbitals}"
        )
    if len(psi_prime_gs_full) != 2**num_spin_orbitals:
        raise ValueError(
            f"psi_prime_gs length {len(psi_prime_gs_full)} ≠ "
            f"2^{num_spin_orbitals} = {2**num_spin_orbitals}"
        )

    jw = JordanWignerMapper()
    K = len(pool_fermi)

    # Precompute D|ψ_GS⟩ (lives in a different sector than ψ_GS)
    D_pauli = jw.map(dipole)
    D_mat = D_pauli.to_matrix(sparse=True)
    D_psi_gs = D_mat @ psi_gs_full

    # Compute ⟨ψ'_GS | T_m† D | ψ_GS⟩ for each pool operator m.
    tmd_matrix_elements = np.zeros(K, dtype=complex)
    for m, T in enumerate(pool_fermi):
        T_pauli = jw.map(T) if isinstance(T, FermionicOp) else T
        T_dag_pauli = T_pauli.adjoint()
        T_dag_mat = T_dag_pauli.to_matrix(sparse=True)
        tmd_matrix_elements[m] = np.vdot(psi_prime_gs_full, T_dag_mat @ D_psi_gs)

    # amplitudes shape: (K', K) — rows are excited states.
    # spectral weight for state n: |Σ_m X_n[m]* · tmd[m]|²
    overlaps = amplitudes.conj() @ tmd_matrix_elements
    weights: np.ndarray = np.abs(overlaps) ** 2
    return weights
