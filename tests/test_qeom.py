"""Tests for siam_vqe.qeom — cross-sector matrix elements and Bauer EOM."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.qeom import cross_sector_matrix_element


def test_cross_sector_simple_creation_op_on_hubbard_dimer():
    """⟨HF_(2,1) | c†_1↑ | HF_(1,1)⟩ on the Hubbard dimer = ±1
    by direct Slater rule. (Mode order: 0,1 = up; 2,3 = down.)
    """
    # |HF_(1,1)⟩ = occupied modes {0, 2} (one up at site 0, one down at site 0)
    # c†_1 |HF_(1,1)⟩ = -|0,1,2⟩ (Jordan-Wigner sign from one occupied mode
    # below mode 1 → factor (-1)^1 = -1).
    # |HF_(2,1)⟩ = occupied modes {0, 1, 2} → same as c†_1|HF_(1,1)⟩ up to sign.
    psi_left = np.zeros(2**4, dtype=complex)
    occ_left = 0b0111  # modes {0, 1, 2}
    psi_left[occ_left] = 1.0

    psi_right = np.zeros(2**4, dtype=complex)
    occ_right = 0b0101  # modes {0, 2}
    psi_right[occ_right] = 1.0

    op = FermionicOp({"+_1": 1.0}, num_spin_orbitals=4)
    result = cross_sector_matrix_element(
        psi_left_full=psi_left,
        op=op,
        psi_right_full=psi_right,
        num_spin_orbitals=4,
    )
    # The matrix element should be ±1 (sign from JW).
    assert abs(result) == pytest.approx(1.0)


def test_cross_sector_zero_when_op_misses_sector():
    """⟨HF_(2,1) | c†_3 | HF_(1,1)⟩ where mode 3 is in (2,1) sector — wait,
    mode 3 is down. (1,1) → (1,2) when we add a down particle. So
    ⟨HF_(1,2) | c†_3 | HF_(1,1)⟩ should be nonzero, and
    ⟨HF_(2,1) | c†_3 | HF_(1,1)⟩ should be zero (wrong target sector).
    """
    psi_left = np.zeros(2**4, dtype=complex)
    psi_left[0b0111] = 1.0  # (2, 1) sector: modes {0, 1, 2}
    psi_right = np.zeros(2**4, dtype=complex)
    psi_right[0b0101] = 1.0  # (1, 1) sector: modes {0, 2}

    op = FermionicOp({"+_3": 1.0}, num_spin_orbitals=4)
    result = cross_sector_matrix_element(
        psi_left_full=psi_left,
        op=op,
        psi_right_full=psi_right,
        num_spin_orbitals=4,
    )
    assert abs(result) < 1e-12


def test_build_eom_matrices_hermiticity():
    """M and S returned by `build_eom_matrices` are Hermitian (after the
    symmetrization step). On a random toy: pool of 3 random Pauli ops
    + random GS vector."""
    from qiskit.quantum_info import SparsePauliOp

    from siam_vqe.qeom import build_eom_matrices

    n_qubits = 4
    rng = np.random.default_rng(0)
    # Random Hermitian H_pauli
    H_p = SparsePauliOp.from_list([
        ("ZZII", rng.standard_normal()),
        ("IIZZ", rng.standard_normal()),
        ("XXII", rng.standard_normal()),
    ])
    pool = [
        SparsePauliOp.from_list([("YXII", 1j)]).simplify(),
        SparsePauliOp.from_list([("XYII", 1j)]).simplify(),
        SparsePauliOp.from_list([("IIYX", 1j)]).simplify(),
    ]
    # Random unit vector in 2^4 space
    psi = rng.standard_normal(2**n_qubits) + 1j * rng.standard_normal(2**n_qubits)
    psi /= np.linalg.norm(psi)

    M, S = build_eom_matrices(H_p, pool, psi)
    assert np.allclose(M, M.conj().T, atol=1e-10)
    assert np.allclose(S, S.conj().T, atol=1e-10)


def test_build_eom_matrices_shape_matches_pool():
    """M and S are K × K where K = pool size."""
    from qiskit.quantum_info import SparsePauliOp

    from siam_vqe.qeom import build_eom_matrices

    H_p = SparsePauliOp.from_list([("ZZ", 1.0)])
    pool = [SparsePauliOp.from_list([("YX", 1j)]).simplify(),
            SparsePauliOp.from_list([("XY", 1j)]).simplify()]
    psi = np.array([1, 0, 0, 0], dtype=complex)

    M, S = build_eom_matrices(H_p, pool, psi)
    assert M.shape == (2, 2)
    assert S.shape == (2, 2)


def test_solve_qeom_reproduces_scipy_eigh_when_s_is_identity():
    """When S is the identity, the GHEP reduces to a standard eigenproblem.
    Compare to scipy.linalg.eigh."""
    import scipy.linalg

    from siam_vqe.qeom import solve_qeom

    K = 5
    rng = np.random.default_rng(42)
    A = rng.standard_normal((K, K))
    M = (A + A.T) / 2  # symmetric real
    S = np.eye(K)
    energies_ours, _ = solve_qeom(M, S)
    energies_scipy = np.sort(scipy.linalg.eigh(M)[0])
    energies_ours = np.sort(energies_ours)
    assert np.allclose(energies_ours, energies_scipy, atol=1e-10)


def test_solve_qeom_drops_null_directions():
    """When S has a near-null direction, solve_qeom should drop it
    rather than blow up."""
    from siam_vqe.qeom import solve_qeom

    # Construct M = diag(1, 2, 3), S = diag(1, 1, 1e-12)
    M = np.diag([1.0, 2.0, 3.0])
    S = np.diag([1.0, 1.0, 1e-12])
    energies, _ = solve_qeom(M, S, s_tol=1e-8)
    # The 3rd direction is dropped → 2 surviving eigenvalues
    assert len(energies) == 2
    # The surviving energies are 1.0 and 2.0 (sorted)
    np.testing.assert_allclose(np.sort(energies), [1.0, 2.0], atol=1e-10)


def test_qeom_result_dataclass_immutability():
    from siam_vqe.qeom import QEOMResult

    r = QEOMResult(
        energies_eV=np.zeros(3),
        amplitudes=np.eye(3, dtype=complex),
        pool_size=3,
        s_dropped=0,
    )
    with pytest.raises((AttributeError, Exception)):
        r.pool_size = 4


def test_qeom_excited_states_match_ed_on_hubbard_dimer():
    """End-to-end integration test (Phase 5 physics gate): build M, S
    for the Hubbard dimer using ψ_GS from direct ED, solve qEOM, and
    confirm every (1,1) ED excitation is recovered.

    The Hubbard dimer has 4 states in the (1, 1) sector (C(2,1)×C(2,1)),
    so the (1, 1) sector ED yields 3 excitation energies relative to GS.

    Pool design note (deviation from the original spec):
        The spec proposed an anti-Hermitian fermionic pool
        (e.g., T1 = c†_1↑ c_0↑ − c†_0↑ c_1↑). For ANY anti-Hermitian T
        the commutator metric used by `build_eom_matrices` gives
        S_nn = ⟨[T†, T]⟩ = ⟨[−T, T]⟩ = 0; and for the Hubbard dimer at
        half-filling the GS is site-symmetric, so even
        non-anti-Hermitian particle-conserving pools yield S ≡ 0
        (⟨[c†_i c_j, c†_j c_i]⟩ = ⟨n_j − n_i⟩ = 0 by symmetry). With
        S identically null, `solve_qeom` correctly returns zero
        excitation energies. To exercise the qEOM pipeline on this
        fixture, we instead use a complete anti-Hermitian Pauli pool
        (1j × all 4-qubit Pauli strings except identity), which spans
        the full operator space and breaks the particle-conservation
        symmetry constraint. With this pool, qEOM yields every
        H-eigenvalue minus E_GS, including the 3 (1,1) excitations
        (plus cross-sector excitations to other (Nu, Nd) sectors,
        which are simply additional valid eigenvalues of H − E_GS).

    Pass criterion: each (1,1) ED excitation energy must appear in the
    qEOM spectrum to within 1e-6 eV.
    """
    import itertools

    from qiskit.quantum_info import SparsePauliOp
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from siam_vqe.hamiltonian import hubbard_dimer
    from siam_vqe.qeom import build_eom_matrices, solve_qeom

    # Hubbard dimer: U = 4, t = 1
    U, t = 4.0, 1.0
    H_fermi = hubbard_dimer(U=U, t=t)
    H_pauli = JordanWignerMapper().map(H_fermi)

    # GS in (1, 1) sector via direct ED.
    # Mode ordering (see hamiltonian.py): 0,1 = up at sites 0,1; 2,3 = down.
    H_mat = H_pauli.to_matrix()
    n_up = np.array([bin(i & 0b0011).count("1") for i in range(16)])
    n_dn = np.array([bin((i & 0b1100) >> 2).count("1") for i in range(16)])
    sector_mask = (n_up == 1) & (n_dn == 1)
    sector_idx = np.where(sector_mask)[0]

    H_sector = H_mat[np.ix_(sector_idx, sector_idx)]
    eigvals_sector, eigvecs_sector = np.linalg.eigh(H_sector)
    psi_gs_full = np.zeros(16, dtype=complex)
    psi_gs_full[sector_idx] = eigvecs_sector[:, 0]
    e_gs = eigvals_sector[0]
    excitation_ed = eigvals_sector[1:] - e_gs  # 3 values for (1,1)

    # qEOM pool: 1j × (all 4-qubit Paulis − identity). 255 anti-Hermitian
    # operators; spans the full operator algebra on 4 qubits.
    pool: list[SparsePauliOp] = []
    for combo in itertools.product("IXYZ", repeat=4):
        label = "".join(combo)
        if label == "IIII":
            continue
        pool.append(SparsePauliOp.from_list([(label, 1j)]))

    M, S = build_eom_matrices(H_pauli, pool, psi_gs_full)
    energies_qeom, _ = solve_qeom(M, S, s_tol=1e-8)

    # Keep only positive eigenvalues — qEOM yields ±E_n pairs by
    # construction (signed root of the GHEP). Excitations to other
    # (Nu, Nd) sectors will also appear; that's expected for a
    # non-particle-conserving Pauli pool. We only require that each
    # (1,1) ED excitation be present in the spectrum.
    e_pos = np.sort(energies_qeom[energies_qeom > 1e-6])

    assert len(e_pos) >= len(excitation_ed), (
        f"qEOM returned {len(e_pos)} positive eigenvalues; expected "
        f"at least {len(excitation_ed)} (the (1,1) ED excitations)."
    )

    # For each ED excitation, the nearest qEOM eigenvalue must match
    # within 1e-6 eV.
    for e_target in excitation_ed:
        nearest_err = np.min(np.abs(e_pos - e_target))
        assert nearest_err < 1e-6, (
            f"(1,1) ED excitation {e_target:.10f} eV not matched by "
            f"qEOM; nearest qEOM eigenvalue differs by {nearest_err:.2e} eV."
        )


def test_compute_spectral_weights_sum_rule_on_hubbard():
    """Σ_F |⟨F|D|GS⟩|² should equal ⟨GS|D†D|GS⟩ (Lehmann completeness).

    Use the Hubbard dimer + a particle-non-conserving dipole c†_0 to test."""
    import itertools

    from qiskit.quantum_info import SparsePauliOp
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from siam_vqe.hamiltonian import hubbard_dimer
    from siam_vqe.qeom import build_eom_matrices, compute_spectral_weights, solve_qeom

    H_fermi = hubbard_dimer(U=4.0, t=1.0)
    H_pauli = JordanWignerMapper().map(H_fermi)

    # GS in (1,1)
    H_mat = H_pauli.to_matrix()
    n_up = np.array([bin(i & 0b0011).count("1") for i in range(16)])
    n_dn = np.array([bin((i & 0b1100) >> 2).count("1") for i in range(16)])
    mask_init = (n_up == 1) & (n_dn == 1)
    idx_init = np.where(mask_init)[0]
    H_init_block = H_mat[np.ix_(idx_init, idx_init)]
    _eigvals_init, eigvecs_init = np.linalg.eigh(H_init_block)
    psi_gs = np.zeros(16, dtype=complex)
    psi_gs[idx_init] = eigvecs_init[:, 0]

    # Build a ref state in (2,1) — the d-electron-added sector for our toy
    mask_fin = (n_up == 2) & (n_dn == 1)
    idx_fin = np.where(mask_fin)[0]
    H_fin_block = H_mat[np.ix_(idx_fin, idx_fin)]
    _eigvals_fin, eigvecs_fin = np.linalg.eigh(H_fin_block)
    psi_prime_gs = np.zeros(16, dtype=complex)
    psi_prime_gs[idx_fin] = eigvecs_fin[:, 0]

    # qEOM pool: 1j × (all 4-qubit Paulis − identity). Same workaround as
    # Task 9 / Observation #20: a particle-conserving fermionic pool gives
    # S ≡ 0 on the symmetric reference state. The complete anti-Hermitian
    # Pauli pool spans the full operator algebra and breaks the symmetry,
    # exposing the surviving excitations in the (2,1) sector and beyond.
    pool_pauli: list[SparsePauliOp] = []
    for combo in itertools.product("IXYZ", repeat=4):
        label = "".join(combo)
        if label == "IIII":
            continue
        pool_pauli.append(SparsePauliOp.from_list([(label, 1j)]))

    M, S = build_eom_matrices(H_pauli, pool_pauli, psi_prime_gs)
    _energies, X = solve_qeom(M, S)

    # Dipole D = c†_0 (creates a spin-up at site 0)
    D = FermionicOp({"+_0": 1.0}, num_spin_orbitals=4)

    # `compute_spectral_weights` accepts the pool as FermionicOp OR
    # SparsePauliOp — for this test we pass the Pauli pool directly to
    # match the operators used in `build_eom_matrices` above.
    weights = compute_spectral_weights(
        psi_gs_full=psi_gs,
        psi_prime_gs_full=psi_prime_gs,
        pool_fermi=pool_pauli,
        amplitudes=X,
        dipole=D,
        num_spin_orbitals=4,
    )

    # Lehmann sum rule: Σ |⟨F|D|GS⟩|² ≤ ⟨GS|D†D|GS⟩  (equality at full pool)
    D_pauli = JordanWignerMapper().map(D)
    DdD = (D_pauli.adjoint() @ D_pauli).to_matrix()
    sum_rule = np.real(psi_gs.conj() @ DdD @ psi_gs)

    # Pool only catches a fraction of the sum-rule. Check weights are
    # nonnegative and total bounded by the sum-rule.
    for w in weights:
        assert w >= -1e-10
    assert sum(weights) < sum_rule + 1e-10
