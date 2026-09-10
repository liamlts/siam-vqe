"""Tests for siam_vqe.adapt_vqe."""
from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.adapt_vqe import (
    AdaptConfig,
    AdaptResult,
    HFState,
    apply_exp_iT,
    build_l3_multistart_seeds,
    build_uccsd_pool,
    hartree_fock_initial_state,
    hf_state_to_tapered_statevector,
    pool_to_tapered_paulis,
    run_adapt_vqe,
    screen_gradients,
)
from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.tapering_l3 import tapered_l3_pauli


def test_build_uccsd_pool_returns_list_of_fermionic_ops():
    """All pool operators must be FermionicOp instances."""
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied,
                            virtual=virtual)
    assert all(isinstance(op, FermionicOp) for op in pool)


def test_build_uccsd_pool_for_l3_size_is_around_99():
    """L3 (9, 9) sector: 9 occ × 1 virt per spin × 2 spins = 18 singles;
    9 × 9 × 1 × 1 = 81 doubles mixed-spin; 0 doubles same-spin → ~99 total."""
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied,
                            virtual=virtual)
    assert 90 <= len(pool) <= 110, f"Pool size = {len(pool)}, expected ~99"


def test_pool_generators_are_anti_hermitian():
    """T = c†c − h.c. should satisfy T† = -T."""
    occupied = [0, 10]
    virtual = [1, 11]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied,
                            virtual=virtual)
    for T in pool:
        T_dag = T.adjoint().simplify()
        sum_op = (T + T_dag).simplify()
        max_coeff = max((abs(c) for c in sum_op.values()), default=0.0)
        assert max_coeff < 1e-10, f"T + T† has nonzero norm: {max_coeff}"


def test_all_pool_generators_nonzero_post_tapering():
    """F3 mitigation: every UCCSD generator must remain non-trivial after
    JW + parity + (N, S_z) tapering."""
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied,
                            virtual=virtual)
    paulis = pool_to_tapered_paulis(pool, num_particles=(9, 9))
    for i, P in enumerate(paulis):
        norm = float(np.sqrt(np.sum(np.abs(P.coeffs)**2)))
        assert norm > 1e-10, f"Pool generator {i} vanishes post-tapering"


def test_all_tapered_pool_generators_are_anti_hermitian():
    """Anti-Hermiticity: P† = -P (note pool_to_tapered_paulis multiplies by i,
    making each iT Hermitian, so we check P + P† = 0)."""
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied,
                            virtual=virtual)
    paulis = pool_to_tapered_paulis(pool, num_particles=(9, 9))
    for i, P in enumerate(paulis):
        # Each output Pauli is iT, which should be Hermitian: P = P†
        # Equivalently (P - P†) should have norm 0.
        P_diff = (P - P.adjoint()).simplify()
        max_coef = max(np.abs(P_diff.coeffs)) if len(P_diff.coeffs) else 0.0
        assert max_coef < 1e-10, f"Pool generator {i} not Hermitian after iT multiplication: {max_coef}"


def test_hf_state_is_in_target_sector():
    """For L3 (9, 9): HF determinant must have 9 up-spins and 9 down-spins."""
    hf = hartree_fock_initial_state(num_spin_orbitals=20, num_particles=(9, 9))
    occ = hf.occupied
    n_up = sum(1 for m in occ if m < 10)
    n_dn = sum(1 for m in occ if m >= 10)
    assert (n_up, n_dn) == (9, 9), f"HF state in sector {(n_up, n_dn)}, expected (9, 9)"


def test_hf_state_occupies_lowest_modes_by_default():
    hf = hartree_fock_initial_state(num_spin_orbitals=20, num_particles=(9, 9))
    expected_occ = set(range(9)) | set(range(10, 19))
    assert set(hf.occupied) == expected_occ
    assert isinstance(hf, HFState)


def test_hf_state_with_swap_perturbation_stays_in_sector():
    """Perturbations are occupation swaps within the (9, 9) sector."""
    hf = hartree_fock_initial_state(
        num_spin_orbitals=20, num_particles=(9, 9),
        swap_pairs=[(8, 9), (18, 19)],
    )
    occ = set(hf.occupied)
    n_up = sum(1 for m in occ if m < 10)
    n_dn = sum(1 for m in occ if m >= 10)
    assert (n_up, n_dn) == (9, 9)
    # Swapping (8, 9) and (18, 19) replaces modes 8 and 18 with 9 and 19
    assert 9 in occ and 8 not in occ
    assert 19 in occ and 18 not in occ


def test_hf_tapered_statevector_is_unit_normalized():
    hf = hartree_fock_initial_state(num_spin_orbitals=20, num_particles=(9, 9))
    psi = hf_state_to_tapered_statevector(hf)
    assert isinstance(psi, np.ndarray)
    assert psi.shape == (2**18,)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_hf_tapered_statevector_supports_only_one_basis_state():
    """An HF state is a single Slater determinant → exactly one nonzero
    entry in the bare JW basis. After tapering, still exactly one nonzero
    entry (since tapering is a unitary on the sector subspace)."""
    hf = hartree_fock_initial_state(num_spin_orbitals=20, num_particles=(9, 9))
    psi = hf_state_to_tapered_statevector(hf)
    nonzero = np.flatnonzero(np.abs(psi) > 1e-10)
    assert len(nonzero) == 1


def test_screen_gradients_returns_real_array():
    """⟨[H, iT_k]⟩ should be real-valued (anti-Hermiticity of iT)."""
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    H_pauli = tapered_l3_pauli(H, num_particles=(9, 9))
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied, virtual=virtual)
    pool_paulis = pool_to_tapered_paulis(pool, num_particles=(9, 9))
    hf = hartree_fock_initial_state()
    psi = hf_state_to_tapered_statevector(hf)
    grads = screen_gradients(H_pauli, pool_paulis, psi)
    assert isinstance(grads, np.ndarray)
    assert grads.shape == (len(pool_paulis),)
    assert grads.dtype.kind == "f", f"grads dtype {grads.dtype} not real"


def test_screen_gradients_zero_on_eigenstate():
    """If |ψ⟩ is an eigenstate of H, all gradients should vanish."""
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    H_pauli = tapered_l3_pauli(H, num_particles=(9, 9))
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied, virtual=virtual)
    pool_paulis = pool_to_tapered_paulis(pool, num_particles=(9, 9))

    # Take an eigenvector of H_pauli (the ground state of the tapered H).
    H_dense = H_pauli.to_matrix(sparse=True)
    import scipy.sparse.linalg as spla
    _, eigvecs = spla.eigsh(H_dense, k=1, which="SA")
    psi = eigvecs[:, 0]
    grads = screen_gradients(H_pauli, pool_paulis, psi)
    assert np.max(np.abs(grads)) < 1e-6, (
        f"Max gradient = {np.max(np.abs(grads))} on eigenstate; expected ~0."
    )


def test_apply_exp_iT_zero_param_is_identity():
    """exp(0 · iT) = I."""
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied, virtual=virtual)
    pool_paulis = pool_to_tapered_paulis(pool, num_particles=(9, 9))
    hf = hartree_fock_initial_state()
    psi_0 = hf_state_to_tapered_statevector(hf)
    psi_out = apply_exp_iT(psi_0, [pool_paulis[0]], [0.0])
    np.testing.assert_allclose(psi_out, psi_0)


def test_apply_exp_iT_preserves_norm():
    occupied = list(range(9)) + list(range(10, 19))
    virtual = [9, 19]
    pool = build_uccsd_pool(num_spin_orbitals=20, occupied=occupied, virtual=virtual)
    pool_paulis = pool_to_tapered_paulis(pool, num_particles=(9, 9))
    hf = hartree_fock_initial_state()
    psi_0 = hf_state_to_tapered_statevector(hf)
    psi_out = apply_exp_iT(psi_0, pool_paulis[:3], [0.1, 0.2, 0.3])
    assert abs(np.linalg.norm(psi_out) - 1.0) < 1e-10


def _h2_hamiltonian_4q() -> SparsePauliOp:
    """STO-3G H₂ Hamiltonian at equilibrium bond length, 4-qubit JW.

    Hardcoded coefficients from O'Malley et al. PRX 2016 / Romero et al. 2018,
    with the orbital ordering q0, q1 = (1s, 2s)-alpha and q2, q3 = (1s, 2s)-beta
    so that the HF determinant is |0011⟩ (both alpha orbitals occupied) and the
    ED ground state has the canonical |0011⟩ + small |1100⟩ doubles structure.
    """
    return SparsePauliOp.from_list([
        ("IIII", -0.81261),
        ("IIIZ",  0.17120),
        ("IIZI",  0.17120),
        ("IZII", -0.22790),
        ("ZIII", -0.22790),
        ("IIZZ",  0.16868),
        ("IZIZ",  0.12054),
        ("IZZI",  0.16567),
        ("ZIIZ",  0.16567),
        ("ZIZI",  0.12054),
        ("ZZII",  0.17434),
        ("XXXX",  0.04532),
        ("YYYY",  0.04532),
        ("XXYY", -0.04532),
        ("YYXX", -0.04532),
    ])


def test_run_adapt_vqe_converges_on_h2():
    """4-qubit H₂ ADAPT-VQE smoke test: should converge to ED ground state."""
    H = _h2_hamiltonian_4q()
    # Pool: build UCCSD pool for (1, 1) sector (2 electrons in 4 spin-orbitals),
    # then taper. For this 4-qubit operator (already parity-tapered), we can use
    # a hand-built 4-qubit Pauli pool instead — simpler for the smoke test:
    pool = [
        SparsePauliOp.from_list([("YXXX", 1.0)]),
        SparsePauliOp.from_list([("XYXX", 1.0)]),
        SparsePauliOp.from_list([("XXYX", 1.0)]),
        SparsePauliOp.from_list([("XXXY", 1.0)]),
    ]
    psi_0 = np.zeros(16, dtype=complex)
    psi_0[3] = 1.0  # HF state |0011⟩ (binary)

    config = AdaptConfig(
        gradient_threshold=1e-6,
        max_operators=10,
        inner_optimizer="cobyla",
        inner_max_iter=200,
    )
    result = run_adapt_vqe(H, pool, psi_0, config)

    # Reference: dense ED of the same 16x16 H.
    eigvals = np.linalg.eigvalsh(H.to_matrix())
    E_ref = float(eigvals[0])

    assert isinstance(result, AdaptResult)
    assert result.final_energy == pytest.approx(E_ref, abs=1e-3), (
        f"ADAPT energy {result.final_energy} vs ED {E_ref}"
    )
    assert len(result.operators_picked) <= 10
    assert result.converged_reason in {"gradient", "max_operators"}


def test_build_l3_multistart_seeds_returns_4_distinct_states():
    seeds = build_l3_multistart_seeds(num_seeds=4)
    assert len(seeds) == 4
    # All seeds should be in the (9, 9) sector.
    for hf in seeds:
        n_up = sum(1 for m in hf.occupied if m < 10)
        n_dn = sum(1 for m in hf.occupied if m >= 10)
        assert (n_up, n_dn) == (9, 9)
    # Seeds should differ from each other.
    occ_sets = [set(hf.occupied) for hf in seeds]
    for i, occ_i in enumerate(occ_sets):
        for j in range(i + 1, 4):
            assert occ_i != occ_sets[j], f"Seeds {i} and {j} are identical"


def test_build_l3_multistart_seeds_first_is_d8_hf():
    """Seed 0 must be the physical d⁸ ³A_2g HF (M_S=0 sublevel).

    With the L3 mode ordering (0-4 = d↑, 5-9 = bath↑, 10-14 = d↓, 15-19 = bath↓)
    and the EDRIXS example_3 charge-transfer parameters, the impurity d
    orbitals lie ~10 eV ABOVE the bath orbitals. Filling lowest-by-index modes
    (`hartree_fock_initial_state` default) lands in d¹⁰, not d⁸ — ~20 eV
    above the true ground state. Multistart from d¹⁰ cannot reach d⁸ because
    the (9, 9) sector has only one virtual per spin and ADAPT can't tunnel.
    """
    seeds = build_l3_multistart_seeds(num_seeds=4)
    expected = {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    assert set(seeds[0].occupied) == expected


def test_build_l3_multistart_seeds_seed0_has_d8_configuration():
    """Seed 0 must have ⟨n_d⟩ = 8 (impurity in d⁸) and ⟨n_bath⟩ = 10."""
    seeds = build_l3_multistart_seeds(num_seeds=1)
    seed0 = seeds[0]
    n_d = sum(1 for m in seed0.occupied if m < 5 or 10 <= m < 15)
    n_bath = sum(1 for m in seed0.occupied if 5 <= m < 10 or m >= 15)
    assert n_d == 8, f"Seed 0 must have 8 d-electrons (³A_2g), got {n_d}"
    assert n_bath == 10, f"Seed 0 must have 10 bath electrons, got {n_bath}"
