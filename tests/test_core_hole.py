"""Tests for siam_vqe.core_hole."""
from __future__ import annotations

import pytest

from siam_vqe.core_hole import CoreHoleParams, v_core_operator
from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian


def test_core_hole_params_defaults():
    """EDRIXS example_3 default: U_dc = 8.5 eV."""
    ch = CoreHoleParams()
    assert ch.U_dc == pytest.approx(8.5)
    assert ch.include_multipoles is False


def test_v_core_operator_zero_when_udc_zero():
    """V_core(U_dc=0) is the zero operator."""
    params = L3Params()
    ch = CoreHoleParams(U_dc=0.0)
    V = v_core_operator(params, ch)
    # SparseLabelOp.simplify removes zero-coeff terms.
    assert len(V.simplify()) == 0


def test_v_core_operator_hermitian():
    """V_core is built from real-coefficient number operators → Hermitian."""
    params = L3Params()
    ch = CoreHoleParams(U_dc=8.5)
    V = v_core_operator(params, ch)
    diff = (V - V.adjoint()).simplify()
    assert len(diff) == 0


def test_v_core_operator_acts_only_on_d_orbitals():
    """V_core touches modes 0..4 (d↑) and 10..14 (d↓), not 5..9 or 15..19."""
    params = L3Params()
    ch = CoreHoleParams(U_dc=8.5)
    V = v_core_operator(params, ch)
    # Every label is a single number op +_m -_m with m ∈ d-modes.
    d_modes = set(range(params.num_d_orbitals)) | set(
        range(params.num_spin_orbitals // 2,
              params.num_spin_orbitals // 2 + params.num_d_orbitals)
    )
    for label, _ in V.terms():
        # label is a list[(op, mode)] like [('+', 3), ('-', 3)]
        modes_in_term = {mode for _, mode in label}
        assert modes_in_term <= d_modes, f"V_core touches non-d mode in {label}"


def test_v_core_operator_coefficient_is_negative_udc():
    """Each n_{d_α} number operator carries coefficient -U_dc."""
    params = L3Params()
    ch = CoreHoleParams(U_dc=8.5)
    V = v_core_operator(params, ch)
    # Expect 10 terms (5 d-orbitals × 2 spins), each with coefficient -8.5.
    coeffs = []
    for _, coeff in V.terms():
        coeffs.append(coeff)
    assert len(coeffs) == 10
    for c in coeffs:
        assert c == pytest.approx(-8.5)


def test_nio_l3_core_hole_equals_h_when_udc_zero():
    """With U_dc=0, H' should equal H exactly."""
    from siam_vqe.core_hole import nio_l3_core_hole_hamiltonian

    params = L3Params()
    ch = CoreHoleParams(U_dc=0.0)
    H = nio_l3_hamiltonian(params)
    H_prime = nio_l3_core_hole_hamiltonian(params, ch)
    diff = (H_prime - H).simplify()
    assert len(diff) == 0


def test_nio_l3_core_hole_lowers_d_full_energy_by_n_udc():
    """The atomic-limit-style gate: for V=0 hybridization, H' adds exactly
    -U_dc × n_d to every Fock basis vector. On a state with all 10 d-modes
    filled, that's -10 × U_dc."""
    from siam_vqe.core_hole import nio_l3_core_hole_hamiltonian

    params = L3Params()
    ch = CoreHoleParams(U_dc=8.5)

    # Build d-only-filled HF state: 10 d-modes (5 each spin), no bath
    # — this is NOT a physical state for our (9,9) sector, just a basis
    # state to probe V_core directly.
    H_prime = nio_l3_core_hole_hamiltonian(params, ch)
    H_bare = nio_l3_hamiltonian(params)
    diff = (H_prime - H_bare).simplify()
    # diff is exactly V_core; check that all coefficients sum to -10 × U_dc
    # when acting on the d-full state (i.e. trace over diagonal terms).
    total_diagonal_coeff = sum(c for _, c in diff.terms())
    assert total_diagonal_coeff == pytest.approx(-10 * ch.U_dc)


def test_tapered_h_prime_pauli_is_18_qubits_in_both_sectors():
    """H' on (10, 9) and (9, 10) should both reduce to 18-qubit
    SparsePauliOp under parity tapering (same as Phase 4 H on (9, 9))."""
    from siam_vqe.core_hole import tapered_l3_h_prime_pauli

    params = L3Params()
    ch = CoreHoleParams(U_dc=8.5)
    H_p_up = tapered_l3_h_prime_pauli(params, ch, num_particles=(10, 9))
    H_p_dn = tapered_l3_h_prime_pauli(params, ch, num_particles=(9, 10))
    assert H_p_up.num_qubits == 18
    assert H_p_dn.num_qubits == 18


def test_tapered_h_prime_groundstate_energy_matches_in_both_spin_sectors():
    """S_z-flip symmetry: ground energies in (10, 9) and (9, 10) match
    within 1 µeV (no SOC, no magnetic field)."""
    import scipy.sparse.linalg as spla

    from siam_vqe.core_hole import tapered_l3_h_prime_pauli

    params = L3Params()
    ch = CoreHoleParams(U_dc=8.5)
    # Use sparse=True: an 18-qubit dense matrix would be ~1 TB. Sparse form
    # is the project-wide convention (see adapt_vqe.py and test_adapt_vqe.py).
    H_up_mat = tapered_l3_h_prime_pauli(params, ch, num_particles=(10, 9)).to_matrix(sparse=True)
    H_dn_mat = tapered_l3_h_prime_pauli(params, ch, num_particles=(9, 10)).to_matrix(sparse=True)
    e0_up, _ = spla.eigsh(H_up_mat, k=1, which="SA")
    e0_dn, _ = spla.eigsh(H_dn_mat, k=1, which="SA")
    assert abs(e0_up[0] - e0_dn[0]) < 1e-6
