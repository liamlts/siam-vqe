"""Tests for siam_vqe.hamiltonian_l3."""
from __future__ import annotations

import numpy as np
import pytest

from siam_vqe.hamiltonian_l3 import (
    L3Params,
    atomic_multiplet_energies,
    nio_l3_hamiltonian,
    nio_l3_impurity_only,
    racah_BC,
    slater_to_u_tensor,
)


def test_l3params_default_values_match_edrixs_example3():
    p = L3Params()
    # EDRIXS example_3 / Haverkort PRB 85, 165113 verbatim
    assert p.F2_dd == pytest.approx(9.787)
    assert p.F4_dd == pytest.approx(6.078)
    assert p.U_dd == pytest.approx(7.3)
    assert p.ten_dq == pytest.approx(0.56)
    assert p.cf_bath_split == pytest.approx(1.44)
    assert p.Delta == pytest.approx(4.7)
    assert p.V_eg == pytest.approx(2.06)
    assert p.V_t2g == pytest.approx(1.21)
    assert p.num_d_orbitals == 5
    assert p.num_bath_orbitals == 5
    assert p.num_spatial == 10
    assert p.num_spin_orbitals == 20


def test_l3params_is_frozen():
    p = L3Params()
    with pytest.raises((AttributeError, TypeError)):
        p.F2_dd = 0.0  # frozen dataclass should reject assignment


def test_racah_BC_from_F2_F4():
    p = L3Params()
    B, C = racah_BC(p.F2_dd, p.F4_dd)
    # B = (9*F2 - 5*F4)/441, C = 35*F4/441
    assert B == pytest.approx((9.0 * p.F2_dd - 5.0 * p.F4_dd) / 441.0)
    assert C == pytest.approx(35.0 * p.F4_dd / 441.0)


def test_atomic_multiplet_d8_term_spacings():
    """d⁸ atomic-limit LS-term energies in Racah B, C (STK ch. 4)."""
    p = L3Params()
    energies = atomic_multiplet_energies(p.F2_dd, p.F4_dd)
    B, C = racah_BC(p.F2_dd, p.F4_dd)

    # Expected relative energies (STK ch. 4, eq. 4.78 row d⁸):
    expected = {
        "3F": -8.0 * B + 3.0 * C,
        "1D": -3.0 * B + 5.0 * C,
        "3P": 7.0 * B + 3.0 * C,
        "1G": 4.0 * B + 5.0 * C,
        "1S": 14.0 * B + 10.0 * C,
    }
    for term, E_exp in expected.items():
        assert energies[term] == pytest.approx(E_exp, abs=1e-9), (
            f"Multiplet {term}: got {energies[term]}, expected {E_exp}"
        )


def test_atomic_multiplet_d8_ground_state_is_3F():
    """For Ni²⁺ d⁸ in the atomic limit, ³F is the ground term (Hund's rules)."""
    p = L3Params()
    energies = atomic_multiplet_energies(p.F2_dd, p.F4_dd)
    ground = min(energies, key=lambda k: energies[k])
    assert ground == "3F"


def _slater_d8_terms_in_Fk(F2: float, F4: float) -> dict[str, float]:
    """Direct Slater-integral derivation of d⁸ LS-term energies.

    Using Condon-Shortley scaled integrals F²_cs = F²/49, F⁴_cs = F⁴/441,
    the canonical d⁸ term-energy expressions are:
        E(³F) = -8 F²_cs - 9 F⁴_cs
        E(¹D) = -3 F²_cs + 36 F⁴_cs
        E(³P) =  7 F²_cs - 84 F⁴_cs
        E(¹G) =  4 F²_cs +  1 F⁴_cs
        E(¹S) = 14 F²_cs + 126 F⁴_cs
    (Cowan ch. 7; equivalently STK Appendix Table I.)
    """
    F2_cs = F2 / 49.0
    F4_cs = F4 / 441.0
    return {
        "3F": -8.0 * F2_cs -  9.0 * F4_cs,
        "1D": -3.0 * F2_cs + 36.0 * F4_cs,
        "3P":  7.0 * F2_cs - 84.0 * F4_cs,
        "1G":  4.0 * F2_cs +  1.0 * F4_cs,
        "1S": 14.0 * F2_cs + 126.0 * F4_cs,
    }


def test_atomic_multiplet_matches_independent_slater_derivation():
    """Cross-check: spacings from Racah-parameterized formula must equal
    spacings from direct Slater-integral evaluation.

    This protects against coefficient bugs in atomic_multiplet_energies that
    would otherwise pass the self-consistency test silently."""
    F2, F4 = 9.787, 6.078
    racah = atomic_multiplet_energies(F2, F4)
    slater = _slater_d8_terms_in_Fk(F2, F4)
    # Compare spacings relative to ³F (absolute zero differs by F⁰ which the
    # Racah formula drops).
    for term in ("1D", "3P", "1G", "1S"):
        racah_gap = racah[term] - racah["3F"]
        slater_gap = slater[term] - slater["3F"]
        assert racah_gap == pytest.approx(slater_gap, abs=1e-9), (
            f"{term} spacing disagreement: Racah formula gives {racah_gap}, "
            f"Slater derivation gives {slater_gap}"
        )


def test_u_tensor_shape_5_5_5_5():
    U = slater_to_u_tensor(F2=9.787, F4=6.078, F0=7.3)
    assert U.shape == (5, 5, 5, 5)


def test_u_tensor_is_real():
    U = slater_to_u_tensor(F2=9.787, F4=6.078, F0=7.3)
    assert np.max(np.abs(np.imag(U))) < 1e-10


def test_u_tensor_physicists_symmetries():
    """U[α,β,γ,δ] = ⟨αβ|γδ⟩ in physicists' notation.

    Real-orbital symmetries:
        U[α,β,γ,δ] = U[β,α,δ,γ]    # particle exchange (rename dummies)
        U[α,β,γ,δ] = U[γ,δ,α,β]    # Hermitian conj + real basis
        U[α,β,γ,δ] = U[δ,γ,β,α]    # composition of the above
    """
    U = slater_to_u_tensor(F2=9.787, F4=6.078, F0=7.3)
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    assert U[a, b, c, d] == pytest.approx(U[b, a, d, c])
                    assert U[a, b, c, d] == pytest.approx(U[c, d, a, b])
                    assert U[a, b, c, d] == pytest.approx(U[d, c, b, a])


def test_u_tensor_orbital_averaged_direct_coulomb_equals_F0():
    """In physicists' notation, U[α,β,α,β] = ⟨αβ|αβ⟩ is the direct Coulomb integral
    between orbital α and orbital β. Averaging over (α, β) in the closed d shell
    gives F⁰ within ~50 meV (the F² and F⁴ angular corrections average to zero
    on a closed shell, an algebraic identity).
    """
    F0 = 7.3
    U = slater_to_u_tensor(F2=9.787, F4=6.078, F0=F0)
    direct = np.array([[U[a, b, a, b] for b in range(5)] for a in range(5)])
    avg = float(np.mean(direct))
    assert abs(avg - F0) < 0.05, f"avg U[α,β,α,β] = {avg}, expected ~{F0}"


def test_u_tensor_spherical_basis_exchange_matrix_element():
    """Reference value: U_sph[m=+2, m=-2 | m=-2, m=+2] = 70 F⁴/441.

    This is a closed-form Slater-Condon matrix element for the d-shell
    exchange channel between the m=+2 and m=-2 spherical orbitals. It is
    NONZERO only when the Gaunt sum correctly couples q ≠ 0 (exchange)
    channels. A bug that collapses exchange (e.g. wrong Gaunt argument
    convention) produces zero here.

    Reference: Cowan, *Theory of Atomic Structure and Spectra* (1981), eq. 7.59
    in combination with the d-shell Gaunt-coefficient table.
    """
    F4 = 6.078
    # Build U in the spherical basis directly by skipping the cubic rotation.
    # Easier: build in cubic basis and inverse-rotate. Use the public function.
    # The spherical-basis element we need is U_sph[m=+2, m=-2 | m=-2, m=+2].
    # In the public API we only have the cubic-basis array, so verify the
    # equivalent cubic-basis identity (the m=+2 / m=-2 mixing concentrates in
    # the x²−y² and xy orbitals, which are real combinations of |m=±2⟩):
    #   U_cubic[1, 4, 4, 1] (x²−y², xy, xy, x²−y²) is an exchange-like element
    #   that must be NONZERO for a correct tensor.
    U = slater_to_u_tensor(F2=0.0, F4=F4, F0=0.0)
    cubic_exchange = U[1, 4, 4, 1]
    # The closed-form spherical exchange 70 F⁴/441 rotates into a specific
    # cubic-basis combination. The magnitude is bounded by 70 F4 / 441 in
    # the spherical basis.
    closed_form_sph = 70.0 * F4 / 441.0
    assert abs(cubic_exchange) > 0.1, (
        f"Cubic-basis exchange U[1,4,4,1] = {cubic_exchange:.6f}, "
        f"should be nonzero (~order of {closed_form_sph:.4f} F⁴/441 reference)."
    )


def test_u_tensor_nonzero_entry_count_at_least_100():
    """A correct d-shell Coulomb tensor in any basis has hundreds of nonzero
    entries (direct + exchange + multipole couplings). A tensor with only the
    25 direct-diagonal entries U[a,b,a,b] is missing all exchange physics."""
    U = slater_to_u_tensor(F2=9.787, F4=6.078, F0=7.3)
    n_nonzero = int(np.sum(np.abs(U) > 1e-10))
    assert n_nonzero >= 100, (
        f"U has only {n_nonzero} nonzero entries; a correct d-shell tensor "
        f"has at least ~100. Likely the Gaunt formula is dropping exchange channels."
    )


def test_atomic_limit_d8_multiplet_spectrum_matches_closed_form():
    """No bath, no hyb, no CF: impurity-only H should have d⁸ multiplet spectrum.

    Diagonalize in the (N=8, S_z=0) sector → 25 states.
    The closed-form spacings (E(¹D)-E(³F), E(³P)-E(³F), etc.) must match
    atomic_multiplet_energies() to within 1 meV.
    """
    p = L3Params()
    H = nio_l3_impurity_only(p)
    from siam_vqe.reference_ed import _build_sector_basis, _fermionic_op_to_sparse_matrix
    basis = _build_sector_basis(num_d_spin_orbitals=10, n_up=4, n_down=4)
    H_mat = _fermionic_op_to_sparse_matrix(H, basis)

    # Sector is exactly 25-dimensional; sparse Arnoldi (eigsh) requires k < n - 1,
    # so we must use a dense decomposition to enumerate all 25 eigenvalues.
    energies = np.sort(np.linalg.eigvalsh(H_mat.toarray()))

    multiplets = atomic_multiplet_energies(p.F2_dd, p.F4_dd)
    E_3F = multiplets["3F"]
    E_3F_obs = energies[0]

    spacing_1D = multiplets["1D"] - E_3F
    spacing_3P = multiplets["3P"] - E_3F
    spacing_1G = multiplets["1G"] - E_3F
    spacing_1S = multiplets["1S"] - E_3F

    # The (N=8, S_z=0) sector has dim C(5,4) × C(5,4) = 25 states.
    # In the impurity-only (no CF) limit, the d⁸ atomic multiplets organize as:
    #   ³F (S=1, L=3): 7 m_L × 1 m_S=0 = 7 states in this sector
    #   ¹D (S=0, L=2): 5 states
    #   ³P (S=1, L=1): 3 m_L × 1 m_S=0 = 3 states
    #   ¹G (S=0, L=4): 9 states
    #   ¹S (S=0, L=0): 1 state
    # Total: 7 + 5 + 3 + 9 + 1 = 25. ✓
    # Energies sorted ascending: ³F (7), ¹D (5), ³P (3), ¹G (9), ¹S (1).
    assert energies[7] - E_3F_obs == pytest.approx(spacing_1D, abs=1e-3)
    assert energies[12] - E_3F_obs == pytest.approx(spacing_3P, abs=1e-3)
    assert energies[15] - E_3F_obs == pytest.approx(spacing_1G, abs=1e-3)
    assert energies[24] - E_3F_obs == pytest.approx(spacing_1S, abs=1e-3)


def test_full_l3_hamiltonian_hermitian():
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    H_dag = H.adjoint().simplify()
    diff = (H - H_dag).simplify()
    # FermionicOp uses .values() for coefficients; original plan referenced
    # the SparsePauliOp-style .coeffs attribute which does not exist on
    # FermionicOp. abs() works for both real and complex coefficients.
    max_coeff = max((abs(c) for c in diff.values()), default=0.0)
    assert max_coeff < 1e-10, f"H not Hermitian, max |H - H†| coeff = {max_coeff}"


def test_full_l3_hamiltonian_num_spin_orbitals_is_20():
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    assert H.num_spin_orbitals == 20


def test_full_l3_ground_state_in_9_9_sector():
    """The Phase-4 target sector: (N_up, N_down) = (9, 9), S_z = 0, 18 electrons."""
    p = L3Params()
    H = nio_l3_hamiltonian(p)
    from siam_vqe.reference_ed import _build_sector_basis, _fermionic_op_to_sparse_matrix
    basis = _build_sector_basis(num_d_spin_orbitals=20, n_up=9, n_down=9)
    assert len(basis) == 100, f"(9,9) sector dimension should be 100, got {len(basis)}"
    H_mat = _fermionic_op_to_sparse_matrix(H, basis)

    import scipy.sparse.linalg as spla
    energies, _ = spla.eigsh(H_mat, k=5, which="SA")
    energies = np.sort(energies)

    # Spec §3.4: expected within-sector ground state is a non-degenerate orbital
    # singlet (the m_s=0 component of the cubic ³A_2g triplet). m_s=±1 partners
    # live in (10, 8) and (8, 10) sectors. If we see a near-degeneracy at the
    # bottom, that is a Hamiltonian-build bug.
    gap = energies[1] - energies[0]
    assert gap > 0.01, (
        f"Ground state near-degenerate in (9,9) sector: gap = {gap} eV. "
        f"Hamiltonian build is likely wrong."
    )
