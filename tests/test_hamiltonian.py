"""Tests for siam_vqe.hamiltonian — Hubbard dimer construction."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian import hubbard_dimer, nio_l1_anderson


def _analytic_dimer_groundstate(U: float, t: float, eps: float, n_electrons: int) -> float:
    """Half-filled (n=2) Sz=0 analytic GS energy: eps*N + U/2 - sqrt((U/2)^2 + 4t^2)."""
    assert n_electrons == 2, "analytic formula assumes half-filling"
    return eps * n_electrons + U / 2 - math.sqrt((U / 2) ** 2 + 4 * t**2)


def test_hubbard_dimer_returns_fermionic_op(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    assert isinstance(fop, FermionicOp)
    assert fop.num_spin_orbitals == 4


def test_hubbard_dimer_groundstate_matches_analytic(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    mapper = JordanWignerMapper()
    pauli_op = mapper.map(fop)
    h_matrix = pauli_op.to_matrix()  # 16x16 dense

    # Build particle-number operator and project H into the N=2 sector.
    # The N=1 single-particle sector has lower global GS energy (-t) than the
    # N=2 half-filling GS when U is large; comparing eigvals[0] to the N=2
    # analytic formula is only valid within the N=2 subspace.
    n_op = FermionicOp(
        {"+_0 -_0": 1.0, "+_1 -_1": 1.0, "+_2 -_2": 1.0, "+_3 -_3": 1.0},
        num_spin_orbitals=4,
    )
    # n_mat is diagonal in JW basis by construction; extract diagonal directly.
    n_mat = mapper.map(n_op).to_matrix()
    n_diag = np.diag(n_mat).real
    n2_mask = np.abs(n_diag - 2.0) < 1e-8
    h_n2 = h_matrix[np.ix_(n2_mask, n2_mask)]
    e0_n2 = np.linalg.eigvalsh(h_n2)[0]

    e_vqe_target = _analytic_dimer_groundstate(
        dimer_params["U"], dimer_params["t"], dimer_params["eps"], n_electrons=2
    )
    assert e0_n2 == pytest.approx(e_vqe_target, abs=1e-10)


def test_hubbard_dimer_hermitian(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    mapper = JordanWignerMapper()
    h = mapper.map(fop).to_matrix()
    assert np.allclose(h, h.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# L1: NiO single-orbital Anderson impurity model
# ---------------------------------------------------------------------------


def test_nio_l1_anderson_returns_fermionic_op() -> None:
    fop = nio_l1_anderson(U=7.3, V=2.06, eps_d=1.0, eps_p=0.0)
    assert isinstance(fop, FermionicOp)
    assert fop.num_spin_orbitals == 4


def test_nio_l1_anderson_hermitian() -> None:
    fop = nio_l1_anderson(U=7.3, V=2.06, eps_d=1.0, eps_p=0.0)
    mapper = JordanWignerMapper()
    h = mapper.map(fop).to_matrix()
    assert np.allclose(h, h.conj().T, atol=1e-12)


def test_nio_l1_anderson_atomic_limit_v_zero() -> None:
    """At V=0, U>0, eps_d>eps_p: empty-d state at eps_p+eps_p (both bath), singlet at
    eps_d+eps_d+U, etc. The lowest state is 'both electrons on bath' = 2*eps_p."""
    fop = nio_l1_anderson(U=4.0, V=0.0, eps_d=2.0, eps_p=0.0)
    mapper = JordanWignerMapper()
    eigvals = np.linalg.eigvalsh(mapper.map(fop).to_matrix())
    assert eigvals[0] == pytest.approx(0.0, abs=1e-10)


def test_nio_l1_anderson_non_interacting_v_only() -> None:
    """At U=0, eps_d=eps_p=0: pure hopping. Single-particle eigenvalues are ±V.
    The 4-spin-orbital Hilbert space at half-filling has GS = 2*(-V) = -2V."""
    v_hop = 2.06
    fop = nio_l1_anderson(U=0.0, V=v_hop, eps_d=0.0, eps_p=0.0)
    mapper = JordanWignerMapper()
    h = mapper.map(fop).to_matrix()
    eigvals = np.linalg.eigvalsh(h)
    assert eigvals[0] == pytest.approx(-2 * v_hop, abs=1e-10)


def test_nio_l1_anderson_edrixs_hopping_sign() -> None:
    """EDRIXS uses hyb[bath, orb] = +V. siam_vqe must use coefficient +V
    on the +_{d} -_{p} term. A sign flip is invisible to the spectrum
    (gauge), so test the FermionicOp label dictionary directly."""
    v_hyb = 1.5
    fop = nio_l1_anderson(U=0.0, V=v_hyb, eps_d=0.0, eps_p=0.0)
    # FermionicOp exposes its terms via the simplify() + iteration; use the
    # dict-style access on the simplified form.
    simplified = fop.simplify()
    # Look up the +_0 -_1 term (d-up -> p-up hopping).
    # FermionicOp.items() yields (label, complex_coeff) pairs.
    terms = dict(simplified.items())
    assert "+_0 -_1" in terms, f"Expected '+_0 -_1' term; found labels: {list(terms.keys())}"
    coeff = complex(terms["+_0 -_1"])
    assert coeff.real == pytest.approx(v_hyb, abs=1e-12)
    assert coeff.imag == pytest.approx(0.0, abs=1e-12)


def test_nio_l1_anderson_half_filling_groundstate_known_values() -> None:
    """Pin a specific (U, V, eps_d, eps_p) parameter set so the GS at half-filling
    is captured for regression. Values from a one-time scipy.linalg.eigh run on
    the JW-mapped Hamiltonian at (U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)."""
    fop = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    mapper = JordanWignerMapper()
    eigvals = np.linalg.eigvalsh(mapper.map(fop).to_matrix())
    # Lowest eigenvalue in the half-filled sector is the global ground state for
    # these parameters (verified offline). Pin to 6 decimal places.
    # Computed via:  np.linalg.eigvalsh(h_matrix)[0] -> -5.687... (will be set
    # at implementation time after running the test once; for now leave as a
    # marker the implementer must fill in. The hermiticity + non-interacting
    # tests above are the strict gates.)
    e0 = eigvals[0]
    assert e0 == pytest.approx(-6.421966047226809, abs=1e-6)


# ---------------------------------------------------------------------------
# L2: NiO e_g² Anderson impurity model (Kanamori)
# ---------------------------------------------------------------------------


def test_nio_l2_kanamori_returns_fermionic_op() -> None:
    from siam_vqe.hamiltonian import nio_l2_kanamori
    fop = nio_l2_kanamori(
        U=8.6, U_prime=6.85, J_H=0.875,
        V=2.06, eps_d=1.0, eps_p=-1.0,
    )
    assert isinstance(fop, FermionicOp)
    assert fop.num_spin_orbitals == 8


def test_nio_l2_kanamori_hermitian() -> None:
    from siam_vqe.hamiltonian import nio_l2_kanamori
    fop = nio_l2_kanamori(
        U=8.6, U_prime=6.85, J_H=0.875,
        V=2.06, eps_d=1.0, eps_p=-1.0,
    )
    h = JordanWignerMapper().map(fop).to_matrix()
    assert np.allclose(h, h.conj().T, atol=1e-12)


def test_nio_l2_kanamori_decouples_at_V_zero_J_zero_Uprime_equals_U() -> None:
    """At V=0, J_H=0, U=U': two independent single-orbital Anderson models
    stacked. Half-filled GS = 4*eps_p (both bath orbitals filled per spin)."""
    from siam_vqe.hamiltonian import nio_l2_kanamori
    fop = nio_l2_kanamori(
        U=4.0, U_prime=4.0, J_H=0.0,
        V=0.0, eps_d=2.0, eps_p=-1.0,
    )
    eigvals = np.linalg.eigvalsh(JordanWignerMapper().map(fop).to_matrix())
    # GS: 4 electrons all on bath (modes 2,3,6,7) at eps_p each = 4*(-1) = -4.
    assert eigvals[0] == pytest.approx(-4.0, abs=1e-10)


def test_nio_l2_kanamori_hund_atomic_limit() -> None:
    """At V=0, U=U'=0, J_H>0, eps_p large positive (bath unoccupiable):
    the 2-impurity-electron sector GS is the high-spin triplet with energy
    2*eps_d - J_H. Projected into the N_imp=2 subspace to isolate the triplet
    from competing sectors (vacuum E=0 and 4-electron E=4*eps_d-2*J_H)."""
    from siam_vqe.hamiltonian import nio_l2_kanamori
    J_H = 0.5
    eps_d = 1.0
    fop = nio_l2_kanamori(
        U=0.0, U_prime=0.0, J_H=J_H,
        V=0.0, eps_d=eps_d, eps_p=20.0,  # bath effectively unoccupiable
    )
    mapper = JordanWignerMapper()
    h = mapper.map(fop).to_matrix()
    # Project to N_imp=2 sector (impurity modes 0,1,4,5).
    n_imp_op = FermionicOp(
        {"+_0 -_0": 1.0, "+_1 -_1": 1.0, "+_4 -_4": 1.0, "+_5 -_5": 1.0},
        num_spin_orbitals=8,
    )
    n_imp_mat = mapper.map(n_imp_op).to_matrix()
    n_imp_diag = np.diag(n_imp_mat).real
    mask = np.abs(n_imp_diag - 2.0) < 1e-8
    h_n2 = h[np.ix_(mask, mask)]
    eigvals_n2 = np.linalg.eigvalsh(h_n2)
    # Triplet GS: <S²>=2, energy = 2*eps_d - J_H = 2*1.0 - 0.5 = 1.5
    assert eigvals_n2[0] == pytest.approx(2 * eps_d - J_H, abs=1e-8)


def test_nio_l2_kanamori_diagonal_hyb_at_J_zero() -> None:
    """At J_H=0, U=U'=0, eps_d=eps_p=0: two independent diagonal-hyb Anderson
    dimers. Single-particle eigenvalues per dimer: ±V. Half-filled GS = -2V
    per dimer × 2 = -4V."""
    from siam_vqe.hamiltonian import nio_l2_kanamori
    V = 2.06
    fop = nio_l2_kanamori(
        U=0.0, U_prime=0.0, J_H=0.0,
        V=V, eps_d=0.0, eps_p=0.0,
    )
    eigvals = np.linalg.eigvalsh(JordanWignerMapper().map(fop).to_matrix())
    assert eigvals[0] == pytest.approx(-4 * V, abs=1e-8)
